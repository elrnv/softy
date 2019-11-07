use super::*;
use crate::constraint::*;
use crate::contact::*;
use crate::fem::problem::Var;
use crate::friction::*;
use crate::matrix::*;
use crate::Error;
use crate::Index;
use crate::TriMesh;
use geo::mesh::topology::*;
use geo::mesh::{Attrib, VertexPositions};
use implicits::*;
use lazycell::LazyCell;
#[cfg(feature = "af")]
use reinterpret::*;
use std::cell::RefCell;
use utils::soap::*;
use utils::soap::{Matrix3, Vector2, Vector3};
use utils::zip;

/// Enforce a contact constraint on a mesh against animated vertices. This constraint prevents
/// vertices from occupying the same space as a smooth representation of the simulation mesh.
/// This is the linearized version of `PointContactConstraint`.
#[derive(Clone, Debug)]
pub struct LinearizedPointContactConstraint {
    /// Implicit surface that represents the deforming object.
    pub implicit_surface: QueryTopo,
    /// Points where collision and contact occurs. I.e. all surface vertex positions on the
    /// collider mesh.
    pub contact_points: RefCell<Chunked3<Vec<f64>>>,

    /// Friction impulses applied during contact.
    pub frictional_contact: Option<FrictionalContact>,
    /// A mass inverse for each vertex in the object mesh.
    /// If the object is fixed, masses are effectively zero and this field
    /// will be `None`.
    pub object_mass_inv: Option<Chunked3<Vec<f64>>>,
    /// A mass inverse for each vertex in the collider mesh.
    /// If the collider is fixed, masses are effectively zero and this field
    /// will be `None`.
    pub collider_mass_inv: Option<Chunked3<Vec<f64>>>,

    /// A flag indicating if the object is fixed. Otherwise it's considered
    /// to be deforming, and thus appropriate derivatives are computed.
    object_is_fixed: bool,
    /// A flag indicating if the collider is fixed. Otherwise it's considered
    /// to be deforming, and thus appropriate derivatives are computed.
    collider_is_fixed: bool,

    /// Internal constraint function buffer.
    constraint_value: Vec<f64>,

    /// Constraint Jacobian in two blocks: first for object Jacobian and second for collider
    /// Jacobian. If one is fixed, it will be `None`.
    constraint_jacobian: LazyCell<[Option<DSBlockMatrix1x3>; 2]>,
}

impl LinearizedPointContactConstraint {
    pub fn new(
        // Main object experiencing contact against its implicit surface representation.
        object: Var<&TriMesh>,
        // Collision object consisting of points pushing against the solid object.
        collider: Var<&TriMesh>,
        kernel: KernelType,
        friction_params: Option<FrictionParams>,
    ) -> Result<Self, Error> {
        let mut surface_builder = ImplicitSurfaceBuilder::new();
        let object_is_fixed = object.is_fixed();
        let collider_is_fixed = collider.is_fixed();
        let object = object.untag();
        let collider = collider.untag();
        surface_builder
            .trimesh(object)
            .kernel(kernel)
            .sample_type(SampleType::Face)
            .background_field(BackgroundFieldParams {
                field_type: BackgroundFieldType::DistanceBased,
                weighted: false,
            });

        if let Some(surface) = surface_builder.build_mls() {
            // Sanity check that the surface is built correctly.
            assert_eq!(
                surface.surface_vertex_positions().len(),
                object.num_vertices()
            );

            let query_points = collider.vertex_positions();

            let object_mass_inv = Self::mass_inv_attribute(&object)?;

            let collider_mass_inv = Self::mass_inv_attribute(&collider)?;

            let mut constraint = LinearizedPointContactConstraint {
                implicit_surface: surface.query_topo(query_points),
                contact_points: RefCell::new(Chunked3::from_array_vec(query_points.to_vec())),
                frictional_contact: friction_params.and_then(|fparams| {
                    if fparams.dynamic_friction > 0.0 {
                        Some(FrictionalContact::new(fparams))
                    } else {
                        None
                    }
                }),
                object_mass_inv,
                collider_mass_inv,
                object_is_fixed,
                collider_is_fixed,
                constraint_value: Vec::new(),
                constraint_jacobian: LazyCell::new(),
            };

            constraint.linearize_constraint(
                Chunked3::from_array_slice(object.vertex_positions()).into(),
                Chunked3::from_array_slice(query_points).into(),
            );

            Ok(constraint)
        } else {
            Err(Error::InvalidImplicitSurface)
        }
    }

    fn mass_inv_attribute(mesh: &TriMesh) -> Result<Option<Chunked3<Vec<f64>>>, Error> {
        match mesh.attrib_as_slice::<MassType, VertexIndex>(MASS_ATTRIB) {
            Ok(attrib) => {
                if !attrib.iter().all(|&x| x > 0.0) {
                    Err(Error::InvalidParameter {
                        name: "Zero mass".to_string(),
                    })
                } else {
                    Ok(Some(attrib.iter().map(|&x| [1.0 / x; 3]).collect()))
                }
            }
            _ => Ok(None),
        }
    }

    fn build_constraint_jacobian(
        &mut self,
        pos: [SubsetView<Chunked3<&[f64]>>; 2],
    ) -> [Option<DSBlockMatrix1x3>; 2] {
        self.update_surface_with_mesh_pos(pos[0]);
        self.update_contact_points(pos[1]);
        let contact_points = self.contact_points.borrow();

        let surf = &self.implicit_surface;
        let neighbourhood_indices = neighbourhood_indices(&surf);
        let row_correction = |((row, col), val)| {
            let idx: Index = neighbourhood_indices[row];
            assert!(idx.is_valid());
            (idx.unwrap(), col, [val])
        };

        let num_rows = surf.num_neighbourhoods();

        let obj_jac = if !self.object_is_fixed {
            let num_cols = surf.surface_vertex_positions().len();
            let iter = surf
                .surface_jacobian_block_indices_iter()
                .zip(
                    self.implicit_surface
                        .surface_jacobian_block_iter(contact_points.view().into()),
                )
                .map(row_correction);
            Some(DSBlockMatrix1x3::from_block_triplets_iter(
                iter, num_rows, num_cols,
            ))
        } else {
            None
        };

        let coll_jac = if !self.collider_is_fixed {
            let num_cols = contact_points.len();
            let iter = surf
                .query_jacobian_block_indices_iter()
                .zip(
                    self.implicit_surface
                        .query_jacobian_block_iter(contact_points.view().into()),
                )
                .map(row_correction);
            Some(DSBlockMatrix1x3::from_block_triplets_iter(
                iter, num_rows, num_cols,
            ))
        } else {
            None
        };

        [obj_jac, coll_jac]
    }

    /// Update implicit surface using the given position data from mesh vertices.
    /// Return the number of positions that were actually updated.
    pub fn update_surface_with_mesh_pos(&mut self, pos: SubsetView<Chunked3<&[f64]>>) -> usize {
        self.implicit_surface.update_surface(pos.iter().cloned())
    }

    pub fn update_contact_points(&mut self, x: SubsetView<Chunked3<&[f64]>>) {
        let mut contact_points = self.contact_points.borrow_mut();
        x.clone_into_other(&mut *contact_points);
    }

    /// Prune contacts with zero contact_impulse and contacts without neighbouring samples.
    /// This function outputs the indices of contacts as well as a pruned vector of impulses.
    fn in_contact_indices(&self, contact_impulse: &[f64]) -> (Vec<usize>, Vec<usize>, Vec<f64>) {
        let surf = &self.implicit_surface;
        let query_points = self.contact_points.borrow();
        let radius = surf.radius() * 0.999;
        let query_indices = self.active_constraint_indices();
        assert_eq!(query_indices.len(), contact_impulse.len());
        let (active_constraint_subset, contact_impulse): (Vec<_>, Vec<_>) = contact_impulse
            .iter()
            .enumerate()
            .filter_map(|(i, &cf)| {
                if cf != 0.0
                    && surf.num_neighbours_within_distance(query_points[query_indices[i]], radius)
                        > 0
                {
                    Some((i, cf))
                } else {
                    None
                }
            })
            .unzip();

        let active_contact_indices: Vec<_> = active_constraint_subset
            .iter()
            .map(|&i| query_indices[i])
            .collect();

        (
            active_constraint_subset,
            active_contact_indices,
            contact_impulse,
        )
    }

    fn compute_contact_jacobian(&self, active_contact_indices: &[usize]) -> ContactJacobian {
        let query_points = self.contact_points.borrow();
        let surf = &self.implicit_surface;
        let active_contact_points =
            Subset::from_unique_ordered_indices(active_contact_indices, query_points.view());

        // Compute contact Jacobian
        let jac_triplets =
            build_triplet_contact_jacobian(&surf, active_contact_points, query_points.view());
        let jac: ContactJacobian = jac_triplets.into();
        let jac = jac.pruned(|_, _, block| block.into_inner() != [[0.0; 3]; 3]);
        //jac.write_img("./out/jac.png");
        jac
    }

    fn compute_effective_mass_inv(
        &self,
        active_contact_indices: &[usize],
        jac: ContactJacobianView,
    ) -> EffectiveMassInv {
        // Construct diagonal mass matrices for object and collider.
        let object_zero_mass_inv = Chunked3::from_array_vec(vec![[0.0; 3]; jac.num_cols()]);
        let collider_zero_mass_inv = Chunked3::from_array_vec(vec![[0.0; 3]; jac.num_rows()]);

        let object_mass_inv = DiagonalBlockMatrix::from_uniform(
            self.object_mass_inv
                .as_ref()
                .map(|mass_inv| mass_inv.view())
                .unwrap_or_else(|| object_zero_mass_inv.view()),
        );

        // Collider mass matrix is constructed at active contacts only.
        let collider_mass_inv = DiagonalBlockMatrix::from_subset(
            self.collider_mass_inv
                .as_ref()
                .map(|mass_inv| {
                    Subset::from_unique_ordered_indices(active_contact_indices, mass_inv.view())
                })
                .unwrap_or_else(|| Subset::all(collider_zero_mass_inv.view())),
        );

        let mut jac_mass = Tensor::new(jac.data.clone().into_owned());
        jac_mass *= object_mass_inv.view();

        //jac_mass.write_img("./out/jac_mass.png");

        let effective_mass_inv = jac_mass.view() * jac.view().transpose();
        let effective_mass_inv = effective_mass_inv.view() + collider_mass_inv.view();

        //effective_mass_inv.write_img("./out/effective_mass_inv.png");
        effective_mass_inv
    }

    fn compute_predictor_impulse(
        v: [SubsetView<Chunked3<&[f64]>>; 2],
        active_contact_indices: &[usize],
        jac: ContactJacobianView,
        effective_mass_inv: EffectiveMassInvView,
    ) -> Chunked3<Vec<f64>> {
        let collider_velocity = Subset::from_unique_ordered_indices(active_contact_indices, v[1]);

        let mut object_velocity = jac.view() * Tensor::new(v[0]);
        let mut relative_velocity = object_velocity.view_mut();
        relative_velocity -= Tensor::new(collider_velocity.view());

        let sprs_effective_mass_inv: sprs::CsMat<f64> = effective_mass_inv.clone().into();
        //        let ldlt_solver =
        //            sprs_ldl::LdlNumeric::<f64, usize>::new(sprs_effective_mass_inv.view()).unwrap();
        //        let predictor_impulse = Chunked3::from_flat(ldlt_solver.solve(rhs.storage()));

        // Prepare velocity vector for llt solve by stripping velocity grouping info.
        let mut rhs = relative_velocity.storage().to_vec();

        if !rhs.is_empty() {
            sprs::linalg::trisolve::lsolve_csr_dense_rhs(sprs_effective_mass_inv.view(), &mut rhs)
                .unwrap();
        }

        // The solve turns our relative velocity into a relative impulse.
        Chunked3::from_flat(rhs)
    }
}

fn neighbourhood_indices(surf: &QueryTopo) -> Vec<Index> {
    let mut neighbourhood_indices = vec![Index::INVALID; surf.num_query_points()];

    let neighbourhood_sizes = surf.neighbourhood_sizes();

    for (i, (idx, _)) in neighbourhood_indices
        .iter_mut()
        .zip(neighbourhood_sizes.iter())
        .filter(|&(_, &s)| s != 0)
        .enumerate()
    {
        *idx = Index::new(i);
    }

    neighbourhood_indices
}

impl ContactConstraint for LinearizedPointContactConstraint {
    // Get the total number of contacts that could potentially occur.
    fn num_potential_contacts(&self) -> usize {
        self.contact_points.borrow().len()
    }
    fn frictional_contact(&self) -> Option<&FrictionalContact> {
        self.frictional_contact.as_ref()
    }
    fn frictional_contact_mut(&mut self) -> Option<&mut FrictionalContact> {
        self.frictional_contact.as_mut()
    }
    fn active_surface_vertex_indices(&self) -> ARef<'_, [usize]> {
        ARef::Plain(&[])
    }

    #[cfg(feature = "af")]
    fn contact_jacobian_af(&self) -> af::Array<f64> {
        // Compute contact jacobian
        let surf = &self.implicit_surface;
        let query_points = self.contact_points.borrow();

        let mut cj_values = vec![0.0; surf.num_contact_jacobian_entries()];
        surf.contact_jacobian_values(
            query_points.view().into(),
            reinterpret_mut_slice(&mut cj_values),
        );
        let cj_indices_iter = surf.contact_jacobian_indices_iter();

        let nnz = self.constraint_jacobian_size();
        let mut rows = vec![0i32; nnz];
        let mut cols = vec![0i32; nnz];

        for ((row, col), (r, c)) in cj_indices_iter.zip(rows.iter_mut().zip(cols.iter_mut())) {
            *r = row as i32;
            *c = col as i32;
        }

        // Build ArrayFire matrix
        let nnz = nnz as u64;
        let num_rows = 3 * query_points.len() as u64;
        let num_cols = 3 * surf.surface_vertex_positions().len() as u64;

        let values = af::Array::new(&cj_values, af::Dim4::new(&[nnz, 1, 1, 1]));
        let row_indices = af::Array::new(&rows, af::Dim4::new(&[nnz, 1, 1, 1]));
        let col_indices = af::Array::new(&cols, af::Dim4::new(&[nnz, 1, 1, 1]));

        af::sparse(
            num_rows,
            num_cols,
            &values,
            &row_indices,
            &col_indices,
            af::SparseFormat::COO,
        )
    }

    fn update_frictional_contact_impulse(
        &mut self,
        orig_contact_impulse_n: &[f64],
        x: [SubsetView<Chunked3<&[f64]>>; 2],
        v: [SubsetView<Chunked3<&[f64]>>; 2],
        _potential_values: &[f64],
        mut friction_steps: u32,
    ) -> u32 {
        if self.frictional_contact.is_none() || friction_steps == 0 {
            return 0;
        }

        self.update_surface_with_mesh_pos(x[0]);
        self.update_contact_points(x[1]);

        // Note that there is a distinction between active *contacts* and active *constraints*.
        // Active *constraints* correspond to to those points that are in the MLS neighbourhood of
        // influence to be part of the optimization. Active *contacts* are a subset of those that
        // are considered in contact.
        let (active_constraint_subset, active_contact_indices, orig_contact_impulse_n) =
            self.in_contact_indices(orig_contact_impulse_n);
        let normals = self.contact_normals();
        let normals_subset = Subset::from_unique_ordered_indices(active_constraint_subset, normals);
        let mut normals = Chunked3::from_array_vec(vec![[0.0; 3]; normals_subset.len()]);
        normals_subset.clone_into_other(&mut normals);

        self.frictional_contact
            .as_mut()
            .unwrap()
            .contact_basis
            .update_from_normals(normals.into());

        let jac = self.compute_contact_jacobian(&active_contact_indices);
        let effective_mass_inv =
            self.compute_effective_mass_inv(&active_contact_indices, jac.view());

        let FrictionalContact {
            contact_basis,
            params,
            object_impulse,
            collider_impulse, // for active point contacts
        } = self.frictional_contact.as_mut().unwrap();

        // A new set of contacts have been determined. We should remap the previous friction
        // impulses to match new impulses.
        let mut prev_friction_impulse: Chunked3<Vec<f64>> = crate::constraints::remap_values_iter(
            collider_impulse.source_iter().map(|x| *x.1),
            [0.0; 3], // Previous impulse for unmatched contacts.
            collider_impulse.selection().index_iter().cloned(),
            active_contact_indices.iter().cloned(),
        )
        .collect();

        // Initialize the new friction impulse in physical space at active contacts.
        let mut friction_impulse =
            Chunked3::from_array_vec(vec![[0.0; 3]; active_contact_indices.len()]);

        if active_contact_indices.is_empty() {
            // If there are no active contacts, there is nothing to update.
            // Clear object_impulse before returning.
            object_impulse.iter_mut().for_each(|(x, y)| {
                *x = [0.0; 3];
                *y = [0.0; 3]
            });
            return 0;
        }

        let mut contact_impulse: Chunked3<Vec<f64>> = contact_basis
            .from_normal_space(&orig_contact_impulse_n)
            .collect();
        // Prepare true predictor for the friction solve.
        let predictor_impulse = Self::compute_predictor_impulse(
            v,
            &active_contact_indices,
            jac.view(),
            effective_mass_inv.view(),
        );
        // Project out the normal component.
        let predictor_impulse: Vec<_> = contact_basis
            .to_tangent_space(&predictor_impulse.into_arrays())
            .collect();
        let predictor_impulse: Chunked3<Vec<_>> = contact_basis
            .from_tangent_space(&predictor_impulse)
            .collect();

        let predictor_impulse: Chunked3<Vec<f64>> =
            (predictor_impulse.expr() + contact_impulse.expr() + prev_friction_impulse.expr())
                .eval();
        let success = if false {
            // Polar coords
            let predictor_impulse_t: Vec<_> = predictor_impulse
                .iter()
                .enumerate()
                .map(|(aqi, &predictor_imp)| {
                    let r = contact_basis.to_cylindrical_contact_coordinates(predictor_imp, aqi);
                    r.tangent
                })
                .collect();
            if true {
                // switch between implicit solver and explicit solver here.
                match FrictionPolarSolver::new(
                    &predictor_impulse_t,
                    &orig_contact_impulse_n,
                    &contact_basis,
                    effective_mass_inv.view(),
                    *params,
                    jac.view(),
                ) {
                    Ok(mut solver) => {
                        eprintln!("#### Solving Friction");
                        if let Ok(FrictionSolveResult { solution: r_t, .. }) = solver.step() {
                            for ((aqi, &r), r_out) in
                                r_t.iter().enumerate().zip(friction_impulse.iter_mut())
                            {
                                let r_polar = Polar2 {
                                    radius: r[0],
                                    angle: r[1],
                                };
                                *r_out = contact_basis
                                    .from_cylindrical_contact_coordinates(r_polar.into(), aqi)
                                    .into();
                            }
                            true
                        } else {
                            eprintln!("Failed friction solve");
                            false
                        }
                    }
                    Err(err) => {
                        dbg!(err);
                        false
                    }
                }
            } else {
                for (aqi, (&pred_r_t, &cr, r_out)) in zip!(
                    predictor_impulse_t.iter(),
                    orig_contact_impulse_n.iter(),
                    friction_impulse.iter_mut()
                )
                .enumerate()
                {
                    let r_t = if pred_r_t.radius > 0.0 {
                        Polar2 {
                            radius: params.dynamic_friction * cr.abs(),
                            angle: negate_angle(pred_r_t.angle),
                        }
                    } else {
                        Polar2 {
                            radius: 0.0,
                            angle: 0.0,
                        }
                    };
                    *r_out = contact_basis
                        .from_cylindrical_contact_coordinates(r_t.into(), aqi)
                        .into();
                }
                true
            }
        } else {
            let mut contact_impulse_n = orig_contact_impulse_n.clone();
            let prev_friction_impulse_t: Vec<_> = contact_basis
                .to_tangent_space(&prev_friction_impulse.view().into_arrays())
                .collect();

            //TODO: undo tmp change
            let prev_friction_impulse_t = vec![[0.0; 2]; prev_friction_impulse_t.len()];

            // Euclidean coords
            if true {
                // Switch between implicit solver and explicit solver here.
                loop {
                    //println!("predictor: {:?}", predictor_impulse.view());
                    let friction_predictor: Chunked3<Vec<f64>> =
                        (predictor_impulse.expr() - contact_impulse.expr()).eval();
                    //println!("f_predictor: {:?}", friction_predictor.view());
                    match crate::friction::solver::FrictionSolver::new(
                        friction_predictor.view().into(),
                        &prev_friction_impulse_t,
                        &contact_impulse_n,
                        &contact_basis,
                        effective_mass_inv.view(),
                        *params,
                    ) {
                        Ok(mut solver) => {
                            eprintln!("#### Solving Friction");

                            if let Ok(FrictionSolveResult { solution: r_t, .. }) = solver.step() {
                                friction_impulse = contact_basis.from_tangent_space(&r_t).collect();
                            } else {
                                eprintln!("Failed friction solve");
                                break false;
                            }
                        }
                        Err(err) => {
                            eprintln!("Failed to create friction solver: {:?}", err);
                            break false;
                        }
                    }

                    //println!("c_before: {:?}", contact_impulse_n);
                    let contact_predictor: Chunked3<Vec<f64>> =
                        (predictor_impulse.expr() - friction_impulse.expr()).eval();
                    //println!("c_predictor: {:?}", contact_predictor.view());

                    let contact_impulse_n_copy = contact_impulse_n.clone();
                    match crate::friction::contact_solver::ContactSolver::new(
                        contact_predictor.view().into(),
                        &contact_impulse_n_copy,
                        &contact_basis,
                        effective_mass_inv.view(),
                        *params,
                    ) {
                        Ok(mut solver) => {
                            eprintln!("#### Solving Contact");

                            if let Ok(r_n) = solver.step() {
                                contact_impulse_n.copy_from_slice(&r_n);
                                contact_impulse.clear();
                                contact_impulse
                                    .extend(contact_basis.from_normal_space(&contact_impulse_n));
                            } else {
                                eprintln!("Failed contact solve");
                                break false;
                            }
                        }
                        Err(err) => {
                            eprintln!("Failed to create contact solver: {:?}", err);
                            break false;
                        }
                    }

                    //println!("c_after: {:?}", contact_impulse_n);
                    //println!("c_after_full: {:?}", contact_impulse.view());

                    let f_prev = Tensor::new(prev_friction_impulse.view());
                    let f_cur = Tensor::new(friction_impulse.view());
                    //println!("prev friction impulse: {:?}", f_prev.norm());
                    //println!("cur friction impulse: {:?}", f_cur.norm());

                    let f_delta = f_prev - f_cur;
                    let rel_err_numerator = f_delta
                        .expr()
                        .dot((effective_mass_inv.view() * f_delta.view()).expr());
                    let rel_err = rel_err_numerator
                        / f_prev
                            .expr()
                            .dot((effective_mass_inv.view() * f_prev.view()).expr());

                    dbg!(rel_err);
                    if rel_err < 1e-3 {
                        friction_steps = 0;
                        break true;
                    }

                    // Update prev_friction_impulse for computing error subsequent iterations.
                    // Note that this should not and does not affect the "prev_friction_impulse_t"
                    // variable which is used in friciton forwarding and set outside the loop.
                    prev_friction_impulse = friction_impulse.clone();

                    friction_steps -= 1;

                    if friction_steps == 0 {
                        break true;
                    }
                }
            } else {
                let predictor_impulse_t: Vec<_> = contact_basis
                    .to_tangent_space(predictor_impulse.view().into())
                    .collect();
                for (aqi, (&pred_r_t, &cr, r_out)) in zip!(
                    predictor_impulse_t.iter(),
                    orig_contact_impulse_n.iter(),
                    friction_impulse.iter_mut(),
                )
                .enumerate()
                {
                    let pred_r_t = Vector2::new(pred_r_t);
                    let pred_r_norm = pred_r_t.norm();
                    let r_t = if pred_r_norm > 0.0 {
                        pred_r_t * (params.dynamic_friction * cr.abs() / pred_r_norm)
                    } else {
                        Vector2::zero()
                    };
                    *r_out = contact_basis
                        .from_contact_coordinates([0.0, r_t[0], r_t[1]], aqi)
                        .into();
                }
                true
            }
        };

        if !success {
            return 0;
        }

        // Now we apply the contact jacobian to map the frictional impulses at contact points (on
        // collider vertices) to the vertices of the simulation mesh. Given contact jacobian J, and
        // frictional impulses r (in physical space), J^T*r produces frictional impulses on the
        // deforming surface mesh. An additional remapping puts these impulses on the volume mesh
        // vertices, but this is applied when the friction impulses are actually used.
        // Compute transpose product J^T*f
        let prev_contact_impulse: Chunked3<Vec<f64>> = contact_basis
            .from_normal_space(&orig_contact_impulse_n)
            .collect();
        let impulse: Chunked3<Vec<f64>> =
            (friction_impulse.expr() + contact_impulse.expr() - prev_contact_impulse.expr()).eval();
        // Correct friction_impulse by subtracting previous friction impulse
        let impulse_corrector: Chunked3<Vec<f64>> =
            (impulse.expr() - prev_friction_impulse.expr()).eval();
        let mut object_impulse_tensor = jac.view().transpose() * Tensor::new(impulse.view());
        object_impulse_tensor.negate();

        let mut object_impulse_corrector_tensor =
            jac.view().transpose() * Tensor::new(impulse_corrector.view());
        object_impulse_corrector_tensor.negate();

        *object_impulse = Chunked3::from_flat((
            object_impulse_corrector_tensor.data.into_flat(),
            object_impulse_tensor.data.into_flat(),
        ));

        *collider_impulse = Sparse::from_dim(
            active_contact_indices.clone(),
            self.contact_points.borrow().len(),
            Chunked3::from_flat((impulse_corrector.into_flat(), impulse.into_flat())),
        );

        if friction_steps > 0 {
            friction_steps - 1
        } else {
            0
        }
    }

    fn add_mass_weighted_frictional_contact_impulse(
        &self,
        [object_vel, collider_vel]: [SubsetView<Chunked3<&mut [f64]>>; 2],
    ) {
        if let Some(ref frictional_contact) = self.frictional_contact {
            if !frictional_contact.object_impulse.is_empty() {
                if let Some(masses) = self.object_mass_inv.as_ref() {
                    let mass_mtx = DiagonalBlockMatrix::new(masses.view());
                    let corrector =
                        Chunked3::from_flat(frictional_contact.object_impulse.view().into_flat().0);
                    let add_vel = mass_mtx.view() * Tensor::new(corrector);
                    let mut out_vel = Tensor::new(object_vel);
                    out_vel += add_vel.view();
                }
            }

            if frictional_contact.collider_impulse.is_empty() || self.collider_mass_inv.is_none() {
                return;
            }
            let indices = self.active_constraint_indices();

            let collider_mass_inv =
                DiagonalBlockMatrix::from_subset(Subset::from_unique_ordered_indices(
                    indices.as_slice(),
                    self.collider_mass_inv.as_ref().unwrap().view(),
                ));
            let corrector = Chunked3::from_flat(
                frictional_contact
                    .collider_impulse
                    .source()
                    .view()
                    .into_flat()
                    .0,
            );
            let add_vel = collider_mass_inv * Tensor::new(corrector);
            let mut out_vel = Tensor::new(Subset::from_unique_ordered_indices(
                indices.as_slice(),
                collider_vel,
            ));
            out_vel += add_vel.view();
        }
    }

    fn add_friction_impulse(
        &self,
        mut grad: [SubsetView<Chunked3<&mut [f64]>>; 2],
        multiplier: f64,
    ) {
        if let Some(ref frictional_contact) = self.frictional_contact() {
            if !frictional_contact.object_impulse.is_empty() && !grad[0].is_empty() {
                for (i, (_, &r)) in frictional_contact.object_impulse.iter().enumerate() {
                    grad[0][i] = (Vector3::new(grad[0][i]) + Vector3::new(r) * multiplier).into();
                }
            }

            if frictional_contact.collider_impulse.is_empty() || grad[1].is_empty() {
                return;
            }

            for (contact_idx, (i, (_, &r))) in frictional_contact
                .collider_impulse
                .indexed_source_iter()
                .enumerate()
            {
                // Project out the normal component
                let r_t = if !frictional_contact.contact_basis.is_empty() {
                    let f = frictional_contact
                        .contact_basis
                        .to_contact_coordinates(r, contact_idx);
                    Vector3::new(
                        frictional_contact
                            .contact_basis
                            .from_contact_coordinates([0.0, f[1], f[2]], contact_idx)
                            .into(),
                    )
                } else {
                    Vector3::zero()
                };

                grad[1][i] = (Vector3::new(grad[1][i]) + r_t * multiplier).into();
            }
        }
    }

    fn frictional_dissipation(&self, v: [SubsetView<Chunked3<&[f64]>>; 2]) -> f64 {
        let mut dissipation = 0.0;
        if let Some(ref frictional_contact) = self.frictional_contact {
            for (i, (_, f)) in frictional_contact.object_impulse.iter().enumerate() {
                for j in 0..3 {
                    dissipation += v[0][i][j] * f[j];
                }
            }

            if frictional_contact.collider_impulse.is_empty() {
                return dissipation;
            }

            for (contact_idx, (i, (_, &r))) in frictional_contact
                .collider_impulse
                .indexed_source_iter()
                .enumerate()
            {
                if let Some(i) = i.into() {
                    // Project out normal component.
                    let r_t = if !frictional_contact.contact_basis.is_empty() {
                        let f = frictional_contact
                            .contact_basis
                            .to_contact_coordinates(r, contact_idx);
                        Vector3::new(
                            frictional_contact
                                .contact_basis
                                .from_contact_coordinates([0.0, f[1], f[2]], contact_idx)
                                .into(),
                        )
                    } else {
                        Vector3::zero()
                    };

                    dissipation += Vector3::new(v[1][i]).dot(r_t);
                }
            }
        }

        dissipation
    }

    fn remap_frictional_contact(&mut self, _old_set: &[usize], _new_set: &[usize]) {}

    /// For visualization purposes.
    fn add_contact_impulse(
        &mut self,
        x: [SubsetView<Chunked3<&[f64]>>; 2],
        contact_impulse: &[f64],
        mut impulse: [Chunked3<&mut [f64]>; 2],
    ) {
        self.update_surface_with_mesh_pos(x[0]);
        self.update_contact_points(x[1]);

        let (active_constraint_subset, active_contact_indices, contact_impulse) =
            self.in_contact_indices(contact_impulse);

        let normals = self.contact_normals();
        let normals = Subset::from_unique_ordered_indices(active_constraint_subset, normals);

        assert_eq!(contact_impulse.len(), normals.len());
        assert_eq!(active_contact_indices.len(), normals.len());

        for (surf_idx, &nml, &cr) in zip!(
            active_contact_indices.into_iter(),
            normals.iter(),
            contact_impulse.iter()
        ) {
            impulse[1][surf_idx] = (Vector3::new(nml) * cr).into();
        }

        let query_points = self.contact_points.borrow();
        assert_eq!(impulse[1].len(), query_points.len());

        let surf = &self.implicit_surface;
        let mut cj_matrices = vec![[[0.0; 3]; 3]; surf.num_contact_jacobian_matrices()];

        surf.contact_jacobian_matrices(query_points.view().into(), &mut cj_matrices);

        let cj_indices_iter = surf.contact_jacobian_matrix_indices_iter();

        for ((row, col), jac) in cj_indices_iter.zip(cj_matrices.into_iter()) {
            let imp = Vector3::new(impulse[0][col]);
            impulse[0][col] =
                (imp + Matrix3::new(jac).transpose() * Vector3::new(impulse[1][row])).into()
        }
    }

    fn contact_normals(&self) -> Chunked3<Vec<f64>> {
        // Contacts occur at the vertex positions of the colliding mesh.
        let surf = &self.implicit_surface;
        let contact_points = self.contact_points.borrow_mut();

        let mut normal_coords = vec![0.0; surf.num_query_jacobian_entries()];
        surf.query_jacobian_values(contact_points.view().into(), &mut normal_coords);
        let mut normals = Chunked3::from_flat(normal_coords);

        // Normalize normals
        // Contact normals point away from the surface being collided against.
        // In this case the gradient is opposite of this direction.
        for n in normals.iter_mut() {
            let nml = Vector3::new(*n);
            let len = nml.norm();
            if len > 0.0 {
                *n = (nml / -len).into();
            }
        }

        normals
    }

    fn contact_radius(&self) -> f64 {
        self.implicit_surface.radius()
    }

    fn update_radius_multiplier(&mut self, rad: f64) {
        self.implicit_surface.update_radius_multiplier(rad);
    }

    fn update_max_step(&mut self, step: f64) {
        self.implicit_surface.update_max_step(step);
    }

    fn active_constraint_indices(&self) -> Vec<usize> {
        self.implicit_surface.nonempty_neighbourhood_indices()
    }

    fn update_neighbours(
        &mut self,
        object_pos: SubsetView<Chunked3<&[f64]>>,
        collider_pos: SubsetView<Chunked3<&[f64]>>,
    ) -> bool {
        let num_vertices_updated = self.update_surface_with_mesh_pos(object_pos);
        assert_eq!(
            num_vertices_updated,
            self.implicit_surface.surface_vertex_positions().len()
        );
        self.update_contact_points(collider_pos);
        let contact_points = self.contact_points.borrow();
        let topo_updated = self.implicit_surface.reset(contact_points.as_arrays());

        // Recompute constraint jacobian.

        topo_updated
    }

    fn linearize_constraint(
        &mut self,
        object_pos: SubsetView<Chunked3<&[f64]>>,
        collider_pos: SubsetView<Chunked3<&[f64]>>,
    ) {
        let jac = self.build_constraint_jacobian([object_pos, collider_pos]);
        self.constraint_jacobian.replace(jac);
        let contact_points = self.contact_points.borrow();

        let num_non_zero_constraints = self.implicit_surface.num_neighbourhoods();
        self.constraint_value.resize(num_non_zero_constraints, 0.0);
        self.implicit_surface.local_potential(
            contact_points.view().into(),
            self.constraint_value.as_mut_slice(),
        );
    }
}

impl<'a> Constraint<'a, f64> for LinearizedPointContactConstraint {
    type Input = [SubsetView<'a, Chunked3<&'a [f64]>>; 2]; // Object and collider vertices

    fn constraint_size(&self) -> usize {
        self.implicit_surface.num_neighbourhoods()
    }

    fn constraint_bounds(&self) -> (Vec<f64>, Vec<f64>) {
        let m = self.constraint_size();
        (vec![0.0; m], vec![2e10; m])
    }

    fn constraint(&mut self, _x0: Self::Input, x1: Self::Input, value: &mut [f64]) {
        debug_assert_eq!(value.len(), self.constraint_size());
        self.update_surface_with_mesh_pos(x1[0]);
        self.update_contact_points(x1[1]);

        let jac = self
            .constraint_jacobian
            .borrow()
            .expect("Constraint Jacobian not initialized");
        let mut constraint = self.constraint_value.as_tensor().clone();
        if let Some(jac) = &jac[0] {
            constraint += jac.view() * *x1[0].as_tensor();
        }
        if let Some(jac) = &jac[1] {
            constraint += jac.view() * *x1[1].as_tensor();
        }
        for (res, out) in constraint.data.iter().zip(value.iter_mut()) {
            *out = *res;
        }
    }
}

impl ConstraintJacobian<'_, f64> for LinearizedPointContactConstraint {
    fn constraint_jacobian_size(&self) -> usize {
        let jac = self
            .constraint_jacobian
            .borrow()
            .expect("Constraint Jacobian not initialized");

        jac[0].as_ref().map_or(0, |jac| jac.num_non_zeros())
            + jac[1].as_ref().map_or(0, |jac| jac.num_non_zeros())
    }

    fn constraint_jacobian_indices_iter<'a>(
        &'a self,
    ) -> Result<Box<dyn Iterator<Item = MatrixElementIndex> + 'a>, Error> {
        let col_offset = self.implicit_surface.surface_vertex_positions().len() * 3;
        let jac = self
            .constraint_jacobian
            .borrow()
            .expect("Constraint Jacobian not initialized");
        let jac_indices = move |jac: DSBlockMatrix1x3View<'a>| {
            jac.data
                .into_iter()
                .enumerate()
                .flat_map(move |(row_idx, row)| {
                    row.into_iter().flat_map(move |(col_idx, _)| {
                        (0..3).map(move |component_idx| MatrixElementIndex {
                            row: row_idx,
                            col: 3 * col_idx + component_idx,
                        })
                    })
                })
        };
        Ok(Box::new(
            jac[0]
                .as_ref()
                .map(|jac| jac_indices(jac.view()))
                .into_iter()
                .flatten()
                .chain(
                    jac[1]
                        .as_ref()
                        .map(|jac| jac_indices(jac.view()))
                        .into_iter()
                        .flatten()
                        .map(move |MatrixElementIndex { row, col }| MatrixElementIndex {
                            row,
                            col: col + col_offset,
                        }),
                ),
        ))
    }

    fn constraint_jacobian_values(
        &mut self,
        _x0: Self::Input,
        _x1: Self::Input,
        values: &mut [f64],
    ) -> Result<(), Error> {
        let mut value_blocks = Chunked3::from_flat(values);
        let jac = self
            .constraint_jacobian
            .borrow()
            .expect("Constraint Jacobian not initialized");

        for (v, v_out) in jac[0]
            .as_ref()
            .map(|jac| {
                jac.view()
                    .data
                    .into_iter()
                    .flat_map(move |row| row.into_iter())
                    .map(|(_, val)| val)
            })
            .into_iter()
            .flatten()
            .chain(
                jac[1]
                    .as_ref()
                    .map(|jac| {
                        jac.view()
                            .data
                            .into_iter()
                            .flat_map(move |row| row.into_iter())
                            .map(|(_, val)| val)
                    })
                    .into_iter()
                    .flatten(),
            )
            .zip(value_blocks.iter_mut())
        {
            *v_out = v.into_arrays()[0];
        }
        Ok(())
    }
}

impl<'a> ConstraintHessian<'a, f64> for LinearizedPointContactConstraint {
    type InputDual = &'a [f64];
    fn constraint_hessian_size(&self) -> usize {
        0
    }

    fn constraint_hessian_indices_iter<'b>(
        &'b self,
    ) -> Result<Box<dyn Iterator<Item = MatrixElementIndex> + 'b>, Error> {
        Ok(Box::new(std::iter::empty()))
    }

    fn constraint_hessian_values(
        &mut self,
        _x0: Self::Input,
        _x1: Self::Input,
        _lambda: Self::InputDual,
        _scale: f64,
        _values: &mut [f64],
    ) -> Result<(), Error> {
        Ok(())
    }
}
