use super::*;
use crate::constraint::*;
use crate::contact::*;
use crate::fem::problem::Var;
use crate::friction::*;
use crate::matrix::*;
use crate::Error;
use crate::Index;
use crate::TriMesh;
use geo::math::{Matrix3, Vector2, Vector3};
use geo::mesh::topology::*;
use geo::mesh::{Attrib, VertexPositions};
use implicits::*;
#[cfg(feature = "af")]
use reinterpret::*;
use std::cell::RefCell;
use utils::soap::*;
use utils::zip;

/// Enforce a contact constraint on a mesh against animated vertices. This constraint prevents
/// vertices from occupying the same space as a smooth representation of the simulation mesh.
#[derive(Clone, Debug)]
pub struct PointContactConstraint {
    /// Implicit surface that represents the deforming object.
    pub implicit_surface: RefCell<ImplicitSurface>,
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

    /// Internal constraint function buffer used to store temporary constraint computations.
    constraint_buffer: RefCell<Vec<f64>>,
}

impl PointContactConstraint {
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

        if let Some(surface) = surface_builder.build() {
            // Sanity check that the surface is built correctly.
            assert_eq!(
                surface.surface_vertex_positions().len(),
                object.num_vertices()
            );

            let query_points = collider.vertex_positions();

            let object_mass_inv = object
                .attrib_as_slice::<MassType, VertexIndex>(MASS_ATTRIB)
                .map(|attrib| {
                    attrib
                        .iter()
                        .map(|&x| {
                            assert!(x > 0.0);
                            [1.0 / x; 3]
                        })
                        .collect()
                })
                .ok();

            let collider_mass_inv = collider
                .attrib_as_slice::<MassType, VertexIndex>(MASS_ATTRIB)
                .map(|attrib| {
                    attrib
                        .iter()
                        .map(|&x| {
                            assert!(x > 0.0);
                            [1.0 / x; 3]
                        })
                        .collect()
                })
                .ok();

            let constraint = PointContactConstraint {
                implicit_surface: RefCell::new(surface),
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
                constraint_buffer: RefCell::new(vec![0.0; query_points.len()]),
            };

            constraint
                .implicit_surface
                .borrow()
                .cache_neighbours(query_points);

            Ok(constraint)
        } else {
            Err(Error::InvalidImplicitSurface)
        }
    }

    /// Update implicit surface using the given position data from mesh vertices.
    pub fn update_surface_with_mesh_pos(&self, pos: SubsetView<Chunked3<&[f64]>>) {
        self.implicit_surface
            .borrow_mut()
            .update(pos.iter().cloned());
    }

    pub fn update_contact_points(&self, x: SubsetView<Chunked3<&[f64]>>) {
        let mut contact_points = self.contact_points.borrow_mut();
        x.clone_into_other(&mut *contact_points);
    }

    ///// Update implicit surface using the given position and displacement data.
    //pub fn update_surface_with_displacement(&self, x: &[f64], dx: &[f64]) {
    //    let all_displacements: &[Vector3<f64>] = reinterpret_slice(dx);
    //    let all_positions: &[Vector3<f64>] = reinterpret_slice(x);
    //    let points_iter = self
    //        .sim_verts
    //        .iter()
    //        .map(|&i| (all_positions[i] + all_displacements[i]).into());

    //    self.implicit_surface.borrow_mut().update(points_iter);
    //}

    /* Needed for the Linear constraint
    /// Update implicit surface using the given position data.
    pub fn update_surface(&self, x: &[f64]) {
        let all_positions: &[[f64; 3]] = reinterpret_slice(x);
        let points_iter = self.sim_verts.iter().map(|&i| all_positions[i]);

        self.implicit_surface.borrow_mut().update(points_iter);
    }
    */

    #[allow(dead_code)]
    fn background_points(&self) -> Vec<bool> {
        let cached_neighbourhood_sizes = self
            .implicit_surface
            .borrow()
            .cached_neighbourhood_sizes()
            .unwrap();

        let mut background_points = vec![true; cached_neighbourhood_sizes.len()];

        for (_, bg) in cached_neighbourhood_sizes
            .iter()
            .zip(background_points.iter_mut())
            .filter(|&(&c, _)| c != 0)
        {
            *bg = false;
        }

        background_points
    }

    /// This function fills the non-local values of the constraint function with a constant signed
    /// value (equal to the contact radius in magnitude) to help the optimization determine
    /// feasible regions. This is done using a flood fill algorithm as follows.
    /// 1. Identify non-local query poitns with `cached_neighbourhood_sizes`.
    /// 2. Partition the primitives of the kinematic object (from which the points are from) into
    ///    connected components of non-local points. This means that points with a valid local
    ///    potential serve as boundaries.
    /// 3. During the splitting above, record whether a component must be inside or outside
    ///    depending on the sign of its boundary points (local points).
    /// 4. It could happen that a connected component has no local points, in which case we do a
    ///    ray cast in the x direction from any point and intersect it with our mesh to determine
    ///    the winding number. (TODO)
    /// 5. It could also happen that the local points don't separate the primitives into inside
    ///    and outside partitions if the radius is not sufficiently large. This is a problem for
    ///    the future (FIXME)
    #[allow(dead_code)]
    fn fill_background_potential(
        mesh: &TriMesh,
        background_points: &[bool],
        abs_fill_val: f64,
        values: &mut [f64],
    ) {
        debug_assert!(abs_fill_val >= 0.0);

        let mut hedge_dest_indices = vec![Vec::new(); mesh.num_vertices()];
        for f in mesh.face_iter() {
            for vtx in 0..3 {
                // Get an edge with vertices in sorted order.
                let edge = [f[vtx], f[(vtx + 1) % 3]];

                let neighbourhood = &mut hedge_dest_indices[edge[0]];

                if let Err(idx) = neighbourhood.binary_search_by(|x: &usize| x.cmp(&edge[1])) {
                    neighbourhood.insert(idx, edge[1]);
                }
            }
        }
        //println!("edges:");
        //for (i, v) in hedge_dest_indices.iter().enumerate() {
        //    for &vtx in v.iter() {
        //        println!("({}, {})", i, vtx);
        //    }
        //}
        //dbg!(background_points);

        let mut vertex_is_inside = vec![false; mesh.num_vertices()];
        for vidx in (0..mesh.num_vertices()).filter(|&i| !background_points[i]) {
            vertex_is_inside[vidx] = values[vidx] < 0.0;
        }

        let mut seen_vertices = vec![false; mesh.num_vertices()];

        let mut queue = std::collections::VecDeque::new();

        for vidx in (0..mesh.num_vertices()).filter(|&i| !background_points[i]) {
            if seen_vertices[vidx] {
                continue;
            }

            let is_inside = vertex_is_inside[vidx];

            queue.push_back(vidx);

            while let Some(vidx) = queue.pop_front() {
                if background_points[vidx] {
                    if seen_vertices[vidx] {
                        continue;
                    } else {
                        vertex_is_inside[vidx] = is_inside;
                    }
                }

                seen_vertices[vidx] = true;

                queue.extend(
                    hedge_dest_indices[vidx]
                        .iter()
                        .filter(|&&i| background_points[i])
                        .filter(|&&i| !seen_vertices[i]),
                );
            }
        }

        for ((&is_inside, &bg), val) in vertex_is_inside
            .iter()
            .zip(background_points.iter())
            .zip(values.iter_mut())
        {
            if bg {
                if is_inside {
                    *val = -abs_fill_val;
                } else {
                    *val = abs_fill_val;
                }
            }
        }
    }

    /// Prune contacts with zero contact_impulse and contacts without neighbouring samples.
    /// This function outputs the indices of contacts as well as a pruned vector of impulses.
    fn in_contact_indices(&self, contact_impulse: &[f64]) -> (Vec<usize>, Vec<usize>, Vec<f64>) {
        let surf = self.implicit_surface.borrow();
        let query_points = self.contact_points.borrow();
        let radius = surf.radius() * 0.999;
        let query_indices = self
            .active_constraint_indices()
            .expect("Failed to retrieve constraint indices.");
        assert_eq!(query_indices.len(), contact_impulse.len());
        let (active_constraint_subset, contact_impulse): (Vec<_>, Vec<_>) = contact_impulse
            .iter()
            .enumerate()
            .filter_map(|(i, &cf)| {
                if cf != 0.0
                    && surf.num_neighbours_within_distance(
                        query_points[query_indices[i]],
                        radius,
                    ) > 0
                {
                    Some((i, cf))
                } else {
                    None
                }
            })
            .unzip();

        let active_contact_indices: Vec<_> = active_constraint_subset
            .iter()
            .map(|&i| query_indices[i]).collect();

        (active_constraint_subset, active_contact_indices, contact_impulse)
    }
}

impl ContactConstraint for PointContactConstraint {
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
        let surf = self.implicit_surface.borrow();
        let query_points = self.contact_points.borrow();

        let mut cj_values = vec![
            0.0;
            surf.num_contact_jacobian_entries()
                .expect("Failed to get contact Jacobian size.")
        ];
        surf.contact_jacobian_values(
            query_points.view().into(),
            reinterpret_mut_slice(&mut cj_values),
        )
        .expect("Failed to compute contact Jacobian.");
        let cj_indices_iter = surf
            .contact_jacobian_indices_iter()
            .expect("Failed to get contact Jacobian indices.");

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
        contact_impulse: &[f64],
        x: [SubsetView<Chunked3<&[f64]>>; 2],
        v: [SubsetView<Chunked3<&[f64]>>; 2],
        _potential_values: &[f64],
        friction_steps: u32,
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
        let (active_constraint_subset, active_contact_indices, contact_impulse) = self.in_contact_indices(contact_impulse);
        let normals = self
            .contact_normals()
            .expect("Failed to compute contact normals.");
        let normals_subset = Subset::from_unique_ordered_indices(active_constraint_subset, normals);
        let mut normals = Chunked3::from_array_vec(vec![[0.0; 3]; normals_subset.len()]);
        normals_subset.clone_into_other(&mut normals);

        let FrictionalContact {
            contact_basis,
            params,
            object_impulse: object_friction_impulse,
            collider_impulse: collider_friction_impulse, // for active point contacts
        } = self.frictional_contact.as_mut().unwrap();

        let query_points = self.contact_points.borrow();

        // Friction impulse in physical space at active contacts.
        *collider_friction_impulse =
            Sparse::from_dim(
                active_contact_indices.clone(),
                query_points.len(),
                Chunked3::from_array_vec(vec![[0.0; 3]; active_contact_indices.len()])
                );

        if active_contact_indices.is_empty() {
            // If there are no active contacts, there is nothing to update.
            // Clear object_friction_impulse before returning.
            object_friction_impulse
                .iter_mut()
                .for_each(|x| *x = [0.0; 3]);
            return 0;
        }

        contact_basis.update_from_normals(normals.into());

        let surf = self.implicit_surface.borrow();

        // Construct diagonal mass matrices for object and collider.
        let object_zero_mass_inv = Chunked3::from_array_vec(vec![[0.0; 3]; v[0].len()]);
        let collider_zero_mass_inv = Chunked3::from_array_vec(vec![[0.0; 3]; v[1].len()]);
        let object_mass_inv = DiagonalBlockMatrix::from_uniform(
            self.object_mass_inv
                .as_ref()
                .map(|mass_inv| mass_inv.view())
                .unwrap_or_else(|| object_zero_mass_inv.view()),
        );

        // Collider mass matrix is constructed at active contacts only.
        let collider_mass_inv =
            DiagonalBlockMatrix::from_subset(Subset::from_unique_ordered_indices(
                active_contact_indices.as_slice(),
                self.collider_mass_inv
                    .as_ref()
                    .map(|mass_inv| mass_inv.view())
                    .unwrap_or_else(|| collider_zero_mass_inv.view()),
            ));

        let active_contact_points = Subset::from_unique_ordered_indices(
            active_contact_indices.as_slice(),
            query_points.view(),
        );

        // Compute contact jacobian
        let jac_triplets =
            build_triplet_contact_jacobian(&surf, active_contact_points, query_points.view());
        let jac: ContactJacobian = jac_triplets.into();

        let collider_velocity =
            Subset::from_unique_ordered_indices(active_contact_indices.as_slice(), v[1]);

        let mut velocity = jac.view() * Tensor::new(v[0]);
        let mut rhs = velocity.view_mut();
        rhs -= Tensor::new(collider_velocity.view());

        jac.write_img("./out/jac.png");
        let mut jac_mass = jac.clone();
        jac_mass *= object_mass_inv.view();

        //jac_mass.write_img("./out/jac_mass.png");

        let effective_mass_inv = jac_mass.view() * jac.view().transpose();
        let effective_mass_inv = effective_mass_inv.view() + collider_mass_inv.view();
        effective_mass_inv.write_img("./out/effective_mass_inv.png");

        let sprs_effective_mass_inv: sprs::CsMat<f64> = effective_mass_inv.clone().into();
        //        let ldlt_solver =
        //            sprs_ldl::LdlNumeric::<f64, usize>::new(sprs_effective_mass_inv.view()).unwrap();
        //        let predictor_impulse = Chunked3::from_flat(ldlt_solver.solve(rhs.storage()));

        let mut rhs = rhs.storage().to_vec();
        sprs::linalg::trisolve::lsolve_csr_dense_rhs(sprs_effective_mass_inv.view(), &mut rhs)
            .unwrap();
        let predictor_impulse = Chunked3::from_flat(rhs);

        let success = if false {
            // Polar coords
            let predictor_impulse_t: Vec<_> =
                predictor_impulse.iter().enumerate()
                .map(|(aqi, &predictor_imp)| {
                    let r = contact_basis.to_cylindrical_contact_coordinates(predictor_imp, aqi);
                    r.tangent
                })
                .collect();
            if true {
                // switch between implicit solver and explicit solver here.
                match FrictionPolarSolver::new(
                    &predictor_impulse_t,
                    &contact_impulse,
                    &contact_basis,
                    effective_mass_inv.view(),
                    *params,
                    jac.view(),
                ) {
                    Ok(mut solver) => {
                        eprintln!("#### Solving Friction");
                        if let Ok(FrictionSolveResult { solution: r_t, .. }) = solver.step() {
                            for ((aqi, &r), r_out) in 
                                r_t.iter().enumerate()
                                .zip(collider_friction_impulse.source_iter_mut())
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
                    contact_impulse.iter(),
                    collider_friction_impulse.source_iter_mut()
                ).enumerate() {
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
            // Euclidean coords
            let predictor_impulse_t: Vec<_> =
                predictor_impulse.iter().enumerate()
                .map(|(aqi, &pred_r)| {
                    let r = contact_basis.to_contact_coordinates(pred_r, aqi);
                    [r[1], r[2]]
                })
                .collect();
            if true {
                // Switch between implicit solver and explicit solver here.
                let mut solver = FrictionSolver::new(
                    &predictor_impulse_t,
                    &contact_impulse,
                    &contact_basis,
                    effective_mass_inv.view(),
                    *params,
                    jac.view(),
                );
                eprintln!("#### Solving Friction");
                let r_t = solver.step();
                for ((aqi, &r), r_out) in
                    r_t.iter().enumerate()
                    .zip(collider_friction_impulse.source_iter_mut()) {
                    *r_out = contact_basis
                        .from_contact_coordinates([0.0, r[0], r[1]], aqi)
                        .into();
                }
                true
            } else {
                for (aqi, (&pred_r_t, &cr, r_out)) in zip!(
                    predictor_impulse_t.iter(),
                    contact_impulse.iter(),
                    collider_friction_impulse.source_iter_mut(),
                ).enumerate() {
                    let pred_r_t = Vector2(pred_r_t);
                    let pred_r_norm = pred_r_t.norm();
                    let r_t = if pred_r_norm > 0.0 {
                        pred_r_t * (-params.dynamic_friction * cr.abs() / pred_r_norm)
                    } else {
                        Vector2::zeros()
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
        *object_friction_impulse =
            (jac.view().transpose() * Tensor::new(collider_friction_impulse.source().view().into())).data;

        // The last thing to do is to ensure that collider friction impulses are
        // the impulses ON the collider and not BY the collider.
        collider_friction_impulse
            .view_mut()
            .into_flat()
            .iter_mut()
            .for_each(|imp| *imp = -*imp);

        friction_steps - 1
    }

    fn add_mass_weighted_frictional_contact_impulse(
        &self,
        [object_vel, collider_vel]: [SubsetView<Chunked3<&mut [f64]>>; 2],
    ) {
        if let Some(ref frictional_contact) = self.frictional_contact {
            if !frictional_contact.object_impulse.is_empty() {
                if let Some(masses) = self.object_mass_inv.as_ref() {
                    let mass_mtx = DiagonalBlockMatrix::new(masses.view());
                    let add_vel =
                        mass_mtx.view() * Tensor::new(frictional_contact.object_impulse.view());
                    let mut out_vel = Tensor::new(object_vel);
                    out_vel += add_vel.view();
                }
            }

            if frictional_contact.collider_impulse.is_empty() || self.collider_mass_inv.is_none() {
                return;
            }
            let indices = self
                .active_constraint_indices()
                .expect("Failed to retrieve constraint indices.");

            let add_vel = DiagonalBlockMatrix::new(self.collider_mass_inv.as_ref().unwrap().view())
                * Tensor::new(frictional_contact.collider_impulse.source().view());
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
                for (i, &r) in frictional_contact.object_impulse.iter().enumerate() {
                    grad[0][i] = (Vector3(grad[0][i]) + Vector3(r) * multiplier).into();
                }
            }

            if frictional_contact.collider_impulse.is_empty() || grad[1].is_empty() {
                return;
            }

            for (contact_idx, (i, &r)) in
                    frictional_contact.collider_impulse.indexed_source_iter()
                .enumerate()
            {
                // Project out the normal component
                let r_t = if !frictional_contact.contact_basis.is_empty() {
                    let f = frictional_contact
                        .contact_basis
                        .to_contact_coordinates(r, contact_idx);
                    Vector3(
                        frictional_contact
                            .contact_basis
                            .from_contact_coordinates([0.0, f[1], f[2]], contact_idx)
                            .into(),
                    )
                } else {
                    Vector3::zeros()
                };

                grad[1][i] = (Vector3(grad[1][i]) + r_t * multiplier).into();
            }
        }
    }

    fn frictional_dissipation(&self, v: [SubsetView<Chunked3<&[f64]>>; 2]) -> f64 {
        let mut dissipation = 0.0;
        if let Some(ref frictional_contact) = self.frictional_contact {
            for (i, f) in frictional_contact.object_impulse.iter().enumerate() {
                for j in 0..3 {
                    dissipation += v[0][i][j] * f[j];
                }
            }

            if frictional_contact.collider_impulse.is_empty() {
                return dissipation;
            }

            for (contact_idx, (i, &r)) in 
                    frictional_contact.collider_impulse.indexed_source_iter()
                .enumerate()
            {
                if let Some(i) = i.into() {
                    // Project out normal component.
                    let r_t = if !frictional_contact.contact_basis.is_empty() {
                        let f = frictional_contact
                            .contact_basis
                            .to_contact_coordinates(r, contact_idx);
                        Vector3(
                            frictional_contact
                                .contact_basis
                                .from_contact_coordinates([0.0, f[1], f[2]], contact_idx)
                                .into(),
                        )
                    } else {
                        Vector3::zeros()
                    };

                    dissipation += Vector3(v[1][i]).dot(r_t);
                }
            }
        }

        dissipation
    }

    fn remap_frictional_contact(&mut self, _old_set: &[usize], _new_set: &[usize]) {
        // Remap friction forces the same way we remap constraint multipliers for the contact
        // solve.
        //if let Some(ref mut frictional_contact) = self.frictional_contact {
            // Remap collider contacts (since the set of contact points may have
            // changed).
            //let new_friction_impulses = crate::constraints::remap_values(
            //    frictional_contact.collider_impulse.iter().cloned(),
            //    [0.0; 3],
            //    old_set.iter().cloned(),
            //    new_set.iter().cloned(),
            //);

            //std::mem::replace(
            //    &mut frictional_contact.collider_impulse,
            //    Chunked3::from_array_vec(new_friction_impulses),
            //);

            // Object impulses don't need to be remapped because we store them
            // on all the surface vertices regardless of the contact set.

            //frictional_contact.contact_basis.remap(old_set, new_set);
        //}
    }

    /// For visualization purposes.
    fn add_contact_impulse(
        &self,
        x: [SubsetView<Chunked3<&[f64]>>; 2],
        contact_impulse: &[f64],
        mut impulse: [Chunked3<&mut [f64]>; 2],
    ) {
        self.update_surface_with_mesh_pos(x[0]);
        self.update_contact_points(x[1]);

        let (active_constraint_subset, active_contact_indices, contact_impulse) = self.in_contact_indices(contact_impulse);

        let normals = self
            .contact_normals()
            .expect("Failed to retrieve contact normals.");
        let normals = Subset::from_unique_ordered_indices(active_constraint_subset, normals);

        assert_eq!(contact_impulse.len(), normals.len());
        assert_eq!(active_contact_indices.len(), normals.len());

        for (surf_idx, &nml, &cr) in zip!(
            active_contact_indices.into_iter(),
            normals.iter(),
            contact_impulse.iter()
        ) {
            impulse[1][surf_idx] = (Vector3(nml) * cr).into();
        }

        let query_points = self.contact_points.borrow();
        assert_eq!(impulse[1].len(), query_points.len());

        let surf = self.implicit_surface.borrow();
        let mut cj_matrices = vec![
            [[0.0; 3]; 3];
            surf.num_contact_jacobian_matrices()
                .expect("Failed to get contact jacobian size")
        ];

        surf.contact_jacobian_matrices(query_points.view().into(), &mut cj_matrices)
            .expect("Failed to compute contact Jacobian");

        let cj_indices_iter = surf
            .contact_jacobian_matrix_indices_iter()
            .expect("Failed to get contact Jacobian indices");

        for ((row, col), jac) in cj_indices_iter.zip(cj_matrices.into_iter()) {
            let imp = Vector3(impulse[0][col]);
            impulse[0][col] = (imp + Matrix3(jac).transpose() * Vector3(impulse[1][row])).into()
        }
    }

    fn contact_normals(
        &self,
    ) -> Result<Chunked3<Vec<f64>>, Error> {
        // Contacts occur at the vertex positions of the colliding mesh.
        let surf = self.implicit_surface.borrow();
        let contact_points = self.contact_points.borrow_mut();

        let mut normal_coords = vec![0.0; surf.num_query_jacobian_entries()?];
        surf.query_jacobian_values(contact_points.view().into(), &mut normal_coords)?;
        let mut normals = Chunked3::from_flat(normal_coords);

        // Normalize normals
        // Contact normals point away from the surface being collided against.
        // In this case the gradient is opposite of this direction.
        for n in normals.iter_mut() {
            let nml = Vector3(*n);
            let len = nml.norm();
            if len > 0.0 {
                *n = (nml / -len).into();
            }
        }

        Ok(normals)
    }

    fn contact_radius(&self) -> f64 {
        self.implicit_surface.borrow_mut().radius()
    }

    fn update_radius_multiplier(&mut self, rad: f64) {
        self.implicit_surface
            .borrow_mut()
            .update_radius_multiplier(rad);
    }

    fn update_max_step(&mut self, step: f64) {
        self.implicit_surface.borrow_mut().update_max_step(step);
    }

    fn active_constraint_indices(&self) -> Result<Vec<usize>, Error> {
        self.implicit_surface
            .borrow()
            .nonempty_neighbourhood_indices()
            .map_err(|_| Error::InvalidImplicitSurface)
    }

    fn update_cache(
        &mut self,
        object_pos: SubsetView<Chunked3<&[f64]>>,
        collider_pos: SubsetView<Chunked3<&[f64]>>,
    ) -> bool {
        self.update_contact_points(collider_pos);
        let contact_points = self.contact_points.borrow_mut();

        let mut surf = self.implicit_surface.borrow_mut();
        let num_vertices_updated = surf.update(object_pos.iter().cloned());
        assert_eq!(num_vertices_updated, surf.surface_vertex_positions().len());

        surf.invalidate_query_neighbourhood();
        surf.cache_neighbours(contact_points.view().into())
    }

    fn cached_neighbourhood_indices(&self) -> Vec<Index> {
        let surf = self.implicit_surface.borrow();
        let mut cached_neighbourhood_indices = if let Ok(n) = surf.num_cached_query_points() {
            vec![Index::INVALID; n]
        } else {
            return Vec::new();
        };

        let cached_neighbourhood_sizes = match surf.cached_neighbourhood_sizes() {
            Ok(c) => c,
            Err(_) => return cached_neighbourhood_indices,
        };

        for (i, (idx, _)) in cached_neighbourhood_indices
            .iter_mut()
            .zip(cached_neighbourhood_sizes.iter())
            .filter(|&(_, &s)| s != 0)
            .enumerate()
        {
            *idx = Index::new(i);
        }

        cached_neighbourhood_indices
    }
}

impl<'a> Constraint<'a, f64> for PointContactConstraint {
    type Input = [SubsetView<'a, Chunked3<&'a [f64]>>; 2]; // Object and collider vertices

    #[inline]
    fn constraint_size(&self) -> usize {
        self.implicit_surface
            .borrow()
            .num_cached_neighbourhoods()
            .unwrap_or(0)
    }

    #[inline]
    fn constraint_bounds(&self) -> (Vec<f64>, Vec<f64>) {
        let m = self.constraint_size();
        (vec![0.0; m], vec![2e10; m])
    }

    #[inline]
    fn constraint(&self, _x0: Self::Input, x1: Self::Input, value: &mut [f64]) {
        debug_assert_eq!(value.len(), self.constraint_size());
        self.update_surface_with_mesh_pos(x1[0]);

        self.update_contact_points(x1[1]);
        let contact_points = self.contact_points.borrow_mut();

        let mut cbuf = self.constraint_buffer.borrow_mut();
        let radius = self.contact_radius();

        let surf = self.implicit_surface.borrow();
        for (val, q) in cbuf.iter_mut().zip(contact_points.iter()) {
            // Clear potential value.
            let closest_sample = surf.nearest_neighbour_lookup(*q).unwrap();
            if closest_sample.nml.dot(Vector3(*q) - closest_sample.pos) > 0.0 {
                *val = radius;
            } else {
                *val = -radius;
            }
        }

        surf.potential(contact_points.view().into(), &mut cbuf)
            .unwrap();

        //let bg_pts = self.background_points();
        //let collider_mesh = self.collision_object.borrow();
        //Self::fill_background_potential(&collider_mesh, &bg_pts, radius, &mut cbuf);

        let cached_neighbourhood_sizes = surf.cached_neighbourhood_sizes().unwrap();

        //println!("cbuf = ");
        //for c in cbuf.iter() {
        //    print!("{:9.5} ", *c);
        //}
        //println!("");

        // Because `value` tracks only the values for which the neighbourhood is not empty.
        for ((_, new_v), v) in cached_neighbourhood_sizes
            .iter()
            .zip(cbuf.iter())
            .filter(|&(&c, _)| c != 0)
            .zip(value.iter_mut())
        {
            *v = *new_v;
        }
        //dbg!(&value);
    }
}

impl ConstraintJacobian<'_, f64> for PointContactConstraint {
    #[inline]
    fn constraint_jacobian_size(&self) -> usize {
        let num_obj = if !self.object_is_fixed {
            self.implicit_surface
                .borrow()
                .num_surface_jacobian_entries()
                .unwrap_or(0)
        } else {
            0
        };

        let num_coll = if !self.collider_is_fixed {
            self.implicit_surface
                .borrow()
                .num_query_jacobian_entries()
                .unwrap_or(0)
        } else {
            0
        };
        num_obj + num_coll
    }
    fn constraint_jacobian_indices_iter(
        &self,
    ) -> Result<Box<dyn Iterator<Item = MatrixElementIndex>>, Error> {
        let idx_iter = {
            let surf = self.implicit_surface.borrow();
            let col_offset = surf.surface_vertex_positions().len() * 3;
            let obj_indices_iter = if !self.object_is_fixed {
                Some(surf.surface_jacobian_indices_iter()?)
            } else {
                None
            };

            let coll_indices_iter = if !self.collider_is_fixed {
                Some(surf.query_jacobian_indices_iter()?)
            } else {
                None
            };
            obj_indices_iter.into_iter().flatten().chain(
                coll_indices_iter
                    .into_iter()
                    .flatten()
                    .map(move |(row, col)| (row, col + col_offset)),
            )
        };

        let cached_neighbourhood_indices = self.cached_neighbourhood_indices();
        Ok(Box::new(idx_iter.map(move |(row, col)| {
            assert!(cached_neighbourhood_indices[row].is_valid());
            MatrixElementIndex {
                row: cached_neighbourhood_indices[row].unwrap(),
                col,
            }
        })))
    }

    fn constraint_jacobian_values(
        &self,
        _x0: Self::Input,
        x1: Self::Input,
        values: &mut [f64],
    ) -> Result<(), Error> {
        debug_assert_eq!(values.len(), self.constraint_jacobian_size());

        self.update_surface_with_mesh_pos(x1[0]);
        self.update_contact_points(x1[1]);

        let contact_points = self.contact_points.borrow_mut();

        let num_obj_jac_nnz;

        if !self.object_is_fixed {
            num_obj_jac_nnz = self
                .implicit_surface
                .borrow()
                .num_surface_jacobian_entries()
                .unwrap_or(0);

            self.implicit_surface.borrow().surface_jacobian_values(
                contact_points.view().into(),
                &mut values[..num_obj_jac_nnz],
            )?;
        } else {
            num_obj_jac_nnz = 0;
        }

        if !self.collider_is_fixed {
            self.implicit_surface.borrow().query_jacobian_values(
                contact_points.view().into(),
                &mut values[num_obj_jac_nnz..],
            )?;
        }
        Ok(())
    }
}

impl<'a> ConstraintHessian<'a, f64> for PointContactConstraint {
    type InputDual = &'a [f64];
    #[inline]
    fn constraint_hessian_size(&self) -> usize {
        0 + if !self.object_is_fixed {
            self.implicit_surface
                .borrow()
                .num_surface_hessian_product_entries()
                .unwrap_or(0)
        } else {
            0
        } + if !self.collider_is_fixed {
            self.implicit_surface
                .borrow()
                .num_query_hessian_product_entries()
                .unwrap_or(0)
        } else {
            0
        }
    }

    fn constraint_hessian_indices_iter(
        &self,
    ) -> Result<Box<dyn Iterator<Item = MatrixElementIndex>>, Error> {
        let idx_iter = {
            let surf = self.implicit_surface.borrow();
            let offset = surf.surface_vertex_positions().len() * 3;
            let obj_indices_iter = if !self.object_is_fixed {
                Some(surf.surface_hessian_product_indices_iter()?)
            } else {
                None
            };
            let coll_indices_iter = if !self.collider_is_fixed {
                Some(surf.query_hessian_product_indices_iter()?)
            } else {
                None
            };
            obj_indices_iter.into_iter().flatten().chain(
                coll_indices_iter
                    .into_iter()
                    .flatten()
                    .map(move |(row, col)| (row + offset, col + offset)),
            )
        };
        Ok(Box::new(
            idx_iter.map(move |(row, col)| MatrixElementIndex { row, col }),
        ))
    }

    fn constraint_hessian_values(
        &self,
        _x0: Self::Input,
        x1: Self::Input,
        lambda: Self::InputDual,
        scale: f64,
        values: &mut [f64],
    ) -> Result<(), Error> {
        self.update_surface_with_mesh_pos(x1[0]);
        let surf = self.implicit_surface.borrow();
        self.update_contact_points(x1[1]);
        let contact_points = self.contact_points.borrow();

        let mut obj_hess_nnz = 0;

        if !self.object_is_fixed {
            obj_hess_nnz = self
                .implicit_surface
                .borrow()
                .num_surface_hessian_product_entries()
                .unwrap_or(0);

            surf.surface_hessian_product_scaled_values(
                contact_points.view().into(),
                lambda,
                scale,
                &mut values[..obj_hess_nnz],
            )?;
        }

        if !self.collider_is_fixed {
            surf.query_hessian_product_scaled_values(
                contact_points.view().into(),
                lambda,
                scale,
                &mut values[obj_hess_nnz..],
            )?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use utils::*;

    /// Test the `fill_background_potential` function on a small grid.
    #[test]
    fn background_fill_test() {
        // Make a small grid.
        let mut grid = TriMesh::from(make_grid(Grid {
            rows: 4,
            cols: 6,
            orientation: AxisPlaneOrientation::ZX,
        }));

        let mut values = vec![0.0; grid.num_vertices()];
        let mut bg_pts = vec![true; grid.num_vertices()];
        let radius = 0.5;
        for ((p, v), bg) in grid
            .vertex_position_iter()
            .zip(values.iter_mut())
            .zip(bg_pts.iter_mut())
        {
            if p[0] == 0.0 {
                *bg = false;
            } else if p[0] < radius && p[0] > 0.0 {
                *v = radius;
                *bg = false;
            } else if p[0] > -radius && p[0] < 0.0 {
                *v = -radius;
                *bg = false;
            }
        }

        PointContactConstraint::fill_background_potential(&grid, &bg_pts, radius, &mut values);

        grid.set_attrib_data::<_, VertexIndex>("potential", &values)
            .expect("Failed to set potential field on grid");

        //geo::io::save_polymesh(&geo::mesh::PolyMesh::from(grid.clone()), &std::path::PathBuf::from("out/background_test.vtk")).unwrap();

        for (&p, &v) in grid.vertex_position_iter().zip(values.iter()) {
            if p[0] > 0.0 {
                assert!(v > 0.0);
            } else {
                assert!(v <= 0.0);
            }
        }
    }
}
