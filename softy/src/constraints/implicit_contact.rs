use super::*;
use crate::constraint::*;
use crate::contact::*;
use crate::friction::*;
use crate::matrix::*;
use crate::Error;
use crate::Index;
use crate::TriMesh;
use geo::math::{Matrix3, Vector2, Vector3};
use geo::mesh::topology::*;
use geo::mesh::{Attrib, VertexPositions};
use implicits::*;
use reinterpret::*;
use std::cell::RefCell;
use utils::zip;

/// Enforce a contact constraint on a mesh against an animated implicit surface. This constraint prevents
/// vertices of the simulation mesh from penetrating through the implicit surface.
#[derive(Clone, Debug)]
pub struct ImplicitContactConstraint {
    /// Implicit surface that represents the collision object.
    pub implicit_surface: RefCell<ImplicitSurface>,
    /// A buffer of vertex positions on the simulation mesh. This is used to avoid reallocating
    /// contiguous space for these positions every time the constraint is evaluated.
    pub contact_points: RefCell<Chunked3<Vec<f64>>>,
    /// Friction impulses applied during contact.
    pub frictional_contact: Option<FrictionalContact>,
    /// A mass for each vertex in the object mesh.
    pub vertex_masses: Vec<f64>,

    /// Internal constraint function buffer used to store temporary constraint computations.
    constraint_buffer: RefCell<Vec<f64>>,

    /// Worspace vector to keep track of active constraint indices.
    active_constraint_indices: RefCell<Vec<usize>>,
}

impl ImplicitContactConstraint {
    /// Build an implicit surface from the given trimesh, and constrain the tetmesh vertices to lie
    /// strictly outside of it.
    pub fn new(
        // Main object experiencing contact with its vertices.
        object: &TriMesh,
        // Collision object generating an implicit contact potential.
        collider: &TriMesh,
        kernel: KernelType,
        friction_params: Option<FrictionParams>,
    ) -> Result<Self, Error> {
        let mut surface_builder = ImplicitSurfaceBuilder::new();
        surface_builder
            .trimesh(&collider)
            .kernel(kernel)
            .sample_type(SampleType::Face)
            .background_field(BackgroundFieldParams {
                field_type: BackgroundFieldType::DistanceBased,
                weighted: false,
            });

        if let Some(surface) = surface_builder.build() {
            let contact_points = object.vertex_positions();

            let vertex_masses = object
                .attrib_as_slice::<MassType, VertexIndex>(MASS_ATTRIB)?
                .to_vec();

            let constraint = ImplicitContactConstraint {
                implicit_surface: RefCell::new(surface),
                contact_points: RefCell::new(Chunked3::from_array_vec(contact_points.to_vec())),
                frictional_contact: friction_params.and_then(|friction_params| {
                    if friction_params.dynamic_friction > 0.0 {
                        Some(FrictionalContact::new(friction_params))
                    } else {
                        None
                    }
                }),
                vertex_masses,
                constraint_buffer: RefCell::new(vec![0.0; contact_points.len()]),
                active_constraint_indices: RefCell::new(Vec::new()),
            };

            constraint
                .implicit_surface
                .borrow()
                .cache_neighbours(contact_points);

            Ok(constraint)
        } else {
            Err(Error::InvalidImplicitSurface)
        }
    }

    pub fn update_contact_points(&self, x: SubsetView<Chunked3<&[f64]>>) {
        let mut contact_points = self.contact_points.borrow_mut();
        x.clone_into_other(&mut *contact_points);
    }
}

impl ContactConstraint for ImplicitContactConstraint {
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
        {
            let mut active_constraint_indices = self.active_constraint_indices.borrow_mut();
            active_constraint_indices.clear();
            if let Ok(indices) = self.active_constraint_indices() {
                active_constraint_indices.extend(indices.into_iter());
            }
        }
        ARef::Cell(std::cell::Ref::map(
            self.active_constraint_indices.borrow(),
            |v| v.as_slice(),
        ))
    }

    #[cfg(feature = "af")]
    fn contact_jacobian_af(&self) -> af::Array<f64> {
        // The contact jacobian for implicit collisions is just a selection matrix of vertices that
        // are in contact, since contacts are colocated with vertex positions.

        let surf_indices = self
            .active_constraint_indices()
            .expect("Failed to retrieve constraint indices.");

        let nnz = surf_indices.len();
        let values = vec![1.0; nnz];
        let rows: Vec<_> = (0i32..nnz as i32).collect();
        let cols: Vec<_> = surf_indices.iter().map(|&i| i as i32).collect();

        let num_contacts = self.contact_points.borrow().len();

        // Build ArrayFire matrix
        let nnz = nnz as u64;
        let num_rows = nnz as u64;
        let num_cols = num_contacts as u64;

        let values = af::Array::new(&values, af::Dim4::new(&[nnz, 1, 1, 1]));
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

    //fn contact_jacobian_sprs(&self) -> sprs::CsMat<f64> {
    //    // The contact jacobian for implicit collisions is just a selection matrix of vertices that
    //    // are in contact, since contacts are colocated with vertex positions.

    //    let surf_indices = self
    //        .active_constraint_indices()
    //        .expect("Failed to retrieve constraint indices.");

    //    let nnz = surf_indices.len();
    //    let values = vec![1.0; nnz];
    //    let rows: Vec<_> = (0..nnz).collect();
    //    let cols: Vec<_> = surf_indices;

    //    let num_rows = nnz;
    //    let num_cols = self.contact_points.borrow().len();

    //    sprs::TriMat::from_triplets((num_rows, num_cols), rows, cols, values).to_csr()
    //}

    fn update_frictional_contact_impulse(
        &mut self,
        contact_impulse: &[f64],
        x: [SubsetView<Chunked3<&[f64]>>; 2],
        v: [SubsetView<Chunked3<&[f64]>>; 2],
        _constraint_values: &[f64],
        mut friction_steps: u32,
    ) -> u32 {
        if self.frictional_contact.is_none() {
            return 0;
        }

        self.update_contact_points(x[0]);
        let normals = self
            .contact_normals()
            .expect("Failed to compute contact normals.");
        let surf_indices = self
            .active_constraint_indices()
            .expect("Failed to retrieve constraint indices.");

        // A set of masses on active contact vertices.
        let object_mass_inv: EffectiveMassInv = From::from(DiagonalBlockMatrix::new(
            surf_indices
                .iter()
                .map(|&surf_idx| {
                    let m = self.vertex_masses[surf_idx];
                    assert!(m > 0.0);
                    [1.0 / m; 3]
                })
                .collect::<Chunked3<Vec<f64>>>(),
        ));

        let ImplicitContactConstraint {
            ref mut frictional_contact,
            ..
        } = *self;

        let frictional_contact = frictional_contact.as_mut().unwrap(); // Must be checked above.

        let mu = frictional_contact.params.dynamic_friction;

        frictional_contact
            .contact_basis
            .update_from_normals(normals.into());
        frictional_contact.object_impulse.clear();
        assert_eq!(contact_impulse.len(), surf_indices.len());

        if false {
            // Polar coords
            let velocity_t: Vec<_> = surf_indices
                .iter()
                .enumerate()
                .map(|(contact_idx, &surf_idx)| {
                    let v = frictional_contact
                        .contact_basis
                        .to_cylindrical_contact_coordinates(v[0][surf_idx], contact_idx);
                    v.tangent
                })
                .collect();

            if true {
                // switch between implicit solver and explicit solver here.
                match FrictionPolarSolver::without_contact_jacobian(
                    &velocity_t,
                    &contact_impulse,
                    &frictional_contact.contact_basis,
                    object_mass_inv.view(),
                    frictional_contact.params,
                ) {
                    Ok(mut solver) => {
                        eprintln!("#### Solving Friction");
                        if let Ok(FrictionSolveResult { solution: r_t, .. }) = solver.step() {
                            frictional_contact.object_impulse.extend_from_slice(
                                &frictional_contact
                                    .contact_basis
                                    .from_polar_tangent_space(reinterpret_vec(r_t)),
                            );
                            friction_steps -= 1;
                        } else {
                            eprintln!("Failed friction solve");
                            friction_steps = 0;
                        }
                    }
                    Err(err) => {
                        dbg!(err);
                        friction_steps = 0;
                    }
                }
            } else {
                for (contact_idx, (&v_t, &cr)) in
                    zip!(velocity_t.iter(), contact_impulse.iter()).enumerate()
                {
                    let r_t = if v_t.radius > 0.0 {
                        Polar2 {
                            radius: mu * cr.abs(),
                            angle: negate_angle(v_t.angle),
                        }
                    } else {
                        Polar2 {
                            radius: 0.0,
                            angle: 0.0,
                        }
                    };
                    let r = frictional_contact
                        .contact_basis
                        .from_cylindrical_contact_coordinates(r_t.into(), contact_idx);
                    frictional_contact.object_impulse.push(r.into());
                }
                friction_steps -= 1;
            }
        } else {
            // Euclidean coords
            let velocity_t: Vec<[f64; 2]> = surf_indices
                .iter()
                .enumerate()
                .map(|(contact_idx, &surf_idx)| {
                    let v = frictional_contact
                        .contact_basis
                        .to_contact_coordinates(v[0][surf_idx], contact_idx);
                    [v[1], v[2]]
                })
                .collect();

            if true {
                // Implicit Friction

                let mut solver = FrictionSolver::without_contact_jacobian(
                    &velocity_t,
                    &contact_impulse,
                    &frictional_contact.contact_basis,
                    object_mass_inv.view(),
                    frictional_contact.params,
                );

                eprintln!("#### Solving Friction");
                let r_t = solver.step();

                frictional_contact.object_impulse.extend_from_slice(
                    &frictional_contact
                        .contact_basis
                        .from_tangent_space(reinterpret_vec(r_t)),
                );

            //// This contact jacobian is a selection matrix or a mapping from contact vertices to
            //// simulation vertices, because contacts are colocated with a subset of  simulation
            //// vertices on the surface.
            //let contact_jacobian_indices: Vec<_> = surf_indices.map(|idx| sim_verts[idx]).collect();
            //let elastic_energy = crate::friction::ElasticEnergyParams {
            //    energy_model: energy_model.clone(),
            //    time_step,
            //};

            //let solver = ElasticFrictionSolver::selection_contact_jacobian(
            //    v,
            //    &contact_impulse,
            //    &frictional_contact.contact_basis,
            //    &vertex_masses,
            //    frictional_contact.params,
            //    &contact_jacobian_indices,
            //    Some(elastic_energy),
            //);

            //eprintln!("#### Solving Friction");
            //let r = solver.step();

            //frictional_contact.impulse.append(&mut reinterpret_vec(r));
            } else {
                // Explicit Friction
                for (contact_idx, (&v_t, &cr)) in
                    zip!(velocity_t.iter(), contact_impulse.iter()).enumerate()
                {
                    let v_t = Vector2(v_t);
                    let v_norm = v_t.norm();
                    let r_t = if v_norm > 0.0 {
                        v_t * (-mu * cr.abs() / v_norm)
                    } else {
                        Vector2::zeros()
                    };
                    let r = frictional_contact
                        .contact_basis
                        .from_contact_coordinates([0.0, r_t[0], r_t[1]], contact_idx);
                    frictional_contact.object_impulse.push(r.into());
                }
            }

            friction_steps -= 1;
        };

        friction_steps
    }

    fn add_mass_weighted_frictional_contact_impulse(
        &self,
        mut vel: [SubsetView<Chunked3<&mut [f64]>>; 2],
    ) {
        if let Some(ref frictional_contact) = self.frictional_contact {
            if frictional_contact.object_impulse.is_empty() {
                return;
            }

            let indices = self
                .active_constraint_indices()
                .expect("Failed to retrieve constraint indices.");

            assert_eq!(indices.len(), frictional_contact.object_impulse.len());

            for (&i, &r) in indices.iter().zip(frictional_contact.object_impulse.iter()) {
                let m = self.vertex_masses[i];
                let v = Vector3(vel[0][i]);
                vel[0][i] = (v + Vector3(r) / m).into();
            }
        }
    }

    fn remap_frictional_contact(&mut self, old_set: &[usize], new_set: &[usize]) {
        // Remap friction forces the same way we remap constraint multipliers for the contact
        // solve.
        if let Some(ref mut frictional_contact) = self.frictional_contact {
            let new_friction_impulses = crate::constraints::remap_values(
                frictional_contact.object_impulse.iter().cloned(),
                [0.0; 3],
                old_set.iter().cloned(),
                new_set.iter().cloned(),
            );

            std::mem::replace(
                &mut frictional_contact.object_impulse,
                Chunked3::from_array_vec(new_friction_impulses),
            );

            frictional_contact.contact_basis.remap(old_set, new_set);
        }
    }

    /// For visualization purposes.
    fn add_contact_impulse(
        &self,
        x: [SubsetView<Chunked3<&[f64]>>; 2],
        contact_impulse: &[f64],
        mut impulse: [Chunked3<&mut [f64]>; 2],
    ) {
        self.update_contact_points(x[0]);
        let normals = self
            .contact_normals()
            .expect("Failed to retrieve contact normals.");
        let surf_indices = self
            .active_constraint_indices()
            .expect("Failed to retrieve constraint indices.");

        assert_eq!(contact_impulse.len(), normals.len());
        assert_eq!(surf_indices.len(), normals.len());

        for (surf_idx, nml, &cr) in zip!(
            surf_indices.into_iter(),
            normals.into_iter(),
            contact_impulse.iter()
        ) {
            impulse[0][surf_idx] = (Vector3(nml) * cr).into();
        }

        let query_points = self.contact_points.borrow();
        assert_eq!(impulse[0].len(), query_points.len());

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
            let imp = Vector3(impulse[1][col]);
            impulse[1][col] = (imp + Matrix3(jac).transpose() * Vector3(impulse[0][row])).into()
        }
    }

    fn contact_normals(
        &self,
    ) -> Result<Chunked3<Vec<f64>>, Error> {
        // Contacts occur at vertex positions of the deforming volume mesh.
        let surf = self.implicit_surface.borrow();
        let contact_points = self.contact_points.borrow_mut();

        let mut normal_coords = vec![0.0; surf.num_query_jacobian_entries()?];
        surf.query_jacobian_values(contact_points.view().into(), &mut normal_coords)?;
        let mut normals = Chunked3::from_flat(normal_coords);

        // Normalize normals.
        // Contact normals point away from the surface being collided against.
        // In this case the gradient coincides with this direction.
        for n in normals.iter_mut() {
            let nml = Vector3(*n);
            let len = nml.norm();
            if len > 0.0 {
                *n = (nml / len).into();
            }
        }

        Ok(normals)
    }

    fn contact_radius(&self) -> f64 {
        self.implicit_surface.borrow().radius()
    }

    fn update_radius_multiplier(&mut self, rad_mult: f64) {
        self.implicit_surface
            .borrow_mut()
            .update_radius_multiplier(rad_mult);
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
        obj_pos: SubsetView<Chunked3<&[f64]>>,
        col_pos: SubsetView<Chunked3<&[f64]>>,
    ) -> bool {
        // Recall: Here, implicit surface is generated by the collision mesh, and query points are
        // coming from the simulation mesh.
        self.update_contact_points(obj_pos);

        let mut surf = self.implicit_surface.borrow_mut();

        let num_vertices_updated = surf.update(col_pos.iter().cloned());
        assert_eq!(num_vertices_updated, surf.surface_vertex_positions().len());

        surf.invalidate_query_neighbourhood();
        surf.cache_neighbours(self.contact_points.borrow().view().into())
    }

    /// Get a list of indices of the surface vertices on the simulation mesh which are neighbouring
    /// the implicit surface. These indices are with respect to the `sim_verts` vector,
    /// not the actual simulation mesh vertices.
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

impl<'a> Constraint<'a, f64> for ImplicitContactConstraint {
    type Input = [SubsetView<'a, Chunked3<&'a [f64]>>; 2]; // Object and Collider vertices

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
        self.update_contact_points(x1[0]);

        let contact_points = self.contact_points.borrow();
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

impl ConstraintJacobian<'_, f64> for ImplicitContactConstraint {
    #[inline]
    fn constraint_jacobian_size(&self) -> usize {
        self.implicit_surface
            .borrow()
            .num_query_jacobian_entries()
            .unwrap_or(0)
    }

    fn constraint_jacobian_indices_iter(
        &self,
    ) -> Result<Box<dyn Iterator<Item = MatrixElementIndex>>, Error> {
        let idx_iter = {
            let surf = self.implicit_surface.borrow();
            surf.query_jacobian_indices_iter()?
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
        self.update_contact_points(x1[0]);
        let contact_points = self.contact_points.borrow();

        Ok(self
            .implicit_surface
            .borrow()
            .query_jacobian_values(contact_points.view().into(), values)?)
    }
}

impl<'a> ConstraintHessian<'a, f64> for ImplicitContactConstraint {
    type InputDual = &'a [f64];

    #[inline]
    fn constraint_hessian_size(&self) -> usize {
        self.implicit_surface
            .borrow()
            .num_query_hessian_product_entries()
            .unwrap_or(0)
    }

    fn constraint_hessian_indices_iter(
        &self,
    ) -> Result<Box<dyn Iterator<Item = MatrixElementIndex>>, Error> {
        let surf = self.implicit_surface.borrow();
        Ok(Box::new(
            surf.query_hessian_product_indices_iter()?
                .map(move |(row, col)| MatrixElementIndex { row, col }),
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
        self.update_contact_points(x1[0]);
        let contact_points = self.contact_points.borrow();

        Ok(self
            .implicit_surface
            .borrow()
            .query_hessian_product_scaled_values(
                contact_points.view().into(),
                lambda,
                scale,
                values,
            )?)
    }
}
