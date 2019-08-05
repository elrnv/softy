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
use utils::soap::*;
use utils::zip;

/// Enforce a contact constraint on a mesh against animated vertices. This constraint prevents
/// vertices from occupying the same space as a smooth representation of the simulation mesh.
#[derive(Clone, Debug)]
pub struct PointContactConstraint {
    /// Implicit surface that represents the deforming object.
    pub implicit_surface: RefCell<ImplicitSurface>,
    /// Points where collision and contact occurs.
    pub contact_points: RefCell<Chunked3<Vec<f64>>>,

    /// Friction impulses applied during contact.
    pub frictional_contact: Option<FrictionalContact>,
    /// A mass for each vertex in the object mesh.
    pub vertex_masses: Vec<f64>,

    /// Store the indices to the Hessian here. These will be served through the constraint
    /// interface.
    surface_hessian_rows: RefCell<Vec<usize>>,
    surface_hessian_cols: RefCell<Vec<usize>>,

    /// Internal constraint function buffer used to store temporary constraint computations.
    constraint_buffer: RefCell<Vec<f64>>,
}

impl PointContactConstraint {
    pub fn new(
        // Main object experiencing contact against its implicit surface representation.
        object: &TriMesh,
        // Collision object consisting of points pushing against the solid object.
        collider: &TriMesh,
        kernel: KernelType,
        friction_params: Option<FrictionParams>,
    ) -> Result<Self, Error> {
        let mut surface_builder = ImplicitSurfaceBuilder::new();
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

            let vertex_masses = object
                .attrib_as_slice::<MassType, VertexIndex>(MASS_ATTRIB)?
                .to_vec();

            let constraint = PointContactConstraint {
                implicit_surface: RefCell::new(surface),
                contact_points: RefCell::new(Chunked3::from_grouped_vec(query_points.to_vec())),
                frictional_contact: friction_params.and_then(|fparams| {
                    if fparams.dynamic_friction > 0.0 {
                        Some(FrictionalContact::new(fparams))
                    } else {
                        None
                    }
                }),
                vertex_masses,
                surface_hessian_rows: RefCell::new(Vec::new()),
                surface_hessian_cols: RefCell::new(Vec::new()),
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
    fn in_contact_indices(&self, contact_impulse: &[f64]) -> Vec<usize> {
        contact_impulse
            .iter()
            .enumerate()
            .map(|(i, _)| i)
            //.filter_map(|(i, &cf)| {
            //    if cf.abs() > tolerance {
            //        Some(i)
            //    } else {
            //        None
            //    }
            //})
            .collect()
    }
}

impl ContactConstraint for PointContactConstraint {
    fn num_contacts(&self) -> usize {
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
        let num_rows = query_points.len() as u64;
        let num_cols = surf.surface_vertex_positions().len() as u64;

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

    fn contact_jacobian_sprs(&self) -> sprs::CsMat<f64> {
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

        let (rows, cols) = cj_indices_iter.unzip();

        let num_cols = surf.surface_vertex_positions().len();
        let num_rows = query_points.len();

        sprs::TriMat::from_triplets((num_rows, num_cols), rows, cols, cj_values).to_csr()
    }

    fn update_frictional_contact_impulse(
        &mut self,
        contact_impulse: &[f64],
        x: [SubsetView<Chunked3<&[f64]>>; 2],
        v: [SubsetView<Chunked3<&[f64]>>; 2],
        potential_values: &[f64],
        friction_steps: u32,
    ) -> u32 {
        if self.frictional_contact.is_none() || friction_steps == 0 {
            return 0;
        }

        let normals = self
            .contact_normals(x)
            .expect("Failed to compute contact normals.");
        let query_indices = self
            .active_constraint_indices()
            .expect("Failed to retrieve constraint indices.");

        let active_query_indices = self.in_contact_indices(contact_impulse);

        let FrictionalContact {
            impulse: vertex_friction_impulse,
            contact_basis,
            params,
        } = self.frictional_contact.as_mut().unwrap();

        contact_basis.update_from_normals(normals.into());

        let query_points = self.contact_points.borrow();

        // Compute contact jacobian
        let surf = self.implicit_surface.borrow();

        let mut cj_matrices = vec![
            Matrix3::zeros();
            surf.num_contact_jacobian_matrices()
                .expect("Failed to get contact Jacobian size.")
        ];
        surf.contact_jacobian_matrices(
            query_points.view().into(),
            reinterpret_mut_slice(&mut cj_matrices),
        )
        .expect("Failed to compute contact Jacobian.");
        let cj_indices_iter = surf
            .contact_jacobian_matrix_indices_iter()
            .expect("Failed to get contact Jacobian indices.");

        // Friction impulse in physical space at contact positions.
        let mut friction_impulse = vec![Vector3::zeros(); query_points.len()];
        // Velocity in physical space at contact positions.
        let mut velocity = vec![Vector3::zeros(); query_points.len()];

        // Compute product J*u to produce velocities (displacements) at each point of contact in
        // physical space.
        for ((r, c), &m) in cj_indices_iter.clone().zip(cj_matrices.iter()) {
            velocity[r] += m * Vector3(v[0][c]);
        }

        assert_eq!(query_indices.len(), contact_impulse.len());
        assert_eq!(potential_values.len(), contact_impulse.len());

        // Compute contact impulse on active query indices.
        let contact_impulse: Vec<_> = active_query_indices
            .iter()
            .map(|&aqi| contact_impulse[aqi])
            .collect();

        let vertex_masses = &self.vertex_masses;

        let success = if false {
            // Polar coords
            let velocity_t: Vec<_> = active_query_indices
                .iter()
                .map(|&aqi| {
                    let vel: [f64; 3] = velocity[query_indices[aqi]].into();
                    let v = contact_basis.to_cylindrical_contact_coordinates(vel, aqi);
                    v.tangent
                })
                .collect();
            if true {
                // switch between implicit solver and explicit solver here.
                match FrictionPolarSolver::new(
                    &velocity_t,
                    &contact_impulse,
                    &contact_basis,
                    vertex_masses,
                    *params,
                    (&cj_matrices, cj_indices_iter.clone()),
                ) {
                    Ok(mut solver) => {
                        eprintln!("#### Solving Friction");
                        if let Ok(FrictionSolveResult { solution: r_t, .. }) = solver.step() {
                            for (&aqi, &r) in active_query_indices.iter().zip(r_t.iter()) {
                                let r_polar = Polar2 {
                                    radius: r[0],
                                    angle: r[1],
                                };
                                friction_impulse[query_indices[aqi]] = Vector3(
                                    contact_basis
                                        .from_cylindrical_contact_coordinates(r_polar.into(), aqi)
                                        .into(),
                                );
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
                for (contact_idx, (&aqi, &v_t, &cr)) in zip!(
                    active_query_indices.iter(),
                    velocity_t.iter(),
                    contact_impulse.iter()
                )
                .enumerate()
                {
                    let r_t = if v_t.radius > 0.0 {
                        Polar2 {
                            radius: params.dynamic_friction * cr.abs(),
                            angle: negate_angle(v_t.angle),
                        }
                    } else {
                        Polar2 {
                            radius: 0.0,
                            angle: 0.0,
                        }
                    };
                    let r = Vector3(
                        contact_basis
                            .from_cylindrical_contact_coordinates(r_t.into(), contact_idx)
                            .into(),
                    );
                    friction_impulse[query_indices[aqi]] = r;
                }
                true
            }
        } else {
            // Euclidean coords
            let velocity_t: Vec<_> = active_query_indices
                .iter()
                .map(|&aqi| {
                    let vel: [f64; 3] = velocity[query_indices[aqi]].into();
                    let v = contact_basis.to_contact_coordinates(vel, aqi);
                    [v[1], v[2]]
                })
                .collect();
            if false {
                // switch between implicit solver and explicit solver here.
                let mut solver = FrictionSolver::new(
                    &velocity_t,
                    &contact_impulse,
                    &contact_basis,
                    vertex_masses,
                    *params,
                    (&cj_matrices, cj_indices_iter.clone()),
                );
                eprintln!("#### Solving Friction");
                let r_t = solver.step();
                for (&aqi, &r) in active_query_indices.iter().zip(r_t.iter()) {
                    friction_impulse[query_indices[aqi]] = Vector3(
                        contact_basis
                            .from_contact_coordinates([0.0, r[0], r[1]], aqi)
                            .into(),
                    );
                }
                true
            } else {
                for (contact_idx, (&aqi, &v_t, &cr)) in zip!(
                    active_query_indices.iter(),
                    velocity_t.iter(),
                    contact_impulse.iter()
                )
                .enumerate()
                {
                    let v_t = Vector2(v_t);
                    let v_norm = v_t.norm();
                    let r_t = if v_norm > 0.0 {
                        v_t * (-params.dynamic_friction * cr.abs() / v_norm)
                    } else {
                        Vector2::zeros()
                    };
                    let r = Vector3(
                        contact_basis
                            .from_contact_coordinates([0.0, r_t[0], r_t[1]], contact_idx)
                            .into(),
                    );
                    friction_impulse[query_indices[aqi]] = r;
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
        vertex_friction_impulse.clear();
        vertex_friction_impulse.resize(surf.surface_vertex_positions().len(), [0.0; 3]);

        // Copute transpose product J^T*f
        for ((r, c), &m) in cj_indices_iter.zip(cj_matrices.iter()) {
            vertex_friction_impulse[c] =
                (Vector3(vertex_friction_impulse[c]) + m.transpose() * friction_impulse[r]).into();
        }

        friction_steps - 1
    }

    fn add_mass_weighted_frictional_contact_impulse(
        &self,
        mut vel: SubsetView<Chunked3<&mut [f64]>>,
    ) {
        if let Some(ref frictional_contact) = self.frictional_contact {
            if frictional_contact.impulse.is_empty() {
                return;
            }
            for (v, (f, m)) in vel.iter_mut().zip(
                frictional_contact
                    .impulse
                    .iter()
                    .zip(self.vertex_masses.iter()),
            ) {
                for j in 0..3 {
                    v[j] += f[j] / m;
                }
            }
        }
    }

    fn add_friction_impulse(&self, mut grad: SubsetView<Chunked3<&mut [f64]>>, multiplier: f64) {
        if let Some(ref frictional_contact) = self.frictional_contact() {
            if frictional_contact.impulse.is_empty() {
                return;
            }

            for (i, &r) in frictional_contact.impulse.iter().enumerate() {
                grad[i] = (Vector3(grad[i]) + Vector3(r) * multiplier).into();
            }
        }
    }

    fn frictional_dissipation(&self, v: [SubsetView<Chunked3<&[f64]>>; 2]) -> f64 {
        let mut dissipation = 0.0;
        if let Some(ref frictional_contact) = self.frictional_contact {
            for (i, f) in frictional_contact.impulse.iter().enumerate() {
                for j in 0..3 {
                    dissipation += v[0][i][j] * f[j];
                }
            }
        }

        dissipation
    }

    fn remap_frictional_contact(&mut self, old_set: &[usize], new_set: &[usize]) {
        // Remap friction forces the same way we remap constraint multipliers for the contact
        // solve.
        if let Some(ref mut frictional_contact) = self.frictional_contact {
            // No need to remap friction impulse because we store one for every vertex on the
            // surface of the deforming mesh instead of just at contact points.
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
        let normals = self
            .contact_normals(x)
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
        x: [SubsetView<Chunked3<&[f64]>>; 2],
    ) -> Result<Chunked3<Vec<f64>>, Error> {
        // Contacts occur at the vertex positions of the colliding mesh.
        self.update_surface_with_mesh_pos(x[0]);

        let surf = self.implicit_surface.borrow();

        self.update_contact_points(x[1]);
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
        self.implicit_surface
            .borrow_mut()
            .update(object_pos.iter().cloned());

        self.update_contact_points(collider_pos);
        let contact_points = self.contact_points.borrow_mut();

        let surf = self.implicit_surface.borrow_mut();
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
            .num_cached_neighbourhoods().unwrap_or(0)
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
        self.implicit_surface
            .borrow()
            .num_surface_jacobian_entries()
            .unwrap_or(0)
    }
    fn constraint_jacobian_indices_iter(
        &self,
    ) -> Result<Box<dyn Iterator<Item = MatrixElementIndex>>, Error> {
        let idx_iter = {
            let surf = self.implicit_surface.borrow();
            surf.surface_jacobian_indices_iter()?
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
        Ok(self
            .implicit_surface
            .borrow()
            .surface_jacobian_values(contact_points.view().into(), values)?)
    }
}

impl<'a> ConstraintHessian<'a, f64> for PointContactConstraint {
    type InputDual = &'a [f64];
    #[inline]
    fn constraint_hessian_size(&self) -> usize {
        let num = self
            .implicit_surface
            .borrow()
            .num_surface_hessian_product_entries()
            .unwrap_or(0);

        // Allocate the space for the Hessian indices.
        {
            let mut hess_rows = self.surface_hessian_rows.borrow_mut();
            hess_rows.clear();
            hess_rows.resize(num, 0);
        }

        {
            let mut hess_cols = self.surface_hessian_cols.borrow_mut();
            hess_cols.clear();
            hess_cols.resize(num, 0);
        }

        num
    }

    fn constraint_hessian_indices_iter(
        &self,
    ) -> Result<Box<dyn Iterator<Item = MatrixElementIndex>>, Error> {
        let surf = self.implicit_surface.borrow();
        Ok(Box::new(
            surf.surface_hessian_product_indices_iter()?
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
        self.update_surface_with_mesh_pos(x1[0]);
        let surf = self.implicit_surface.borrow();
        self.update_contact_points(x1[1]);
        let contact_points = self.contact_points.borrow_mut();
        surf.surface_hessian_product_scaled_values(
            contact_points.view().into(),
            lambda,
            scale,
            values,
        )?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::*;
    use crate::*;
    use std::path::PathBuf;
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

    /// Pinch a box between two probes.
    /// Given sufficient friction, the box should not fall.
    fn pinch_tester(fc_params: FrictionalContactParams) -> Result<(), Error> {
        use geo::mesh::topology::*;

        let params = SimParams {
            max_iterations: 200,
            max_outer_iterations: 20,
            gravity: [0.0f32, -9.81, 0.0],
            time_step: Some(0.01),
            print_level: 5,
            friction_iterations: 1,
            ..DYNAMIC_PARAMS
        };

        let material = SOLID_MATERIAL
            .with_id(0)
            .with_elasticity(ElasticityParameters::from_young_poisson(1e6, 0.45));

        let clamps = geo::io::load_polymesh(&PathBuf::from("assets/clamps.vtk"))?;
        let mut box_mesh = geo::io::load_tetmesh(&PathBuf::from("assets/box.vtk"))?;
        box_mesh.remove_attrib::<VertexIndex>("fixed")?;

        let mut solver = fem::SolverBuilder::new(params.clone())
            .add_solid(box_mesh, material)
            .add_fixed(clamps, 1)
            .add_frictional_contact(fc_params, (0, 1))
            .build()?;

        for iter in 0..50 {
            let res = solver.step()?;

            //println!("res = {:?}", res);
            assert!(
                res.iterations <= params.max_outer_iterations,
                "Exceeded max outer iterations."
            );

            // Check that the mesh hasn't fallen.
            let tetmesh = &solver.solid(0).tetmesh;

            geo::io::save_tetmesh(tetmesh, &PathBuf::from(&format!("out/mesh_{}.vtk", iter)))?;

            for v in tetmesh.vertex_position_iter() {
                assert!(v[1] > -0.6);
            }
        }

        Ok(())
    }

    /// Pinch a box against a couple of implicit surfaces.
    #[allow(dead_code)]
    fn pinch_against_implicit() -> Result<(), Error> {
        let fc_params = FrictionalContactParams {
            contact_type: ContactType::Point,
            kernel: KernelType::Cubic {
                radius_multiplier: 1.5,
            },
            friction_params: Some(FrictionParams {
                dynamic_friction: 0.4,
                inner_iterations: 40,
                tolerance: 1e-5,
                print_level: 5,
            }),
        };

        pinch_tester(fc_params)
    }
}
