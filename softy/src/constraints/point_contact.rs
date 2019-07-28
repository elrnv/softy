use super::*;
use crate::constraint::*;
use crate::constraints::compute_vertex_masses;
use crate::contact::*;
use crate::friction::*;
use crate::matrix::*;
use crate::Error;
use crate::Index;
use crate::TetMesh;
use crate::TriMesh;
use geo::math::{Matrix3, Vector2, Vector3};
use geo::mesh::topology::*;
use geo::mesh::{Attrib, VertexPositions};
use implicits::*;
use reinterpret::*;
use std::{cell::RefCell, rc::Rc};
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
        friction_params: FrictionParams,
        density: f64,
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
            let query_points = collider.vertex_positions();

            let constraint = PointContactConstraint {
                implicit_surface: RefCell::new(surface),
                contact_points: RefCell::new(Chunked3::from_grouped_slice(query_points.to_vec())),
                frictional_contact: friction_params.and_then(|fparams| {
                    if fparams.dynamic_friction > 0.0 {
                        Some(FrictionalContact::new(fparams))
                    } else {
                        None
                    }
                }),
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

    /// Given an index into the surface point position coordinates, return the corresponding index
    /// into the original `TetMesh`.
    pub fn tetmesh_coordinate_index(&self, idx: usize) -> usize {
        3 * self.sim_verts[idx / 3] + idx % 3
    }

    /// Update implicit surface using the given position data from mesh vertices.
    pub fn update_surface_with_mesh_pos(&self, x: &[f64]) {
        let pos: &[[f64; 3]] = reinterpret_slice(x);
        let points_iter = self.sim_verts.iter().map(|&i| pos[i].into());

        self.implicit_surface.borrow_mut().update(points_iter);
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
        self.contact_points.len()
    }
    fn frictional_contact(&self) -> Option<&FrictionalContact> {
        self.frictional_contact.as_ref()
    }
    fn frictional_contact_mut(&mut self) -> Option<&mut FrictionalContact> {
        self.frictional_contact.as_mut()
    }
    fn vertex_index_mapping(&self) -> Option<&[usize]> {
        None
    }
    fn active_surface_vertex_indices(&self) -> ARef<'_, [usize]> {
        ARef::Plain(&self.sim_verts)
    }

    fn contact_jacobian_af(&self) -> af::Array<f64> {
        // Compute contact jacobian
        let surf = self.implicit_surface.borrow();
        let collider = self.collision_object.borrow();
        let query_points = collider.vertex_positions();

        let mut cj_values = vec![
            0.0;
            surf.num_contact_jacobian_entries()
                .expect("Failed to get contact Jacobian size.")
        ];
        surf.contact_jacobian_values(query_points, reinterpret_mut_slice(&mut cj_values))
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

        let collider = self.collision_object.borrow();

        // Build ArrayFire matrix
        let nnz = nnz as u64;
        let num_rows = self.sim_verts.len() as u64;
        let num_cols = collider.num_vertices() as u64;

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
        let collider = self.collision_object.borrow();
        let query_points = collider.vertex_positions();

        let mut cj_values = vec![
            0.0;
            surf.num_contact_jacobian_entries()
                .expect("Failed to get contact Jacobian size.")
        ];
        surf.contact_jacobian_values(query_points, reinterpret_mut_slice(&mut cj_values))
            .expect("Failed to compute contact Jacobian.");
        let cj_indices_iter = surf
            .contact_jacobian_indices_iter()
            .expect("Failed to get contact Jacobian indices.");

        let (rows, cols) = cj_indices_iter.unzip();

        let collider = self.collision_object.borrow();

        let num_cols = self.sim_verts.len();
        let num_rows = collider.num_vertices();

        sprs::TriMat::from_triplets((num_rows, num_cols), rows, cols, cj_values).to_csr()
    }

    fn update_frictional_contact_impulse(
        &mut self,
        contact_impulse: &[f64],
        x: &[[f64; 3]],
        v: &[[f64; 3]],
        potential_values: &[f64],
        friction_steps: u32,
    ) -> u32 {
        if self.frictional_contact.is_none() {
            return 0;
        }

        let normals = self
            .contact_normals(reinterpret_slice(x))
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

        contact_basis.update_from_normals(normals);

        let collider = self.collision_object.borrow();
        let query_points = collider.vertex_positions();

        // Compute contact jacobian
        let surf = self.implicit_surface.borrow();

        let mut cj_matrices = vec![
            Matrix3::zeros();
            surf.num_contact_jacobian_matrices()
                .expect("Failed to get contact Jacobian size.")
        ];
        surf.contact_jacobian_matrices(query_points, reinterpret_mut_slice(&mut cj_matrices))
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
            let vtx_idx = self.sim_verts[c];
            velocity[r] += m * Vector3(v[vtx_idx]);
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
        vertex_friction_impulse.resize(self.sim_verts.len(), [0.0; 3]);

        // Copute transpose product J^T*f
        for ((r, c), &m) in cj_indices_iter.zip(cj_matrices.iter()) {
            vertex_friction_impulse[c] =
                (Vector3(vertex_friction_impulse[c]) + m.transpose() * friction_impulse[r]).into();
        }

        friction_steps - 1
    }

    fn add_mass_weighted_frictional_contact_impulse(&self, vel: Subset<Chunked3<&mut [f64]>>) {
        if let Some(ref frictional_contact) = self.frictional_contact {
            if frictional_contact.impulse.is_empty() {
                return;
            }
            for (v, f, m) in zip!(
                vel.iter_mut(),
                frictional_contact.impulse.iter(),
                self.vertex_masses.iter()
            ) {
                for j in 0..3 {
                    v[j] += f[j] / m;
                }
            }
        }
    }

    fn add_friction_impulse(&self, grad: &mut [f64], multiplier: f64) {
        let grad: &mut [Vector3<f64>] = reinterpret_mut_slice(grad);
        if let Some(ref frictional_contact) = self.frictional_contact() {
            if frictional_contact.impulse.is_empty() {
                return;
            }

            assert_eq!(self.sim_verts.len(), frictional_contact.impulse.len());
            for (&i, &r) in self.sim_verts.iter().zip(frictional_contact.impulse.iter()) {
                grad[i] += Vector3(r) * multiplier;
            }
        }
    }

    fn frictional_dissipation(&self, v: &[f64]) -> f64 {
        let mut dissipation = 0.0;
        if let Some(ref frictional_contact) = self.frictional_contact {
            for (&i, f) in self.sim_verts.iter().zip(frictional_contact.impulse.iter()) {
                for j in 0..3 {
                    dissipation += v[3 * i + j] * f[j];
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
    fn compute_contact_impulse(
        &self,
        x: Chunked3<&[f64]>,
        contact_impulse: &[f64],
        impulse: Chunked3<&mut [f64]>,
    ) {
        let normals = self
            .contact_normals(x)
            .expect("Failed to retrieve contact normals.");
        let surf_indices = self
            .active_constraint_indices()
            .expect("Failed to retrieve constraint indices.");

        assert_eq!(contact_impulse.len(), normals.len());
        assert_eq!(surf_indices.len(), normals.len());

        let collider = self.collision_object.borrow();
        let query_points = collider.vertex_positions();

        let surf = self.implicit_surface.borrow();
        let mut cj_matrices = vec![
            [[0.0; 3]; 3];
            surf.num_contact_jacobian_matrices()
                .expect("Failed to get contact jacobian size")
        ];
        surf.contact_jacobian_matrices(query_points, &mut cj_matrices)
            .expect("Failed to compute contact jacobian");

        let collider = self.collision_object.borrow();
        let mut surface_impulse = vec![Vector3::zeros(); collider.num_vertices()];

        for (surf_idx, nml, &cf) in zip!(
            surf_indices.into_iter(),
            normals.into_iter(),
            contact_impulse.iter()
        ) {
            surface_impulse[surf_idx] = Vector3(nml) * cf;
        }

        for r in impulse.iter_mut() {
            *r = [0.0; 3];
        }

        let cj_indices_iter = surf
            .contact_jacobian_matrix_indices_iter()
            .expect("Failed to get contact jacobian indices");

        for ((r, c), m) in cj_indices_iter.zip(cj_matrices.into_iter()) {
            let vtx_idx = self.sim_verts[c];
            impulse[vtx_idx] =
                (Vector3(impulse[vtx_idx]) + Matrix3(m).transpose() * surface_impulse[r]).into()
        }
    }

    fn contact_normals(&self, x: Chunked3<&[f64]>) -> Result<Chunked3<Vec<f64>>, Error> {
        // Contacts occur at the vertex positions of the colliding mesh.
        self.update_surface_with_mesh_pos(x);

        let surf = self.implicit_surface.borrow();

        let collider = self.collision_object.borrow();
        let query_points = collider.vertex_positions();

        let mut normal_coords = vec![0.0; surf.num_query_jacobian_entries()?];
        surf.query_jacobian_values(&query_points, &mut normal_coords)?;
        let mut normals: Vec<Vector3<f64>> = reinterpret_vec(normal_coords);

        // Normalize normals
        // Contact normals point away from the surface being collided against.
        // In this case the gradient is opposite of this direction.
        for n in normals.iter_mut() {
            let len = n.norm();
            if len > 0.0 {
                *n /= -len;
            }
        }

        Ok(reinterpret_vec(normals))
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
        object_pos: Subset<Chunked3<&[f64]>>,
        collider_pos: Subset<Chunked3<&[f64]>>,
    ) -> bool {
        self.implicit_surface.borrow_mut().update(object_pos.iter());

        let surf = self.implicit_surface.borrow_mut();
        surf.invalidate_query_neighbourhood();
        surf.cache_neighbours(collider_pos.into())
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

impl Constraint<f64> for PointContactConstraint {
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
    fn constraint(&self, _x0: &[f64], x1: &[f64], value: &mut [f64]) {
        debug_assert_eq!(value.len(), self.constraint_size());
        self.update_surface_with_mesh_pos(x1);
        let collider = self.collision_object.borrow();
        let query_points = collider.vertex_positions();

        let mut cbuf = self.constraint_buffer.borrow_mut();
        let radius = self.contact_radius();

        let surf = self.implicit_surface.borrow();
        for (val, q) in cbuf.iter_mut().zip(query_points.iter()) {
            // Clear potential value.
            let closest_sample = surf.nearest_neighbour_lookup(*q).unwrap();
            if closest_sample.nml.dot(Vector3(*q) - closest_sample.pos) > 0.0 {
                *val = radius;
            } else {
                *val = -radius;
            }
        }

        surf.potential(query_points, &mut cbuf).unwrap();

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

impl ConstraintJacobian<f64> for PointContactConstraint {
    #[inline]
    fn constraint_jacobian_size(&self) -> usize {
        self.implicit_surface
            .borrow()
            .num_surface_jacobian_entries()
            .unwrap_or(0)
    }
    fn constraint_jacobian_indices_iter<'a>(
        &'a self,
    ) -> Result<Box<dyn Iterator<Item = MatrixElementIndex> + 'a>, Error> {
        let idx_iter = {
            let surf = self.implicit_surface.borrow();
            surf.surface_jacobian_indices_iter()?
        };

        let cached_neighbourhood_indices = self.cached_neighbourhood_indices();
        Ok(Box::new(idx_iter.map(move |(row, col)| {
            assert!(cached_neighbourhood_indices[row].is_valid());
            MatrixElementIndex {
                row: cached_neighbourhood_indices[row].unwrap(),
                col: self.tetmesh_coordinate_index(col),
            }
        })))
    }

    fn constraint_jacobian_values(
        &self,
        _x0: &[f64],
        x1: &[f64],
        values: &mut [f64],
    ) -> Result<(), Error> {
        debug_assert_eq!(values.len(), self.constraint_jacobian_size());
        self.update_surface_with_mesh_pos(x1);
        let collider = self.collision_object.borrow();
        let query_points = collider.vertex_positions();
        Ok(self
            .implicit_surface
            .borrow()
            .surface_jacobian_values(query_points, values)?)
    }
}

impl ConstraintHessian<f64> for PointContactConstraint {
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

    fn constraint_hessian_indices_iter<'a>(
        &'a self,
    ) -> Result<Box<dyn Iterator<Item = MatrixElementIndex> + 'a>, Error> {
        let surf = self.implicit_surface.borrow();
        Ok(Box::new(surf.surface_hessian_product_indices_iter()?.map(
            move |(row, col)| MatrixElementIndex {
                row: self.tetmesh_coordinate_index(row),
                col: self.tetmesh_coordinate_index(col),
            },
        )))
    }

    fn constraint_hessian_values(
        &self,
        _x0: &[f64],
        x1: &[f64],
        lambda: &[f64],
        scale: f64,
        values: &mut [f64],
    ) -> Result<(), Error> {
        self.update_surface_with_mesh_pos(x1);
        let surf = self.implicit_surface.borrow();
        let collider = self.collision_object.borrow();
        let query_points = collider.vertex_positions();
        surf.surface_hessian_product_scaled_values(query_points, lambda, scale, values)?;
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
    fn pinch_tester(sc_params: SmoothContactParams) -> Result<(), Error> {
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

        let material = Material {
            elasticity: ElasticityParameters::from_young_poisson(1e6, 0.45),
            ..SOLID_MATERIAL
        };

        let clamps = geo::io::load_polymesh(&PathBuf::from("assets/clamps.vtk"))?;
        let mut box_mesh = geo::io::load_tetmesh(&PathBuf::from("assets/box.vtk"))?;
        box_mesh.remove_attrib::<VertexIndex>("fixed")?;

        let mut solver = fem::SolverBuilder::new(params.clone())
            .solid_material(material)
            .add_solid(box_mesh)
            .add_shell(clamps)
            .smooth_contact_params(sc_params)
            .build()?;

        for iter in 0..50 {
            let res = solver.step()?;

            //println!("res = {:?}", res);
            assert!(
                res.iterations <= params.max_outer_iterations,
                "Exceeded max outer iterations."
            );

            // Check that the mesh hasn't fallen.
            let tetmesh = solver.borrow_mesh();

            geo::io::save_tetmesh(&tetmesh, &PathBuf::from(&format!("out/mesh_{}.vtk", iter)))?;

            for v in tetmesh.vertex_position_iter() {
                assert!(v[1] > -0.6);
            }
        }

        Ok(())
    }

    /// Pinch a box against a couple of implicit surfaces.
    #[allow(dead_code)]
    fn pinch_against_implicit() -> Result<(), Error> {
        let sc_params = SmoothContactParams {
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

        pinch_tester(sc_params)
    }
}
