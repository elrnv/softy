use super::*;
use crate::constraints::compute_vertex_masses;
use crate::constraint::*;
use crate::friction::*;
use crate::contact::*;
use crate::matrix::*;
use crate::Error;
use crate::Index;
use crate::TetMesh;
use crate::TriMesh;
use geo::math::{Vector2, Vector3};
use geo::mesh::topology::*;
use geo::mesh::{Attrib, VertexPositions};
use implicits::*;
use reinterpret::*;
use std::{cell::RefCell, rc::Rc};
use utils::zip;

/// Enforce a contact constraint on a mesh against an animated implicit surface. This constraint prevents
/// vertices of the simulation mesh from penetrating through the implicit surface.
#[derive(Clone, Debug)]
pub struct SPImplicitContactConstraint {
    /// Implicit surface that represents the collision object.
    pub implicit_surface: RefCell<ImplicitSurface>,
    pub simulation_mesh: Rc<RefCell<TetMesh>>,
    pub collision_object: Rc<RefCell<TriMesh>>,
    /// Mapping from points on the surface of the simulation mesh to the actual
    /// vertices on the tetrahedron mesh.
    pub sim_verts: Vec<usize>,
    /// A buffer of vertex positions on the simulation mesh. This is used to avoid reallocating
    /// contiguous space for these positions every time the constraint is evaluated.
    pub query_points: RefCell<Vec<[f64; 3]>>,
    /// Friction impulses applied during contact.
    pub frictional_contact: Option<FrictionalContact>,
    /// A mass for each vertex in the simulation mesh.
    pub vertex_masses: Vec<f64>,

    /// Internal constraint function buffer used to store temporary constraint computations.
    constraint_buffer: RefCell<Vec<f64>>,

    /// Worspace vector to keep track of active constraint indices.
    active_constraint_indices: RefCell<Vec<usize>>,
}

impl SPImplicitContactConstraint {
    /// Build an implicit surface from the given trimesh, and constrain the tetmesh vertices to lie
    /// strictly outside of it.
    pub fn new(
        tetmesh_rc: &Rc<RefCell<TetMesh>>,
        trimesh_rc: &Rc<RefCell<TriMesh>>,
        kernel: KernelType,
        friction_params: Option<FrictionParams>,
        density: f64,
    ) -> Result<Self, Error> {
        let trimesh = trimesh_rc.borrow();

        let mut surface_builder = ImplicitSurfaceBuilder::new();
        surface_builder
            .trimesh(&trimesh)
            .kernel(kernel)
            .sample_type(SampleType::Face)
            .background_field(BackgroundFieldParams {
                field_type: BackgroundFieldType::DistanceBased,
                weighted: false,
            });

        if let Some(surface) = surface_builder.build() {
            let tetmesh = tetmesh_rc.borrow();
            let mut surf_mesh = tetmesh.surface_trimesh_with_mapping(Some("i"), None, None, None);
            let sim_verts = surf_mesh
                .remove_attrib::<VertexIndex>("i")
                .expect("Failed to map indices.")
                .into_buffer()
                .into_vec::<usize>()
                .expect("Incorrect index type: not usize");

            let query_points = surf_mesh.vertex_positions();

            let vertex_masses = compute_vertex_masses(&tetmesh, density);

            let constraint = SPImplicitContactConstraint {
                implicit_surface: RefCell::new(surface),
                simulation_mesh: Rc::clone(tetmesh_rc),
                collision_object: Rc::clone(trimesh_rc),
                sim_verts,
                query_points: RefCell::new(query_points.to_vec()),
                frictional_contact: friction_params.and_then(|fparams| {
                    if fparams.dynamic_friction > 0.0 {
                        Some(FrictionalContact::new(fparams))
                    } else {
                        None
                    }
                }),
                vertex_masses,
                constraint_buffer: RefCell::new(vec![0.0; query_points.len()]),
                active_constraint_indices: RefCell::new(Vec::new()),
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

    pub fn update_query_points_with_mesh_pos(&self, x: &[f64]) {
        let pos = reinterpret_slice(x);
        self.update_query_points(self.sim_verts.iter().map(|&i| pos[i]));
    }

    pub fn update_query_points(&self, q_iter: impl Iterator<Item = [f64; 3]>) {
        let mut query_points = self.query_points.borrow_mut();
        query_points.clear();
        query_points.extend(q_iter);
    }
}

impl ContactConstraint for SPImplicitContactConstraint {
    fn frictional_contact(&self) -> Option<&FrictionalContact> {
        self.frictional_contact.as_ref()
    }
    fn frictional_contact_mut(&mut self) -> Option<&mut FrictionalContact> {
        self.frictional_contact.as_mut()
    }
    fn vertex_index_mapping(&self) -> Option<&[usize]> {
        Some(&self.sim_verts)
    }
    fn active_surface_vertex_indices(&self) -> ARef<'_, [usize]> {
        {
            let mut active_constraint_indices = self.active_constraint_indices.borrow_mut();
            active_constraint_indices.clear();
            if let Ok(mut indices) = self.active_constraint_indices() {
                active_constraint_indices.append(&mut indices);
            }
        }
        ARef::Cell(std::cell::Ref::map(self.active_constraint_indices.borrow(), |v| v.as_slice()))
    }

    fn update_frictional_contact_impulse(
        &mut self,
        contact_impulse: &[f64],
        x: &[[f64; 3]],
        v: &[[f64; 3]],
        _constraint_values: &[f64],
        mut friction_steps: u32,
    ) -> u32 {
        if self.frictional_contact.is_none() {
            return 0;
        }

        let normals = self
            .contact_normals(reinterpret::reinterpret_slice(x))
            .expect("Failed to compute contact normals.");
        let surf_indices = self
            .active_constraint_indices()
            .expect("Failed to retrieve constraint indices.");

        // A set of masses on active contact vertices.
        let contact_masses: Vec<_> = surf_indices
            .iter()
            .map(|&surf_idx| self.vertex_masses[self.sim_verts[surf_idx]])
            .collect();

        let SPImplicitContactConstraint {
            ref mut frictional_contact,
            ref mut sim_verts,
            ..
        } = *self;

        let frictional_contact = frictional_contact.as_mut().unwrap(); // Must be checked above.

        let mu = frictional_contact.params.dynamic_friction;

        frictional_contact.contact_basis.update_from_normals(normals);
        frictional_contact.impulse.clear();
        let contact_impulse = if contact_impulse.is_empty() {
            vec![0.0; surf_indices.len()]
        } else {
            contact_impulse.to_vec()
        };

        assert_eq!(contact_impulse.len(), surf_indices.len());
        
        eprintln!("#### Solveing Friction");
        let success = if false {
            // Polar coords
            let velocity_t: Vec<_> = surf_indices
                .iter()
                .enumerate()
                .map(|(contact_idx, &surf_idx)| {
                    let vtx_idx = sim_verts[surf_idx];
                    let v = frictional_contact.contact_basis.to_cylindrical_contact_coordinates(v[vtx_idx], contact_idx);
                    v.tangent
                }).collect();

            if true {
                // switch between implicit solver and explicit solver here.
                match FrictionPolarSolver::without_contact_jacobian(
                    &velocity_t, &contact_impulse, &frictional_contact.contact_basis, &contact_masses, frictional_contact.params) {
                    Ok(mut solver) => {
                        eprintln!("#### Solving Friction");
                        if let Ok(FrictionSolveResult {
                            solution: r_t,
                            ..
                        }) = solver.step()
                        {
                            frictional_contact.impulse.append(&mut frictional_contact.contact_basis.from_polar_tangent_space(reinterpret_vec(r_t)));
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
                        Polar2 { radius: 0.0, angle: 0.0 }
                    };
                    let r = 
                        frictional_contact.contact_basis
                            .from_cylindrical_contact_coordinates(r_t.into(), contact_idx);
                    frictional_contact.impulse.push(r.into());
                }
                friction_steps -= 1;
           }
        } else {
            // Euclidean coords
            let (velocity_n, velocity_t): (Vec<_>, Vec<_>) = surf_indices
                .iter()
                .enumerate()
                .map(|(contact_idx, &surf_idx)| {
                    let vtx_idx = sim_verts[surf_idx];
                    let v = frictional_contact.contact_basis.to_contact_coordinates(v[vtx_idx], contact_idx);
                    (v[0], [v[1], v[2]])
                }).unzip();

            // switch between implicit solver and explicit solver here.
            if true {
                let mut r_n = contact_impulse.clone();
                let mut r_t = Vec::new();
                while friction_steps > 0 {
                    friction_steps -= 1;
                    let mut solver = ContactSolver::without_contact_jacobian(
                        &velocity_n, &r_t, &frictional_contact.contact_basis, &contact_masses);
                    r_n.clear();
                    r_n.append(&mut solver.step());

                    let mut solver = FrictionSolver::without_contact_jacobian(
                        &velocity_t, &r_n, &frictional_contact.contact_basis, &contact_masses, frictional_contact.params);
                    r_t.clear();
                    r_t.append(&mut solver.step());
                }

                assert_eq!(r_n.len(), r_t.len());
                frictional_contact.impulse.resize(r_n.len(), [0.0; 3]);
                for (i, (&r_n, &r_t, r)) in zip!(r_n.iter(), r_t.iter(), frictional_contact.impulse.iter_mut()).enumerate() {
                    *r = frictional_contact.contact_basis.from_contact_coordinates([r_n, r_t[0], r_t[1]], i).into();
                }
            } else {
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
                        let r = 
                            frictional_contact.contact_basis
                            .from_contact_coordinates([0.0, r_t[0], r_t[1]], contact_idx);
                        frictional_contact.impulse.push(r.into());
                    }
            }
            friction_steps -= 1;
        };

        friction_steps
    }

    fn add_mass_weighted_frictional_contact_impulse(&self, x: &mut [f64]) {
        if let Some(ref frictional_contact) = self.frictional_contact {
            if frictional_contact.impulse.is_empty() {
                return;
            }

            let indices = self
                .active_constraint_indices()
                .expect("Failed to retrieve constraint indices.");

            assert_eq!(indices.len(), frictional_contact.impulse.len());

            for (&i, r) in indices.iter().zip(frictional_contact.impulse.iter()) {
                let surf_vert_idx = self.sim_verts[i];
                let m = self.vertex_masses[surf_vert_idx];
                for (j, impulse) in r.iter().enumerate().take(3) {
                    let idx = 3 * surf_vert_idx + j;
                    x[idx] += impulse / m;
                }
            }
        }
    }

    fn remap_frictional_contact(&mut self, old_set: &[usize], new_set: &[usize]) {
        // Remap friction forces the same way we remap constraint multipliers for the contact
        // solve.
        if let Some(ref mut frictional_contact) = self.frictional_contact {
            let new_friction_impulses = crate::constraints::remap_values(
                frictional_contact.impulse.iter().cloned(),
                [0.0; 3],
                old_set.iter().cloned(),
                new_set.iter().cloned(),
            );
            std::mem::replace(&mut frictional_contact.impulse, new_friction_impulses);

            frictional_contact.contact_basis.remap(old_set, new_set);
        }
    }

    /// For visualization purposes.
    fn compute_contact_impulse(
        &self,
        x: &[f64],
        contact_impulse: &[f64],
        impulse: &mut [[f64; 3]],
    ) {
        let normals = self
            .contact_normals(x)
            .expect("Failed to retrieve contact normals.");
        let indices = self
            .active_constraint_indices()
            .expect("Failed to retrieve constraint indices.");

        let contact_impulse = if contact_impulse.is_empty() {
            vec![0.0; normals.len()]
        } else {
            contact_impulse.to_vec()
        };

        assert_eq!(contact_impulse.len(), normals.len());
        assert_eq!(indices.len(), normals.len());
        for (i, (n, &f)) in indices
            .into_iter()
            .zip(normals.into_iter().zip(contact_impulse.iter()))
        {
            impulse[self.sim_verts[i]] = (Vector3(n) * f).into();
        }
    }

    fn contact_normals(&self, x: &[f64]) -> Result<Vec<[f64; 3]>, Error> {
        // Contacts occur at vertex positions of the deforming volume mesh.
        let surf = self.implicit_surface.borrow();
        self.update_query_points_with_mesh_pos(x);
        let query_points = self.query_points.borrow();

        let mut normal_coords = vec![0.0; surf.num_query_jacobian_entries()?];
        surf.query_jacobian_values(&query_points, &mut normal_coords)?;
        let mut normals: Vec<Vector3<f64>> = reinterpret::reinterpret_vec(normal_coords);

        // Normalize normals.
        // These should be pointing away from the deforming mesh.
        for n in normals.iter_mut() {
            let len = n.norm();
            if len > 0.0 {
                *n /= -len;
            }
        }

        Ok(reinterpret::reinterpret_vec(normals))
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

    fn update_cache(&mut self, x: Option<&[f64]>) -> bool {
        // If positions are given, then use them, otherwise, update cache using the vertex
        // positions in the simulation mesh.
        if let Some(x) = x {
            self.update_query_points_with_mesh_pos(x);
        } else {
            let sim_mesh = self.simulation_mesh.borrow();
            self.update_query_points_with_mesh_pos(reinterpret_slice(sim_mesh.vertex_positions()));
        }

        let collision_mesh = self.collision_object.borrow();
        let mut surf = self.implicit_surface.borrow_mut();

        let num_vertices_updated = surf.update(collision_mesh.vertex_position_iter().cloned());
        assert_eq!(num_vertices_updated, surf.surface_vertex_positions().len());

        surf.invalidate_query_neighbourhood();
        surf.cache_neighbours(&self.query_points.borrow())
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

impl Constraint<f64> for SPImplicitContactConstraint {
    #[inline]
    fn constraint_size(&self) -> usize {
        0
    }

    #[inline]
    fn constraint_bounds(&self) -> (Vec<f64>, Vec<f64>) {
        let m = self.constraint_size();
        (vec![0.0; m], vec![2e10; m])
    }

    #[inline]
    fn constraint(&self, _x0: &[f64], _x1: &[f64], _value: &mut [f64]) {
    }
}

impl ConstraintJacobian<f64> for SPImplicitContactConstraint {
    #[inline]
    fn constraint_jacobian_size(&self) -> usize {
        0
    }

    fn constraint_jacobian_indices_iter<'a>(
        &'a self,
    ) -> Result<Box<dyn Iterator<Item = MatrixElementIndex> + 'a>, Error> {
        Ok(Box::new(std::iter::empty()))
    }

    fn constraint_jacobian_values(
        &self,
        _x0: &[f64],
        _x1: &[f64],
        _values: &mut [f64],
    ) -> Result<(), Error> {
        Ok(())
    }
}

impl ConstraintHessian<f64> for SPImplicitContactConstraint {
    #[inline]
    fn constraint_hessian_size(&self) -> usize {
        0
    }

    fn constraint_hessian_indices_iter<'a>(
        &'a self,
    ) -> Result<Box<dyn Iterator<Item = MatrixElementIndex> + 'a>, Error> {
        Ok(Box::new(std::iter::empty()))
    }

    fn constraint_hessian_values(
        &self,
        _x0: &[f64],
        _x1: &[f64],
        _lambda: &[f64],
        _scale: f64,
        _values: &mut [f64],
    ) -> Result<(), Error> {
        Ok(())
    }
}
