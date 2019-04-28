use super::ContactConstraint;
use crate::constraint::*;
use crate::matrix::*;
use crate::Index;
use crate::TetMesh;
use crate::TriMesh;
use geo::math::{Vector3, Vector2};
use geo::mesh::topology::*;
use geo::mesh::{Attrib, VertexPositions};
use implicits::*;
use reinterpret::*;
use std::{cell::RefCell, rc::Rc};
use crate::Error;
use utils::zip;
use crate::contact::*;

/// Enforce a contact constraint on a mesh against an animated implicit surface. This constraint prevents
/// vertices of the simulation mesh from penetrating through the implicit surface.
#[derive(Clone, Debug)]
pub struct ImplicitContactConstraint {
    /// Implicit surface that represents the collision object.
    pub implicit_surface: RefCell<ImplicitSurface>,
    pub simulation_mesh: Rc<RefCell<TetMesh>>,
    pub collision_object: Rc<RefCell<TriMesh>>,
    /// Mapping from points on the surface of the simulation mesh to the actual
    /// vertices on the tetrahedron mesh.
    pub simulation_surf_verts: Vec<usize>,
    /// A buffer of vertex positions on the simulation mesh. This is used to avoid reallocating
    /// contiguous space for these positions every time the constraint is evaluated.
    pub query_points: RefCell<Vec<[f64; 3]>>,
    /// Friction impulses applied during contact.
    pub friction: Option<Friction>,

    /// Internal constraint function buffer used to store temporary constraint computations.
    constraint_buffer: RefCell<Vec<f64>>,
}

impl ImplicitContactConstraint {
    /// Build an implicit surface from the given trimesh, and constrain the tetmesh vertices to lie
    /// strictly outside of it.
    pub fn new(
        tetmesh_rc: &Rc<RefCell<TetMesh>>,
        trimesh_rc: &Rc<RefCell<TriMesh>>,
        kernel: KernelType,
        friction_params: Option<FrictionParams>,
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
            let surf_verts = surf_mesh
                .remove_attrib::<VertexIndex>("i")
                .expect("Failed to map indices.")
                .into_buffer()
                .into_vec::<usize>()
                .expect("Incorrect index type: not usize");

            let query_points = surf_mesh.vertex_positions();

            let constraint = ImplicitContactConstraint {
                implicit_surface: RefCell::new(surface),
                simulation_mesh: Rc::clone(tetmesh_rc),
                collision_object: Rc::clone(trimesh_rc),
                simulation_surf_verts: surf_verts,
                query_points: RefCell::new(query_points.to_vec()),
                friction: friction_params.and_then(|fparams|
                    if fparams.dynamic_friction > 0.0 {
                        Some(Friction::new(fparams))
                    } else {
                        None
                    }),
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
        3 * self.simulation_surf_verts[idx / 3] + idx % 3
    }

    pub fn update_query_points_with_displacement(&self, x: &[f64], dx: &[f64]) {
        let pos: &[Vector3<f64>] = reinterpret_slice(x);
        let disp: &[Vector3<f64>] = reinterpret_slice(dx);
        self.update_query_points(
            self.simulation_surf_verts
                .iter()
                .map(|&i| (pos[i] + disp[i]).into()),
        );
    }

    pub fn update_query_points(&self, q_iter: impl Iterator<Item = [f64; 3]>) {
        let mut query_points = self.query_points.borrow_mut();
        query_points.clear();
        query_points.extend(q_iter);
    }
}

impl ContactConstraint for ImplicitContactConstraint {
    fn update_friction_force(&mut self, contact_force: &[f64], x: &[[f64;3]], dx: &[[f64;3]]) -> bool {
        if self.friction.is_none() {
            return false;
        }

        let normals = self.contact_normals(reinterpret::reinterpret_slice(x), reinterpret::reinterpret_slice(dx))
            .expect("Failed to compute contact normals.");
        let surf_indices = self.active_constraint_indices().expect("Failed to retrieve constraint indices.");

        let friction = self.friction.as_mut().unwrap(); // Must be checked above.

        let mu = friction.params.dynamic_friction;

        friction.update_contact_basis_from_normals(normals);
        friction.force.clear();
        assert_eq!(contact_force.len(), surf_indices.len());

        for (contact_idx, (surf_idx, &cf)) in zip!(surf_indices.into_iter(), contact_force.iter()).enumerate() {
            let vtx_idx = self.simulation_surf_verts[surf_idx];
            let v = friction.to_contact_coordinates(dx[vtx_idx], contact_idx);
            let f = if v[0] <= 0.0 {
                let v_t = Vector2([v[1], v[2]]); // Tangential component
                let mag = v_t.norm();
                let dir = if mag > 0.0 { v_t / mag } else { Vector2::zeros() };
                let f_t = dir * (mu * cf);
                Vector3(friction.to_physical_coordinates([0.0, f_t[0], f_t[1]], contact_idx).into())
            } else {
                Vector3::zeros()
            };
            friction.force.push(f.into());
        }
        true
    }

    fn subtract_friction_force(&self, grad: &mut [f64]) {
        if let Some(ref friction) = self.friction {
            if friction.force.is_empty() {
                return;
            }

            let indices = self.active_constraint_indices().expect("Failed to retrieve constraint indices.");

            assert_eq!(indices.len(), friction.force.len());

            for (&i, f) in indices.iter().zip(friction.force.iter()) {
                for j in 0..3 {
                    let idx = 3*self.simulation_surf_verts[i] + j;
                    grad[idx] -= f[j];
                }
            }
        }
    }
    fn frictional_dissipation(&self, dx: &[f64]) -> f64 {
        let mut dissipation = 0.0;
        if let Some(ref friction) = self.friction {
            if friction.force.is_empty() {
                return dissipation;
            }

            let indices = self.active_constraint_indices().expect("Failed to retrieve constraint indices.");

            assert_eq!(indices.len(), friction.force.len());

            for (&i, f) in indices.iter().zip(friction.force.iter()) {
                for j in 0..3 {
                    dissipation += dx[3*self.simulation_surf_verts[i] + j] * f[j];
                }
            }
        }
        dissipation
    }

    fn remap_friction(&mut self, old_indices: &[Index]) {
        // Remap friction forces the same way we remap constraint multipliers for the contact
        // solve.
        if let Some(ref mut friction) = self.friction {
            let mut new_friction_forces = friction.force.clone();
            new_friction_forces.resize(old_indices.len(), [0.0; 3]);
            for (old_idx, new_f) in old_indices.iter().zip(new_friction_forces.iter_mut())
                .filter_map(|(&idx, f)| idx.into_option().map(|i| (i, f)))
            {
                *new_f = friction.force[old_idx];
            }
            std::mem::replace(&mut friction.force, new_friction_forces);
        }
    }

    /// For visualization purposes.
    fn compute_contact_impulse(&self, x: &[f64], contact_force: &[f64], dt: f64, impulse: &mut [[f64;3]]) {
        let normals = self.contact_normals(x, &vec![0.0; x.len()])
            .expect("Failed to retrieve contact normals.");
        let indices = self.active_constraint_indices()
            .expect("Failed to retrieve constraint indices.");
        assert_eq!(contact_force.len(), normals.len());
        assert_eq!(indices.len(), normals.len());
        for (i, (n, &f)) in indices.into_iter().zip(normals.into_iter().zip(contact_force.iter())) {
            impulse[self.simulation_surf_verts[i]] = (Vector3(n) * f * dt).into();
        }
    }

    fn contact_normals(&self, x: &[f64], dx: &[f64]) -> Result<Vec<[f64; 3]>, Error> {
        // Contacts occur at vertex positions of the deforming volume mesh.
        let surf = self.implicit_surface.borrow();
        self.update_query_points_with_displacement(x, dx);
        let query_points = self.query_points.borrow();

        let mut normal_coords = vec![0.0; surf.num_query_jacobian_entries()?];
        surf.query_jacobian_values(&query_points, &mut normal_coords)?;
        let mut normals: Vec<Vector3<f64>> = reinterpret::reinterpret_vec(normal_coords);

        // Normalize normals and reverse direction so that normals are pointing away from the
        // deforming mesh.
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
        self.implicit_surface.borrow_mut().update_radius_multiplier(rad_mult);
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

    fn update_cache(&mut self, pos: Option<&[f64]>, disp: Option<&[f64]>) -> bool {
        if let Some(pos) = pos {
            if let Some(disp) = disp {
                self.update_query_points_with_displacement(pos, disp);
            } else {
                let pos: &[[f64;3]] = reinterpret_slice(pos);
                self.update_query_points(self.simulation_surf_verts.iter().map(|&i| pos[i]));
            }
        } else {
            let sim_mesh = self.simulation_mesh.borrow();
            let vert_pos: &[Vector3<f64>] = reinterpret_slice(sim_mesh.vertex_positions());
            if let Some(disp) = disp {
                let disp: &[Vector3<f64>] = reinterpret_slice(disp);
                self.update_query_points(
                    self.simulation_surf_verts
                        .iter()
                        .map(|&i| (vert_pos[i] + disp[i]).into())
                );
            } else {
                self.update_query_points(self.simulation_surf_verts.iter().map(|&i| vert_pos[i].into()));
            }
        }

        let collision_mesh = self.collision_object.borrow();
        let mut surf = self.implicit_surface.borrow_mut();

        let num_vertices_updated = surf.update(collision_mesh.vertex_position_iter().cloned());
        assert_eq!(num_vertices_updated, surf.surface_vertex_positions().len());

        surf.invalidate_query_neighbourhood();
        surf.cache_neighbours(&self.query_points.borrow())
    }

    /// Get a list of indices of the surface vertices on the simulation mesh which are neighbouring
    /// the implicit surface. These indices are with respect to the `simulation_surf_verts` vector,
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

impl Constraint<f64> for ImplicitContactConstraint {
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
    fn constraint(&self, x: &[f64], dx: &[f64], value: &mut [f64]) {
        debug_assert_eq!(value.len(), self.constraint_size());
        self.update_query_points_with_displacement(x, dx);

        let query_points = self.query_points.borrow();
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

        surf.potential(&query_points, &mut cbuf).unwrap();

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

impl ConstraintJacobian<f64> for ImplicitContactConstraint {
    #[inline]
    fn constraint_jacobian_size(&self) -> usize {
        self.implicit_surface
            .borrow()
            .num_query_jacobian_entries()
            .unwrap_or(0)
    }

    fn constraint_jacobian_indices_iter<'a>(
        &'a self,
    ) -> Result<Box<dyn Iterator<Item = MatrixElementIndex> + 'a>, Error> {
        let idx_iter = {
            let surf = self.implicit_surface.borrow();
            surf.query_jacobian_indices_iter()?
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

    fn constraint_jacobian_values(&self, x: &[f64], dx: &[f64], values: &mut [f64]) -> Result<(), Error> {
        debug_assert_eq!(values.len(), self.constraint_jacobian_size());
        self.update_query_points_with_displacement(x, dx);
        let query_points = self.query_points.borrow();

        Ok(self.implicit_surface
            .borrow()
            .query_jacobian_values(&query_points, values)?)
    }
}

impl ConstraintHessian<f64> for ImplicitContactConstraint {
    #[inline]
    fn constraint_hessian_size(&self) -> usize {
        self.implicit_surface
            .borrow()
            .num_query_hessian_product_entries()
            .unwrap_or(0)
    }

    fn constraint_hessian_indices_iter<'a>(
        &'a self,
    ) -> Result<Box<dyn Iterator<Item = MatrixElementIndex> + 'a>, Error> {
        let surf = self.implicit_surface.borrow();
        Ok(Box::new(
            surf.query_hessian_product_indices_iter()?
                .map(move |(row, col)| MatrixElementIndex {
                    row: self.tetmesh_coordinate_index(row),
                    col: self.tetmesh_coordinate_index(col),
                }),
        ))
    }

    fn constraint_hessian_values(&self, x: &[f64], dx: &[f64], lambda: &[f64], values: &mut [f64]) -> Result<(), Error> {
        self.update_query_points_with_displacement(x, dx);
        let query_points = self.query_points.borrow();

        Ok(self.implicit_surface
            .borrow()
            .query_hessian_product_values(&query_points, lambda, values)?)
    }
}
