use crate::constraint::*;
use geo::math::Vector3;
use geo::mesh::topology::*;
use geo::mesh::{Attrib, VertexPositions};
use crate::matrix::*;
use crate::TetMesh;
use crate::TriMesh;
use implicits::*;
use reinterpret::*;
use std::{cell::RefCell, rc::Rc};
use crate::Index;
use super::ContactConstraint;

/// Enforce a contact constraint on a mesh against an animated implicit surface. This constraint prevents
/// vertices of the simulation mesh from penetrating through the implicit surface.
#[derive(Clone, Debug)]
pub struct ImplicitContactConstraint {
    /// Implicit surface that represents the collision object.
    pub implicit_surface: RefCell<ImplicitSurface>,
    pub simulation_mesh: Rc<RefCell<TetMesh>>,
    /// Mapping from constrained points on the surface of the simulation mesh to the actual
    /// vertices on the tetrahedron mesh.
    pub simulation_surf_verts: Vec<usize>,
    /// A buffer of vertex positions on the simulation mesh. This is used to avoid reallocating
    /// contiguous space for these positions every time the constraint is evaluated.
    pub query_points: RefCell<Vec<[f64;3]>>,

    /// Internal constraint function buffer used to store temporary constraint computations.
    constraint_buffer: RefCell<Vec<f64>>,
}

impl ImplicitContactConstraint {
    /// Build an implicit surface from the given trimesh, and constrain the tetmesh vertices to lie
    /// strictly outside of it.
    pub fn new(
        tetmesh_rc: &Rc<RefCell<TetMesh>>,
        trimesh_rc: &Rc<RefCell<TriMesh>>,
        radius: f64,
        tolerance: f64,
    ) -> Result<Self, crate::Error> {
        let trimesh = trimesh_rc.borrow();

        let mut surface_builder = ImplicitSurfaceBuilder::new();
        surface_builder
            .trimesh(&trimesh)
            .kernel(KernelType::Approximate {
                radius,
                tolerance,
            })
            .sample_type(SampleType::Face)
            .background_field(BackgroundFieldParams { field_type: BackgroundFieldType::Zero, weighted: false });

        if let Some(surface) = surface_builder.build() {
            let tetmesh = tetmesh_rc.borrow();
            let mut surf_mesh = tetmesh.surface_trimesh_with_mapping(Some("i"), None, None, None);
            let surf_verts = surf_mesh.remove_attrib::<VertexIndex>("i")
                .expect("Failed to map indices.")
                .into_buffer()
                .into_vec::<usize>()
                .expect("Incorrect index type: not usize");

            let query_points = trimesh.vertex_positions();

            let constraint = ImplicitContactConstraint {
                implicit_surface: RefCell::new(surface),
                simulation_mesh: Rc::clone(tetmesh_rc),
                simulation_surf_verts: surf_verts,
                query_points: RefCell::new(query_points.to_vec()),
                constraint_buffer: RefCell::new(vec![0.0; query_points.len()]),
            };

            constraint
                .implicit_surface
                .borrow()
                .cache_neighbours(query_points);

            Ok(constraint)
        } else {
            Err(crate::Error::InvalidImplicitSurface)
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
        self.update_query_points(pos.iter().zip(disp.iter()).map(|(&p, &d)| (p + d).into()));
    }

    pub fn update_query_points(&self, q_iter: impl Iterator<Item = [f64;3]>) {
        let mut query_points = self.query_points.borrow_mut();
        for (q, new_q) in query_points.iter_mut().zip(q_iter) {
            *q = new_q;
        }
    }
}

impl ContactConstraint for ImplicitContactConstraint {
    fn contact_radius(&self) -> f64 {
        self.implicit_surface.borrow_mut().radius()
    }

    fn update_radius(&mut self, rad: f64) {
        self.implicit_surface.borrow_mut().update_radius(rad);
    }

    fn update_max_step(&mut self, step: f64) {
        self.implicit_surface.borrow_mut().update_max_step(step);
    }

    fn update_cache(&mut self) -> bool {
        let sim_mesh = self.simulation_mesh.borrow();
        let vert_pos = sim_mesh.vertex_positions();
        self.update_query_points(self.simulation_surf_verts.iter().map(|&i| vert_pos[i]));
        let surf = self.implicit_surface.borrow_mut();
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

        for (i, (idx, _))  in cached_neighbourhood_indices.iter_mut().zip(cached_neighbourhood_sizes.iter())
            .filter(|&(_, &s)| s != 0)
                .enumerate() {
            *idx = Index::new(i);
        }

        cached_neighbourhood_indices
    }


    fn build_constraint_mapping(&self, old_constrained_points: &[Index], new_to_old_constraint_mapping: &mut [Index]) {
        let surf = self.implicit_surface.borrow();
        let neighbourhoods = surf.cached_neighbourhoods().expect("Failed to retrieve cached neighbourhoods");

        assert_eq!(new_to_old_constraint_mapping.len(), neighbourhoods.len());

        // Initialize all mapping indices to invalid.
        for i in new_to_old_constraint_mapping.iter_mut() {
            *i = Index::INVALID;
        }

        // Find the corresponding neighbourhoods of the old constraint points.
        let remapped_neighbourhoods = neighbourhoods.into_iter().map(|i| old_constrained_points[i]);

        // Construct the new-to-told constraint mapping.
        for (mapping, neigh) in new_to_old_constraint_mapping.iter_mut()
            .zip(remapped_neighbourhoods)
            .filter(|(_, i)| i.is_valid())
        {
            *mapping = neigh;
        }
    }
}

impl Constraint<f64> for ImplicitContactConstraint {
    #[inline]
    fn constraint_size(&self) -> usize {
        self.implicit_surface.borrow().num_cached_neighbourhoods().unwrap_or(0)
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

        surf.potential(&query_points, &mut cbuf)
            .unwrap();

        //let bg_pts = self.background_points();
        //let collider_mesh = self.collision_object.borrow();
        //Self::fill_background_potential(&collider_mesh, &bg_pts, radius, &mut cbuf);

        let cached_neighbourhood_sizes =
            surf.cached_neighbourhood_sizes().unwrap();

        //println!("cbuf = ");
        //for c in cbuf.iter() {
        //    print!("{:9.5} ", *c);
        //}
        //println!("");

        // Because `value` tracks only the values for which the neighbourhood is not empty.
        for ((_, new_v), v) in cached_neighbourhood_sizes.iter()
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
            .num_query_jacobian_entries().unwrap_or(0)
    }

    fn constraint_jacobian_indices_iter<'a>(
        &'a self,
    ) -> Box<dyn Iterator<Item = MatrixElementIndex> + 'a> {
        let n = self.constraint_jacobian_size();
        Box::new((0..n).map(move |i| MatrixElementIndex {
            row: i/3,
            col: self.tetmesh_coordinate_index(i),
        }))
    }

    fn constraint_jacobian_values(&self, x: &[f64], dx: &[f64], values: &mut [f64]) {
        debug_assert_eq!(values.len(), self.constraint_jacobian_size());
        self.update_query_points_with_displacement(x, dx);
        let query_points = self.query_points.borrow();

        self.implicit_surface
            .borrow()
            .query_jacobian(&query_points, reinterpret_mut_slice(values))
            .unwrap();
    }
}

impl ConstraintHessian<f64> for ImplicitContactConstraint {
    #[inline]
    fn constraint_hessian_size(&self) -> usize {
        self.implicit_surface
            .borrow()
            .num_query_hessian_product_entries().unwrap_or(0)
    }

    fn constraint_hessian_indices_iter<'a>(
        &'a self,
    ) -> Box<dyn Iterator<Item = MatrixElementIndex> + 'a>
    {
        let surf = self.implicit_surface.borrow();
        Box::new(
            surf.query_hessian_product_indices_iter().unwrap()
            .map(move |(row, col)| MatrixElementIndex {
                row: self.tetmesh_coordinate_index(row),
                col: self.tetmesh_coordinate_index(col),
            })
        )
    }

    fn constraint_hessian_values(&self, x: &[f64], dx: &[f64], lambda: &[f64], values: &mut [f64]) {
        self.update_query_points_with_displacement(x, dx);
        let query_points = self.query_points.borrow();

        self.implicit_surface
            .borrow()
            .surface_hessian_product_values(&query_points, lambda, values)
            .expect("Failed to compute query Hessian values");
    }
}
