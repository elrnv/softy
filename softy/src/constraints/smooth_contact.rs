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

///// Linearization of a constraint given by `C`.
//pub struct Linearized<C> {
//    constraint: C,
//}
//
//impl<T, C> Constraint<T> for Linearized<C> {
//    #[inline]
//    fn constraint_size(&self) -> usize {
//        // Forward to full implementation
//        self.0.constraint_size()
//    }
//
//    #[inline]
//    fn constraint_bounds(&self) -> (Vec<f64>, Vec<f64>) {
//        // Forward to full implementation
//        self.0.constraint_bounds()
//    }
//
//    #[inline]
//    fn constraint(&self, dx: &[f64], value: &mut [f64]) {
//        self.0.constraint(dx, value);
//
//        // Get Jacobian index iterator.
//        let surf = self.0.implicit_surface.borrow();
//        let jac_idx_iter = surf.surface_jacobian_indices_iter().unwrap();
//
//        // Compute Jacobian . dx (dot product).
//        let mut jac_values = vec![0.0; self.constraint_jacobian_size()];
//        self.current_constraint_jacobian_values(jac_values.as_mut_slice());
//        for ((row, col), &jac_val) in jac_idx_iter.zip(jac_values.iter()) {
//            value[row] += jac_val*dx[col];
//        }
//        //println!("g = {:?}", value);
//    }
//}

/// A linearized version of the smooth contact constraint.
#[derive(Clone, Debug)]
pub struct LinearSmoothContactConstraint(pub SmoothContactConstraint);

impl LinearSmoothContactConstraint {
    pub fn new(
        tetmesh_rc: &Rc<RefCell<TetMesh>>,
        trimesh_rc: &Rc<RefCell<TriMesh>>,
        params: SmoothContactParams,
    ) -> Self {
        LinearSmoothContactConstraint(SmoothContactConstraint::new(tetmesh_rc, trimesh_rc, params))
    }

    pub fn update_max_step(&mut self, step: f64) {
        self.0.update_max_step(step);
    }

    ///// This function computes the constraint value for the current configuration.
    //pub fn current_constraint_value(&self, values: &mut [f64]) {
    //    debug_assert_eq!(values.len(), self.constraint_size());
    //    let collider = self.0.collision_object.borrow();
    //    let query_points = collider.vertex_positions();
    //    self.0.implicit_surface.borrow().potential(query_points, values).unwrap();
    //}

    pub fn update_cache(&mut self, query_points: &[[f64; 3]]) {
        self.0.update_cache(query_points);
    }

    #[inline]
    pub fn reset_iter_count(&mut self) {
        self.0.reset_iter_count();
    }
}

impl Constraint<f64> for LinearSmoothContactConstraint {
    #[inline]
    fn constraint_size(&self) -> usize {
        // Forward to full implementation
        self.0.constraint_size()
    }

    #[inline]
    fn constraint_bounds(&self) -> (Vec<f64>, Vec<f64>) {
        // Forward to full implementation
        self.0.constraint_bounds()
    }

    #[inline]
    fn constraint(&self, x: &[f64], dx: &[f64], value: &mut [f64]) {
        debug_assert_eq!(value.len(), self.constraint_size());
        let collider = self.0.collision_object.borrow();
        let query_points = collider.vertex_positions();

        for val in value.iter_mut() {
            *val = 0.0; // Clear potential value.
        }

        // Set our surface to be in the previous configuration.
        self.0.update_surface(x);
        let surf = self.0.implicit_surface.borrow();

        // Compute the potential at the given query points
        surf.potential(query_points, value).unwrap();

        // Get Jacobian index iterator.
        let jac_idx_iter = surf.surface_jacobian_indices_iter().unwrap();

        // Compute Jacobian . dx (dot product).
        let mut jac_values = vec![0.0; self.constraint_jacobian_size()];
        surf.surface_jacobian_values(query_points, &mut jac_values)
            .unwrap();
        for ((row, col), &jac_val) in jac_idx_iter.zip(jac_values.iter()) {
            value[row] += jac_val * dx[self.0.tetmesh_coordinate_index(col)];
        }
        //println!("g = {:?}", value);
    }
}

impl ConstraintJacobian<f64> for LinearSmoothContactConstraint {
    #[inline]
    fn constraint_jacobian_size(&self) -> usize {
        self.0.constraint_jacobian_size()
    }
    fn constraint_jacobian_indices_iter<'a>(
        &'a self,
    ) -> Box<dyn Iterator<Item = MatrixElementIndex> + 'a> {
        self.0.constraint_jacobian_indices_iter()
    }

    /// The jacobian of a linear constraint is constant.
    fn constraint_jacobian_values(&self, x: &[f64], _dx: &[f64], values: &mut [f64]) {
        debug_assert_eq!(values.len(), self.constraint_jacobian_size());
        self.0.update_surface(x);
        let collider = self.0.collision_object.borrow();
        let query_points = collider.vertex_positions();
        self.0
            .implicit_surface
            .borrow()
            .surface_jacobian_values(query_points, values)
            .unwrap();
    }
}

// No hessian for linearized constraints
impl ConstraintHessian<f64> for LinearSmoothContactConstraint {
    #[inline]
    fn constraint_hessian_size(&self) -> usize {
        0
    }
    fn constraint_hessian_indices_iter<'a>(
        &'a self,
    ) -> Box<dyn Iterator<Item = MatrixElementIndex> + 'a> {
        Box::new(std::iter::empty())
    }
    fn constraint_hessian_values(
        &self,
        _x: &[f64],
        _dx: &[f64],
        _lambda: &[f64],
        _values: &mut [f64],
    ) {
        debug_assert_eq!(_values.len(), self.constraint_hessian_size());
    }
}

/// Enforce a contact constraint on a mesh against animated vertices. This constraint prevents
/// vertices from occupying the same space as a smooth representation of the simulation mesh.
#[derive(Clone, Debug)]
pub struct SmoothContactConstraint {
    /// Indices to the original tetmesh. This is the mapping from surface mesh vertices to the
    /// original tetmesh vertices. This mapping is important for computing correct Jacobians and
    /// mapping incoming tetmesh vertex positions and displacements to sample positions and
    /// displacements.
    pub sample_verts: Vec<usize>,
    pub implicit_surface: RefCell<ImplicitSurface>,
    pub collision_object: Rc<RefCell<TriMesh>>,
    pub iter_count: usize,
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct SmoothContactParams {
    pub radius: f64,
    pub tolerance: f64,

    /// Maximal displacement length. This prevents the displacement
    /// from being too large as to change the sparsity pattern of the smooth contact constraint
    /// Jacobian. If this is zero, then no limit is assumed (not recommended in contact scenarios).
    pub max_step: f64,
}

impl SmoothContactConstraint {
    pub fn new(
        tetmesh_rc: &Rc<RefCell<TetMesh>>,
        trimesh_rc: &Rc<RefCell<TriMesh>>,
        params: SmoothContactParams,
    ) -> Self {
        let tetmesh = tetmesh_rc.borrow();
        let mut surf_mesh = tetmesh.surface_trimesh_with_mapping(Some("i"), None, None, None);
        let sample_verts = surf_mesh
            .remove_attrib::<VertexIndex>("i")
            .expect("Failed to map indices.")
            .into_buffer()
            .into_vec::<usize>()
            .expect("Incorrect index type: not usize");

        let mut surface_builder = ImplicitSurfaceBuilder::new();
        surface_builder
            .trimesh(&surf_mesh)
            .kernel(KernelType::Approximate {
                radius: params.radius,
                tolerance: params.tolerance,
            })
            .max_step(params.max_step * 2.0) // double it because the step can be by samples or query points
            .sample_type(SampleType::Face)
            .background_field(BackgroundFieldType::None);

        let surface = surface_builder
            .build()
            .expect("No surface points detected.");

        let constraint = SmoothContactConstraint {
            sample_verts,
            implicit_surface: RefCell::new(surface),
            collision_object: Rc::clone(trimesh_rc),
            iter_count: 0,
        };

        let trimesh = trimesh_rc.borrow();
        let query_points = trimesh.vertex_positions();
        constraint
            .implicit_surface
            .borrow()
            .cache_neighbours(query_points);
        constraint
    }

    pub fn reset_iter_count(&mut self) {
        self.iter_count = 0;
    }

    pub fn update_max_step(&mut self, step: f64) {
        self.implicit_surface.borrow_mut().update_max_step(step);
    }

    /// Given an index into the surface point position coordinates, return the corresponding index
    /// into the original `TetMesh`.
    pub fn tetmesh_coordinate_index(&self, idx: usize) -> usize {
        3 * self.sample_verts[idx / 3] + idx % 3
    }

    /// Update implicit surface using the given position and displacement data.
    pub fn update_surface_with_displacement(&self, x: &[f64], dx: &[f64]) {
        let all_displacements: &[Vector3<f64>] = reinterpret_slice(dx);
        let all_positions: &[Vector3<f64>] = reinterpret_slice(x);
        let points_iter = self
            .sample_verts
            .iter()
            .map(|&i| (all_positions[i] + all_displacements[i]).into());

        self.implicit_surface.borrow_mut().update(points_iter);
    }

    /// Update implicit surface using the given position data.
    pub fn update_surface(&self, x: &[f64]) {
        let all_positions: &[[f64; 3]] = reinterpret_slice(x);
        let points_iter = self.sample_verts.iter().map(|&i| all_positions[i]);

        self.implicit_surface.borrow_mut().update(points_iter);
    }

    pub fn update_cache(&mut self, query_points: &[[f64; 3]]) {
        let surf = self.implicit_surface.borrow_mut();
        surf.invalidate_query_neighbourhood();
        surf.cache_neighbours(query_points);
    }
}

impl Constraint<f64> for SmoothContactConstraint {
    #[inline]
    fn constraint_size(&self) -> usize {
        self.implicit_surface.borrow().num_cached_query_points()
    }

    #[inline]
    fn constraint_bounds(&self) -> (Vec<f64>, Vec<f64>) {
        let m = self.constraint_size();
        (vec![0.0; m], vec![2e10; m])
    }

    #[inline]
    fn constraint(&self, x: &[f64], dx: &[f64], value: &mut [f64]) {
        debug_assert_eq!(value.len(), self.constraint_size());
        self.update_surface_with_displacement(x, dx);
        let collider = self.collision_object.borrow();
        let query_points = collider.vertex_positions();
        for val in value.iter_mut() {
            *val = 0.0; // Clear potential value.
        }
        self.implicit_surface
            .borrow()
            .potential(query_points, value)
            .unwrap();
    }
}

impl ConstraintJacobian<f64> for SmoothContactConstraint {
    #[inline]
    fn constraint_jacobian_size(&self) -> usize {
        self.implicit_surface
            .borrow()
            .num_surface_jacobian_entries().unwrap_or(0)
    }
    fn constraint_jacobian_indices_iter<'a>(
        &'a self,
    ) -> Box<dyn Iterator<Item = MatrixElementIndex> + 'a> {
        let idx_iter = {
            let surf = self.implicit_surface.borrow();
            surf.surface_jacobian_indices_iter().unwrap()
        };
        Box::new(idx_iter.map(move |(row, col)| MatrixElementIndex {
            row,
            col: self.tetmesh_coordinate_index(col),
        }))
    }
    fn constraint_jacobian_values(&self, x: &[f64], dx: &[f64], values: &mut [f64]) {
        debug_assert_eq!(values.len(), self.constraint_jacobian_size());
        self.update_surface_with_displacement(x, dx);
        let collider = self.collision_object.borrow();
        let query_points = collider.vertex_positions();
        self.implicit_surface
            .borrow()
            .surface_jacobian_values(query_points, values)
            .unwrap();
    }
}

//impl ConstraintHessian<f64> for SmoothContactConstraint {
//    #[inline]
//    fn constraint_hessian_size(&self) -> usize {
//        0
//    }
//    fn constraint_hessian_indices_iter<'a>(
//        &'a self,
//    ) -> Box<dyn Iterator<Item = MatrixElementIndex> + 'a>
//    {
//        Box::new(std::iter::empty())
//    }
//    fn constraint_hessian_values(&self, _x: &[f64], _lambda: &[f64], _values: &mut [f64]) {
//        debug_assert_eq!(_values.len(), self.constraint_hessian_size());
//    }
//}
