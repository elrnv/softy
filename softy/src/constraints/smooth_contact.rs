use crate::constraint::*;
use crate::geo::mesh::topology::*;
use crate::geo::mesh::{VertexPositions, Attrib};
use crate::matrix::*;
use crate::geo::math::Vector3;
use crate::TetMesh;
use crate::TriMesh;
use reinterpret::*;
use std::{cell::RefCell, rc::Rc};
use implicits::*;

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
    pub fn new(tetmesh_rc: &Rc<RefCell<TetMesh>>,
               trimesh_rc: &Rc<RefCell<TriMesh>>,
               params: SmoothContactParams) -> Self {
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

    pub fn update_cache(&mut self, query_points: &[[f64;3]]) {
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
        surf.surface_jacobian_values(query_points, &mut jac_values).unwrap();
        for ((row, col), &jac_val) in jac_idx_iter.zip(jac_values.iter()) {
            value[row] += jac_val*dx[col];
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
    ) -> Box<dyn Iterator<Item = MatrixElementIndex> + 'a>
    {
        self.0.constraint_jacobian_indices_iter()
    }

    /// The jacobian of a linear constraint is constant.
    fn constraint_jacobian_values(&self, x: &[f64], _dx: &[f64], values: &mut [f64]) {
        debug_assert_eq!(values.len(), self.constraint_jacobian_size());
        self.0.update_surface(x);
        let collider = self.0.collision_object.borrow();
        let query_points = collider.vertex_positions();
        self.0.implicit_surface.borrow().surface_jacobian_values(query_points, values).unwrap();
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
    ) -> Box<dyn Iterator<Item = MatrixElementIndex> + 'a>
    {
        Box::new(std::iter::empty())
    }
    fn constraint_hessian_values(&self, _x: &[f64], _dx: &[f64], _lambda: &[f64], _values: &mut [f64]) {
        debug_assert_eq!(_values.len(), self.constraint_hessian_size());
    }
}

/// Enforce a contact constraint on a mesh against animated vertices. This constraint prevents
/// vertices from occupying the same space as a smooth representation of the simulation mesh.
#[derive(Clone, Debug)]
pub struct SmoothContactConstraint {
    /// Indices to surface vertices of the simulation mesh.
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
    pub fn new(tetmesh_rc: &Rc<RefCell<TetMesh>>, trimesh_rc: &Rc<RefCell<TriMesh>>, params: SmoothContactParams) -> Self {
        let tetmesh = tetmesh_rc.borrow();
        let triangles: Vec<[usize; 3]> = tetmesh.surface_topo();
        let surf_verts = tetmesh.surface_vertices();
        let all_points = tetmesh.vertex_positions();
        let points = surf_verts.iter().map(|&i| all_points[i]).collect();

        let mut surface_builder = ImplicitSurfaceBuilder::new();
        surface_builder
            .triangles(triangles)
            .vertices(points)
            .kernel(KernelType::Approximate { radius: params.radius, tolerance: params.tolerance })
            .max_step(params.max_step*2.0) // double it because the step can be by samples or query points
            .sample_type(SampleType::Face)
            .background_potential(BackgroundPotentialType::None);

        if let Ok(all_offsets) = tetmesh.attrib_as_slice::<f32, VertexIndex>("offset") {
            let offsets = surf_verts.iter()
                .map(|&i| all_offsets[i] as f64).collect();
            surface_builder.offsets(offsets);
        }

        let surface = surface_builder.build().expect("No surface points detected");

        let constraint = SmoothContactConstraint {
            sample_verts: surf_verts,
            implicit_surface: RefCell::new(surface),
            collision_object: Rc::clone(trimesh_rc),
            iter_count: 0,
        };

        let trimesh = trimesh_rc.borrow();
        let query_points = trimesh.vertex_positions();
        constraint.implicit_surface.borrow().cache_neighbours(query_points);
        constraint
    }

    pub fn reset_iter_count(&mut self) {
        self.iter_count = 0;
    }

    pub fn update_max_step(&mut self, step: f64) {
        self.implicit_surface.borrow_mut().update_max_step(step);
    }

    /// Update implicit surface using the given position and displacement data.
    pub fn update_surface_with_displacement(&self, x: &[f64], dx: &[f64]) {
        let all_displacements: &[Vector3<f64>] = reinterpret_slice(dx);
        let all_points: &[Vector3<f64>] = reinterpret_slice(x);
        let points_iter = self.sample_verts.iter().map(|&i| (all_points[i] + all_displacements[i]).into());

        self.implicit_surface.borrow_mut()
            .update(points_iter);
    }

    /// Update implicit surface using the given position data.
    pub fn update_surface(&self, x: &[f64]) {
        let all_points: &[[f64;3]] = reinterpret_slice(x);
        let points_iter = self.sample_verts.iter().map(|&i| all_points[i]);

        self.implicit_surface.borrow_mut()
            .update(points_iter);
    }

    pub fn update_cache(&mut self, query_points: &[[f64;3]]) {
        let surf = self.implicit_surface.borrow_mut();
        surf.invalidate_neighbour_cache();
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
        self.implicit_surface.borrow().potential(query_points, value).unwrap();
    }
}

impl ConstraintJacobian<f64> for SmoothContactConstraint {
    #[inline]
    fn constraint_jacobian_size(&self) -> usize {
        self.implicit_surface.borrow().num_surface_jacobian_entries()
    }
    fn constraint_jacobian_indices_iter<'a>(
        &'a self,
    ) -> Box<dyn Iterator<Item = MatrixElementIndex> + 'a>
    {
        let idx_iter = {
            let surf = self.implicit_surface.borrow();
            surf.surface_jacobian_indices_iter().unwrap()
        };
        Box::new(idx_iter.map(|(row, col)| MatrixElementIndex { row, col }))
    }
    fn constraint_jacobian_values(&self, x: &[f64], dx: &[f64], values: &mut [f64]) {
        debug_assert_eq!(values.len(), self.constraint_jacobian_size());
        self.update_surface_with_displacement(x, dx);
        let collider = self.collision_object.borrow();
        let query_points = collider.vertex_positions();
        self.implicit_surface.borrow().surface_jacobian_values(query_points, values).unwrap();
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
