use crate::constraint::*;
use crate::geo::mesh::topology::*;
use crate::geo::mesh::{VertexPositions, Attrib};
use crate::matrix::*;
use crate::TetMesh;
use crate::TriMesh;
use reinterpret::*;
use std::{cell::RefCell, rc::Rc};
use implicits::*;

/// A linearized version of the smooth contact constraint.
#[derive(Clone, Debug)]
pub struct LinearSmoothContactConstraint(SmoothContactConstraint);

impl LinearSmoothContactConstraint {
    pub fn new(tetmesh_rc: &Rc<RefCell<TetMesh>>,
               trimesh_rc: &Rc<RefCell<TriMesh>>,
               params: SmoothContactParams) -> Self {
        let mut scc = LinearSmoothContactConstraint(SmoothContactConstraint::new(tetmesh_rc, trimesh_rc, params));
        scc.update_surface();
        scc
    }

    pub fn update_max_step(&mut self, step: f64) {
        self.0.update_max_step(step);
    }

    /// Compute the constraint jacobian at the current configuration.
    pub fn current_constraint_jacobian_values(&self, values: &mut [f64]) {
        debug_assert_eq!(values.len(), self.constraint_jacobian_size());
        let collider = self.0.collision_object.borrow();
        let query_points = collider.vertex_positions();
        self.0.implicit_surface.borrow().surface_jacobian_values(query_points, values).unwrap();
    }

    /// Update implicit surface using the stored reference to the simulation object. This is to be
    /// called after the simulation mesh is updated by the simulator.
    pub fn update_surface(&mut self) {
        let tetmesh = self.0.simulation_object.borrow();
        let x: &[f64] = reinterpret_slice(tetmesh.vertex_positions());
        self.0.update_surface_with(x);
    }


    pub fn invalidate_neighbour_data(&mut self) {
        self.0.invalidate_neighbour_data();
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
    fn constraint(&mut self, dx: &[f64], value: &mut [f64]) {
        debug_assert_eq!(value.len(), self.constraint_size());

        let collider = self.0.collision_object.borrow();
        let query_points = collider.vertex_positions();

        for val in value.iter_mut() {
            *val = 0.0; // Clear potential value.
        }

        self.0.implicit_surface.borrow().potential(query_points, value).unwrap();
        println!("vals before jac = {:?}", value);

        // Get Jacobian index iterator.
        let jac_idx_iter = {
            let surf = self.0.implicit_surface.borrow();
            surf.surface_jacobian_indices_iter().unwrap()
        };

        // Compute Jacobian . dx (dot product).
        let mut jac_values = vec![0.0; self.constraint_jacobian_size()];
        self.current_constraint_jacobian_values(jac_values.as_mut_slice());
        for ((row, col), &jac_val) in jac_idx_iter.zip(jac_values.iter()) {
            value[row] += jac_val*dx[col];
        }
        println!("vals after jac = {:?}", value);
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
    fn constraint_jacobian_values(&self, _x: &[f64], values: &mut [f64]) {
        debug_assert_eq!(values.len(), self.constraint_jacobian_size());
        self.current_constraint_jacobian_values(values); // Jacobian is constant.
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
    fn constraint_hessian_values(&self, _dx: &[f64], _lambda: &[f64], _values: &mut [f64]) {
        debug_assert_eq!(_values.len(), self.constraint_hessian_size());
    }
}

/// Enforce a contact constraint on a mesh against animated vertices. This constraint prevents
/// vertices from occupying the same space as a smooth representation of the simulation mesh.
#[derive(Clone, Debug)]
pub struct SmoothContactConstraint {
    pub simulation_object: Rc<RefCell<TetMesh>>,
    /// Indices to surface vertices of the simulation mesh.
    pub sample_verts: Vec<usize>,
    pub implicit_surface: RefCell<ImplicitSurface>,
    pub collision_object: Rc<RefCell<TriMesh>>,
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
            .with_triangles(triangles)
            .with_points(points)
            .with_kernel(KernelType::Approximate { radius: params.radius, tolerance: params.tolerance })
            .with_max_step(params.max_step*2.0) // double it because the step can be by samples or query points
            .with_background_potential(false);

        if let Ok(all_offsets) = tetmesh.attrib_as_slice::<f32, VertexIndex>("offset") {
            let offsets = surf_verts.iter()
                .map(|&i| all_offsets[i] as f64).collect();
            surface_builder.with_offsets(offsets);
        }

        let surface = surface_builder.build().expect("No surface points detected");

        let constraint = SmoothContactConstraint {
            simulation_object: Rc::clone(tetmesh_rc),
            sample_verts: surf_verts,
            implicit_surface: RefCell::new(surface),
            collision_object: Rc::clone(trimesh_rc),
        };

        let trimesh = trimesh_rc.borrow();
        let query_points = trimesh.vertex_positions();
        constraint.implicit_surface.borrow().cache_neighbours(query_points);
        constraint
    }

    pub fn update_max_step(&mut self, step: f64) {
        self.implicit_surface.borrow_mut().update_max_step(step);
    }

    /// Update implicit surface using the given position vector.
    pub fn update_surface_with(&self, x: &[f64]) {
        let all_points: &[[f64;3]] = reinterpret_slice(x);
        let points_iter = self.sample_verts.iter().map(|&i| all_points[i]);

        self.implicit_surface.borrow_mut()
            .update_points(points_iter);
    }

    pub fn invalidate_neighbour_data(&mut self) {
        self.implicit_surface.borrow_mut().invalidate_neighbour_cache();
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
    fn constraint(&mut self, x: &[f64], value: &mut [f64]) {
        debug_assert_eq!(value.len(), self.constraint_size());
        self.update_surface_with(x);
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
    fn constraint_jacobian_values(&self, x: &[f64], values: &mut [f64]) {
        debug_assert_eq!(values.len(), self.constraint_jacobian_size());
        self.update_surface_with(x);
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
