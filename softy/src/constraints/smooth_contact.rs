use crate::constraint::*;
use crate::geo::mesh::topology::*;
use crate::geo::mesh::{VertexPositions, Attrib};
use crate::geo::math::{Matrix3, Vector3};
use crate::geo::ops::Volume;
use crate::geo::prim::Tetrahedron;
use crate::matrix::*;
use crate::PointCloud;
use crate::TetMesh;
use crate::TriMesh;
use reinterpret::*;
use std::collections::BTreeSet;
use std::ops::Add;
use std::{cell::RefCell, rc::Rc};
use implicits::*;

/// Enforce a contact constraint on a mesh against animated vertices. This constraint prevents
/// vertices from occupying the same space as a smooth representation of the simulation mesh.
#[derive(Clone, Debug)]
pub struct SmoothContactConstraint {
    pub simulation_object: Rc<RefCell<TetMesh>>,
    /// Indices to surface vertices of the simulation mesh.
    pub sample_verts: Vec<usize>,
    pub implicit_surface: ImplicitSurface,
    pub collision_object: Rc<RefCell<TriMesh>>,
}

impl SmoothContactConstraint {
    pub fn new(tetmesh_rc: &Rc<RefCell<TetMesh>>, trimesh_rc: &Rc<RefCell<TriMesh>>) -> Self {
        let tetmesh = tetmesh_rc.borrow();
        let surf_verts = tetmesh.surface_vertices();
        let all_points = tetmesh.vertex_positions();
        let points = surf_verts.iter().map(|&i| all_points[i]).collect();

        let mut surface_builder = ImplicitSurfaceBuilder::new();
        surface_builder
            .with_points(points)
            .with_kernel(KernelType::Approximate { radius: 1.0, tolerance: 0.001 })
            .with_background_potential(true);

        if let Ok(all_normals) = tetmesh.attrib_as_slice::<[f32; 3], VertexIndex>("N") {
            let normals = surf_verts.iter()
                .map(|&i| Vector3(all_normals[i]).cast::<f64>().unwrap().into())
                .collect();
            surface_builder.with_normals(normals);
        }

        if let Ok(all_offsets) = tetmesh.attrib_as_slice::<f32, VertexIndex>("offset") {
            let offsets = surf_verts.iter()
                .map(|&i| all_offsets[i] as f64).collect();
            surface_builder.with_offsets(offsets);
        }

        SmoothContactConstraint {
            simulation_object: Rc::clone(tetmesh_rc),
            sample_verts: surf_verts,
            implicit_surface: surface_builder.build(),
            collision_object: Rc::clone(trimesh_rc),
        }
    }

    /// Update implicit surface using points from the deformed tetmesh.
    pub fn update_surface(&mut self) {
        let tetmesh = self.simulation_object.borrow();

        let all_points = tetmesh.vertex_positions();
        let points_iter = self.sample_verts.iter().map(|&i| all_points[i]);

        self.implicit_surface
            .update_points(points_iter);
    }

    /// Update implicit surface using the given position vector.
    pub fn update_surface_with(&mut self, x: &[f64]) {
        let all_points: &[[f64;3]] = reinterpret_slice(x);
        let points_iter = self.sample_verts.iter().map(|&i| all_points[i]);
        let sample_verts_iter = self.sample_verts.iter();

        // TODO: Manually recompute normals, this requires storing the whole topology, not just
        // verts
        let tetmesh = self.simulation_object.borrow();
        let normals_iter = tetmesh
            .attrib_as_slice::<[f32; 3], VertexIndex>("N")
            .ok()
            .map(move |all_normals| sample_verts_iter
                .map(move |&i| Vector3(all_normals[i]).cast::<f64>().unwrap().into()));

        self.implicit_surface
            .update_points_and_normals(points_iter, normals_iter);
    }
}

impl Constraint<f64> for SmoothContactConstraint {
    #[inline]
    fn constraint_size(&self) -> usize {
        self.collision_object.borrow().num_vertices()
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
        let query_points = self.collision_object.borrow().vertex_positions();
        self.implicit_surface.compute_potential(query_points, || false, value);
    }
}

//impl SmoothContactConstraint {
//    /// Compute the indices of the sparse matrix entries of the constraint Jacobian.
//    pub fn constraint_jacobian_indices_iter<'a>(&'a self) -> impl Iterator<Item=MatrixElementIndex> + 'a {
//
//    }
//
//    /// Compute the values of the constraint Jacobian.
//    pub fn constraint_jacobian_values_iter<'a>(&'a self, x: &'a [f64]) -> impl Iterator<Item=f64> + 'a {
//
//    }
//}

impl ConstraintJacobian<f64> for SmoothContactConstraint {
    #[inline]
    fn constraint_jacobian_size(&self) -> usize {
        0
    }
    fn constraint_jacobian_indices_iter<'a>(
        &'a self,
    ) -> Box<dyn Iterator<Item = MatrixElementIndex> + 'a>
    {
        debug_assert_eq!(indices.len(), self.constraint_jacobian_size());
        Box::new(std::iter::empty())
    }
    fn constraint_jacobian_values(&self, x: &[f64], values: &mut [f64]) {
        debug_assert_eq!(values.len(), self.constraint_jacobian_size());
    }
}

//impl SmoothContactConstraint {
//    pub fn constraint_hessian_indices_iter<'a>(&'a self) -> impl Iterator<Item=MatrixElementIndex> + 'a {
//    }
//
//    pub fn constraint_hessian_values_iter<'a>(&'a self, x: &'a [f64], lambda: &'a [f64]) -> impl Iterator<Item=f64> + 'a {
//    }
//}

impl ConstraintHessian<f64> for SmoothContactConstraint {
    #[inline]
    fn constraint_hessian_size(&self) -> usize {
        0
    }
    fn constraint_hessian_indices_iter<'a>(
        &'a self,
    ) -> Box<dyn Iterator<Item = MatrixElementIndex> + 'a>
    {
        debug_assert_eq!(indices.len(), self.constraint_hessian_size());
        Box::new(std::iter::empty())
    }
    fn constraint_hessian_values(&self, x: &[f64], lambda: &[f64], values: &mut [f64]) {
        debug_assert_eq!(values.len(), self.constraint_hessian_size());
    }
}
