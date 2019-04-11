use super::ContactConstraint;
use crate::constraint::*;
use crate::matrix::*;
use crate::Index;
use crate::TetMesh;
use crate::TriMesh;
use geo::math::Vector3;
use geo::mesh::topology::*;
use geo::mesh::{Attrib, VertexPositions};
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

/*

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
*/

/// Enforce a contact constraint on a mesh against animated vertices. This constraint prevents
/// vertices from occupying the same space as a smooth representation of the simulation mesh.
#[derive(Clone, Debug)]
pub struct PointContactConstraint {
    /// Indices to the original tetmesh. This is the mapping from surface mesh vertices to the
    /// original tetmesh vertices. This mapping is important for computing correct Jacobians and
    /// mapping incoming tetmesh vertex positions and displacements to sample positions and
    /// displacements.
    pub sample_verts: Vec<usize>,
    /// Implicit surface that represents the deforming object.
    pub implicit_surface: RefCell<ImplicitSurface>,
    pub collision_object: Rc<RefCell<TriMesh>>,

    /// Store the indices to the Hessian here. These will be served through the constraint
    /// interface.
    surface_hessian_rows: RefCell<Vec<usize>>,
    surface_hessian_cols: RefCell<Vec<usize>>,

    /// Internal constraint function buffer used to store temporary constraint computations.
    constraint_buffer: RefCell<Vec<f64>>,
}

impl PointContactConstraint {
    pub fn new(
        tetmesh_rc: &Rc<RefCell<TetMesh>>,
        trimesh_rc: &Rc<RefCell<TriMesh>>,
        radius_multiplier: f64,
        tolerance: f64,
    ) -> Result<Self, crate::Error> {
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
            .kernel(KernelType::Approximate { radius_multiplier, tolerance })
            .sample_type(SampleType::Face)
            .background_field(BackgroundFieldParams {
                field_type: BackgroundFieldType::DistanceBased,
                weighted: false,
            });

        if let Some(surface) = surface_builder.build() {
            let trimesh = trimesh_rc.borrow();
            let query_points = trimesh.vertex_positions();

            let constraint = PointContactConstraint {
                sample_verts,
                implicit_surface: RefCell::new(surface),
                collision_object: Rc::clone(trimesh_rc),
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
            Err(crate::Error::InvalidImplicitSurface)
        }
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

    /* Needed for the Linear constraint
    /// Update implicit surface using the given position data.
    pub fn update_surface(&self, x: &[f64]) {
        let all_positions: &[[f64; 3]] = reinterpret_slice(x);
        let points_iter = self.sample_verts.iter().map(|&i| all_positions[i]);

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
            vertex_is_inside[vidx] = if values[vidx] < 0.0 { true } else { false };
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
}

impl ContactConstraint for PointContactConstraint {
    fn contact_radius(&self) -> f64 {
        self.implicit_surface.borrow_mut().radius()
    }

    fn update_radius_multiplier(&mut self, rad: f64) {
        self.implicit_surface.borrow_mut().update_radius_multiplier(rad);
    }

    fn update_max_step(&mut self, step: f64) {
        self.implicit_surface.borrow_mut().update_max_step(step);
    }

    fn active_constraint_indices(&self) -> Result<Vec<usize>, crate::Error> {
        self.implicit_surface
            .borrow()
            .nonempty_neighbourhood_indices()
            .map_err(|_| crate::Error::InvalidImplicitSurface)
    }

    fn update_cache(&mut self) -> bool {
        let mesh = self.collision_object.borrow();
        let query_points = reinterpret_slice::<_, [f64; 3]>(mesh.vertex_positions());
        let surf = self.implicit_surface.borrow_mut();
        surf.invalidate_query_neighbourhood();
        surf.cache_neighbours(query_points)
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
    fn constraint(&self, x: &[f64], dx: &[f64], value: &mut [f64]) {
        debug_assert_eq!(value.len(), self.constraint_size());
        self.update_surface_with_displacement(x, dx);
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
    ) -> Box<dyn Iterator<Item = MatrixElementIndex> + 'a> {
        let idx_iter = {
            let surf = self.implicit_surface.borrow();
            surf.surface_jacobian_indices_iter().unwrap()
        };

        let cached_neighbourhood_indices = self.cached_neighbourhood_indices();
        Box::new(idx_iter.map(move |(row, col)| {
            assert!(cached_neighbourhood_indices[row].is_valid());
            MatrixElementIndex {
                row: cached_neighbourhood_indices[row].unwrap(),
                col: self.tetmesh_coordinate_index(col),
            }
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
    ) -> Box<dyn Iterator<Item = MatrixElementIndex> + 'a> {
        self.constraint_hessian_size(); // allocate hessian index vectors.
        let surf = self.implicit_surface.borrow();
        let mut rows = self.surface_hessian_rows.borrow_mut();
        let mut cols = self.surface_hessian_cols.borrow_mut();
        surf.surface_hessian_product_indices(&mut rows, &mut cols)
            .unwrap();

        Box::new(
            rows.clone()
                .into_iter()
                .zip(cols.clone().into_iter())
                .map(move |(row, col)| MatrixElementIndex {
                    row: self.tetmesh_coordinate_index(row),
                    col: self.tetmesh_coordinate_index(col),
                }),
        )
    }

    fn constraint_hessian_values(&self, x: &[f64], dx: &[f64], lambda: &[f64], values: &mut [f64]) {
        self.update_surface_with_displacement(x, dx);
        let surf = self.implicit_surface.borrow();
        let collider = self.collision_object.borrow();
        let query_points = collider.vertex_positions();
        surf.surface_hessian_product_values(query_points, lambda, values)
            .expect("Failed to compute surface Hessian values");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
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

}
