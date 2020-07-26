use unroll::unroll_for_loops;

use geo::mesh::topology::*;
use geo::prim::Triangle;
use tensr::*;

use crate::attrib_defines::*;
use crate::TriMesh;

/// An `InteriorEdge` is an manifold edge with exactly two neighbouring faces.
///
/// For triangle meshes, an interior edge can be drawn as follows:
///
/// ```verbatim
///     x3
///     /\
///    /f1\e2
/// x1/_e0_\x0
///   \    /
///    \f0/e1
///     \/
///     x2
/// ```
///
/// Here `e0` is the interior edge itself; `e1` and `e2` are used to compute the derivatives of the
/// angle between the adjacent faces. `x0` to `x3` are the vertices making up all the degrees of
/// freedom that affect the edge reflex angle.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) struct InteriorEdge {
    pub faces: [usize; 2],
    /// Each `edge_start[i]` is a vertex in `faces[i]` that marks the start of the
    /// edge. This value is either `0`, `1` or `2` for a triangle face.
    pub edge_start: [u8; 2],
}

impl InteriorEdge {
    /// Index a slice of face data (e.g. reference positions) with a face index (0 or 1) and a
    /// vertex index within the face counting from the `edge_start` vertex.
    ///
    /// For example if `ref_pos: Vec<[[f32; 3]; 3]>` is a collection of reference positions per
    /// triangle face, then getting the reference position of `x3` from edge `e` can be done with:
    ///
    /// ```ignore
    /// e.face_vert(&ref_pos, 1, 2);
    /// ```
    ///
    /// We could get the reference position of `x0` and `x1` either from face 0 or face 1, which
    /// implies the following identities when the two faces have the same orientation:
    ///
    /// ```ignore
    /// assert_eq!(e.face_vert(&ref_pos, 0, 0), e.face_vert(&ref_pos, 1, 1)); // x0
    /// assert_eq!(e.face_vert(&ref_pos, 0, 1), e.face_vert(&ref_pos, 1, 0)); // x1
    /// ```
    #[inline]
    pub fn face_vert<T: Copy>(&self, data: &[[T; 3]], face: usize, vert: u8) -> T {
        data[self.faces[face]][((self.edge_start[face] + vert) % 3) as usize]
    }

    #[inline]
    pub fn new(faces: [usize; 2], edge_start: [u8; 2]) -> Self {
        InteriorEdge { faces, edge_start }
    }

    /// Get edge verts followed by tangent verts `x0` to `x3`.
    #[inline]
    pub fn verts(&self, faces: &[[usize; 3]]) -> [usize; 4] {
        let [v0, v1] = self.edge_verts(faces);
        let [v2, v3] = self.tangent_verts(faces);
        [v0, v1, v2, v3]
    }

    /// Compute the span of the tile.
    ///
    /// The given positions are expected to be of the undeformed configuration. There should be
    /// exactly 3 positions per element in `ref_pos`, which corresponds to one face.
    ///
    /// This corresponds to the quantity `\bar{h}_e` in the "Discrete Shells" paper
    /// [[Grinspun et al. 2003]](http://www.cs.columbia.edu/cg/pdfs/10_ds.pdf).
    #[inline]
    pub fn tile_span<T: Real>(&self, ref_pos: &[[[T; 3]; 3]]) -> T {
        let [f0x0, f0x1, f0x2] = [
            self.face_vert(ref_pos, 0, 0).into_tensor(),
            self.face_vert(ref_pos, 0, 1).into_tensor(),
            self.face_vert(ref_pos, 0, 2).into_tensor(),
        ];
        let [f1x1, f1x0, f1x3] = [
            self.face_vert(ref_pos, 1, 0).into_tensor(),
            self.face_vert(ref_pos, 1, 1).into_tensor(),
            self.face_vert(ref_pos, 1, 2).into_tensor(),
        ];
        debug_assert_ne!((f0x1 - f0x0).norm_squared(), T::zero());
        debug_assert_ne!((f1x1 - f1x0).norm_squared(), T::zero());
        let f0e0 = (f0x1 - f0x0).normalized();
        let f1e0 = (f1x1 - f1x0).normalized();

        let h0 = (f0x2 - f0x0).cross(f0e0).norm();
        let h1 = (f1x3 - f1x0).cross(f1e0).norm();
        // 6 = 3 (third of the triangle) * 2 (to make h0 and h1 *triangle* areas)
        (h0 + h1) / T::from(6.0).unwrap()
    }

    /// Get the vertex indices of the edge endpoints.
    #[inline]
    pub fn edge_verts(&self, faces: &[[usize; 3]]) -> [usize; 2] {
        [self.face_vert(faces, 0, 0), self.face_vert(faces, 0, 1)]
    }

    /// Get the vertex positions of the edge vertices in reference configuration for each face.
    #[inline]
    pub fn ref_edge_verts<T: Real>(&self, ref_pos: &[[[T; 3]; 3]]) -> [[[T; 3]; 2]; 2] {
        [
            [self.face_vert(ref_pos, 0, 0), self.face_vert(ref_pos, 0, 1)],
            [self.face_vert(ref_pos, 1, 0), self.face_vert(ref_pos, 1, 1)],
        ]
    }

    /// Get the vertex positions of the tangent vertices.
    ///
    /// `faces` can be either triplets of vertex indices or reference positions.
    #[inline]
    pub fn tangent_verts<U: Copy>(&self, faces: &[[U; 3]]) -> [U; 2] {
        [self.face_vert(faces, 0, 2), self.face_vert(faces, 1, 2)]
    }

    /// Compute the reference edge length.
    ///
    /// This is the average of the lengths of the corresponding edges from the two adjacent faces.
    /// Note that faces may not be stitched in the reference configuration.
    ///
    /// The given positions are expected to be of the undeformed configuration. There should be
    /// exactly 3 positions per element in `ref_pos`, which corresponds to one face.
    #[inline]
    pub fn ref_length<T: Real>(&self, ref_pos: &[[[T; 3]; 3]]) -> T {
        let [[f0x0, f0x1], [f1x0, f1x1]] = self.ref_edge_verts(ref_pos);
        ((Vector3::new(f0x1) - Vector3::new(f0x0)).norm()
            + (Vector3::new(f1x1) - Vector3::new(f1x0)).norm())
            * T::from(0.5).unwrap()
    }

    /// Produce the 3D vector corresponding to this edge.
    #[inline]
    pub fn edge_vector<T: Real>(&self, pos: &[[T; 3]], faces: &[[usize; 3]]) -> Vector3<T> {
        let [v0, v1] = self.edge_verts(faces);
        Vector3::new(pos[v1]) - Vector3::new(pos[v0])
    }

    /// Produce a vector that is tangent to faces[0] (so orthogonal to its normal), but not colinear
    /// to the edge itself.
    #[inline]
    pub fn face0_tangent<T: Real>(&self, pos: &[[T; 3]], faces: &[[usize; 3]]) -> Vector3<T> {
        let [v0, v1] = [
            faces[self.faces[0]][self.edge_start[0] as usize],
            self.tangent_verts(faces)[0],
        ];
        Vector3::new(pos[v1]) - Vector3::new(pos[v0])
    }

    /// Produce a vector that is tangent to faces[1] (so orthogonal to its normal), but not colinear
    /// to the edge itself.
    #[inline]
    pub fn face1_tangent<T: Real>(&self, pos: &[[T; 3]], faces: &[[usize; 3]]) -> Vector3<T> {
        let [v0, v1] = [self.face_vert(faces, 0, 0), self.tangent_verts(faces)[1]];
        debug_assert!(v0 == self.face_vert(faces, 1, 1) || v0 == self.face_vert(faces, 1, 0));
        Vector3::new(pos[v1]) - Vector3::new(pos[v0])
    }

    /// Return `true` if the adjacent faces have the same orientation.
    #[inline]
    pub fn is_oriented(&self, faces: &[[usize; 3]]) -> bool {
        self.face_vert(faces, 0, 0) != self.face_vert(faces, 1, 0)
    }

    /// Compute the area weighted normals of adjacent faces.
    ///
    /// This function will reverse the normal of `faces[1]` if it's orientation is opposite to
    /// the orientation of `faces[0]`.
    #[inline]
    pub fn face_area_normals<T: Real>(
        &self,
        pos: &[[T; 3]],
        faces: &[[usize; 3]],
    ) -> [Vector3<T>; 2] {
        let an0 = Triangle::from_indexed_slice(&faces[self.faces[0]], &pos)
            .area_normal()
            .into_tensor();
        let f1 = faces[self.faces[1]];
        let is_oriented = self.is_oriented(faces);
        let idx = [usize::from(!is_oriented), usize::from(is_oriented)];
        let an1 = Triangle::new([pos[f1[idx[0]]], pos[f1[idx[1]]], pos[f1[2]]])
            .area_normal()
            .into_tensor();
        [an0, an1]
    }

    ///// Compute the areas of adjacent faces.
    //#[inline]
    //pub fn face_areas<T: Real>(&self, pos: &[[T; 3]], faces: &[[usize; 3]]) -> [T; 2] {
    //    use geo::ops::Area;
    //    let a0 = Triangle::from_indexed_slice(&faces[self.faces[0]], &pos).area();
    //    let a1 = Triangle::from_indexed_slice(&faces[self.faces[1]], &pos).area();
    //    [a0, a1]
    //}

    /// Compute the reflex of the dihedral angle made by the faces neighbouring this edge.
    #[inline]
    pub(crate) fn edge_angle<T: Real>(&self, pos: &[[T; 3]], faces: &[[usize; 3]]) -> T {
        let [an0, an1] = self.face_area_normals(pos, faces);
        let t = self.face0_tangent(pos, faces);
        an0.cross(an1).norm().atan2(an0.dot(an1)) * -an1.dot(t).signum()
    }

    /// Compute the reflex of the dihedral angle made by the faces neighbouring this edge from
    /// the reference configuration.
    ///
    /// If the two edge vertex positions are coincident in the reference configuration,
    /// then we compute the edge angle between the two faces, otherwise this is zero.
    #[inline]
    pub(crate) fn ref_edge_angle<T: Real>(&self, ref_pos: &[[[T; 3]; 3]]) -> T {
        let an0 = Vector3::new(Triangle::new(ref_pos[self.faces[0]]).area_normal());
        let an1 = Vector3::new(Triangle::new(ref_pos[self.faces[1]]).area_normal());
        let rv = self.ref_edge_verts(ref_pos);

        // Check if the edge is coincident between the two faces. If so, use the angle between them
        // as the reference angle.
        if rv[0] == rv[1] || rv[0][0] == rv[1][1] && rv[0][1] == rv[1][0] {
            let t = self.tangent_verts(ref_pos)[0].into_tensor() - rv[0][0].into_tensor();
            an0.cross(an1).norm().atan2(an0.dot(an1)) * -an1.dot(t).signum()
        } else {
            T::zero()
        }
    }

    /// Compute the equivalent angle in the range `[-π, π)`.
    ///
    /// Note that `π` is not exactly representable with floats, so one should really not rely on
    /// the inclusion/exclusion of `π` and `-π` at the boundaries.
    #[inline]
    pub(crate) fn project_angle<T: Real>(angle: T) -> T {
        let pi = T::from(std::f64::consts::PI).unwrap();
        let two_pi = T::from(2.0 * std::f64::consts::PI).unwrap();

        let mut out = (angle + pi) % two_pi - pi;
        if out < -pi {
            out += two_pi;
        }
        out
    }

    /// Compute the edge angle to be at most `π` away from `ref_angle`.
    ///
    /// Note that `ref_angle` is not necessarily the rest shape angle. It is simply the closest
    /// angle --- the best guess for the current angle.
    #[inline]
    pub(crate) fn incremental_angle<T: Real>(
        &self,
        ref_angle: T,
        pos: &[[T; 3]],
        faces: &[[usize; 3]],
    ) -> T {
        let pi = T::from(std::f64::consts::PI).unwrap();
        let two_pi = T::from(2.0 * std::f64::consts::PI).unwrap();

        let ref_angle_proj = Self::project_angle(ref_angle);
        let mut angle_diff = self.edge_angle(pos, faces) - ref_angle_proj;
        if angle_diff < -pi {
            angle_diff += two_pi;
        } else if angle_diff > pi {
            angle_diff -= two_pi;
        }
        ref_angle + angle_diff
    }

    /// Compute the gradient of the reflex angle given by `edge_angle` for the four significant
    /// vertices that control the angle.
    ///
    /// The area weighted normals are assumed to be computed as `an0 = e0 x e1` and `an1 = e2 x e0`.
    /// The gradient is zero if any of the triangles are found to be degenerate.
    #[inline]
    pub(crate) fn edge_angle_gradient<T: Real>(
        &self,
        pos: &[[T; 3]],
        faces: &[[usize; 3]],
    ) -> [[T; 3]; 4] {
        let [an0, an1] = self.face_area_normals(pos, faces);
        let e0 = self.edge_vector(pos, faces);
        let e1 = self.face0_tangent(pos, faces);
        let e2 = self.face1_tangent(pos, faces);
        let a0_squared = an0.norm_squared();
        let a1_squared = an1.norm_squared();
        if a0_squared == T::zero() || a1_squared == T::zero() {
            return [[T::zero(); 3]; 4];
        }

        let e0_norm = e0.norm();
        // None of the triangles are degenerate, thus e0 must have non-zero length.
        debug_assert_ne!(e0_norm, T::zero());

        let e0n = e0 / e0_norm;
        let e1e0n = e1.dot(e0n);
        let e2e0n = e2.dot(e0n);

        // Compute normals per unit area
        let an0nn = an0 / a0_squared;
        let an1nn = an1 / a1_squared;

        let dx0 = an0nn * (e0_norm - e1e0n) + an1nn * (e0_norm - e2e0n);
        let dx1 = an0nn * e1e0n + an1nn * e2e0n;
        let dx2 = an0nn * -e0_norm;
        let dx3 = an1nn * -e0_norm;

        [
            dx0.into_data(),
            dx1.into_data(),
            dx2.into_data(),
            dx3.into_data(),
        ]
    }

    /// Compute the hessian of the reflex angle given by `edge_angle` for the four significant
    /// vertices that control the angle.
    ///
    /// The area weighted normals are assumed to be computed as `an0 = e0 x e1` and `an1 = e2 x e0`.
    ///
    /// The hessian is zero if any of the triangles are degenerate.
    ///
    /// The resulting hessian is returned as a lower-triangular row-major block matrix as a
    ///  - diagonal part with lower triangular blocks and a
    ///  - lower triangular off-diagonal part as full 3x3 blocks.
    ///
    /// NOTE: The 3-2 (and 2-3) block is zero so it's omitted from the output array of off-diagonal
    ///       part.
    #[inline]
    pub(crate) fn edge_angle_hessian<T: Real>(
        &self,
        pos: &[[T; 3]],
        faces: &[[usize; 3]],
    ) -> ([[T; 6]; 4], [[[T; 3]; 3]; 5]) {
        let [an0, an1] = self.face_area_normals(pos, faces);
        let e0 = self.edge_vector(pos, faces);
        let e1 = self.face0_tangent(pos, faces);
        let e2 = self.face1_tangent(pos, faces);
        let e0_norm_squared = e0.norm_squared();
        let a0_squared = an0.norm_squared();
        let a1_squared = an1.norm_squared();
        if a0_squared == T::zero() || a1_squared == T::zero() {
            return ([[T::zero(); 6]; 4], [[[T::zero(); 3]; 3]; 5]);
        }

        // None of the triangles are degenerate, thus e0 must have non-zero length.
        debug_assert_ne!(e0_norm_squared, T::zero());

        let e0_norm = e0_norm_squared.sqrt();
        let e0u = e0.transpose() / e0_norm;
        let e0u_e1 = (e0u * e1).into_data()[0];
        let e0u_e2 = (e0u * e2).into_data()[0];

        // Compute normals per unit area
        let an0nn = an0 / a0_squared;
        let an1nn = an1 / a1_squared;
        let two = T::from(2.0).unwrap();
        let dan0nn = Matrix3::from_diag_iter(std::iter::repeat(T::one() / a0_squared))
            - an0nn * an0nn.transpose() * two;
        let dan1nn = Matrix3::from_diag_iter(std::iter::repeat(T::one() / a1_squared))
            - an1nn * an1nn.transpose() * two;

        let dn0de0 = (-e1).skew();
        let dn0de1 = (e0).skew();
        let dn1de0 = (e2).skew();
        let dn1de2 = (-e0).skew();

        let dth_de = [
            dan0nn * dn0de0 * e0u_e1
                + an0nn * ((e1.transpose() - e0u * e0u_e1) / e0_norm)
                + dan1nn * dn1de0 * e0u_e2
                + an1nn * ((e2.transpose() - e0u * e0u_e2) / e0_norm), // e0e0
            dan0nn * dn0de1 * e0u_e1 + an0nn * e0u, // e1e0 or transpose for e0e1
            dan1nn * dn1de2 * e0u_e2 + an1nn * e0u, // e2e0 or transpose for e0e2
            dan0nn * dn0de1 * -e0_norm,             // e1e1
            dan1nn * dn1de2 * -e0_norm,             // e2e2
        ];
        let dx0dx0 = dth_de[0] // from e0e0
            + dth_de[1] + dth_de[1].transpose()  // from e0e1 and e1e0
            + dth_de[2] + dth_de[2].transpose() // from e0e2 and e2e0
            + dth_de[3] // from e1e1
            + dth_de[4]; // from e2e2
        let dx0dx1 = -dth_de[0] // from e0e0
            - dth_de[1] // from e1e0
            - dth_de[2]; // from e2e0
        let dx0dx2 = -dth_de[1].transpose() - dth_de[3]; // e0e1 and e1e1
        let dx0dx3 = -dth_de[2].transpose() - dth_de[4]; // e0e2 and e2e2
        let dx1dx1 = dth_de[0]; // e0e0
        let dx1dx2 = dth_de[1].transpose(); // e0e1
        let dx1dx3 = dth_de[2].transpose(); // e0e2
        let dx2dx2 = dth_de[3]; // e1e1
                                //let dx2dx3 = Zero
        let dx3dx3 = dth_de[4]; // e2e2

        // Output format:
        // -> (diagonal lower triangular blocks, lower triangular full blocks)
        // -> ([[T; 6]; 4], [[[T; 3]; 3]; 5])
        (
            [
                dx0dx0.lower_triangular_vec().into_data(),
                dx1dx1.lower_triangular_vec().into_data(),
                dx2dx2.lower_triangular_vec().into_data(),
                dx3dx3.lower_triangular_vec().into_data(),
            ],
            [
                dx0dx1.into_data(),
                dx0dx2.into_data(),
                dx1dx2.into_data(),
                dx0dx3.into_data(),
                dx1dx3.into_data(),
            ],
        )
    }
}

/// An edge can be one of:
/// 1. A boundary edge produced by a single triangle,
/// 2. A pair of adjacent triangles producing a manifold edge,
/// 3. Multiple triangles caused by a non-trivial stitch in the triangle mesh.
///
/// In either case the edge will be adjacent to a least one face index.
/// We use a `Vec` to record non-manifold stitched, which would not cause allocations in most
/// cases.
enum EdgeTopo {
    Boundary(usize),
    Manifold([usize; 2]),
    NonManifold(Vec<usize>),
}

impl EdgeTopo {
    fn into_manifold_edge(&self) -> Option<[usize; 2]> {
        match self {
            EdgeTopo::Manifold(e) => Some(*e),
            _ => None,
        }
    }
}

struct EdgeData {
    vertices: [usize; 2],
    topo: EdgeTopo,
}

impl EdgeData {
    fn new(vertices: [usize; 2], face_idx: usize) -> Self {
        EdgeData {
            vertices,
            topo: EdgeTopo::Boundary(face_idx),
        }
    }
    fn into_manifold_edge(&self) -> Option<([usize; 2], [usize; 2])> {
        self.topo.into_manifold_edge().map(|e| (self.vertices, e))
    }
}

/// Get reference triangle.
/// This routine assumes that there is a face vertex attribute called `ref` of type `[f32;3]`.
pub fn ref_tri(ref_tri: &[RefPosType]) -> Triangle<f64> {
    Triangle::new([
        Vector3::new(ref_tri[0]).cast::<f64>().into(),
        Vector3::new(ref_tri[1]).cast::<f64>().into(),
        Vector3::new(ref_tri[2]).cast::<f64>().into(),
    ])
}

/// Compute a set of interior edges of a given triangle mesh.
///
/// Interior edges are edges which have exactly two adjacent faces.
#[unroll_for_loops]
pub(crate) fn compute_interior_edge_topology(trimesh: &TriMesh) -> Vec<InteriorEdge> {
    // TODO: Move this algorithm to gut.
    // An edge is actually defined by a pair of vertices.
    // We iterate through all the faces and register each half edge (sorted by vertex index)
    // into a hashmap along with the originating face index.
    #[cfg(test)]
    let mut edges = {
        // We want our tests to be deterministic, so we opt for hardcoding the seeds here.
        let hash_builder = hashbrown::hash_map::DefaultHashBuilder::with_seeds(7, 47);
        hashbrown::HashMap::with_capacity_and_hasher(trimesh.num_faces(), hash_builder)
    };
    #[cfg(not(test))]
    let mut edges = hashbrown::HashMap::with_capacity(trimesh.num_faces());

    let add_face_edges = |(face_idx, face): (usize, &[usize; 3])| {
        for i in 0..3 {
            let [v0, v1] = [face[i], face[(i + 1) % 3]];

            let key = if v0 < v1 { [v0, v1] } else { [v1, v0] }; // Sort edge
            edges
                .entry(key)
                .and_modify(|e: &mut EdgeData| {
                    match &mut e.topo {
                        EdgeTopo::Boundary(i) => e.topo = EdgeTopo::Manifold([*i, face_idx]),
                        EdgeTopo::Manifold([a, b]) => {
                            e.topo = EdgeTopo::NonManifold(vec![*a, *b, face_idx])
                        }
                        EdgeTopo::NonManifold(v) => {
                            v.push(face_idx);
                        }
                    };
                })
                .or_insert(EdgeData::new([v0, v1], face_idx));
        }
    };

    trimesh.face_iter().enumerate().for_each(add_face_edges);

    let mut interior_edges = Vec::with_capacity(trimesh.num_faces()); // Estimate capacity

    // Given a pair of verts marking an edge, find which of v0, v1 and v2 corresponds to
    // one of the verts such that the next face vertex is also an edge vertex.
    let find_triangle_edge_start = |verts: [usize; 2], &[v0, v1, v2]: &[usize; 3]| {
        if v0 == verts[0] {
            if v1 == verts[1] {
                Some(0)
            } else if v2 == verts[1] {
                Some(2)
            } else {
                None
            }
        } else if v1 == verts[0] {
            if v2 == verts[1] {
                Some(1)
            } else if v0 == verts[1] {
                Some(0)
            } else {
                None
            }
        } else if v2 == verts[0] {
            if v0 == verts[1] {
                Some(2)
            } else if v1 == verts[1] {
                Some(1)
            } else {
                None
            }
        } else {
            None
        }
        .unwrap_or_else(|| unreachable!("Corrupt edge adjacency detected"))
    };

    for edge in edges.values() {
        // We only consider manifold edges with strictly two adjacent faces.
        // Boundary edges are ignored as are non-manifold edges.
        if let Some((verts, faces)) = edge.into_manifold_edge() {
            // Determine the source vertex for this edge in faces[0].
            let edge_start = [
                find_triangle_edge_start(verts, trimesh.face(faces[0])),
                find_triangle_edge_start(verts, trimesh.face(faces[1])),
            ];

            interior_edges.push(InteriorEdge::new(faces, edge_start));
        }
    }

    interior_edges
}

#[cfg(test)]
mod tests {
    use approx::*;
    use autodiff::F1;
    use hashbrown::HashSet;

    use super::*;

    /// Create a test case for interior edge functions
    fn make_test_interior_edge() -> (InteriorEdge, [[f64; 3]; 4], [[usize; 3]; 2]) {
        let x = [
            [0.0; 3],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.5, -1.0, 0.5], // slight bend and asymmetry
        ];

        let faces = [[0, 1, 2], [0, 3, 1]];

        let e = InteriorEdge::new([0, 1], [0, 2]);

        (e, x, faces)
    }

    fn make_test_interior_edge_alt() -> (InteriorEdge, [[f64; 3]; 4], [[usize; 3]; 2]) {
        let x = [[1.0, 0.0, 0.25], [0.0, 1.0, 0.0], [0.0; 3], [1.0, 1.0, 0.0]];

        let faces = [[2, 0, 1], [0, 3, 1]];

        let e = InteriorEdge::new([0, 1], [1, 2]);

        (e, x, faces)
    }

    /// A test example of two adjacent triangles with opposite orientations.
    fn make_test_interior_edge_unoriented() -> (InteriorEdge, [[f64; 3]; 4], [[usize; 3]; 2]) {
        let x = [[1.0, 0.0, 0.25], [0.0, 1.0, 0.0], [0.0; 3], [1.0, 1.0, 0.0]];

        let faces = [[2, 0, 1], [0, 1, 3]];

        let e = InteriorEdge::new([0, 1], [1, 0]);

        (e, x, faces)
    }

    #[test]
    fn interior_edge_structure() {
        let (e, x, faces) = make_test_interior_edge();

        assert_eq!(e.edge_start, [0, 2]);
        assert_eq!(e.faces, [0, 1]);
        assert_eq!(e.edge_verts(&faces[..]), [0, 1]);
        assert_eq!(e.tangent_verts(&faces[..]), [2, 3]);
        assert_eq!(e.verts(&faces[..]), [0, 1, 2, 3]);
        assert_eq!(
            e.edge_vector(&x[..], &faces[..]),
            Vector3::new([1.0, 0.0, 0.0])
        );
        assert_eq!(
            e.face0_tangent(&x[..], &faces[..]),
            Vector3::new([0.0, 1.0, 0.0])
        );
        assert_eq!(
            e.face1_tangent(&x[..], &faces[..]),
            Vector3::new([0.5, -1.0, 0.5])
        );
    }

    #[test]
    fn project_angle() {
        assert_eq!(InteriorEdge::project_angle(0.0), 0.0);
        let pi = std::f64::consts::PI;
        assert_eq!(InteriorEdge::project_angle(pi / 2.0), pi / 2.0);
        assert_eq!(InteriorEdge::project_angle(-pi / 2.0), -pi / 2.0);
        assert_eq!(InteriorEdge::project_angle(3.0 * pi / 2.0), -pi / 2.0);
        assert_eq!(InteriorEdge::project_angle(-3.0 * pi / 2.0), pi / 2.0);
        assert_eq!(InteriorEdge::project_angle(pi + 0.01), -pi + 0.01);
        assert_eq!(InteriorEdge::project_angle(pi - 0.01), pi - 0.01);
        assert_eq!(InteriorEdge::project_angle(2.0 * pi), 0.0);
        assert_relative_eq!(
            InteriorEdge::project_angle(7.0 * pi + 0.01),
            -pi + 0.01,
            max_relative = 1e-8
        );
        assert_relative_eq!(
            InteriorEdge::project_angle(-7.0 * pi - 0.01),
            pi - 0.01,
            max_relative = 1e-8
        );
    }

    #[test]
    fn incremental_angle() {
        // Test for idempotency (approximate).
        let (e, x, faces) = make_test_interior_edge();
        let inc = |a| e.incremental_angle(a, &x[..], &faces[..]);
        let a = e.edge_angle(&x[..], &faces[..]);
        assert_relative_eq!(inc(a), a, max_relative = 1e-8);

        let (e, x, faces) = make_test_interior_edge_alt();
        let inc = |a| e.incremental_angle(a, &x[..], &faces[..]);
        let a = e.edge_angle(&x[..], &faces[..]);
        assert_relative_eq!(inc(a), a, max_relative = 1e-8);
    }

    #[test]
    fn edge_angle_gradient() {
        let (e, x, f) = make_test_interior_edge();
        edge_angle_gradient_tester(e, &x, &f);
        let (e, x, f) = make_test_interior_edge_alt();
        edge_angle_gradient_tester(e, &x, &f);
        let (e, x, f) = make_test_interior_edge_unoriented();
        edge_angle_gradient_tester(e, &x, &f);
    }

    fn edge_angle_gradient_tester(e: InteriorEdge, x: &[[f64; 3]; 4], faces: &[[usize; 3]; 2]) {
        let mut x_ad = [
            Vector3::new(x[0]).cast::<F1>().into_data(),
            Vector3::new(x[1]).cast::<F1>().into_data(),
            Vector3::new(x[2]).cast::<F1>().into_data(),
            Vector3::new(x[3]).cast::<F1>().into_data(),
        ];

        let verts = e.verts(&faces[..]);

        let grad = e.edge_angle_gradient(&x[..], &faces[..]);
        let mut grad_ad = [[0.0; 3]; 4]; // Autodiff version of the grad for debugging

        let mut success = true;
        for vtx in 0..4 {
            for i in 0..3 {
                x_ad[verts[vtx]][i] = F1::var(x_ad[verts[vtx]][i]);
                let a = e.edge_angle(&x_ad[..], &faces[..]);
                grad_ad[vtx][i] = a.deriv();
                let ret = relative_eq!(grad[vtx][i], a.deriv(), max_relative = 1e-8);
                if !ret {
                    success = false;
                    eprintln!("{:?} vs. {:?}", grad[vtx][i], a.deriv());
                }
                x_ad[verts[vtx]][i] = F1::cst(x_ad[verts[vtx]][i]);
            }
        }

        eprintln!("Actual:");
        for vtx in 0..4 {
            for i in 0..3 {
                eprint!("{:10.2e}", grad[vtx][i]);
            }
        }
        eprintln!("\n");
        eprintln!("Expected:");
        for vtx in 0..4 {
            for i in 0..3 {
                eprint!("{:10.2e}", grad_ad[vtx][i]);
            }
        }
        eprintln!("");

        assert!(success);
    }

    #[test]
    fn edge_angle_hessian() {
        let (e, x, f) = make_test_interior_edge();
        edge_angle_hessian_tester(e, &x, &f);
        let (e, x, f) = make_test_interior_edge_alt();
        edge_angle_hessian_tester(e, &x, &f);
        let (e, x, f) = make_test_interior_edge_unoriented();
        edge_angle_hessian_tester(e, &x, &f);
    }

    fn edge_angle_hessian_tester(e: InteriorEdge, x: &[[f64; 3]; 4], faces: &[[usize; 3]; 2]) {
        let mut x_ad = [
            Vector3::new(x[0]).cast::<F1>().into_data(),
            Vector3::new(x[1]).cast::<F1>().into_data(),
            Vector3::new(x[2]).cast::<F1>().into_data(),
            Vector3::new(x[3]).cast::<F1>().into_data(),
        ];

        let verts = e.verts(&faces[..]);

        let hess = e.edge_angle_hessian(&x[..], &faces[..]);
        let mut hess_ad = ([[0.0; 6]; 4], [[[0.0; 3]; 3]; 5]); // Autodiff version of the hessian for debugging

        let vtx_map = [&[][..], &[0][..], &[1, 2][..], &[3, 4][..]];
        let idx_map = [&[0][..], &[1, 2][..], &[3, 4, 5][..]];
        let mut success = true;
        for col_vtx in 0..4 {
            for col in 0..3 {
                x_ad[verts[col_vtx]][col] = F1::var(x_ad[verts[col_vtx]][col]);
                let g = e.edge_angle_gradient(&x_ad[..], &faces[..]);
                for row_vtx in col_vtx..4 {
                    if (row_vtx == 2 && col_vtx == 3) || (row_vtx == 3 && col_vtx == 2) {
                        for row in 0..3 {
                            assert_eq!(g[row_vtx][row].deriv(), 0.0);
                        }
                    } else if row_vtx == col_vtx {
                        for row in col..3 {
                            let i = idx_map[row][col];
                            let ad = g[row_vtx][row].deriv();
                            hess_ad.0[row_vtx][i] = ad;
                            success &= relative_eq!(hess.0[row_vtx][i], ad, max_relative = 1e-8);
                        }
                    } else {
                        for row in 0..3 {
                            let ad = g[row_vtx][row].deriv();
                            let vtx = vtx_map[row_vtx][col_vtx];
                            hess_ad.1[vtx][row][col] = ad;
                            success &= relative_eq!(hess.1[vtx][row][col], ad, max_relative = 1e-8);
                        }
                    }
                }
                x_ad[verts[col_vtx]][col] = F1::cst(x_ad[verts[col_vtx]][col]);
            }
        }

        eprintln!("Actual:");
        for row_vtx in 0..4 {
            for row in 0..3 {
                for col_vtx in 0..4 {
                    for col in 0..3 {
                        if row_vtx < col_vtx
                            || (row_vtx == col_vtx && row < col)
                            || row_vtx == 3 && col_vtx == 2
                            || row_vtx == 2 && col_vtx == 3
                        {
                            eprint!("          ");
                        } else if row_vtx == col_vtx {
                            let i = idx_map[row][col];
                            eprint!("{:10.2e}", hess.0[row_vtx][i]);
                        } else {
                            let vtx = vtx_map[row_vtx][col_vtx];
                            eprint!("{:10.2e}", hess.1[vtx][row][col]);
                        }
                    }
                }
                eprintln!("");
            }
        }
        eprintln!("\n");
        eprintln!("Expected:");
        for row_vtx in 0..4 {
            for row in 0..3 {
                for col_vtx in 0..4 {
                    for col in 0..3 {
                        if row_vtx < col_vtx
                            || (row_vtx == col_vtx && row < col)
                            || row_vtx == 3 && col_vtx == 2
                            || row_vtx == 2 && col_vtx == 3
                        {
                            eprint!("          ");
                        } else if row_vtx == col_vtx {
                            let i = idx_map[row][col];
                            eprint!("{:10.2e}", hess_ad.0[row_vtx][i]);
                        } else {
                            let vtx = vtx_map[row_vtx][col_vtx];
                            eprint!("{:10.2e}", hess_ad.1[vtx][row][col]);
                        }
                    }
                }
                eprintln!("");
            }
        }
        eprintln!("");

        assert!(success);
    }

    #[test]
    fn compute_interior_edge_topology_small_test() {
        // Make a test mesh.

        let pos = vec![
            [-2.0, -0.795, 0.0],
            [0.0, -0.795, 0.0],
            [2.0, -0.795, 0.0],
            [-2.0, 1.205, 0.0],
            [0.0, 1.205, 0.0],
            [2.0, 1.205, 0.0],
            [-2.0, 3.205, 0.0],
            [0.0, 3.205, 0.0],
            [2.0, 3.205, 0.0],
        ];

        let verts = vec![
            [0, 1, 4],
            [0, 4, 3],
            [1, 2, 4],
            [2, 5, 4],
            [3, 4, 6],
            [4, 7, 6],
            [4, 5, 8],
            [4, 8, 7],
        ];

        let trimesh = TriMesh::new(pos, verts);

        let interior_edges: HashSet<_> = compute_interior_edge_topology(&trimesh)
            .into_iter()
            .collect();

        let expected_interior_edges: HashSet<_> = vec![
            InteriorEdge {
                faces: [1, 4],
                edge_start: [1, 0],
            },
            InteriorEdge {
                faces: [2, 3],
                edge_start: [1, 2],
            },
            InteriorEdge {
                faces: [0, 2],
                edge_start: [1, 2],
            },
            InteriorEdge {
                faces: [0, 1],
                edge_start: [2, 0],
            },
            InteriorEdge {
                faces: [3, 6],
                edge_start: [1, 0],
            },
            InteriorEdge {
                faces: [4, 5],
                edge_start: [1, 2],
            },
            InteriorEdge {
                faces: [5, 7],
                edge_start: [0, 2],
            },
            InteriorEdge {
                faces: [6, 7],
                edge_start: [2, 0],
            },
        ]
        .into_iter()
        .collect();

        assert_eq!(interior_edges, expected_interior_edges);
    }

    #[test]
    fn compute_interior_edge_topology_large_test() {
        // Make a test mesh.

        let pos = vec![
            [-2.0, -0.795, -0.513003],
            [-0.666667, -0.795, -0.513003],
            [0.666667, -0.795, -0.513003],
            [2.0, -0.795, -0.513003],
            [-2.0, 0.538333, -0.513003],
            [-0.666667, 0.538333, -0.513003],
            [0.666667, 0.538333, -0.513003],
            [2.0, 0.538333, -0.513003],
            [-2.0, 1.87167, -0.513003],
            [-0.666667, 1.87167, -0.513003],
            [0.666667, 1.87167, -0.513003],
            [2.0, 1.87167, -0.513003],
            [-2.0, 3.205, -0.513003],
            [-0.666667, 3.205, -0.513003],
            [0.666667, 3.205, -0.513003],
            [2.0, 3.205, -0.513003],
        ];

        let verts = vec![
            [0, 1, 5],
            [0, 5, 4],
            [1, 2, 5],
            [2, 6, 5],
            [2, 3, 7],
            [2, 7, 6],
            [4, 5, 8],
            [5, 9, 8],
            [5, 6, 10],
            [5, 10, 9],
            [6, 7, 10],
            [7, 11, 10],
            [8, 9, 13],
            [8, 13, 12],
            [9, 10, 13],
            [10, 14, 13],
            [10, 11, 15],
            [10, 15, 14],
        ];

        let trimesh = TriMesh::new(pos, verts);

        let interior_edges: HashSet<_> = compute_interior_edge_topology(&trimesh)
            .into_iter()
            .collect();

        let expected_interior_edges: HashSet<_> = vec![
            InteriorEdge {
                faces: [3, 5],
                edge_start: [0, 2],
            },
            InteriorEdge {
                faces: [3, 8],
                edge_start: [1, 0],
            },
            InteriorEdge {
                faces: [9, 14],
                edge_start: [1, 0],
            },
            InteriorEdge {
                faces: [8, 9],
                edge_start: [2, 0],
            },
            InteriorEdge {
                faces: [6, 7],
                edge_start: [1, 2],
            },
            InteriorEdge {
                faces: [8, 10],
                edge_start: [1, 2],
            },
            InteriorEdge {
                faces: [10, 11],
                edge_start: [1, 2],
            },
            InteriorEdge {
                faces: [15, 17],
                edge_start: [0, 2],
            },
            InteriorEdge {
                faces: [2, 3],
                edge_start: [1, 2],
            },
            InteriorEdge {
                faces: [14, 15],
                edge_start: [1, 2],
            },
            InteriorEdge {
                faces: [0, 1],
                edge_start: [2, 0],
            },
            InteriorEdge {
                faces: [0, 2],
                edge_start: [1, 2],
            },
            InteriorEdge {
                faces: [4, 5],
                edge_start: [2, 0],
            },
            InteriorEdge {
                faces: [1, 6],
                edge_start: [1, 0],
            },
            InteriorEdge {
                faces: [5, 10],
                edge_start: [1, 0],
            },
            InteriorEdge {
                faces: [7, 12],
                edge_start: [1, 0],
            },
            InteriorEdge {
                faces: [7, 9],
                edge_start: [0, 2],
            },
            InteriorEdge {
                faces: [11, 16],
                edge_start: [1, 0],
            },
            InteriorEdge {
                faces: [12, 14],
                edge_start: [1, 2],
            },
            InteriorEdge {
                faces: [12, 13],
                edge_start: [2, 0],
            },
            InteriorEdge {
                faces: [16, 17],
                edge_start: [2, 0],
            },
        ]
        .into_iter()
        .collect();

        assert_eq!(interior_edges, expected_interior_edges);
    }
}
