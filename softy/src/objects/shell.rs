use unroll::unroll_for_loops;

use geo::mesh::{attrib, topology::*, VertexPositions};
use geo::ops::*;
use geo::prim::Triangle;
use utils::soap::*;
use utils::zip;

use crate::attrib_defines::*;
use crate::energy_models::elasticity::*;
use crate::energy_models::gravity::*;
use crate::energy_models::inertia::*;
use crate::fem::problem::Var;
use crate::objects::*;
use crate::TriMesh;

/// A soft shell represented by a trimesh. It is effectively a triangle mesh decorated by
/// physical material properties that govern how it behaves.
#[derive(Clone, Debug)]
pub struct TriMeshShell {
    pub trimesh: TriMesh,
    /// A list of interior edges taken from the deformed topology, defined by a pair of faces and
    /// indices to the opposing vertices in those faces.
    ///
    /// Using deformed topology allows us to capture bending energy at seams and using face pairs
    /// enables capturing bending in non-manifold edges.
    pub(crate) interior_edges: Vec<InteriorEdge>,
    pub(crate) interior_edge_ref_angles: Vec<f64>,
    pub(crate) interior_edge_angles: Vec<f64>,
    pub(crate) interior_edge_ref_length: Vec<f64>,
    pub(crate) interior_edge_bending_stiffness: Vec<f64>,
    pub material: ShellMaterial,
}

// TODO: This impl can be automated with a derive macro
impl Object for TriMeshShell {
    type Mesh = TriMesh;
    type Material = ShellMaterial;
    type ElementIndex = FaceIndex;
    fn num_elements(&self) -> usize {
        self.trimesh.num_faces()
    }
    fn mesh(&self) -> &TriMesh {
        &self.trimesh
    }
    fn material(&self) -> &ShellMaterial {
        &self.material
    }
    fn mesh_mut(&mut self) -> &mut TriMesh {
        &mut self.trimesh
    }
    fn material_mut(&mut self) -> &mut ShellMaterial {
        &mut self.material
    }
}

/// An `InteriorEdge` is an manifold edge with exactly two neighbouring faces.
///
/// For triangle meshes, an interior edge can be drawn as follows:
///
/// ```verbatim
///     x3
///     /\
///    /  \e2
/// x1/_e0_\x0
///   \    /
///    \  /e1
///     \/
///     x2
/// ```
///
/// Here `e0` is the interior edge itself; `e1` and `e2` are used to compute the derivatives of the
/// angle between the adjacent faces. `x0` to `x3` are the vertices making up all the degrees of
/// freedom that affect the edge reflex angle.
#[derive(Copy, Clone, Debug, PartialEq)]
pub(crate) struct InteriorEdge {
    pub faces: [usize; 2],
    /// Each edge_start[i] is a vertex in `faces[i]` that marks the start of the
    /// edge. This value is either `0`, `1` or `2` for a triangle face.
    edge_start: [u8; 2],
}

impl InteriorEdge {
    #[inline]
    pub fn new(faces: [usize; 2], edge_start: [u8; 2]) -> Self {
        InteriorEdge {
            faces,
            edge_start,
        }
    }

    /// Get edge verts followed by tangent verts `x0` to `x3`.
    #[inline]
    pub fn verts(&self, faces: &[[usize; 3]]) -> [usize; 4] {
        let [x0, x1] = self.edge_verts(faces);
        let [x2, x3] = self.tangent_verts(faces);
        [x0, x1, x2, x3]
    }

    /// Get the vertex indices of the edge endpoints.
    #[inline]
    pub fn edge_verts(&self, faces: &[[usize; 3]]) -> [usize; 2] {
        [faces[self.faces[0]][self.edge_start[0] as usize],
         faces[self.faces[0]][((self.edge_start[0] + 1)%3) as usize]]
    }

    #[inline]
    pub fn tangent_verts(&self, faces: &[[usize; 3]]) -> [usize; 2] {
        [faces[self.faces[0]][((self.edge_start[0] + 2)%3) as usize],
         faces[self.faces[1]][((self.edge_start[1] + 2)%3) as usize]]
    }

    /// Compute the edge length.
    #[inline]
    pub fn length<T: Real>(&self, pos: &[[T; 3]], faces: &[[usize; 3]]) -> T {
        self.edge_vector(pos, faces).norm()
    }

    /// Produce the 3D vector corresponding to this edge.
    #[inline]
    pub fn edge_vector<T: Real>(&self, pos: &[[T; 3]], faces: &[[usize; 3]]) -> Vector3<T> {
        let [v0, v1] = self.edge_verts(faces);
        (Vector3::new(pos[v1]) - Vector3::new(pos[v0]))
    }

    /// Produce a vector that is tangent to faces[0] (so orthogonal to its normal), but not colinear
    /// to the edge itself.
    #[inline]
    pub fn face0_tangent<T: Real>(&self, pos: &[[T; 3]], faces: &[[usize; 3]]) -> Vector3<T> {
        let [v0, v1] = [
            faces[self.faces[0]][self.edge_start[0] as usize],
            self.tangent_verts(faces)[0]
        ];
        (Vector3::new(pos[v1]) - Vector3::new(pos[v0]))
    }

    /// Produce a vector that is tangent to faces[1] (so orthogonal to its normal), but not colinear
    /// to the edge itself.
    #[inline]
    pub fn face1_tangent<T: Real>(&self, pos: &[[T; 3]], faces: &[[usize; 3]]) -> Vector3<T> {
        let [v0, v1] = [
            faces[self.faces[1]][((self.edge_start[1] + 1)%3) as usize],
            self.tangent_verts(faces)[1]
        ];
        (Vector3::new(pos[v1]) - Vector3::new(pos[v0]))
    }

    /// Compute the reflex of the dihedral angle made by the faces neighbouring this edge.
    #[inline]
    pub(crate) fn edge_angle<T: Real>(&self, pos: &[[T; 3]], faces: &[[usize; 3]]) -> T {
        let an0 = Vector3::new(Triangle::from_indexed_slice(&faces[self.faces[0]], &pos).area_normal());
        let an1 = Vector3::new(Triangle::from_indexed_slice(&faces[self.faces[1]], &pos).area_normal());
        let t = self.face0_tangent(pos, faces);
        an0.cross(an1).norm().atan2(an0.dot(an1)) * -an1.dot(t).signum()
        //let e0 = self.edge_vector(pos, faces);
        //(e0.dot(an0.cross(an1))/e0.norm()).atan2(an0.dot(an1))
    }

    /// Compute the gradient of the reflex angle given by `edge_angle` for the four significant
    /// vertices that control the angle.
    ///
    /// The area weighted normals are assumed to be computed as `an0 = e0 x e1` and `an1 = e2 x e0`.
    /// The gradient is zero if any of the triangles are found to be degenerate.
    #[inline]
    pub(crate) fn edge_angle_gradient<T: Real>(&self, pos: &[[T; 3]], faces: &[[usize; 3]]) -> [[T; 3]; 4] {
        let an0 = Vector3::new(Triangle::from_indexed_slice(&faces[self.faces[0]], &pos).area_normal());
        let an1 = Vector3::new(Triangle::from_indexed_slice(&faces[self.faces[1]], &pos).area_normal());
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

        [dx0.into_data(), dx1.into_data(), dx2.into_data(), dx3.into_data()]
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
    pub(crate) fn edge_angle_hessian<T: Real>(&self, pos: &[[T; 3]], faces: &[[usize; 3]])
                                              -> ([[T; 6]; 4], [[[T; 3]; 3]; 5])
    {
        let an0 = Vector3::new(Triangle::from_indexed_slice(&faces[self.faces[0]], &pos).area_normal());
        let an1 = Vector3::new(Triangle::from_indexed_slice(&faces[self.faces[1]], &pos).area_normal());
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
        let e1u = e1.transpose() / e0_norm;
        let e2u = e2.transpose() / e0_norm;
        let e0u_e1 = (e0u*e1).into_data()[0];
        let e0u_e2 = (e0u*e2).into_data()[0];

        // Compute normals per unit area
        let an0nn = an0 / a0_squared;
        let an1nn = an1 / a1_squared;
        let _2 = T::from(2.0).unwrap();
        let dan0nn = Matrix3::from_diag_iter(std::iter::repeat(T::one() / a0_squared)) - an0nn * an0nn.transpose() * _2;
        let dan1nn = Matrix3::from_diag_iter(std::iter::repeat(T::one() / a1_squared)) - an1nn * an1nn.transpose() * _2;

        let dn0de0 = (-e1).skew();
        let dn0de1 = (e0).skew();
        let dn1de0 = (e2).skew();
        let dn1de2 = (-e0).skew();

        let dth_de = [
            dan0nn*dn0de0*e0u_e1 + an0nn*(e1u - e0u*e0u_e1)
                + dan1nn*dn1de0*e0u_e2 + an1nn*(e2u - e0u*e0u_e2), // e0e0
            dan0nn*dn0de1*e0u_e1 + an0nn*e0u, // e1e0 or transpose for e0e1
            dan1nn*dn1de2*e0u_e2 + an1nn*e0u, // e2e0 or transpose for e0e2
            dan0nn*dn0de1*-e0_norm,// e1e1
            dan1nn*dn1de2*-e0_norm, // e2e2
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
        ([
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
         ])
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
            EdgeTopo::Manifold(e) => {
                Some(*e)
            }
            _ => None
        }
    }
}

struct EdgeData {
    vertices: [usize; 2],
    topo: EdgeTopo
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
        Vector3::new(ref_tri[2]).cast::<f64>().into()
    ])
}

impl TriMeshShell {
    pub fn new(trimesh: TriMesh, material: ShellMaterial) -> TriMeshShell {
        let pos = if let Ok(ref_pos) = trimesh.attrib_iter::<RefPosType, FaceVertexIndex>(REFERENCE_POSITION_ATTRIB) {
            ref_pos.map(|&p| Vector3::new(p).cast::<f64>().into_data()).collect()
        } else {
            trimesh.vertex_positions().to_vec()
        };
        let interior_edges = Self::compute_interior_edge_topology(&trimesh);
        let interior_edge_bending_stiffness = vec![0.0; interior_edges.len()];
        let (interior_edge_ref_angles, interior_edge_ref_length): (Vec<_>, Vec<_>) =
            interior_edges.iter().map(|e|
            (e.edge_angle(&pos, trimesh.faces()), e.length(&pos, trimesh.faces()))
        ).unzip();

        TriMeshShell {
            trimesh,
            material,
            interior_edges,
            interior_edge_angles: interior_edge_ref_angles.clone(),
            interior_edge_ref_angles,
            interior_edge_ref_length,
            interior_edge_bending_stiffness,
        }
    }

    /// Given a set of new vertex positions update the set of interior edge angles.
    pub(crate) fn update_interior_edge_angles<T: Real>(&mut self, x1: &[[T; 3]]) {
        let Self {
            ref trimesh,
            ref interior_edges,
            ref mut interior_edge_angles,
            ..
        } = *self;

        interior_edges.iter().zip(interior_edge_angles.iter_mut()).for_each(|(e, t)| {
            *t = e.incremental_angle(T::from(*t).unwrap(), x1, trimesh.faces()).to_f64().unwrap();
        });
    }

    /// Precompute attributes necessary for FEM simulation on the given mesh.
    pub(crate) fn with_fem_attributes(mut self) -> Result<TriMeshShell, Error> {
        self.material = self.material.normalized();

        self.init_source_index_attribute()?;

        match self.material.properties {
            ShellProperties::Fixed => {
                // Kinematic meshes don't have material properties.
                self.init_kinematic_vertex_attributes()?;
            }
            ShellProperties::Rigid { .. } => {
                self.init_dynamic_vertex_attributes()?;
            }
            ShellProperties::Deformable { .. } => {
                self.init_deformable_vertex_attributes()?;

                self.init_deformable_attributes()?;

                {
                    // Add elastic strain energy attribute.
                    // This will be computed at the end of the time step.
                    self.trimesh.set_attrib::<StrainEnergyType, FaceIndex>(STRAIN_ENERGY_ATTRIB, 0f64)?;
                }

                // Below we prepare attributes that give elasticity and density parameters. If such were
                // already provided on the mesh, then any given global parameters are ignored. This
                // behaviour is justified because variable material properties are most likely more
                // accurate and probably determined from a data driven method.

                self.init_elasticity_attributes()?;

                self.init_density_attribute()?;

                // Compute vertex masses.
                self.compute_vertex_masses()?;
            }
        };

        Ok(self)
    }

    pub(crate) fn init_deformable_attributes(&mut self) -> Result<(), Error> {
        self.init_vertex_face_ref_pos_attribute()?;
        self.init_bending_stiffness()?;

        let mesh = &mut self.trimesh;

        let ref_areas = Self::compute_ref_tri_areas(mesh)?;
        mesh.set_attrib_data::<RefAreaType, FaceIndex>(
            REFERENCE_AREA_ATTRIB,
            ref_areas.as_slice(),
        )?;

        let ref_shape_mtx_inverses = Self::compute_ref_tri_shape_matrix_inverses(mesh)?;
        mesh.set_attrib_data::<_, FaceIndex>(
            REFERENCE_SHAPE_MATRIX_INV_ATTRIB,
            ref_shape_mtx_inverses.as_slice(),
        )?;
        Ok(())
    }

    /// Compute signed volume for reference elements in the given `TetMesh`.
    fn compute_ref_tri_areas(mesh: &mut TriMesh) -> Result<Vec<f64>, Error> {
        let ref_pos = mesh.attrib_as_slice::<RefPosType, FaceVertexIndex>(REFERENCE_POSITION_ATTRIB)?;
        Ok(ref_pos.chunks_exact(3).map(|tri| ref_tri(tri).area()).collect())
    }

    /// Convert a 3D triangle shape matrix into a 2D matrix assuming an isotropic deformation
    /// model.
    ///
    /// Assume that reference triangles are non-degenerate.
    pub fn isotropic_tri_shape_matrix<T: Real>(m: Matrix2x3<T>) -> Matrix2<T> {
        // Project (orthogonally) second row onto the first.
        let scale = m[0].dot(m[1]) / m[0].norm_squared();
        let r1_proj = m[0] * scale;

        let q = Matrix2x3 {
            data: [
                m[0].normalized(),
                (m[1] - r1_proj).normalized(),
            ]
        };

        m * q.transpose()
    }

    /// Compute shape matrix inverses for reference elements in the given `TriMesh`.
    fn compute_ref_tri_shape_matrix_inverses(mesh: &mut TriMesh) -> Result<Vec<Matrix2<f64>>, Error> {
        let ref_pos = mesh.attrib_as_slice::<RefPosType, FaceVertexIndex>(REFERENCE_POSITION_ATTRIB)?;
        // Compute reference shape matrix inverses
        Ok(ref_pos.chunks_exact(3)
            .map(|tri| {
                let ref_shape_matrix = Matrix2x3::new(ref_tri(tri).shape_matrix());
                Self::isotropic_tri_shape_matrix(ref_shape_matrix).inverse().unwrap()
            })
            .collect())
    }

    pub(crate) fn init_bending_stiffness(&mut self) -> Result<(), Error> {
        let num_elements = self.num_elements();
        if let Some(bending_stiffness) = self.material().scaled_bending_stiffness() {
            match self.mesh_mut()
                .add_attrib_data::<BendingStiffnessType, FaceIndex>(
                    BENDING_STIFFNESS_ATTRIB,
                    vec![bending_stiffness; num_elements]
                ) {
                Err(attrib::Error::AlreadyExists(_)) => {
                    crate::objects::scale_param(self.material().scale(), self.mesh_mut()
                        .attrib_iter_mut::<BendingStiffnessType, FaceIndex>(BENDING_STIFFNESS_ATTRIB)
                        .expect("Internal error: Missing bending stiffness"));
                }
                Err(e) => return Err(e.into()),
                _ => {}
            }
        } else {
            // If no bending stiffness was provided, simply scale what is already there. If there
            // is nothing on the mesh, simply initialize bending stiffness to zero. This is
            // a reasonable default.
            let scale = self.material().scale();
            let attrib = self.mesh_mut().attrib_or_add::<BendingStiffnessType, FaceIndex>(
                BENDING_STIFFNESS_ATTRIB,
                0.0,
            )?;
            crate::objects::scale_param(scale, attrib.iter_mut::<BendingStiffnessType>()?);
        }

        // At this point we are confident that bending stiffness is correctly initialized on the mesh.
        // Now it remains to move it to interior edges
        // by averaging of the bending stiffnesses of the adjacent faces.
        let Self {
            trimesh,
            interior_edges,
            interior_edge_bending_stiffness,
            ..
        } = self;
        let face_bending_stiffnesses = trimesh.attrib_as_slice::<BendingStiffnessType, FaceIndex>(BENDING_STIFFNESS_ATTRIB)?;
        for (e, mult) in interior_edges.iter().zip(interior_edge_bending_stiffness.iter_mut()) {
            *mult = 0.5*(face_bending_stiffnesses[e.faces[0]] as f64 +
                         face_bending_stiffnesses[e.faces[1]] as f64);
        }
        Ok(())
    }

    /// A helper function to populate the vertex face reference position attribute.
    #[unroll_for_loops]
    pub(crate) fn init_vertex_face_ref_pos_attribute(&mut self) -> Result<(), Error> {
        let mesh = &mut self.trimesh;
        let mut ref_pos = vec![[0.0; 3]; mesh.num_face_vertices()];
        let pos = if let Ok(vtx_ref_pos) = mesh.attrib_as_slice::<RefPosType, VertexIndex>(REFERENCE_POSITION_ATTRIB) {
            // There is a reference attribute on the vertices themselves, just transfer these to
            // face vertices instead of using mesh position.
            vtx_ref_pos.to_vec()
        } else {
            mesh.vertex_position_iter()
                .map(|&x| Vector3::new(x).cast::<f32>().into())
                .collect()
        };

        for (face_idx, face) in mesh.face_iter().enumerate() {
            for i in 0..3 {
                let face_vtx_idx: usize = mesh.face_vertex(face_idx, i).unwrap().into();
                for j in 0..3 {
                    ref_pos[face_vtx_idx][j] = pos[face[i]][j];
                }
            }
        }

        mesh.attrib_or_add_data::<RefPosType, FaceVertexIndex>(
            REFERENCE_POSITION_ATTRIB,
            ref_pos.as_slice(),
        )?;
        Ok(())
    }

    /// Compute vertex masses on the given shell. The shell is assumed to have
    /// area and density attributes already.
    pub(crate) fn compute_vertex_masses(&mut self) -> Result<(), Error> {
        let trimesh = &mut self.trimesh;
        let mut masses = vec![0.0; trimesh.num_vertices()];

        for (&area, density, face) in zip!(
            trimesh
                .attrib_iter::<RefAreaType, FaceIndex>(REFERENCE_AREA_ATTRIB)
                .unwrap(),
            trimesh
                .attrib_iter::<DensityType, FaceIndex>(DENSITY_ATTRIB)
                .unwrap()
                .map(|&x| f64::from(x)),
            trimesh.face_iter()
        ) {
            for i in 0..3 {
                masses[face[i]] += area * density / 3.0;
            }
        }

        trimesh.add_attrib_data::<MassType, VertexIndex>(MASS_ATTRIB, masses)?;
        Ok(())
    }


    #[unroll_for_loops]
    fn compute_interior_edge_topology(trimesh: &TriMesh) -> Vec<InteriorEdge> {
        // TODO: Move this algorithm to gut.
        // An edge is actually defined by a pair of vertices.
        // We iterate through all the faces and register each half edge (sorted by vertex index)
        // into a hashmap along with the originating face index.
        let mut edges =
            fnv::FnvHashMap::with_capacity_and_hasher(trimesh.num_faces(), Default::default());

        let add_face_edges = |(face_idx, face): (usize, &[usize; 3])| {
            for i in 0..3 {
                let [v0, v1] = [face[i], face[(i+1)%3]];

                let key = if v0 < v1 { [v0, v1] } else { [v1, v0] }; // Sort edge
                edges.entry(key)
                    .and_modify(|e: &mut EdgeData| {
                        match &mut e.topo {
                            EdgeTopo::Boundary(i) => 
                                e.topo = EdgeTopo::Manifold([*i, face_idx]),
                            EdgeTopo::Manifold([a, b]) =>
                                e.topo = EdgeTopo::NonManifold(vec![*a, *b, face_idx]),
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
                    0
                } else if v2 == verts[1] {
                    2
                } else {
                    unreachable!("Corrupt edge adjacency detected");
                }
            } else if v1 == verts[0] {
                if v2 == verts[1] {
                    1
                } else if v0 == verts[1] {
                    0
                } else {
                    unreachable!("Corrupt edge adjacency detected");
                }
            } else if v2 == verts[0] {
                if v0 == verts[1] {
                    2
                } else if v1 == verts[1] {
                    1
                } else {
                    unreachable!("Corrupt edge adjacency detected");
                }
            } else {
                unreachable!("Corrupt edge adjacency detected");
            }
        };

        for edge in edges.values() {
            // We only consider manifold edges with strictly two adjacent faces.
            // Boundary edges are ignored as are non-manifold edges.
            if let Some((verts, faces)) = edge.into_manifold_edge() {
                // Determine the source vertex for this edge in faces[0].

                let edge_start = [
                    find_triangle_edge_start(verts, trimesh.face(faces[0])),
                    find_triangle_edge_start(verts, trimesh.face(faces[1]))
                ];

                interior_edges.push(InteriorEdge::new(faces, edge_start));
            }
        }

        interior_edges
    }

    pub fn tagged_mesh(&self) -> Var<&TriMesh> {
        match self.material.properties {
            ShellProperties::Fixed => Var::Fixed(&self.trimesh),
            _ => Var::Variable(&self.trimesh),
        }
    }
}

impl<'a> Elasticity<'a, Option<TriMeshNeoHookean<'a, f64>>> for TriMeshShell {
    fn elasticity(&'a self) -> Option<TriMeshNeoHookean<'a, f64>> {
        match self.material.properties {
            ShellProperties::Deformable { .. } => Some(TriMeshNeoHookean::new(self)),
            _ => None,
        }
    }
}

impl<'a> Inertia<'a, Option<TriMeshInertia<'a>>> for TriMeshShell {
    fn inertia(&'a self) -> Option<TriMeshInertia<'a>> {
        match self.material.properties {
            ShellProperties::Fixed => None,
            _ => Some(TriMeshInertia(self)),
        }
    }
}

impl<'a> Gravity<'a, Option<TriMeshGravity<'a>>> for TriMeshShell {
    fn gravity(&'a self, g: [f64; 3]) -> Option<TriMeshGravity<'a>> {
        match self.material.properties {
            ShellProperties::Fixed => None,
            _ => Some(TriMeshGravity::new(self, g)),
        }
    }
}

#[cfg(test)]
mod tests {
    use approx::*;
    use autodiff::F;

    use super::*;

    /// Create a test case for interior edge functions
    fn make_test_interior_edge() -> (InteriorEdge, [[f64; 3]; 4], [[usize; 3]; 2]) {
        let x = [
            [0.0; 3],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.5, -1.0, 0.5], // slight bend and asymmetry
        ];

        let faces = [
            [0, 1, 2],
            [0, 3, 1]
        ];

        let e = InteriorEdge::new([0,1], [0, 2]);

        (e, x, faces)
    }

    #[test]
    fn interior_edge_structure() {
        let (e, x, faces) = make_test_interior_edge();

        assert_eq!(e.edge_start, [0, 2]);
        assert_eq!(e.faces, [0, 1]);
        assert_eq!(e.edge_verts(&faces[..]), [0, 1]);
        assert_eq!(e.tangent_verts(&faces[..]), [2, 3]);
        assert_eq!(e.length(&x[..], &faces[..]), 1.0);
        assert_eq!(e.edge_vector(&x[..], &faces[..]), Vector3::new([1.0, 0.0, 0.0]));
        assert_eq!(e.face0_tangent(&x[..], &faces[..]), Vector3::new([0.0, 1.0, 0.0]));
        assert_eq!(e.face1_tangent(&x[..], &faces[..]), Vector3::new([0.5, -1.0, 0.5]));
    }

    #[test]
    fn edge_angle_gradient() {
        let (e, x, faces) = make_test_interior_edge();

        let mut x_ad = [
            Vector3::new(x[0]).cast::<F>().into_data(),
            Vector3::new(x[1]).cast::<F>().into_data(),
            Vector3::new(x[2]).cast::<F>().into_data(),
            Vector3::new(x[3]).cast::<F>().into_data(),
        ];

        let grad = e.edge_angle_gradient(&x[..], &faces[..]);
        let mut grad_ad = [[0.0; 3]; 4]; // Autodiff version of the grad for debugging

        let mut success = true;
        for vtx in 0..4 {
            for i in 0..3 {
                x_ad[vtx][i] = F::var(x_ad[vtx][i]);
                let a = e.edge_angle(&x_ad[..], &faces[..]);
                grad_ad[vtx][i] = a.deriv();
                success &= relative_eq!(grad[vtx][i], a.deriv());
                x_ad[vtx][i] = F::cst(x_ad[vtx][i]);
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
        let (e, x, faces) = make_test_interior_edge();

        let mut x_ad = [
            Vector3::new(x[0]).cast::<F>().into_data(),
            Vector3::new(x[1]).cast::<F>().into_data(),
            Vector3::new(x[2]).cast::<F>().into_data(),
            Vector3::new(x[3]).cast::<F>().into_data(),
        ];

        let hess = e.edge_angle_hessian(&x[..], &faces[..]);
        let mut hess_ad = ([[0.0; 6]; 4], [[[0.0; 3]; 3]; 5]); // Autodiff version of the hessian for debugging

        let vtx_map = [&[][..], &[0][..], &[1, 2][..], &[3, 4][..]];
        let idx_map = [&[0][..], &[1, 2][..], &[3, 4, 5][..]];
        let mut success = true;
        for col_vtx in 0..4 {
            for col in 0..3 {
                x_ad[col_vtx][col] = F::var(x_ad[col_vtx][col]);
                let g = e.edge_angle_gradient(&x_ad[..], &faces[..]);
                for row_vtx in col_vtx..4 {
                    if (row_vtx == 2 && col_vtx == 3) || (row_vtx == 3 && col_vtx == 2) {
                        continue;
                    }
                    if row_vtx == col_vtx {
                        for row in col..3 {
                            let i = idx_map[row][col];
                            let ad = g[row_vtx][row].deriv();
                            hess_ad.0[row_vtx][i] = ad;
                            success &= relative_eq!(hess.0[row_vtx][i], ad);
                        }
                    } else {
                        for row in 0..3 {
                            let ad = g[row_vtx][row].deriv();
                            let vtx = vtx_map[row_vtx][col_vtx];
                            hess_ad.1[vtx][row][col] = ad;
                            success &= relative_eq!(hess.1[vtx][row][col], ad);
                        }
                    }
                }
                x_ad[col_vtx][col] = F::cst(x_ad[col_vtx][col]);
            }
        }

        eprintln!("Actual:");
        for row_vtx in 0..4 {
            for row in 0..3 {
                for col_vtx in 0..4 {
                    for col in 0..3 {
                        if row_vtx < col_vtx || (row_vtx == col_vtx && row < col)
                            || row_vtx == 3 && col_vtx == 2 || row_vtx == 2 && col_vtx == 3 {
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
                        if row_vtx < col_vtx || (row_vtx == col_vtx && row < col)
                            || row_vtx == 3 && col_vtx == 2 || row_vtx == 2 && col_vtx == 3 {
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
}
