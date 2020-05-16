use unroll::unroll_for_loops;

use geo::mesh::{attrib, topology::*, VertexPositions};
use geo::ops::*;
use num_traits::Zero;
use tensr::*;
use utils::*;

use crate::attrib_defines::*;
use crate::energy_models::elasticity::*;
use crate::energy_models::gravity::*;
use crate::energy_models::inertia::*;
use crate::energy_models::Either;
use crate::fem::problem::Var;
use crate::objects::*;
use crate::TriMesh;

mod interior_edge;
pub use interior_edge::*;

#[derive(Copy, Clone, Debug)]
pub(crate) enum FixedVerts {
    Zero,
    #[allow(dead_code)]
    One(usize),
    #[allow(dead_code)]
    Two([usize; 2]),
}

/// Data used for simulating shells. This is used in conjunction with attributes on the mesh itself.
#[derive(Clone, Debug)]
pub(crate) enum ShellData {
    Soft {
        material: SoftShellMaterial,
        /// A list of interior edges taken from the deformed topology, defined by a pair of faces and
        /// indices to the opposing vertices in those faces.
        ///
        /// Using deformed topology allows us to capture bending energy at seams and using face pairs
        /// enables capturing bending in non-manifold edges.
        interior_edges: Vec<InteriorEdge>,
        interior_edge_ref_angles: Vec<f64>,
        interior_edge_angles: Vec<f64>,
        /// A normalized (unitless) reference length.
        interior_edge_ref_length: Vec<f64>,
        interior_edge_bending_stiffness: Vec<f64>,
    },
    Rigid {
        material: RigidMaterial,
        transform: Matrix3<f64>,
        fixed: FixedVerts,
        mass: f64,
        cm: Vector3<f64>,
        inertia: Matrix3<f64>,
    },
    Fixed {
        material: FixedMaterial,
    },
}

/// A soft shell represented by a trimesh. It is effectively a triangle mesh decorated by
/// physical material properties that govern how it behaves.
#[derive(Clone, Debug)]
pub struct TriMeshShell {
    pub trimesh: TriMesh,
    pub(crate) data: ShellData,
}

// TODO: This impl can be automated with a derive macro
impl Object for TriMeshShell {
    type Mesh = TriMesh;
    type ElementIndex = FaceIndex;

    fn num_elements(&self) -> usize {
        self.trimesh.num_faces()
    }
    fn mesh(&self) -> &TriMesh {
        &self.trimesh
    }
    fn mesh_mut(&mut self) -> &mut TriMesh {
        &mut self.trimesh
    }
    fn material_id(&self) -> usize {
        match self.data {
            ShellData::Soft { material, .. } => material.id,
            ShellData::Rigid { material, .. } => material.id,
            ShellData::Fixed { material, .. } => material.id,
        }
    }
}

impl DynamicObject for TriMeshShell {
    fn density(&self) -> Option<f32> {
        match self.data {
            ShellData::Soft { material, .. } => material.density(),
            ShellData::Rigid { material, .. } => material.density(),
            ShellData::Fixed { .. } => None,
        }
    }
}

impl DeformableObject for TriMeshShell {}

impl ElasticObject for TriMeshShell {
    fn elasticity_parameters(&self) -> Option<ElasticityParameters> {
        match self.data {
            ShellData::Soft { material, .. } => material.elasticity(),
            _ => None,
        }
    }
}

impl TriMeshShell {
    /// A generic `TriMeshShell` constructor that takes a dynamic reference to a material type
    /// and constructs a new mesh with the given material if it can be recognized.
    ///
    /// If the given material is not recognized, `None` is returned to prevent unexpected
    /// behaviour. If the type of material is known ahead of time, use one of `soft`, `rigid` or
    /// `fixed` constructors.
    pub fn new(trimesh: TriMesh, material: &dyn std::any::Any) -> Option<TriMeshShell> {
        match material.downcast_ref::<FixedMaterial>() {
            None => match material.downcast_ref::<RigidMaterial>() {
                None => match material.downcast_ref::<SoftShellMaterial>() {
                    None => None,
                    Some(m) => Some(Self::soft(trimesh, *m)),
                },
                Some(m) => Some(Self::rigid(trimesh, *m)),
            },
            Some(m) => Some(Self::fixed(trimesh, *m)),
        }
    }
    pub fn soft(trimesh: TriMesh, material: SoftShellMaterial) -> TriMeshShell {
        TriMeshShell {
            trimesh,
            data: ShellData::Soft {
                material,
                interior_edges: Vec::new(),
                interior_edge_angles: Vec::new(),
                interior_edge_ref_angles: Vec::new(),
                interior_edge_ref_length: Vec::new(),
                interior_edge_bending_stiffness: Vec::new(),
            },
        }
    }
    pub fn rigid(trimesh: TriMesh, material: RigidMaterial) -> TriMeshShell {
        let (mass, cm, inertia) =
            Self::integrate_rigid_properties(&trimesh, f64::from(material.properties.density));
        TriMeshShell {
            trimesh,
            data: ShellData::Rigid {
                material,
                transform: Matrix3::identity(),
                fixed: FixedVerts::Zero,
                mass,
                cm,
                inertia,
            },
        }
    }
    pub fn fixed(trimesh: TriMesh, material: FixedMaterial) -> TriMeshShell {
        TriMeshShell {
            trimesh,
            data: ShellData::Fixed { material },
        }
    }

    /// Compute the mass, center of mass and inertia tensor for a rigid body represented
    /// by the given triangle mesh.
    ///
    /// It is assumed that the mesh forms a closed surface.
    pub(crate) fn integrate_rigid_properties(
        mesh: &TriMesh,
        density: f64,
    ) -> (f64, Vector3<f64>, Matrix3<f64>) {
        let (one, xyz, x2y2z2, xy_xz_yz) = Self::integrate_polynomial_basis(mesh);
        let mass = one * density;
        let cm = xyz / one;
        let cm2 = cm.cwise_mul(cm);
        let xx = x2y2z2[1] + x2y2z2[2] - one * (cm2[1] + cm2[2]);
        let yy = x2y2z2[0] + x2y2z2[2] - one * (cm2[2] + cm2[0]);
        let xy = one * cm[0] * cm[1] - xy_xz_yz[0];
        let zz = x2y2z2[0] + x2y2z2[1] - one * (cm2[0] + cm2[1]);
        let yz = one * cm[1] * cm[2] - xy_xz_yz[1];
        let xz = one * cm[2] * cm[0] - xy_xz_yz[2];
        let inertia = [[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]].into_tensor() * density;
        (mass, cm, inertia)
    }

    /// Integrate a polynomial basis {1, x, y, z, x^2, y^2, z^2, xy, xz, yz} over the
    /// volume represented by this triangle mesh.
    ///
    /// This is a helper function for `integrate_rigid_properties`.
    ///
    /// The method for integrating quantities is taken from
    /// [David Eberly's writeup](https://www.geometrictools.com/Documentation/PolyhedralMassProperties.pdf)
    fn integrate_polynomial_basis(
        mesh: &TriMesh,
    ) -> (f64, Vector3<f64>, Vector3<f64>, Vector3<f64>) {
        let mut one = 0.0;
        let mut xyz = Vector3::zero();
        let mut x2y2z2 = Vector3::zero();
        let mut xy_xz_yz = Vector3::zero();
        for tri in mesh.tri_iter() {
            let an = tri.area_normal().into_tensor();
            let v = tri.into_array().into_tensor();
            let t0 = v[0] + v[1];
            let f1 = t0 + v[2];
            let t1 = v[0].cwise_mul(v[0]);
            let t2 = t1 + v[1].cwise_mul(t0);
            let f2 = t2 + v[2].cwise_mul(f1);
            let f3 = v[0].cwise_mul(t1) + v[1].cwise_mul(t2) + v[2].cwise_mul(f2);
            let g = Tensor {
                data: [
                    f2 + v[0].cwise_mul(f1 + v[0]),
                    f2 + v[1].cwise_mul(f1 + v[1]),
                    f2 + v[2].cwise_mul(f1 + v[2]),
                ],
            }
            .transpose();

            one += an[0] * f1[0];
            xyz += an.cwise_mul(f2);
            x2y2z2 += an.cwise_mul(f3);
            let w = v.transpose();
            xy_xz_yz +=
                an.cwise_mul([w[1].dot(g[0]), w[2].dot(g[1]), w[0].dot(g[2])].into_tensor());
        }

        (one / 6.0, xyz / 24.0, x2y2z2 / 60.0, xy_xz_yz / 120.0)
    }

    /// Given a set of new vertex positions update the set of interior edge angles.
    pub(crate) fn update_interior_edge_angles<T: Real>(&mut self, x1: &[[T; 3]]) {
        let Self {
            ref trimesh,
            ref mut data,
        } = *self;

        if let ShellData::Soft {
            ref interior_edges,
            ref mut interior_edge_angles,
            ..
        } = data
        {
            interior_edges
                .iter()
                .zip(interior_edge_angles.iter_mut())
                .for_each(|(e, t)| {
                    *t = e
                        .incremental_angle(T::from(*t).unwrap(), x1, trimesh.faces())
                        .to_f64()
                        .unwrap();
                });
        }
    }

    /// Precompute attributes necessary for FEM simulation on the given mesh.
    pub(crate) fn with_simulation_attributes(mut self) -> Result<TriMeshShell, Error> {
        self.init_source_index_attribute()?;

        match self.data {
            ShellData::Fixed { .. } => {
                // Kinematic meshes don't have material properties.
                self.init_kinematic_vertex_attributes()?;
            }
            ShellData::Rigid { cm, mass, .. } => {
                self.init_dynamic_vertex_attributes()?;
                self.init_rest_pos_vertex_attribute(cm)?;
                // Vertex masses are needed for the friction solve.
                self.trimesh
                    .set_attrib::<MassType, VertexIndex>(MASS_ATTRIB, mass)?;
            }
            ShellData::Soft { .. } => {
                self.init_deformable_vertex_attributes()?;

                self.init_fixed_element_attribute()?;

                self.init_deformable_attributes()?;

                {
                    // Add elastic strain energy attribute.
                    // This will be computed at the end of the time step.
                    self.trimesh
                        .set_attrib::<StrainEnergyType, FaceIndex>(STRAIN_ENERGY_ATTRIB, 0f64)?;
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

    /// A helper function to flag all elements with all vertices fixed as fixed.
    pub(crate) fn init_fixed_element_attribute(&mut self) -> Result<(), Error> {
        let fixed_verts = self
            .mesh()
            .attrib_as_slice::<FixedIntType, VertexIndex>(FIXED_ATTRIB)?;

        let fixed_elements: Vec<_> = self
            .mesh()
            .face_iter()
            .map(|face| face.iter().map(|&vi| fixed_verts[vi]).sum::<i8>() / 3)
            .collect();

        self.mesh_mut()
            .set_attrib_data::<FixedIntType, FaceIndex>(FIXED_ATTRIB, &fixed_elements)?;

        Ok(())
    }

    pub(crate) fn init_rest_pos_vertex_attribute(&mut self, cm: Vector3<f64>) -> Result<(), Error> {
        // Translate every vertex such that the object's center of mass is at the origin.
        // This is done for rigid bodies.
        let ref_pos: Vec<_> = self
            .trimesh
            .vertex_position_iter()
            .map(|&x| (x.into_tensor() - cm).into_data())
            .collect();
        self.trimesh
            .set_attrib_data::<RigidRefPosType, VertexIndex>(
                REFERENCE_VERTEX_POS_ATTRIB,
                &ref_pos,
            )?;
        Ok(())
    }

    pub(crate) fn init_deformable_attributes(&mut self) -> Result<(), Error> {
        self.init_vertex_face_ref_pos_attribute()?;

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

        self.init_bending_attributes()?;

        Ok(())
    }

    /// Compute areas for reference triangles in the given `TriMesh`.
    fn compute_ref_tri_areas(mesh: &mut TriMesh) -> Result<Vec<f64>, Error> {
        let ref_pos =
            mesh.attrib_as_slice::<RefPosType, FaceVertexIndex>(REFERENCE_FACE_VERTEX_POS_ATTRIB)?;
        let areas: Vec<_> = ref_pos
            .chunks_exact(3)
            .map(|tri| ref_tri(tri).area())
            .collect();
        let degens: Vec<_> = areas
            .iter()
            .enumerate()
            .filter_map(|(i, &a)| if a <= 0.0 { Some(i) } else { None })
            .collect();
        if !degens.is_empty() {
            return Err(Error::DegenerateReferenceElement { degens });
        }
        Ok(areas)
    }

    /// Convert a 3D triangle shape matrix into a 2D matrix assuming an isotropic deformation
    /// model.
    ///
    /// Assume that reference triangles are non-degenerate.
    pub fn isotropic_tri_shape_matrix<T: Real>(m: Matrix2x3<T>) -> Matrix2<T> {
        // Project (orthogonally) second row onto the first.
        let m0_norm = m[0].norm();
        let e0 = m[0] / m0_norm;
        let m1_e0 = e0.dot(m[1]);
        let m1_e1 = (m[1] - e0 * m1_e0).norm();

        Matrix2::new([[m0_norm, T::zero()], [m1_e0, m1_e1]])
    }

    /// Compute shape matrix inverses for reference elements in the given `TriMesh`.
    fn compute_ref_tri_shape_matrix_inverses(
        mesh: &mut TriMesh,
    ) -> Result<Vec<Matrix2<f64>>, Error> {
        let ref_pos =
            mesh.attrib_as_slice::<RefPosType, FaceVertexIndex>(REFERENCE_FACE_VERTEX_POS_ATTRIB)?;
        // Compute reference shape matrix inverses
        Ok(ref_pos
            .chunks_exact(3)
            .map(|tri| {
                let ref_shape_matrix = Matrix2x3::new(ref_tri(tri).shape_matrix());
                Self::isotropic_tri_shape_matrix(ref_shape_matrix)
                    .inverse()
                    .unwrap()
            })
            .collect())
    }

    pub(crate) fn init_bending_attributes(&mut self) -> Result<(), Error> {
        let material = match self.data {
            ShellData::Soft { material, .. } => material,
            _ => return Err(Error::ObjectMaterialMismatch),
        };
        // Initialize bending stiffness face attribute
        let num_elements = self.num_elements();
        if let Some(bending_stiffness) = material.bending_stiffness() {
            match self
                .mesh_mut()
                .add_attrib_data::<BendingStiffnessType, FaceIndex>(
                    BENDING_STIFFNESS_ATTRIB,
                    vec![bending_stiffness; num_elements],
                ) {
                Err(attrib::Error::AlreadyExists(_)) => {}
                Err(e) => return Err(e.into()),
                _ => {}
            }
        } else {
            // If no bending stiffness was provided use what is already there. If there
            // is nothing on the mesh, simply initialize bending stiffness to zero. This is
            // a reasonable default.
            self.mesh_mut()
                .attrib_or_add::<BendingStiffnessType, FaceIndex>(BENDING_STIFFNESS_ATTRIB, 0.0)?;
        }

        let mesh = self.mesh();

        // Initialize edge topology and reference quantities.
        let ref_pos = Chunked3::from_flat(
            mesh.attrib_as_slice::<RefPosType, FaceVertexIndex>(REFERENCE_FACE_VERTEX_POS_ATTRIB)?,
        )
        .into_arrays();

        let mut interior_edges = compute_interior_edge_topology(&mesh);
        let mut interior_edge_bending_stiffness = vec![0.0; interior_edges.len()];
        let (mut interior_edge_ref_angles, mut interior_edge_ref_length): (Vec<_>, Vec<_>) =
            interior_edges
                .iter()
                .map(|e| {
                    let length = f64::from(e.ref_length(&ref_pos));
                    // A triangle height measure used to normalize the length. This allows the energy
                    // model to correctly approximate mean curvature.
                    let h_e = f64::from(e.tile_span(&ref_pos));
                    let ref_angle = f64::from(e.ref_edge_angle(&ref_pos));
                    (ref_angle, length / h_e)
                })
                .unzip();

        // Check if there are any additional reference angle attributes in the mesh, and ADD them
        // to the computed reference angles. This allows for non-coincident edges in reference
        // configuration to have a non-zero reference angle.
        if let Ok(ref_angles) = mesh
            .attrib_as_slice::<RefAngleType, FaceEdgeIndex>(REFERENCE_ANGLE_ATTRIB)
            .or_else(|_| {
                mesh.attrib_as_slice::<RefAngleType, FaceVertexIndex>(REFERENCE_ANGLE_ATTRIB)
            })
        {
            let ref_angles = Chunked3::from_flat(ref_angles).into_arrays();
            interior_edges
                .iter()
                .zip(interior_edge_ref_angles.iter_mut())
                .for_each(|(e, ref_angle)| {
                    *ref_angle += f64::from(ref_angles[e.faces[0]][e.edge_start[0] as usize])
                });
        }

        // Initialize interior_edge_angles.
        let mut interior_edge_angles: Vec<_> = interior_edges
            .iter()
            .map(|e| e.edge_angle(mesh.vertex_positions(), mesh.faces()))
            .collect();

        // At this point we are confident that bending stiffness is correctly initialized on the mesh.
        // Now it remains to move it to interior edges by averaging of the bending stiffnesses of
        // the adjacent faces.
        let face_bending_stiffnesses =
            mesh.attrib_as_slice::<BendingStiffnessType, FaceIndex>(BENDING_STIFFNESS_ATTRIB)?;
        for (e, mult) in interior_edges
            .iter()
            .zip(interior_edge_bending_stiffness.iter_mut())
        {
            *mult = 0.5
                * (face_bending_stiffnesses[e.faces[0]] as f64
                    + face_bending_stiffnesses[e.faces[1]] as f64);
        }

        // This should be the last step in initializing parameters for computing bending energy.
        // We can prune all edges for which bending stiffness is zero as to lower the computation
        // cost during simulation as much as possible.
        let mut bs_iter = interior_edge_bending_stiffness.iter().cloned();
        interior_edges.retain(|_| bs_iter.next().unwrap() != 0.0);
        let mut bs_iter = interior_edge_bending_stiffness.iter().cloned();
        interior_edge_angles.retain(|_| bs_iter.next().unwrap() != 0.0);
        let mut bs_iter = interior_edge_bending_stiffness.iter().cloned();
        interior_edge_ref_angles.retain(|_| bs_iter.next().unwrap() != 0.0);
        let mut bs_iter = interior_edge_bending_stiffness.iter().cloned();
        interior_edge_ref_length.retain(|_| bs_iter.next().unwrap() != 0.0);
        interior_edge_bending_stiffness.retain(|&bs| bs != 0.0);
        log::debug!("Number of interior edges: {}", interior_edges.len());

        // Ensure that whatever pruning algorithm used above produces same sized vectors.
        assert_eq!(interior_edges.len(), interior_edge_bending_stiffness.len());
        assert_eq!(interior_edges.len(), interior_edge_angles.len());
        assert_eq!(interior_edges.len(), interior_edge_ref_angles.len());
        assert_eq!(interior_edges.len(), interior_edge_ref_length.len());

        if let ShellData::Soft {
            interior_edges: self_interior_edges,
            interior_edge_bending_stiffness: self_interior_edge_bending_stiffness,
            interior_edge_angles: self_interior_edge_angles,
            interior_edge_ref_angles: self_interior_edge_ref_angles,
            interior_edge_ref_length: self_interior_edge_ref_length,
            ..
        } = &mut self.data
        {
            *self_interior_edges = interior_edges;
            *self_interior_edge_bending_stiffness = interior_edge_bending_stiffness;
            *self_interior_edge_angles = interior_edge_angles;
            *self_interior_edge_ref_angles = interior_edge_ref_angles;
            *self_interior_edge_ref_length = interior_edge_ref_length;
        }
        Ok(())
    }

    /// A helper function to populate the vertex face reference position attribute.
    #[unroll_for_loops]
    pub(crate) fn init_vertex_face_ref_pos_attribute(&mut self) -> Result<(), Error> {
        let mesh = self.mesh_mut();
        let mut ref_pos = vec![[0.0; 3]; mesh.num_face_vertices()];
        let pos = if let Ok(vtx_ref_pos) =
            mesh.attrib_as_slice::<RefPosType, VertexIndex>(REFERENCE_VERTEX_POS_ATTRIB)
        {
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
            REFERENCE_FACE_VERTEX_POS_ATTRIB,
            ref_pos.as_slice(),
        )?;
        Ok(())
    }

    /// Compute vertex masses on the given shell. The shell is assumed to have
    /// area and density attributes already.
    pub(crate) fn compute_vertex_masses(&mut self) -> Result<(), Error> {
        let trimesh = self.mesh_mut();
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

        trimesh.set_attrib_data::<MassType, VertexIndex>(MASS_ATTRIB, &masses)?;
        Ok(())
    }

    pub fn tagged_mesh(&self) -> Var<&TriMesh, f64> {
        match self.data {
            ShellData::Fixed { .. } => Var::Fixed(&self.trimesh),
            ShellData::Rigid { mass, inertia, .. } => Var::Rigid(&self.trimesh, mass, inertia),
            _ => Var::Variable(&self.trimesh),
        }
    }

    pub fn rigid_effective_mass_inv(
        mass: f64,
        translation: Vector3<f64>,
        rot: Vector3<f64>,
        inertia: Matrix3<f64>,
        contact_points: SubsetView<Chunked3<&[f64]>>,
    ) -> DBlockMatrix3<f64> {
        let n = contact_points.len();
        debug_assert!(n > 0);
        let mut out = UniChunked::from_flat_with_stride(
            UniChunked::from_flat(UniChunked::from_flat(vec![0.0; n * n * 9])),
            n,
        );

        let inertia_inv = inertia.inverse().expect("Failed to invert inertia matrix");

        for (&row_p, mut out_row) in contact_points.iter().zip(out.iter_mut()) {
            for (&col_p, out_block) in contact_points.iter().zip(out_row.iter_mut()) {
                let block: Matrix3<f64> = Matrix3::identity() / mass
                    - rotate(row_p.into_tensor() - translation, -rot).skew()
                        * inertia_inv
                        * rotate(col_p.into_tensor() - translation, -rot).skew();
                let out_arrays: &mut [[f64; 3]; 3] = out_block.into_arrays();
                *out_arrays = block.into_data();
            }
        }

        out.into_tensor()
    }
}

impl<'a> Elasticity<'a, Option<TriMeshNeoHookean<'a, f64>>> for TriMeshShell {
    fn elasticity(&'a self) -> Option<TriMeshNeoHookean<'a, f64>> {
        TriMeshNeoHookean::new(self)
    }
}

/// Inertia implementation for trimesh shells.
///
/// Shells can be fixed (no inertia), rigid or soft.
impl<'a> Inertia<'a, Option<Either<SoftShellInertia<'a>, RigidShellInertia>>> for TriMeshShell {
    fn inertia(&'a self) -> Option<Either<SoftShellInertia<'a>, RigidShellInertia>> {
        match self.data {
            ShellData::Fixed { .. } => None,
            ShellData::Rigid {
                mass, cm, inertia, ..
            } => Some(Either::Right(RigidShellInertia { mass, cm, inertia })),
            ShellData::Soft { .. } => Some(Either::Left(SoftShellInertia(self))),
        }
    }
}

impl<'a> Gravity<'a, Option<Either<SoftShellGravity<'a>, RigidShellGravity>>> for TriMeshShell {
    fn gravity(&'a self, g: [f64; 3]) -> Option<Either<SoftShellGravity<'a>, RigidShellGravity>> {
        match self.data {
            ShellData::Fixed { .. } => None,
            ShellData::Rigid { mass, .. } => Some(Either::Right(RigidShellGravity::new(g, mass))),
            ShellData::Soft { .. } => Some(Either::Left(SoftShellGravity::new(self, g))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::*;
    use approx::*;

    /// This test verifies that the correct implementation of integrating rigid properties like
    /// tensor of inertia, center of mass and mass of a rigid body.
    #[test]
    fn rigid_properties() {
        // We can compute these properties by hand of a few meshes with known symmetries.

        // Useful consts:
        let pi = std::f64::consts::PI;
        let sin45 = 1.0 / 2.0_f64.sqrt(); // same as cos(45)

        // Off-center Axis aligned cube
        let mut mesh = make_box(4).surface_trimesh();
        mesh.translate([0.1, 0.2, 0.3]);

        // Expected values
        let exp_mass = 2.0;
        let exp_cm = Vector3::new([0.1, 0.2, 0.3]);
        let moment_of_inertia = 1.0 / 3.0;
        let exp_inertia = Matrix3::identity() * moment_of_inertia;

        let (mass, cm, inertia) = TriMeshShell::integrate_rigid_properties(&mesh, 2.0);
        assert_relative_eq!(mass, exp_mass);
        assert_relative_eq!(cm, exp_cm, max_relative = 1e-7);
        assert_relative_eq!(inertia, exp_inertia);

        // Rotated cube at the origin
        let mut mesh = make_box(4).surface_trimesh();
        mesh.rotate_by_vector([pi / 4.0, 0.0, 0.0]);
        mesh.rotate_by_vector([0.0, pi / 3.0, 0.0]);

        // Expected values
        let moment_of_inertia_y = 2.0 * (2.0 * sin45 * sin45 + 1.0) / 12.0;
        let exp_cm = Vector3::new([0.0; 3]);

        let (mass, cm, inertia) = TriMeshShell::integrate_rigid_properties(&mesh, 2.0);
        assert_relative_eq!(mass, exp_mass);
        assert_relative_eq!(cm, exp_cm, max_relative = 1e-7);
        assert_relative_eq!(inertia[1][1], moment_of_inertia_y, max_relative = 1e-7);

        // Rotated cube off center
        let mut mesh = make_box(4).surface_trimesh();
        mesh.rotate_by_vector([pi / 4.0, 0.0, 0.0]);
        mesh.rotate_by_vector([0.0, pi / 3.0, 0.0]);
        mesh.translate([0.1, 0.2, 0.3]);

        let exp_cm = Vector3::new([0.1, 0.2, 0.3]);

        let (mass, cm, inertia) = TriMeshShell::integrate_rigid_properties(&mesh, 2.0);
        assert_relative_eq!(mass, exp_mass);
        assert_relative_eq!(cm, exp_cm, max_relative = 1e-7);
        assert_relative_eq!(inertia[1][1], moment_of_inertia_y, max_relative = 1e-7);

        // Torus off center
        let inner_radius = 0.25;
        let outer_radius = 0.5;
        let mut torus = TriMesh::from(
            geo::mesh::builder::TorusBuilder {
                outer_radius,
                inner_radius,
                outer_divs: 300,
                inner_divs: 150,
            }
            .build(),
        );

        torus.translate([0.1, 0.2, 0.3]);

        let a2 = inner_radius as f64 * inner_radius as f64;
        let b = outer_radius as f64;
        let b2 = b * b;
        let exp_mass = 2.0 * 2.0 * pi * pi * b * a2;
        let exp_cm = Vector3::new([0.1, 0.2, 0.3]);
        let inertia_y = 0.25 * exp_mass * (4.0 * b2 + 3.0 * a2);
        let inertia_xz = 0.125 * exp_mass * (4.0 * b2 + 5.0 * a2);
        let mut exp_inertia = Matrix3::identity();
        exp_inertia[0][0] = inertia_xz;
        exp_inertia[1][1] = inertia_y;
        exp_inertia[2][2] = inertia_xz;

        let (mass, cm, inertia) = TriMeshShell::integrate_rigid_properties(&torus, 2.0);
        assert!(mass < exp_mass); // discretized torus is smaller
        assert_relative_eq!(mass, exp_mass, max_relative = 1e-3);
        assert_relative_eq!(cm, exp_cm, max_relative = 1e-3);
        assert_relative_eq!(inertia, exp_inertia, max_relative = 1e-3, epsilon = 1e-10);
    }

    /// Test generation of rigid effective mass.
    #[test]
    fn rigid_effective_mass_inv() {
        let mesh = make_box(2).surface_trimesh();
        let material = RigidMaterial::new(0, 1.0);
        let shell = TriMeshShell::rigid(mesh, material);
        let (inertia, mass) = match shell.data {
            ShellData::Rigid { inertia, mass, .. } => (inertia, mass),
            _ => panic!("Looking for a rigid mesh here."),
        };

        let contact_points: Subset<_> = Subset::all(Chunked3::from_array_vec(vec![
            [0.0, 0.5, 0.0],  // top
            [-0.5, 0.0, 0.0], // side
            [0.5, 0.5, 0.0],  // corner coincident with vertex
        ]));

        let effective_mass_inv = TriMeshShell::rigid_effective_mass_inv(
            mass,
            [0.0; 3].into_tensor(),
            [0.0; 3].into_tensor(),
            inertia,
            contact_points.view(),
        )
        .into_data();

        for row in 0..3 {
            let rrow = contact_points[row].into_tensor().skew();
            for col in 0..3 {
                let rcol = contact_points[col].into_tensor().skew();
                let exp_block =
                    Matrix3::identity() / mass - rrow * inertia.inverse().unwrap() * rcol;
                let actual_block = effective_mass_inv.view().at(row).at(col);
                for i in 0..3 {
                    assert_relative_eq!(actual_block[i].into_tensor(), exp_block[i]);
                }
            }
        }
    }
}
