use unroll::unroll_for_loops;

use flatk::*;
use geo::index::Index;
use geo::mesh::{topology::*, CellType, VertexPositions};
use geo::ops::*;
use tensr::*;
use utils::CheckedIndex;

use crate::attrib_defines::*;
use crate::energy_models::elasticity::*;
use crate::energy_models::gravity::*;
use crate::energy_models::inertia::*;
use crate::fem::nl::state::VertexType;
use crate::objects::*;
use crate::Mesh;
use crate::Real;

use super::interior_edge::*;

fn shell_mtl_iter<'a>(
    mesh: &'a Mesh,
    materials: &'a [Material],
    orig_cell_indices: &'a [usize],
) -> Result<impl Iterator<Item = Result<&'a SoftShellMaterial, Error>>, Error> {
    let mtl_id = mesh.attrib_as_slice::<MaterialIdType, CellIndex>(MATERIAL_ID_ATTRIB)?;
    Ok(orig_cell_indices.iter().map(move |&orig_cell_idx| {
        let mtl_id = mtl_id[orig_cell_idx];
        // CAST: safely clamping mtl_id below at 0 before converting to usize.
        if let Material::SoftShell(mtl) = &materials[mtl_id as usize] {
            Ok(mtl)
        } else {
            Err(Error::ObjectMaterialMismatch)
        }
    }))
}

#[derive(Clone, Debug, Default)]
pub struct TriangleElements {
    // Cell indices into the mesh these triangles originated from.
    pub orig_cell_indices: Vec<usize>,
    pub triangles: Vec<[usize; 3]>,
    pub density: Vec<f32>,
    pub ref_area: Vec<f64>,
    pub ref_tri_shape_mtx_inv: Vec<Matrix2<f64>>,
    pub damping: Vec<f32>,
    pub lambda: Vec<f32>,
    pub mu: Vec<f32>,
}

impl TriangleElements {
    fn try_from_mesh_and_materials(
        mesh: &Mesh,
        materials: &[Material],
        vertex_type: &[VertexType],
    ) -> Result<TriangleElements, Error> {
        // Initialize triangle topology.
        let (orig_cell_indices, triangles): (Vec<_>, Vec<_>) = mesh
            .cell_iter()
            .zip(mesh.cell_type_iter())
            .enumerate()
            .filter_map(|(i, (cell, ty))| {
                if ty == CellType::Triangle
                    && cell.len() == 3
                    && (vertex_type[cell[0]] != VertexType::Fixed
                        || vertex_type[cell[1]] != VertexType::Fixed
                        || vertex_type[cell[2]] != VertexType::Fixed)
                {
                    Some((i, [cell[0], cell[1], cell[2]]))
                } else {
                    None
                }
            })
            .unzip();

        // Initialize density.
        let density: Vec<_> = if let Ok(density_attrib) =
            mesh.attrib_as_slice::<DensityType, CellIndex>(DENSITY_ATTRIB)
        {
            orig_cell_indices
                .iter()
                .map(|&orig_cell_idx| density_attrib[orig_cell_idx])
                .collect()
        } else {
            let mtls = shell_mtl_iter(mesh, materials, &orig_cell_indices)?;
            mtls.map(|mtl| mtl?.density().ok_or(Error::MissingDensity))
                .collect::<Result<Vec<_>, Error>>()?
        };

        // Initialize reference areas.
        let ref_area = Self::compute_ref_tri_areas(mesh, orig_cell_indices.as_slice())?;

        // Initialize reference shape matrix inverses.
        let ref_tri_shape_mtx_inv =
            Self::compute_ref_tri_shape_matrix_inverses(mesh, orig_cell_indices.as_slice())?;

        // Initialize elasticity parameters.
        let lambda = if let Ok(lambda_attrib) =
            mesh.attrib_as_slice::<LambdaType, CellIndex>(LAMBDA_ATTRIB)
        {
            orig_cell_indices
                .iter()
                .map(|&orig_cell_idx| lambda_attrib[orig_cell_idx])
                .collect()
        } else {
            let mtls = shell_mtl_iter(mesh, materials, &orig_cell_indices)?;
            mtls.map(|mtl| {
                mtl?.elasticity()
                    .map(|x| x.lambda)
                    .ok_or(Error::MissingElasticityParams)
            })
            .collect::<Result<Vec<_>, Error>>()?
        };

        let mu = if let Ok(mu_attrib) = mesh.attrib_as_slice::<MuType, CellIndex>(MU_ATTRIB) {
            orig_cell_indices
                .iter()
                .map(|&orig_cell_idx| mu_attrib[orig_cell_idx])
                .collect()
        } else {
            let mtls = shell_mtl_iter(mesh, materials, &orig_cell_indices)?;
            mtls.map(|mtl| {
                mtl?.elasticity()
                    .map(|x| x.mu)
                    .ok_or(Error::MissingElasticityParams)
            })
            .collect::<Result<Vec<_>, Error>>()?
        };

        let damping = if let Ok(damping_attrib) =
            mesh.attrib_as_slice::<DampingType, CellIndex>(DAMPING_ATTRIB)
        {
            orig_cell_indices
                .iter()
                .map(|&orig_cell_idx| damping_attrib[orig_cell_idx])
                .collect()
        } else {
            let mtls = shell_mtl_iter(mesh, materials, &orig_cell_indices)?;
            mtls.map(|mtl| Ok(mtl?.damping()))
                .collect::<Result<Vec<_>, Error>>()?
        };

        Ok(TriangleElements {
            orig_cell_indices,
            triangles,
            density,
            ref_area,
            ref_tri_shape_mtx_inv,
            damping,
            lambda,
            mu,
        })
    }

    /// Compute areas for reference triangles in the given `Mesh`.
    fn compute_ref_tri_areas(mesh: &Mesh, orig_cell_indices: &[usize]) -> Result<Vec<f64>, Error> {
        let ref_pos =
            mesh.attrib_as_slice::<RefPosType, CellVertexIndex>(REFERENCE_CELL_VERTEX_POS_ATTRIB)?;
        let areas: Vec<_> = orig_cell_indices
            .iter()
            .map(|&orig_cell_idx| {
                let tri = [
                    ref_pos[mesh.cell_vertex(orig_cell_idx, 0).unwrap().into_inner()],
                    ref_pos[mesh.cell_vertex(orig_cell_idx, 1).unwrap().into_inner()],
                    ref_pos[mesh.cell_vertex(orig_cell_idx, 2).unwrap().into_inner()],
                ];
                ref_tri(&tri).area()
            })
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

    /// Compute shape matrix inverses for reference elements in the given mesh.
    fn compute_ref_tri_shape_matrix_inverses(
        mesh: &Mesh,
        orig_cell_indices: &[usize],
    ) -> Result<Vec<Matrix2<f64>>, Error> {
        let ref_pos =
            mesh.attrib_as_slice::<RefPosType, CellVertexIndex>(REFERENCE_CELL_VERTEX_POS_ATTRIB)?;
        // Compute reference shape matrix inverses
        Ok(orig_cell_indices
            .iter()
            .map(|&orig_cell_idx| {
                let tri = [
                    ref_pos[mesh.cell_vertex(orig_cell_idx, 0).unwrap().into_inner()],
                    ref_pos[mesh.cell_vertex(orig_cell_idx, 1).unwrap().into_inner()],
                    ref_pos[mesh.cell_vertex(orig_cell_idx, 2).unwrap().into_inner()],
                ];
                let ref_shape_matrix = Matrix2x3::new(ref_tri(&tri).shape_matrix());
                Self::isotropic_tri_shape_matrix(ref_shape_matrix)
                    .inverse()
                    .unwrap()
            })
            .collect())
    }

    /// Given a mesh, compute the strain energy per triangle.
    ///
    /// If the strain attribute exists on the mesh, this function will only add
    /// to the existing attribute, and will not overwrite any values not
    /// associated with triangle of this `TriShell`.
    pub(crate) fn compute_strain_energy_attrib(&self, mesh: &mut Mesh) -> Result<(), Error> {
        // Set the "strain_energy" attribute.
        mesh.attrib_or_insert_with_default::<StrainEnergyType, CellIndex>(
            STRAIN_ENERGY_ATTRIB,
            0.0f64,
        )?;
        // No panics since attribute was added above.
        let mut strain_attrib = mesh
            .remove_attrib::<CellIndex>(STRAIN_ENERGY_ATTRIB)
            .unwrap();
        // Panics: No panic since strain is created above as f64.
        let strain = strain_attrib.as_mut_slice::<StrainEnergyType>().unwrap();

        zip!(
            self.orig_cell_indices.iter(),
            self.lambda.iter(),
            self.mu.iter(),
            self.ref_area.iter(),
            self.ref_tri_shape_mtx_inv.iter(),
            self.triangles.iter(),
        )
        .for_each(
            |(&orig_cell_idx, &lambda, &mu, &area, &ref_shape_mtx_inv, tri_indices)| {
                let tri =
                    geo::prim::Triangle::from_indexed_slice(tri_indices, mesh.vertex_positions());
                let shape_mtx = Matrix2x3::new(tri.shape_matrix());
                strain[orig_cell_idx] = NeoHookeanTriEnergy::new(
                    shape_mtx,
                    ref_shape_mtx_inv,
                    area,
                    f64::from(lambda),
                    f64::from(mu),
                )
                .energy()
            },
        );
        mesh.insert_attrib::<CellIndex>(STRAIN_ENERGY_ATTRIB, strain_attrib)?;
        Ok(())
    }

    /// Add triangle vertex masses on the given mesh.
    ///
    /// A mass attribute is added to the given `Mesh` if one doesn't already exist.
    /// Note that if the mesh already has a mass attribute on the vertices, it will be added to.
    /// To get fresh masses please zero out the mass attribute with `MassType` on `VertexIndex` topology (vertices).
    pub(crate) fn add_vertex_masses(&self, mesh: &mut Mesh) -> Result<(), Error> {
        let masses_attrib =
            mesh.attrib_or_insert_with_default::<MassType, VertexIndex>(MASS_ATTRIB, 0.0)?;
        let masses = masses_attrib.as_mut_slice::<MassType>().unwrap();

        for (tri_indices, &area, density) in zip!(
            self.triangles.iter(),
            self.ref_area.iter(),
            self.density.iter().map(|&x| f64::from(x)),
        ) {
            for i in 0..3 {
                masses[tri_indices[i]] += area * density / 3.0;
            }
        }

        Ok(())
    }

    /// Given a mesh, compute the elastic forces per vertex due to triangle
    /// potentials and add it at a vertex attribute.
    ///
    /// If the attribute doesn't already exists, it will be created, otherwise
    /// it will be added to.
    fn add_elastic_forces(&self, mesh: &mut Mesh) -> Result<(), Error> {
        mesh.attrib_or_insert_with_default::<ElasticForceType, VertexIndex>(
            ELASTIC_FORCE_ATTRIB,
            [0.0; 3],
        )?;
        let mut forces_attrib = mesh.remove_attrib::<VertexIndex>(ELASTIC_FORCE_ATTRIB)?;
        let forces = forces_attrib.as_mut_slice::<ElasticForceType>().unwrap();

        zip!(
            self.lambda.iter(),
            self.mu.iter(),
            self.ref_area.iter(),
            self.ref_tri_shape_mtx_inv.iter(),
            self.triangles.iter()
        )
        .for_each(|(&lambda, &mu, &area, &ref_shape_mtx_inv, tri_indices)| {
            let tri = geo::prim::Triangle::from_indexed_slice(tri_indices, mesh.vertex_positions());
            let shape_mtx = Matrix2x3::new(tri.shape_matrix());
            let grad = NeoHookeanTriEnergy::new(
                shape_mtx,
                ref_shape_mtx_inv,
                area,
                f64::from(lambda),
                f64::from(mu),
            )
            .energy_gradient();
            for j in 0..3 {
                let f = Vector3::new(forces[tri_indices[j]]);
                forces[tri_indices[j]] = (f - grad[j]).into();
            }
        });
        mesh.insert_attrib::<VertexIndex>(ELASTIC_FORCE_ATTRIB, forces_attrib)?;
        Ok(())
    }

    pub fn num_elements(&self) -> usize {
        self.triangles.len()
    }
}

#[derive(Clone, Debug, Default)]
pub struct DihedralElements {
    // Cell indices into the mesh from which `triangles` originated.
    pub orig_cell_indices: Vec<usize>,
    /// A list of triangle elements.
    ///
    /// This is different than triangles in `TriangleElements` because dihedrals
    /// can have a face that is completely fixed, which means that there may be
    /// elements here that are not in `TriangleElements`.  We need this vectors
    /// since dihedrals are defined by face topology, not vertex topology. This
    /// allows the same dihedral be represented by different vertex topologies.
    pub triangles: Vec<[usize; 3]>,
    /// A list of dihedral elements taken from the deformed topology, defined by a
    /// pair of faces and indices to the opposing vertices in those faces.
    ///
    /// Using deformed topology allows us to capture bending energy at seams and
    /// using face pairs enables capturing bending in non-manifold edges.
    pub dihedrals: Vec<InteriorEdge>,
    pub ref_angles: Vec<f64>,
    pub angles: Vec<f64>,
    /// A normalized (unitless) reference length.
    pub ref_length: Vec<f64>,
    pub bending_stiffness: Vec<f64>,
}

impl DihedralElements {
    #[unroll_for_loops]
    fn try_from_mesh_and_materials(
        mesh: &Mesh,
        materials: &[Material],
        vertex_type: &[VertexType],
    ) -> Result<DihedralElements, Error> {
        // Initialize edge topology and reference quantities.
        let ref_pos =
            mesh.attrib_as_slice::<RefPosType, CellVertexIndex>(REFERENCE_CELL_VERTEX_POS_ATTRIB)?;
        let get_ref_pos = |f, i| ref_pos[mesh.cell_vertex(f, i).unwrap().into_inner()];

        let mut dihedrals = compute_interior_edge_topology_from_mesh(&mesh, vertex_type)?;
        let (mut ref_angles, mut ref_length): (Vec<_>, Vec<_>) = dihedrals
            .iter()
            .map(|e| {
                let length = f64::from(e.ref_length(get_ref_pos));
                // A triangle height measure used to normalize the length. This allows the energy
                // model to correctly approximate mean curvature.
                let h_e = f64::from(e.tile_span(get_ref_pos));
                let ref_angle = f64::from(e.ref_edge_angle(get_ref_pos));
                (ref_angle, length / h_e)
            })
            .unzip();

        // Check if there are any additional reference angle attributes in the mesh, and ADD them
        // to the computed reference angles. This allows for non-coincident edges in reference
        // configuration to have a non-zero reference angle.
        if let Ok(ref_angles_attrib) =
            mesh.attrib_as_slice::<RefAngleType, CellVertexIndex>(REFERENCE_ANGLE_ATTRIB)
        {
            dihedrals
                .iter()
                .zip(ref_angles.iter_mut())
                .for_each(|(e, ref_angle)| {
                    *ref_angle += f64::from(
                        ref_angles_attrib[mesh
                            .cell_vertex(e.faces[0], e.edge_start[0] as usize)
                            .unwrap()
                            .into_inner()],
                    )
                });
        }

        // Initialize interior_edge_angles.
        let mut angles: Vec<_> = dihedrals
            .iter()
            .map(|e| {
                e.edge_angle(mesh.vertex_positions(), |c, i| {
                    mesh.cell_to_vertex(c, i).unwrap().into_inner()
                })
            })
            .collect();

        // At this point we are confident that bending stiffness is correctly initialized on the mesh.
        // Now it remains to move it to the dihedrals by averaging of the bending stiffnesses of
        // the adjacent faces.
        let mut bending_stiffness = vec![0.0; dihedrals.len()];
        // Initialize bending stiffness
        let cell_bending_stiffnesses = if let Ok(attrib) =
            mesh.attrib_as_slice::<BendingStiffnessType, CellIndex>(BENDING_STIFFNESS_ATTRIB)
        {
            // TODO: Refactor the unnecessary cloning here:
            attrib.to_vec()
        } else {
            mesh.attrib_iter::<MaterialIdType, CellIndex>(MATERIAL_ID_ATTRIB)?
                .map(|&mtl_id| {
                    // CAST: safely clamping mtl_id below at 0 before converting to usize.
                    if let Material::SoftShell(mtl) = &materials[mtl_id.max(0) as usize] {
                        mtl.bending_stiffness().unwrap_or(0.0)
                    } else {
                        0.0
                    }
                })
                .collect()
        };

        for (e, mult) in dihedrals.iter().zip(bending_stiffness.iter_mut()) {
            *mult = 0.5
                * (cell_bending_stiffnesses[e.faces[0]] as f64
                    + cell_bending_stiffnesses[e.faces[1]] as f64);
        }

        // This should be the last step in initializing parameters for computing bending energy.
        // We can prune all edges for which bending stiffness is zero as to lower the computation
        // cost during simulation as much as possible.
        let mut bs_iter = bending_stiffness.iter().cloned();
        dihedrals.retain(|_| bs_iter.next().unwrap() != 0.0);
        let mut bs_iter = bending_stiffness.iter().cloned();
        angles.retain(|_| bs_iter.next().unwrap() != 0.0);
        let mut bs_iter = bending_stiffness.iter().cloned();
        ref_angles.retain(|_| bs_iter.next().unwrap() != 0.0);
        let mut bs_iter = bending_stiffness.iter().cloned();
        ref_length.retain(|_| bs_iter.next().unwrap() != 0.0);
        bending_stiffness.retain(|&bs| bs != 0.0);
        log::debug!("Number of dihedrals: {}", dihedrals.len());

        // Ensure that whatever pruning algorithm used above produces same sized vectors.
        assert_eq!(dihedrals.len(), bending_stiffness.len());
        assert_eq!(dihedrals.len(), angles.len());
        assert_eq!(dihedrals.len(), ref_angles.len());
        assert_eq!(dihedrals.len(), ref_length.len());

        // Now we generate a set of triangles referenced by dihedrals, and remap
        // the dihedral face indices to reference this new set of triangles.
        let mut triangle_map = vec![Index::invalid(); mesh.num_cells()];
        let mut triangles = Vec::new();
        triangles.reserve(mesh.num_cells());
        let mut orig_cell_indices = Vec::new();
        orig_cell_indices.reserve(mesh.num_cells());
        for edge in dihedrals.iter_mut() {
            for i in 0..2 {
                if let Some(valid_idx) = triangle_map[edge.faces[i]].into_option() {
                    edge.faces[i] = valid_idx; // Remap to new index.
                } else {
                    let new_idx = triangles.len();
                    triangle_map[edge.faces[i]] = Index::new(new_idx);

                    // Push the newly discovered triangle.
                    let cell = flatk::View::view(&mesh.indices).at(edge.faces[i]);
                    triangles.push([cell[0], cell[1], cell[2]]);
                    // Update mapping to original mesh.
                    orig_cell_indices.push(edge.faces[i]);

                    edge.faces[i] = new_idx; // Remap to new index.
                }
            }
        }
        triangles.shrink_to_fit();
        orig_cell_indices.shrink_to_fit();

        Ok(DihedralElements {
            orig_cell_indices,
            triangles,
            dihedrals,
            ref_angles,
            angles,
            ref_length,
            bending_stiffness,
        })
    }

    /// Given a set of new vertex positions update the set of dihedral angles.
    pub(crate) fn update_angles<T: Real>(&mut self, x1: &[[T; 3]]) {
        let DihedralElements {
            triangles,
            dihedrals,
            angles,
            ..
        } = self;

        dihedrals.iter().zip(angles.iter_mut()).for_each(|(e, t)| {
            *t = e
                .incremental_angle(T::from(*t).unwrap(), x1, |f, i| triangles[f][i])
                .to_f64()
                .unwrap();
        });
    }

    pub fn num_elements(&self) -> usize {
        self.dihedrals.len()
    }
}

/// Data used for simulating shells.
#[derive(Clone, Debug)]
pub struct TriShell {
    pub triangle_elements: TriangleElements,
    pub dihedral_elements: DihedralElements,
}

impl TriShell {
    /// Returns true if the given cell is recognized by this `TriShell`.
    pub fn is_valid_cell(cell: &[usize], ty: CellType) -> bool {
        ty == CellType::Triangle && cell.len() == 3
    }

    /// A generic `TriShell` constructor that takes a dynamic reference to a material type
    /// and constructs a new mesh with the given material if it can be recognized.
    ///
    /// If the given material is not recognized, `None` is returned to prevent unexpected
    /// behaviour.
    pub fn try_from_mesh_and_materials(
        mesh: &Mesh,
        materials: &[Material],
        vertex_type: &[VertexType],
    ) -> Result<TriShell, Error> {
        let triangle_elements =
            TriangleElements::try_from_mesh_and_materials(mesh, materials, vertex_type)?;
        let dihedral_elements =
            DihedralElements::try_from_mesh_and_materials(mesh, materials, vertex_type)?;
        Ok(TriShell {
            triangle_elements,
            dihedral_elements,
        })
    }

    /// Given a set of new vertex positions update the set of dihedral angles.
    pub(crate) fn update_dihedral_angles<T: Real>(&mut self, x1: &[[T; 3]]) {
        self.dihedral_elements.update_angles(x1);
    }

    /// Add triangle vertex masses on the given mesh.
    ///
    /// A mass attribute is added to the given `Mesh` if one doesn't already exist.
    /// Note that if the mesh already has a mass attribute on the vertices, it will be added to.
    /// To get fresh masses please zero out the mass attribute with `MassType` on `VertexIndex` topology (vertices).
    #[allow(dead_code)]
    pub(crate) fn add_vertex_masses(&self, mesh: &mut Mesh) -> Result<(), Error> {
        self.triangle_elements.add_vertex_masses(mesh)
    }

    /// Given a mesh, compute the strain energy per triangle.
    ///
    /// If the strain attribute exists on the mesh, this function will only add
    /// to the existing attribute, and will not overwrite any values not
    /// associated with triangle of this `TriShell`.
    #[allow(dead_code)]
    fn compute_strain_energy_attrib(&self, mesh: &mut Mesh) -> Result<(), Error> {
        self.triangle_elements.compute_strain_energy_attrib(mesh)
    }

    /// Given a mesh, compute the elastic forces per vertex due to triangle
    /// potentials and add it at a vertex attribute.
    ///
    /// If the attribute doesn't already exists, it will be created, otherwise
    /// it will be added to.
    #[allow(dead_code)]
    fn add_elastic_forces(&self, mesh: &mut Mesh) -> Result<(), Error> {
        self.triangle_elements.add_elastic_forces(mesh)?;
        // TODO: implement forces due to dihedral potential (bending).
        Ok(())
    }

    //pub fn contact_surface(&self) -> crate::constraints::ContactSurface<&TriMesh, f64> {
    //    use crate::constraints::ContactSurface;
    //    match self.data {
    //        ShellData::Fixed { .. } => ContactSurface::fixed(&self.trimesh),
    //        ShellData::Rigid { mass, inertia, .. } => {
    //            ContactSurface::rigid(&self.trimesh, mass, inertia)
    //        }
    //        _ => ContactSurface::deformable(&self.trimesh),
    //    }
    //}

    pub fn rigid_effective_mass_inv<T: Real>(
        mass: T,
        translation: Vector3<T>,
        rot: Vector3<T>,
        inertia: Matrix3<T>,
        contact_points: SelectView<Chunked3<&[T]>>,
    ) -> Tensor![T; D D 3 3] {
        let n = contact_points.len();
        debug_assert!(n > 0);
        let mut out = <Tensor![T; D D 3 3]>::from_shape(&[n, n, 3, 3]);

        let inertia_inv = inertia.inverse().expect("Failed to invert inertia matrix");

        for ((_, &row_p), mut out_row) in contact_points.iter().zip(out.iter_mut()) {
            for ((_, &col_p), out_block) in contact_points.iter().zip(out_row.iter_mut()) {
                let block: Matrix3<T> = Matrix3::identity() / mass
                    - rotate(row_p.into_tensor() - translation, -rot).skew()
                        * inertia_inv
                        * rotate(col_p.into_tensor() - translation, -rot).skew();
                let out_arrays: &mut [[T; 3]; 3] = out_block.into_arrays();
                *out_arrays = block.into_data();
            }
        }

        out
    }
}

impl TriShell {
    pub fn elasticity<'a, T: Real>(&'a self) -> TriShellNeoHookean<'a, T> {
        TriShellNeoHookean::new(self)
    }
}

/// Inertia implementation for triangle shells.
impl<'a> Inertia<'a, SoftTriShellInertia<'a>> for TriShell {
    fn inertia(&'a self) -> SoftTriShellInertia<'a> {
        SoftTriShellInertia(self)
    }
}

/// Gravity implementation for triangle shells.
impl<'a> Gravity<'a, SoftTriShellGravity<'a>> for TriShell {
    fn gravity(&'a self, g: [f64; 3]) -> SoftTriShellGravity<'a> {
        SoftTriShellGravity::new(self, g)
    }
}
