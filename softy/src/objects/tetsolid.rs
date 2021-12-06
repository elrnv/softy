use crate::Real;
use flatk::*;
use geo::attrib::Attrib;
use geo::mesh::{CellType, VertexPositions};
use geo::ops::*;
use tensr::{Matrix3, Vector3};

use crate::attrib_defines::*;
use crate::energy_models::elasticity::*;
use crate::energy_models::gravity::*;
use crate::energy_models::inertia::*;
use crate::fem::nl::state::VertexType;
use crate::fem::ref_tet;
use crate::objects::{material::*, *};
use crate::Mesh;

fn solid_mtl_iter<'a>(
    mesh: &'a Mesh,
    materials: &'a [Material],
    orig_cell_indices: &'a [usize],
) -> Result<impl Iterator<Item = Result<&'a SolidMaterial, Error>>, Error> {
    let mtl_id = mesh.attrib_as_slice::<MaterialIdType, CellIndex>(MATERIAL_ID_ATTRIB)?;
    Ok(orig_cell_indices.iter().map(move |&orig_cell_idx| {
        let mtl_id = mtl_id[orig_cell_idx];
        // CAST: safely clamping mtl_id below at 0 before converting to usize.
        if let Material::Solid(mtl) = &materials[mtl_id.max(0) as usize] {
            Ok(mtl)
        } else {
            Err(Error::ObjectMaterialMismatch)
        }
    }))
}

#[derive(Clone, Debug)]
pub(crate) struct TetElements {
    pub orig_cell_indices: Vec<usize>,
    pub tets: Vec<[usize; 4]>,
    pub density: Vec<f32>,
    pub ref_volume: Vec<f64>,
    pub ref_tet_shape_mtx_inv: Vec<Matrix3<f64>>,
    pub damping: Vec<f32>,
    pub lambda: Vec<f32>,
    pub mu: Vec<f32>,
}

impl TetElements {
    pub fn try_from_mesh_and_materials(
        model: ElasticityModel,
        mesh: &Mesh,
        materials: &[Material],
        vertex_type: &[VertexType],
    ) -> Result<TetElements, Error> {
        // Initialize tet topology.
        let mtl_id_iter = mesh.attrib_iter::<MaterialIdType, CellIndex>(MATERIAL_ID_ATTRIB)?;
        let (orig_cell_indices, tets): (Vec<_>, Vec<_>) = mesh
            .cell_iter()
            .zip(mesh.cell_type_iter())
            .zip(mtl_id_iter)
            .enumerate()
            .filter_map(|(i, ((cell, ty), &mtl_id))| {
                if let Material::Solid(mtl) = &materials[mtl_id.max(0) as usize] {
                    if mtl.model() == model
                        && TetSolid::is_valid_cell(cell, ty)
                        && (vertex_type[cell[0]] != VertexType::Fixed
                            || vertex_type[cell[1]] != VertexType::Fixed
                            || vertex_type[cell[2]] != VertexType::Fixed
                            || vertex_type[cell[3]] != VertexType::Fixed)
                    {
                        return Some((i, [cell[0], cell[1], cell[2], cell[3]]));
                    }
                };

                None
            })
            .unzip();

        let mtls = solid_mtl_iter(mesh, materials, &orig_cell_indices)?;

        // Initialize density.
        let density: Vec<_> = if let Ok(density_attrib) =
            mesh.attrib_as_slice::<DensityType, CellIndex>(DENSITY_ATTRIB)
        {
            orig_cell_indices
                .iter()
                .map(|&orig_cell_idx| density_attrib[orig_cell_idx])
                .collect()
        } else {
            mtls.map(|mtl| mtl?.density().ok_or(Error::MissingDensity))
                .collect::<Result<Vec<_>, Error>>()?
        };

        // Initialize reference areas.
        let ref_volume = Self::compute_ref_tet_volumes(mesh, orig_cell_indices.as_slice())?;

        // Initialize reference shape matrix inverses.
        let ref_tet_shape_mtx_inv =
            Self::compute_ref_tet_shape_matrix_inverses(mesh, orig_cell_indices.as_slice())?;

        // Initialize elasticity parameters.
        let lambda = if let Ok(lambda_attrib) =
            mesh.attrib_as_slice::<LambdaType, CellIndex>(LAMBDA_ATTRIB)
        {
            orig_cell_indices
                .iter()
                .map(|&orig_cell_idx| lambda_attrib[orig_cell_idx])
                .collect()
        } else {
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
            mtls.map(|mtl| Ok(mtl?.damping()))
                .collect::<Result<Vec<_>, Error>>()?
        };

        Ok(TetElements {
            orig_cell_indices,
            tets,
            density,
            ref_volume,
            ref_tet_shape_mtx_inv,
            damping,
            lambda,
            mu,
        })
    }

    /// Compute areas for reference tetrahedra in the given `Mesh`.
    fn compute_ref_tet_volumes(
        mesh: &Mesh,
        orig_cell_indices: &[usize],
    ) -> Result<Vec<f64>, Error> {
        let ref_pos =
            mesh.attrib_as_slice::<RefPosType, CellVertexIndex>(REFERENCE_FACE_VERTEX_POS_ATTRIB)?;
        let ref_volumes: Vec<_> = orig_cell_indices
            .iter()
            .map(|&orig_cell_idx| {
                let cell = mesh.indices.view().at(orig_cell_idx);
                let tet = [
                    ref_pos[cell[0]],
                    ref_pos[cell[1]],
                    ref_pos[cell[2]],
                    ref_pos[cell[3]],
                ];
                ref_tet(&tet).signed_volume()
            })
            .collect();

        let inverted: Vec<_> = ref_volumes
            .iter()
            .enumerate()
            .filter_map(|(i, &v)| if v <= 0.0 { Some(i) } else { None })
            .collect();
        if !inverted.is_empty() {
            return Err(Error::InvertedReferenceElement { inverted });
        }
        Ok(ref_volumes)
    }

    /// Compute shape matrix inverses for reference elements in the given mesh.
    fn compute_ref_tet_shape_matrix_inverses(
        mesh: &Mesh,
        orig_cell_indices: &[usize],
    ) -> Result<Vec<Matrix3<f64>>, Error> {
        let ref_pos =
            mesh.attrib_as_slice::<RefPosType, CellVertexIndex>(REFERENCE_FACE_VERTEX_POS_ATTRIB)?;
        // Compute reference shape matrix inverses
        Ok(orig_cell_indices
            .iter()
            .map(|&orig_cell_idx| {
                let cell = mesh.indices.view().at(orig_cell_idx);
                let tet = [
                    ref_pos[cell[0]],
                    ref_pos[cell[1]],
                    ref_pos[cell[2]],
                    ref_pos[cell[3]],
                ];
                Matrix3::new(ref_tet(&tet).shape_matrix())
                    .inverse()
                    .unwrap()
            })
            .collect())
    }

    /// Given a mesh, compute the strain energy per tet.
    ///
    /// If the strain attribute exists on the mesh, this function will only add
    /// to the existing attribute, and will not overwrite any values not
    /// associated with tets of this `TetSolid`.
    pub(crate) fn compute_strain_energy_attrib(&self, mesh: &mut Mesh) {
        // Set the "strain_energy" attribute.
        let mut strain_attrib = mesh
            .attrib_or_insert_with_default::<StrainEnergyType, CellIndex>(
                STRAIN_ENERGY_ATTRIB,
                0.0f64,
            )
            .unwrap();
        // Panics: No panic since strain is created above as f64.
        let strain = strain_attrib.as_mut_slice::<StrainEnergyType>().unwrap();

        zip!(
            self.orig_cell_indices.iter(),
            self.lambda.iter(),
            self.mu.iter(),
            self.ref_volume.iter(),
            self.ref_tet_shape_mtx_inv.iter(),
            self.tets.iter(),
        )
        .for_each(
            |(&orig_cell_idx, &lambda, &mu, &volume, &ref_shape_mtx_inv, tet_indices)| {
                let tet = geo::prim::Tetrahedron::from_indexed_slice(
                    tet_indices,
                    mesh.vertex_positions(),
                );
                let shape_mtx = Matrix3::new(tet.shape_matrix());
                strain[orig_cell_idx] = NeoHookeanTetEnergy::new(
                    shape_mtx,
                    ref_shape_mtx_inv,
                    volume,
                    f64::from(lambda),
                    f64::from(mu),
                )
                .energy()
            },
        );
    }

    /// Add tet vertex masses on the given mesh.
    ///
    /// A mass attribute is added to the given `Mesh` if one doesn't already
    /// exist.  Note that if the mesh already has a mass attribute on the
    /// vertices, it will be added to.  To get fresh masses please zero out the
    /// mass attribute with `MassType` on `VertexIndex` topology (vertices).
    pub(crate) fn add_vertex_masses(&self, mesh: &mut Mesh) -> Result<(), Error> {
        let masses_attrib =
            mesh.attrib_or_insert_with_default::<MassType, VertexIndex>(MASS_ATTRIB, 0.0)?;
        let masses = masses_attrib.as_mut_slice::<MassType>().unwrap();

        for (tet_indices, &volume, density) in zip!(
            self.tets.iter(),
            self.ref_volume.iter(),
            self.density.iter().map(|&x| f64::from(x)),
        ) {
            for i in 0..3 {
                masses[tet_indices[i]] += volume * density / 3.0;
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
        let forces_attrib = mesh.attrib_or_insert_with_default::<ElasticForceType, VertexIndex>(
            ELASTIC_FORCE_ATTRIB,
            [0.0; 3],
        )?;
        let mut forces = forces_attrib.as_mut_slice::<ElasticForceType>().unwrap();

        zip!(
            self.lambda.iter(),
            self.mu.iter(),
            self.ref_volume.iter(),
            self.ref_tet_shape_mtx_inv.iter(),
            self.tets.iter()
        )
        .for_each(|(&lambda, &mu, &volume, &ref_shape_mtx_inv, tet_indices)| {
            let tet =
                geo::prim::Tetrahedron::from_indexed_slice(tet_indices, mesh.vertex_positions());
            let shape_mtx = Matrix3::new(tet.shape_matrix());
            let grad = NeoHookeanTetEnergy::new(
                shape_mtx,
                ref_shape_mtx_inv,
                volume,
                f64::from(lambda),
                f64::from(mu),
            )
            .energy_gradient();
            for j in 0..3 {
                let f = Vector3::new(forces[tet_indices[j]]);
                forces[tet_indices[j]] = (f - grad[j]).into();
            }
        });
        Ok(())
    }

    pub fn num_elements(&self) -> usize {
        self.tets.len()
    }
}

/// A soft solid represented by a tetrahedra.
///
/// It is effectively a tetrahedral mesh decorated by physical material
/// properties that govern how it behaves.
#[derive(Clone, Debug)]
pub struct TetSolid {
    /// Neo-Hookean tetrahedral elements.
    pub nh_tet_elements: TetElements,
    /// Stable neo-Hookean tetrahedral elements.
    pub snh_tet_elements: TetElements,
}

impl TetSolid {
    /// Returns true if the given cell is recognized by this `TriShell`.
    pub fn is_valid_cell(cell: &[usize], ty: CellType) -> bool {
        ty == CellType::Tetrahedron && cell.len() == 4
    }

    pub fn try_from_mesh_and_materials(
        mesh: &Mesh,
        materials: &[Material],
        vertex_type: &[VertexType],
    ) -> Result<TetSolid, Error> {
        let nh_tet_elements = TetElements::try_from_mesh_and_materials(
            ElasticityModel::NeoHookean,
            mesh,
            materials,
            vertex_type,
        )?;
        let snh_tet_elements = TetElements::try_from_mesh_and_materials(
            ElasticityModel::StableNeoHookean,
            mesh,
            materials,
            vertex_type,
        )?;

        Ok(TetSolid {
            nh_tet_elements,
            snh_tet_elements,
        })
    }

    /// Add triangle vertex masses on the given mesh.
    ///
    /// A mass attribute is added to the given `Mesh` if one doesn't already exist.
    /// Note that if the mesh already has a mass attribute on the vertices, it will be added to.
    /// To get fresh masses please zero out the mass attribute with `MassType` on `VertexIndex` topology (vertices).
    pub(crate) fn add_vertex_masses(&self, mesh: &mut Mesh) -> Result<(), Error> {
        self.nh_tet_elements.add_vertex_masses(mesh)?;
        self.snh_tet_elements.add_vertex_masses(mesh)
    }

    /// Given a mesh, compute the strain energy per tet.
    ///
    /// If the strain attribute exists on the mesh, this function will only add
    /// to the existing attribute, and will not overwrite any values not
    /// associated with tets of this `TetSolid`.
    fn compute_strain_energy_attrib(&self, mesh: &mut Mesh) {
        self.nh_tet_elements.compute_strain_energy_attrib(mesh);
        self.snh_tet_elements.compute_strain_energy_attrib(mesh);
    }

    /// Given a mesh, compute the elastic forces per vertex due to tet
    /// potentials and add it at a vertex attribute.
    ///
    /// If the attribute doesn't already exists, it will be created, otherwise
    /// it will be added to.
    fn add_elastic_forces(&self, mesh: &mut Mesh) -> Result<(), Error> {
        self.nh_tet_elements.add_elastic_forces(mesh)?;
        self.snh_tet_elements.add_elastic_forces(mesh)
    }

    /// Produces an iterator over all tetrahedral cells in this solid.
    pub fn tet_iter(&self) -> impl Iterator<Item = &[usize; 4]> {
        self.nh_tet_elements
            .tets
            .iter()
            .chain(self.snh_tet_elements.tets.iter())
    }
}

impl TetSolid {
    #[inline]
    pub fn elasticity<'a, T: Real>(
        &'a self,
    ) -> (TetSolidNeoHookean<'a, T>, TetSolidStableNeoHookean<'a, T>) {
        (
            TetSolidNeoHookean::new(&self.nh_tet_elements),
            TetSolidStableNeoHookean::new(&self.snh_tet_elements),
        )
    }
}

impl<'a> Inertia<'a, TetSolidInertia<'a>> for TetSolid {
    #[inline]
    fn inertia(&'a self) -> TetSolidInertia<'a> {
        TetSolidInertia(self)
    }
}

impl<'a> Gravity<'a, TetSolidGravity<'a>> for TetSolid {
    #[inline]
    fn gravity(&'a self, g: [f64; 3]) -> TetSolidGravity<'a> {
        TetSolidGravity::new(self, g)
    }
}
