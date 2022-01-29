use geo::attrib::{self, Attrib, AttribIndex, VertexAttrib};
use geo::mesh::{topology::*, VertexPositions};
pub use material::*;
//pub use shell::*;
//pub use solid::*;

use crate::attrib_defines::*;
use crate::Error;

pub mod material;
#[cfg(feature = "optsolver")]
pub mod shell;
#[cfg(feature = "optsolver")]
pub mod solid;
pub mod tetmesh_surface;
pub mod tetsolid;
pub mod trishell;

pub(crate) mod interior_edge;
pub use interior_edge::*;

/// A utility for initializing the source index attribute to use for updating mesh vertices.
///
/// This can be used explicitly by the user on the mesh before building the solver. For
/// instance if the user splits the mesh before adding it to the solver, it is necessary to
/// initialize the source index attribute so that the meshes can be updated via a global vertex
/// position array (point cloud). The attribute is only set if it doesn't already exist.
///
/// This is exposed as a stand-alone function since it should be called on meshes that are
/// sources for the simulation objects.
pub fn init_mesh_source_index_attribute<M>(mesh: &mut M) -> Result<(), Error>
where
    M: NumVertices + Attrib + VertexAttrib,
{
    // Need source index for meshes so that their vertices can be updated.
    // If this index is missing, it is assumed that the source is coincident
    // with the provided mesh.
    let num_verts = mesh.num_vertices();
    match mesh.insert_attrib_data::<SourceIndexType, VertexIndex>(
        SOURCE_INDEX_ATTRIB,
        (0..num_verts).collect::<Vec<_>>(),
    ) {
        Err(attrib::Error::AlreadyExists(_)) => Ok(()),
        Err(e) => Err(e.into()),
        _ => Ok(()),
    }
}

// TODO: Revise the Object trait hierarchy
//       It is confusing to have multiple implementations of different object traits for a single
//       object type because one type should have one type of material. In the case of
//       TriMeshShell, currently it can represent multiple different types of objects.
//       Probably the way forward is to separate different types of objects into different
//       concrete types (e.g. RigidShell, SoftShell, FixedShell) and store them separately in
//       ObjectData in the problem, or store them as trait objects with a more carefully designed
//       Object trait.
//       When/If the "Chunked" types are capable of storing multiple Vecs, ObjectData should be
//       further rafctored and may further includes some common attributes.
//       ObjectData is intended as a data iterface to the simulation objects for the non-linear
//       problem, which draws onto common types of data like position, velocity in contiguous
//       arrays and leaves object specific data like reference positions and matrices to be
//       drawn directly from the object types (TriMeshShell and TetMeshSolid).

/// An object is any entity that can be a part of the simulation in some way.
///
/// This trait helps access and initialize data needed for simulation.
pub trait Object {
    type Mesh: NumVertices + Attrib + VertexAttrib;
    type ElementIndex: AttribIndex<Self::Mesh>;

    fn num_elements(&self) -> usize;
    fn mesh(&self) -> &Self::Mesh;
    fn mesh_mut(&mut self) -> &mut Self::Mesh;
    fn material_id(&self) -> usize;

    fn init_kinematic_vertex_attributes(&mut self) -> Result<(), Error> {
        self.mesh_mut()
            .attrib_or_insert_with_default::<VelType, VertexIndex>(VELOCITY_ATTRIB, [0.0; 3])?;
        Ok(())
    }

    /// Initialize the source index attribute to use for updating mesh vertices.
    ///
    /// This is a wrapper for `init_mesh_source_index_attribute` for use when it hasn't
    /// already been called on the actual source or the source has the same topology and
    /// vertex order as the final simulation mesh.
    fn init_source_index_attribute(&mut self) -> Result<(), Error> {
        init_mesh_source_index_attribute(self.mesh_mut())
    }
}

pub trait DynamicObject: Object {
    fn density(&self) -> Option<f32>;

    /// A helper function to populate vertex attributes for simulation on a dynamic mesh.
    fn init_dynamic_vertex_attributes(&mut self) -> Result<(), Error> {
        self.init_kinematic_vertex_attributes()?;

        // If this attribute doesn't exist, assume no vertices are fixed. This function will
        // return an error if there is an existing Fixed attribute with the wrong type.
        {
            let mesh = self.mesh_mut();
            use geo::attrib::*;
            let fixed_buf = mesh
                .remove_attrib::<VertexIndex>(FIXED_ATTRIB)
                .unwrap_or_else(|_| {
                    Attribute::direct_from_vec(vec![0 as FixedIntType; mesh.num_vertices()])
                })
                .into_data();
            let fixed = fixed_buf
                .cast_into_vec::<FixedIntType>()
                .unwrap_or_else(|| {
                    // If non-numeric type detected, just fill it with zeros.
                    vec![0; mesh.num_vertices()]
                });
            mesh.insert_attrib::<VertexIndex>(FIXED_ATTRIB, Attribute::direct_from_vec(fixed))?;
        }

        Ok(())
    }
    fn init_density_attribute(&mut self) -> Result<(), Error> {
        // Prepare density parameter
        if let Some(density) = self.density() {
            let num_elements = self.num_elements();
            match self
                .mesh_mut()
                .insert_attrib_data::<DensityType, Self::ElementIndex>(
                    DENSITY_ATTRIB,
                    vec![density as f32; num_elements],
                ) {
                // if ok or already exists, everything is ok.
                Err(attrib::Error::AlreadyExists(_)) => {}
                Err(e) => return Err(e.into()),
                _ => {}
            }
        } else {
            // Ensure that there is a density parameter defined on the mesh.
            self.mesh()
                .attrib_check::<DensityType, Self::ElementIndex>(DENSITY_ATTRIB)?;
        }
        Ok(())
    }
}

pub trait DeformableObject: DynamicObject {
    /// A helper function to populate vertex attributes for simulation on a deformable mesh.
    fn init_deformable_vertex_attributes(&mut self) -> Result<(), Error>
    where
        Self::Mesh: VertexPositions<Element = [f64; 3]>,
    {
        // Deformable meshes are dynamic. Prepare dynamic attributes first.
        self.init_dynamic_vertex_attributes()?;

        {
            // Add elastic force attributes.
            // These will be computed at the end of the time step.
            self.mesh_mut()
                .reset_attrib_to_default::<ElasticForceType, VertexIndex>(
                    ELASTIC_FORCE_ATTRIB,
                    [0f64; 3],
                )?;
        }

        Ok(())
    }
}

pub trait ElasticObject: DeformableObject {
    fn elasticity_parameters(&self) -> Option<ElasticityParameters>;

    /// Transfer parameters `lambda` and `mu` from the object material to the
    /// mesh if it hasn't already been populated on the input.
    fn init_elasticity_attributes(&mut self) -> Result<(), Error> {
        if let Some(elasticity) = self.elasticity_parameters() {
            let num_elements = self.num_elements();
            match self
                .mesh_mut()
                .insert_attrib_data::<LambdaType, Self::ElementIndex>(
                    LAMBDA_ATTRIB,
                    vec![elasticity.lambda(); num_elements],
                ) {
                // if ok or already exists, everything is ok.
                Err(attrib::Error::AlreadyExists(_)) => {}
                Err(e) => return Err(e.into()),
                _ => {}
            }
            match self
                .mesh_mut()
                .insert_attrib_data::<MuType, Self::ElementIndex>(
                    MU_ATTRIB,
                    vec![elasticity.mu(); num_elements],
                ) {
                // if ok or already exists, everything is ok.
                Err(attrib::Error::AlreadyExists(_)) => {}
                Err(e) => return Err(e.into()),
                _ => {}
            }
        } else {
            // No global elasticity parameters were given. Check that the mesh has the right
            // parameters.
            if self
                .mesh()
                .attrib_check::<LambdaType, Self::ElementIndex>(LAMBDA_ATTRIB)
                .is_err()
                || self
                    .mesh()
                    .attrib_check::<MuType, Self::ElementIndex>(MU_ATTRIB)
                    .is_err()
            {
                return Err(Error::MissingElasticityParams);
            }
        }
        Ok(())
    }
}
