use num_traits::ToPrimitive;
use std::cell::RefCell;

use geo::attrib::Attrib;
use geo::mesh::{topology::*, VertexPositions};
use tensr::*;

use super::mcp::*;
use super::newton::*;
use super::problem::{FrictionalContactConstraint, NLProblem, NonLinearProblem};
use super::state::*;
use super::{NLSolver, SimParams, SolveResult, Status};
use crate::attrib_defines::*;
use crate::constraints::compute_contact_penalty;
use crate::constraints::*;
use crate::contact::*;
use crate::inf_norm;
use crate::objects::tetsolid::*;
use crate::objects::trishell::*;
use crate::objects::*;
use crate::{Error, Mesh, PointCloud};
use crate::{Real, Real64};

#[derive(Clone, Debug)]
pub struct SolverBuilder {
    sim_params: SimParams,
    mesh: Mesh,
    materials: Vec<Material>,
    frictional_contacts: Vec<(FrictionalContactParams, (usize, usize))>,
}

///// The index of the object subject to the appropriate contact constraint.
/////
///// This enum helps us map from the particular contact constraint to the
///// originating simulation object (shell or solid).
//#[derive(Copy, Clone, Debug, PartialEq)]
//pub enum SourceIndex {
//    Solid(usize),
//    Shell(usize),
//}
//
//impl SourceIndex {
//    #[inline]
//    fn into_source(self, with_fixed: bool) -> SourceObject {
//        match self {
//            SourceIndex::Solid(i) => SourceObject::Solid(i, with_fixed),
//            SourceIndex::Shell(i) => SourceObject::Shell(i),
//        }
//    }
//
//    #[inline]
//    pub fn get(&self) -> usize {
//        match self {
//            SourceIndex::Solid(idx) | SourceIndex::Shell(idx) => *idx,
//        }
//    }
//}

impl SolverBuilder {
    /// Create a `SolverBuilder` with the minimum required parameters, which are the simulation
    /// parameters, `SimParams`.
    pub fn new(sim_params: SimParams) -> Self {
        SolverBuilder {
            sim_params,
            mesh: Mesh::default(),
            materials: Vec::new(),
            frictional_contacts: Vec::new(),
        }
    }

    /// Set the simulation mesh representing all objects in the scene.
    pub fn set_mesh(&mut self, mesh: Mesh) -> &mut Self {
        self.mesh = mesh;
        self
    }

    /// Set the set materials used by the elements in this solver.
    pub fn set_materials(&mut self, materials: Vec<Material>) -> &mut Self {
        self.materials = materials;
        self
    }

    /// Set parameters for frictional contact problems.
    ///
    /// The given two object IDs determine which objects should experience
    /// frictional contact described by the given parameters. To add
    /// self-contact, simply set the two ids to be equal. For one-directional
    /// models, the first index corresponds to the object (affected) while the
    /// second index corresponds to the collider (unaffected). Some
    /// bi-directional constraints treat the two objects differently, and
    /// changing the order of the indices may change the behaviour. In these
    /// cases, the first index corresponds to the `object` (primary) and the
    /// second to the `collider` (secondary).
    pub fn add_frictional_contact(
        &mut self,
        params: FrictionalContactParams,
        obj_ids: (usize, usize),
    ) -> &mut Self {
        // We can already weed out frictional contacts for pure static sims
        // since we already have the `SimParams`.
        if params.friction_params.is_none() || self.sim_params.time_step.is_some() {
            self.frictional_contacts.push((params, obj_ids));
        }
        self
    }

    /// Helper function to initialize volume constraints from a set of solids.
    fn build_volume_constraints(
        mesh: &Mesh,
        materials: &[Material],
    ) -> Result<Vec<RefCell<VolumeConstraint>>, Error> {
        Ok(VolumeConstraint::try_from_mesh(mesh, materials)?
            .into_iter()
            .map(|vc| RefCell::new(vc))
            .collect())
    }

    fn build_frictional_contact_constraints<T: Real>(
        mesh: &Mesh,
        vertex_type: &[VertexType],
        frictional_contacts: Vec<(FrictionalContactParams, (usize, usize))>,
    ) -> Result<Vec<FrictionalContactConstraint<T>>, crate::Error> {
        use super::problem::ObjectId;
        use crate::TriMesh;
        use geo::algo::*;
        use hashbrown::HashMap;

        let object_ids = mesh
            .attrib_as_slice::<ObjectIdType, CellIndex>(OBJECT_ID_ATTRIB)
            .unwrap();
        let (partition, num_parts) = partition_slice(object_ids);

        // Create a lookup table for object ids.
        let mut id_map = vec![0; num_parts];
        for (&part_index, &obj_id) in partition.iter().zip(object_ids.iter()) {
            // As cast is safe since Object id is clamped to 0 during initializtion.
            id_map[part_index] = obj_id as usize;
        }

        let parts = mesh.clone().split_by_cell_partition(partition, num_parts).0;
        let object_surface_meshes: HashMap<usize, TriMesh> = parts
            .into_iter()
            .enumerate()
            .map(|(part_index, part)| {
                // Split a tetmesh from the mesh and construct a surface mesh out of that.
                // Then merge the result with remaining triangles.
                let mut surface_mesh =
                    TriMesh::merge_iter(part.split_into_typed_meshes().into_iter().map(
                        |typed_mesh| match typed_mesh {
                            TypedMesh::Tet(mesh) => mesh.surface_trimesh(),
                            TypedMesh::Tri(mesh) => mesh,
                        },
                    ));
                // Sort indices by original vertex so we can use this attribute to make subsets.
                if let Ok(orig_vertex_index) = surface_mesh
                    .attrib_clone_into_vec::<OriginalVertexIndexType, VertexIndex>(
                        ORIGINAL_VERTEX_INDEX_ATTRIB,
                    )
                {
                    surface_mesh.sort_vertices_by_key(|i| orig_vertex_index[i]);
                }
                (id_map[part_index], surface_mesh)
            })
            .collect();

        let build_contact_surface = |id| -> Result<_, Error> {
            let mesh = object_surface_meshes
                .get(&id)
                .ok_or(Error::ContactObjectIdError { id })?;
            // Detecting if the whole mesh is fixed allows us to skip generating potentially expensive Jacobians for the entire mesh.
            // TODO: Investigate if this is perhaps more efficient if done on a per vertex level.
            let is_fixed = mesh
                .face_iter()
                .all(|face| face.iter().all(|&i| vertex_type[i] == VertexType::Fixed));
            Ok(if is_fixed {
                ContactSurface::fixed(mesh)
            } else {
                ContactSurface::deformable(mesh)
            })
        };

        let num_vertices = mesh.num_vertices();

        frictional_contacts
            .into_iter()
            .map(|(mut params, (object_id, collider_id))| {
                let object = build_contact_surface(object_id)?;
                let collider = build_contact_surface(collider_id)?;
                params.contact_type = ContactType::Point; // linearized not supported on nl solvers.
                Ok(FrictionalContactConstraint {
                    object_id: ObjectId {
                        obj_id: object_id,
                        include_fixed: params.use_fixed,
                    },
                    collider_id: ObjectId {
                        obj_id: collider_id,
                        include_fixed: params.use_fixed,
                    },
                    constraint: build_penalty_contact_constraint(object, collider, params, num_vertices)?,
                })
            })
            .collect::<Result<Vec<_>, crate::Error>>()
    }

    /// A helper function to initialize the object ID attribute if one doesn't already exist.
    ///
    /// This function also sets all ids that are out of bounds to 0, to avoid out of bounds errors.
    pub(crate) fn init_object_id_attribute(mesh: &mut Mesh) -> Result<(), Error> {
        use std::convert::TryFrom;
        let clamp_id = move |id: ObjectIdType| id.max(0);

        // If there is already an attribute with the right type, normalize the ids.
        if let Ok(attrib) = mesh.attrib_mut::<CellIndex>(OBJECT_ID_ATTRIB) {
            if let Ok(attrib_slice) = attrib.as_mut_slice::<ObjectIdType>() {
                attrib_slice.iter_mut().for_each(|id| *id = clamp_id(*id));
                return Ok(());
            }
        }
        // Remove an attribute with the same name if it exists since it will have the wrong type.
        if let Ok(attrib) = mesh.remove_attrib::<CellIndex>(OBJECT_ID_ATTRIB) {
            if let Ok(iter) = attrib.iter::<i32>() {
                let obj_ids = iter
                    .map(|&id| clamp_id(ObjectIdType::try_from(id).unwrap_or(0)))
                    .collect();
                mesh.insert_attrib_data::<ObjectIdType, VertexIndex>(OBJECT_ID_ATTRIB, obj_ids)?;
                return Ok(());
            } else if let Ok(iter) = attrib.iter::<u8>() {
                let obj_ids = iter
                    .map(|&id| clamp_id(ObjectIdType::try_from(id).unwrap_or(0)))
                    .collect();
                mesh.insert_attrib_data::<ObjectIdType, VertexIndex>(OBJECT_ID_ATTRIB, obj_ids)?;
                return Ok(());
            } else if let Ok(iter) = attrib.iter::<u32>() {
                let obj_ids = iter
                    .map(|&id| clamp_id(ObjectIdType::try_from(id).unwrap_or(0)))
                    .collect();
                mesh.insert_attrib_data::<ObjectIdType, VertexIndex>(OBJECT_ID_ATTRIB, obj_ids)?;
                return Ok(());
            }
        }
        // If no attribute exists, just insert the new attribute with a default value.
        // We know that this will not panic because above we removed any attribute with the same name.
        mesh.attrib_or_insert_with_default::<ObjectIdType, CellIndex>(OBJECT_ID_ATTRIB, 0)
            .unwrap();
        Ok(())
    }

    /// A helper function to initialize the material id attribute if one doesn't already exist.
    ///
    /// This function also sets all ids that are out of bounds to 0, to avoid out of bounds errors,
    /// so the attribute can be used to directly index the slice of materials.
    pub(crate) fn init_material_id_attribute(
        mesh: &mut Mesh,
        num_materials: usize,
    ) -> Result<(), Error> {
        use std::convert::TryFrom;
        let normalize_id = move |id: MaterialIdType| {
            if id >= MaterialIdType::try_from(num_materials).unwrap() {
                0
            } else {
                id.max(0)
            }
        };

        // If there is already an attribute with the right type, normalize the ids.
        if let Ok(attrib) = mesh.attrib_mut::<CellIndex>(MATERIAL_ID_ATTRIB) {
            if let Ok(attrib_slice) = attrib.as_mut_slice::<MaterialIdType>() {
                attrib_slice
                    .iter_mut()
                    .for_each(|id| *id = normalize_id(*id));
                return Ok(());
            }
        }
        // Remove an attribute with the same name if it exists since it will have the wrong type.
        if let Ok(attrib) = mesh.remove_attrib::<CellIndex>(MATERIAL_ID_ATTRIB) {
            if let Ok(iter) = attrib.iter::<i32>() {
                let mtl_ids = iter
                    .map(|&id| normalize_id(MaterialIdType::try_from(id).unwrap_or(0)))
                    .collect();
                mesh.insert_attrib_data::<MaterialIdType, VertexIndex>(
                    MATERIAL_ID_ATTRIB,
                    mtl_ids,
                )?;
                return Ok(());
            } else if let Ok(iter) = attrib.iter::<u8>() {
                let mtl_ids = iter
                    .map(|&id| normalize_id(MaterialIdType::try_from(id).unwrap_or(0)))
                    .collect();
                mesh.insert_attrib_data::<MaterialIdType, VertexIndex>(
                    MATERIAL_ID_ATTRIB,
                    mtl_ids,
                )?;
                return Ok(());
            } else if let Ok(iter) = attrib.iter::<u32>() {
                let mtl_ids = iter
                    .map(|&id| normalize_id(MaterialIdType::try_from(id).unwrap_or(0)))
                    .collect();
                mesh.insert_attrib_data::<MaterialIdType, VertexIndex>(
                    MATERIAL_ID_ATTRIB,
                    mtl_ids,
                )?;
                return Ok(());
            }
        }
        // If no attribute exists, just insert the new attribute with a default value.
        // We know that this will not panic because above we removed any attribute with the same name.
        mesh.attrib_or_insert_with_default::<MaterialIdType, CellIndex>(MATERIAL_ID_ATTRIB, 0)
            .unwrap();
        Ok(())
    }

    /// A helper function to initialize a vertex index attribute to keep track of vertex indices
    /// for derivative meshes (after splits and merges).
    pub(crate) fn init_original_vertex_index_attribute(mesh: &mut Mesh) {
        let _ = mesh
            .remove_attrib::<VertexIndex>(ORIGINAL_VERTEX_INDEX_ATTRIB)
            .ok();
        // Add an original vertex index attribute to track vertices after they are split when
        // building surface meshes from tetmeshes.
        mesh.insert_attrib_data::<OriginalVertexIndexType, VertexIndex>(
            ORIGINAL_VERTEX_INDEX_ATTRIB,
            (0..mesh.num_vertices()).collect(),
        )
        .unwrap();
    }

    /// A helper function to initialize the inverse mass vertex attribute.
    /// Any previously added attributes of the same name are overwritten.
    pub(crate) fn init_mass_inv_attribute<T: Real>(mesh: &mut Mesh, mass_inv: &[T]) {
        let _ = mesh.remove_attrib::<VertexIndex>(MASS_INV_ATTRIB).ok();
        mesh.insert_attrib_data::<MassInvType, VertexIndex>(
            MASS_INV_ATTRIB,
            mass_inv.iter().map(|&x| x.to_f64().unwrap()).collect(),
        )
        .unwrap(); // Should not panic, since we removed the attribute beforehand.
    }
    /// A helper function to initialize the fixed attribute if one doesn't already exist.
    pub(crate) fn init_fixed_attribute(mesh: &mut Mesh) -> Result<(), Error> {
        use num_traits::{One, Zero};
        use std::convert::TryFrom;
        let normalize_id = move |id: FixedIntType| {
            if id != FixedIntType::zero() {
                FixedIntType::one()
            } else {
                FixedIntType::zero()
            }
        };

        // If there is already an attribute with the right type, leave values at either 0 or set to 1 if non-zero.
        if let Ok(attrib) = mesh.attrib_mut::<VertexIndex>(FIXED_ATTRIB) {
            if let Ok(attrib_slice) = attrib.as_mut_slice::<FixedIntType>() {
                attrib_slice
                    .iter_mut()
                    .for_each(|id| *id = normalize_id(*id));
                return Ok(());
            }
        }
        // Remove an attribute with the same name if it exists since it will have the wrong type.
        if let Ok(attrib) = mesh.remove_attrib::<VertexIndex>(FIXED_ATTRIB) {
            if let Ok(iter) = attrib.iter::<i32>() {
                let mtl_ids = iter
                    .map(|&id| normalize_id(FixedIntType::try_from(id).unwrap_or(0)))
                    .collect();
                mesh.insert_attrib_data::<FixedIntType, VertexIndex>(FIXED_ATTRIB, mtl_ids)?;
                return Ok(());
            } else if let Ok(iter) = attrib.iter::<i64>() {
                let mtl_ids = iter
                    .map(|&id| normalize_id(FixedIntType::try_from(id).unwrap_or(0)))
                    .collect();
                mesh.insert_attrib_data::<FixedIntType, VertexIndex>(FIXED_ATTRIB, mtl_ids)?;
                return Ok(());
            } else if let Ok(iter) = attrib.iter::<u8>() {
                let mtl_ids = iter
                    .map(|&id| normalize_id(FixedIntType::try_from(id).unwrap_or(0)))
                    .collect();
                mesh.insert_attrib_data::<FixedIntType, VertexIndex>(FIXED_ATTRIB, mtl_ids)?;
                return Ok(());
            } else if let Ok(iter) = attrib.iter::<u32>() {
                let mtl_ids = iter
                    .map(|&id| normalize_id(FixedIntType::try_from(id).unwrap_or(0)))
                    .collect();
                mesh.insert_attrib_data::<FixedIntType, VertexIndex>(FIXED_ATTRIB, mtl_ids)?;
                return Ok(());
            } else if let Ok(iter) = attrib.iter::<u64>() {
                let mtl_ids = iter
                    .map(|&id| normalize_id(FixedIntType::try_from(id).unwrap_or(0)))
                    .collect();
                mesh.insert_attrib_data::<FixedIntType, VertexIndex>(FIXED_ATTRIB, mtl_ids)?;
                return Ok(());
            }
        }
        // If no attribute exists, just insert the new attribute with a default value.
        // We know that this will not panic because above we removed any attribute with the same name.
        mesh.attrib_or_insert_with_default::<FixedIntType, VertexIndex>(FIXED_ATTRIB, 0)
            .unwrap();
        Ok(())
    }

    /// A helper function to initialize the vertex velocity attribute.
    pub(crate) fn init_velocity_attribute(mesh: &mut Mesh) -> Result<(), Error> {
        // If there is already an attribute with the right type, just leave it alone.
        if let Ok(_) = mesh.attrib_check::<VelType, VertexIndex>(VELOCITY_ATTRIB) {
            return Ok(());
        }
        // Remove an attribute with the same name if it exists since it will have the wrong type.
        if let Ok(attrib) = mesh.remove_attrib::<VertexIndex>(VELOCITY_ATTRIB) {
            // If the attribute is [f32;3] we can just convert it to the right type.
            if let Ok(iter32) = attrib.iter::<[f32; 3]>() {
                let vel64 = iter32
                    .map(|v| [f64::from(v[0]), f64::from(v[1]), f64::from(v[2])])
                    .collect();
                mesh.insert_attrib_data::<VelType, VertexIndex>(VELOCITY_ATTRIB, vel64)?;
                return Ok(());
            }
        }
        // If none of the above applies, just insert the new attribute.
        // We know that this will not panic because above we removed any attribute with the same name.
        mesh.attrib_or_insert_with_default::<VelType, VertexIndex>(VELOCITY_ATTRIB, [0.0; 3])
            .unwrap();
        Ok(())
    }

    /// A helper function to populate the vertex face reference position attribute.
    pub(crate) fn init_cell_vertex_ref_pos_attribute(mesh: &mut Mesh) -> Result<(), Error> {
        // If the attribute already exists, just leave it alone.
        if let Ok(_) =
            mesh.attrib_check::<RefPosType, CellVertexIndex>(REFERENCE_CELL_VERTEX_POS_ATTRIB)
        {
            return Ok(());
        }

        // Otherwise we need to build a reference topology.
        let mut ref_pos = vec![[0.0; 3]; mesh.num_cell_vertices()];
        let pos = if let Ok(vtx_ref_pos) =
            mesh.attrib_as_slice::<RefPosType, VertexIndex>(REFERENCE_VERTEX_POS_ATTRIB)
        {
            // There is a reference attribute on the vertices themselves, just
            // transfer these to face vertices instead of using mesh position.
            vtx_ref_pos.to_vec()
        } else {
            mesh.vertex_position_iter()
                .map(|&x| Vector3::new(x).cast::<f32>().into())
                .collect()
        };

        for (cell_idx, cell) in mesh.cell_iter().enumerate() {
            for i in 0..cell.len() {
                let cell_vtx_idx: usize = mesh.cell_vertex(cell_idx, i).unwrap().into();
                ref_pos[cell_vtx_idx] = pos[cell[i]];
            }
        }

        mesh.attrib_or_insert_data::<RefPosType, CellVertexIndex>(
            REFERENCE_CELL_VERTEX_POS_ATTRIB,
            ref_pos.as_slice(),
        )?;
        Ok(())
    }

    //fn build_rigid_bodies(
    //    rigid_bodies: Vec<(PolyMesh, RigidMaterial)>,
    //) -> Result<Vec<TriMeshShell>, Error> {
    //    // Equip `PolyMesh`es with physics parameters, making them bona-fide shells.
    //    let mut out = Vec::new();
    //    for (polymesh, material) in rigid_bodies.into_iter() {
    //        let trimesh = TriMesh::from(polymesh);
    //        // Prepare shell for simulation.
    //        out.push(TriMeshShell::rigid(trimesh, material).with_simulation_attributes()?)
    //    }

    //    Ok(out)
    //}

    /// Helper function to compute the maximum element mass in the problem.
    fn compute_max_element_mass(_solid: &TetSolid, _shell: &TriShell) -> f64 {
        1.0
    }

    /// Helper function to compute the minimum element mass in the problem.
    fn compute_min_element_mass(_solid: &TetSolid, _shell: &TriShell) -> f64 {
        1.0
    }

    /// Helper to compute max object size (diameter) over all deformable or rigid objects.
    /// This is used for normalizing the problem.
    fn compute_max_size(_solids: &TetSolid, _shell: &TriShell) -> f64 {
        1.0
    }

    /// Helper to compute max element size. This is used for normalizing tolerances.
    fn compute_max_element_size(_solid: &TetSolid, _shell: &TriShell) -> f64 {
        1.0
    }

    /// Helper function to compute the maximum elastic modulus of all given meshes.
    /// This aids in figuring out the correct scaling for the problem.
    fn compute_max_modulus(_solid: &TetSolid, _shell: &TriShell) -> Result<f32, Error> {
        Ok(1.0)
    }

    /// Helper function to compute the maximum elastic bending stiffness.
    fn compute_max_bending_stiffness(_shell: &TriShell) -> f64 {
        1.0
    }

    /// Helper to compute min element size. This is used for normalizing tolerances.
    fn compute_min_element_size(_solid: &TetSolid, _shell: &TriShell) -> f64 {
        1.0
    }

    /// Helper function to compute the minimum elastic modulus of all given meshes.
    /// This aids in figuring out the correct scaling for the problem.
    fn compute_min_modulus(_solid: &TetSolid, _shell: &TriShell) -> Result<f32, Error> {
        Ok(1.0)
    }

    /// Helper function to compute the minimum elastic bending stiffness.
    fn compute_min_bending_stiffness(_shell: &TriShell) -> f64 {
        1.0
    }

    pub(crate) fn build_problem<T: Real>(&self) -> Result<NLProblem<T>, Error> {
        let SolverBuilder {
            sim_params: params,
            mut mesh,
            materials,
            frictional_contacts,
        } = self.clone();

        // Keep the original mesh around for easy inspection and visualization purposes.
        let orig_mesh = mesh.clone();

        // Compute the reference position attribute temporarily.
        // This is used when building the simulation elements and constraints of the mesh.
        init_mesh_source_index_attribute(&mut mesh)?;

        Self::init_cell_vertex_ref_pos_attribute(&mut mesh)?;
        Self::init_velocity_attribute(&mut mesh)?;
        Self::init_material_id_attribute(&mut mesh, materials.len())?;
        Self::init_object_id_attribute(&mut mesh)?;
        Self::init_fixed_attribute(&mut mesh)?;

        let vertex_type = crate::fem::nl::state::sort_mesh_vertices_by_type(&mut mesh, &materials);

        // Keeps track of vertices as they appear in state for meshes used in contact.
        Self::init_original_vertex_index_attribute(&mut mesh);

        // Initialize state (but not constraint multipliers).
        let state = State::<T, autodiff::FT<T>>::try_from_mesh_and_materials(
            &mesh,
            &materials,
            &vertex_type,
        )?;

        // Initialize constraints.

        let volume_constraints = Self::build_volume_constraints(&mesh, &materials)?;

        // Early exit if we detect any self contacts.
        if frictional_contacts.iter().any(|(_, (i, j))| i == j) {
            return Err(Error::UnimplementedFeature {
                description: String::from("Self contacts"),
            });
        }

        // Frictional contacts need to be built after state since state initializes the vertex mass inverses.
        Self::init_mass_inv_attribute::<T>(&mut mesh, &state.vtx.mass_inv);

        let frictional_contact_constraints = Self::build_frictional_contact_constraints::<T>(
            &mesh,
            &vertex_type,
            frictional_contacts,
        )?;
        let frictional_contact_constraints_ad = frictional_contact_constraints
            .iter()
            .map(FrictionalContactConstraint::clone_as_autodiff::<T>)
            .collect::<Vec<_>>();

        let gravity = [
            f64::from(params.gravity[0]),
            f64::from(params.gravity[1]),
            f64::from(params.gravity[2]),
        ];

        let time_step = f64::from(params.time_step.unwrap_or(0.0f32));

        // Compute maxes

        let max_size = Self::compute_max_size(&state.solid, &state.shell);

        let max_element_size = Self::compute_max_element_size(&state.solid, &state.shell);

        // The following scales have units of force (N).
        let max_element_modulus_scale = Self::compute_max_modulus(&state.solid, &state.shell)?
            as f64
            * max_element_size
            * max_element_size;

        let max_element_mass = Self::compute_max_element_mass(&state.solid, &state.shell);

        let max_element_gravity_scale = max_element_mass * gravity.as_tensor().norm();

        let max_element_inertia_scale = if time_step > 0.0 {
            max_element_mass * max_element_size / (time_step * time_step)
        } else {
            0.0
        };

        let max_element_bending_scale =
            Self::compute_max_bending_stiffness(&state.shell) / max_element_size;

        log::trace!("max_element_modulus_scale = {}", max_element_modulus_scale);
        log::trace!("max_element_inertia_scale = {}", max_element_inertia_scale);
        log::trace!("max_element_gravity_scale = {}", max_element_gravity_scale);
        log::trace!("max_element_bending_scale = {}", max_element_bending_scale);

        // Determine the most likely dominant force.
        let mut max_scale = max_element_modulus_scale
            .max(max_element_inertia_scale)
            .max(max_element_gravity_scale)
            .max(max_element_bending_scale);

        // max_scale is a denominator. Ensure that it is never zero.
        if max_scale == 0.0 {
            log::warn!("All scaling factors are zero");
            max_scale = 1.0;
        }

        // Compute mins

        let ignore_zero = |x| if x == 0.0 { f64::INFINITY } else { x };

        let min_element_size = Self::compute_min_element_size(&state.solid, &state.shell);

        let min_element_modulus_scale =
            ignore_zero(Self::compute_min_modulus(&state.solid, &state.shell)? as f64)
                * min_element_size
                * min_element_size;

        let min_element_mass = Self::compute_min_element_mass(&state.solid, &state.shell);

        let min_element_gravity_scale = ignore_zero(min_element_mass * gravity.as_tensor().norm());

        let min_element_inertia_scale = if time_step > 0.0 {
            min_element_mass * min_element_size / (time_step * time_step)
        } else {
            f64::INFINITY
        };

        let min_element_bending_scale =
            Self::compute_min_bending_stiffness(&state.shell) / min_element_size;

        log::trace!("min_element_modulus_scale = {}", min_element_modulus_scale);
        log::trace!("min_element_inertia_scale = {}", min_element_inertia_scale);
        log::trace!("min_element_gravity_scale = {}", min_element_gravity_scale);
        log::trace!("min_element_bending_scale = {}", min_element_bending_scale);

        // Determine the least dominant force.
        let min_scale = min_element_modulus_scale
            .min(min_element_inertia_scale)
            .min(min_element_gravity_scale)
            .min(min_element_bending_scale);

        Ok(NLProblem {
            state: RefCell::new(state),
            kappa: 1.0e8 / params.contact_tolerance as f64,
            delta: params.contact_tolerance as f64,
            frictional_contact_constraints,
            frictional_contact_constraints_ad,
            volume_constraints,
            time_step,
            gravity,
            iterations: 0,
            initial_residual_error: std::f64::INFINITY,
            iter_counter: RefCell::new(0),
            max_size,
            max_element_force_scale: max_scale,
            min_element_force_scale: min_scale,
            original_mesh: orig_mesh,
        })
    }

    /// Build the simulation solver.
    pub fn build<T: Real64>(&self) -> Result<Solver<impl NLSolver<NLProblem<T>, T>, T>, Error> {
        self.build_with_interrupt(|| true)
    }

    /// Build the simulation solver with a given interrupter.
    ///
    /// This version of build allows the caller to conditionally interrupt the solver.
    pub fn build_with_interrupt<T: Real64>(
        &self,
        mut interrupt_checker: impl FnMut() -> bool + Send + 'static,
    ) -> Result<Solver<impl NLSolver<NLProblem<T>, T>, T>, Error> {
        let problem = self.build_problem::<T>()?;

        let num_variables = problem.num_variables();

        // Setup Ipopt parameters using the input simulation params.
        let params = self.sim_params.clone();

        let r_scale = problem.min_element_force_scale;
        let r_tol = params.residual_tolerance.unwrap_or(0.0) * r_scale as f32;

        let x_tol = params.velocity_tolerance.unwrap_or(0.0);
        let a_tol = params.acceleration_tolerance.unwrap_or(0.0);

        log::info!("Simulation Parameters:\n{:#?}", params);
        log::info!("r_tol: {:?}", r_tol);
        log::info!("x_tol: {:?}", x_tol);
        log::trace!("r-scale: {:?}", r_scale);

        // Construct the non-linear equation solver.
        let solver = MCPSolver::newton(
            problem,
            NewtonParams {
                r_tol,
                x_tol,
                a_tol,
                max_iter: params.max_iterations,
                linsolve_tol: params.linsolve_tolerance,
                linsolve_max_iter: params.max_linsolve_iterations,
                line_search: params.line_search,
            },
            Box::new(move |_args| {
                //let mesh = _args.problem.mesh_with(_args.x);
                //let meshes = mesh.split_into_typed_meshes();
                //for (i, mesh) in meshes.into_iter().enumerate() {
                //    match mesh {
                //        geo::algo::split::TypedMesh::Tri(mesh) =>
                //            geo::io::save_polymesh(&geo::mesh::PolyMesh::from(mesh), &format!("./out/polymesh{}_{}.vtk", i, _args.iteration)).unwrap(),
                //        geo::algo::split::TypedMesh::Tet(mesh) =>
                //            geo::io::save_tetmesh(&mesh, &format!("./out/tetmesh{}_{}.vtk", i, _args.iteration)).unwrap(),
                //    }
                //}
                interrupt_checker()
            }),
            Box::new(|_| true),
        );

        Ok(Solver {
            solver,
            sim_params: params,
            max_step: 0.0,
            solution: vec![T::zero(); num_variables],
            iteration_count: 0,
        })
    }

    ///// A helper function to populate the edge reference angles used to compute bending energies.
    //#[unroll_for_loops]
    //pub(crate) fn prepare_edge_ref_angle_attribute(mesh: &mut TriMesh) -> Result<(), Error>
    //{
    //    for (face_idx, face) in mesh.edge_iter().enumerate() {
    //        for i in 0..3 {
    //            let face_vtx_idx: usize = mesh.face_vertex(face_idx, i).unwrap().into();
    //            for j in 0..3 {
    //                ref_pos[face_vtx_idx][j] = pos[face[i]][j];
    //            }
    //        }
    //    }

    //    mesh.attrib_or_add_data::<RefAngleType, EdgeIndex>(
    //        REFERENCE_POSITION_ATTRIB,
    //        ref_pos.as_slice(),
    //    )?;
    //    Ok(())
    //}
}

/// Finite element engine.
pub struct Solver<S, T> {
    /// Non-linear solver.
    solver: S,
    /// Simulation parameters. This is kept around for convenience.
    sim_params: SimParams,
    /// Maximal displacement length.
    ///
    /// Used to limit displacement which is necessary in contact scenarios
    /// because it defines how far a step we can take before the constraint
    /// Jacobian sparsity pattern changes. If zero, then no limit is applied but
    /// constraint Jacobian is kept sparse.
    max_step: f64,
    /// Structured solution to the problem in the solver.
    ///
    /// This is also used as warm start for subsequent steps.
    solution: Vec<T>,
    /// Counts the number of times `step` is called.
    iteration_count: u32,
}

impl<S, T> Solver<S, T>
where
    T: Real64 + na::RealField,
    S: NLSolver<NLProblem<T>, T>,
{
    /// Sets the interrupt checker to be called at every outer iteration.
    pub fn set_coarse_interrupter(&self, mut interrupted: impl FnMut() -> bool + Send + 'static) {
        *self.solver.outer_callback().borrow_mut() = Box::new(move |_| !interrupted());
    }

    /// Sets the interrupt checker to be called at every inner iteration.
    pub fn set_fine_interrupter(&self, mut interrupted: impl FnMut() -> bool + Send + 'static) {
        *self.solver.inner_callback().borrow_mut() = Box::new(move |_| !interrupted());
    }

    /// If the time step was not specified or specified to be zero, then this function will return
    /// zero.
    pub fn time_step(&self) -> f64 {
        self.sim_params.time_step.unwrap_or(0.0).into()
    }
    /// Get an immutable reference to the underlying problem.
    fn problem(&self) -> &NLProblem<T> {
        self.solver.problem()
    }

    /// Get a mutable reference to the underlying problem.
    fn problem_mut(&mut self) -> &mut NLProblem<T> {
        self.solver.problem_mut()
    }

    /// Get a slice of solid objects represented in this solver.
    pub fn solid(&self) -> std::cell::Ref<TetSolid> {
        std::cell::Ref::map(self.problem().state.borrow(), |state| &state.solid)
    }

    /// Get a slice of shell objects represented in this solver.
    pub fn shell(&self) -> std::cell::Ref<TriShell> {
        std::cell::Ref::map(self.problem().state.borrow(), |state| &state.shell)
    }

    pub fn mesh(&self) -> Mesh {
        self.problem().mesh()
    }

    /// Get simulation parameters.
    pub fn params(&self) -> SimParams {
        self.sim_params.clone()
    }

    /// Update the maximal displacement allowed. If zero, no limit is applied.
    pub fn update_max_step(&mut self, step: f64) {
        self.max_step = step;
        self.problem_mut().update_max_step(step);
    }

    pub fn update_radius_multiplier(&mut self, rad: f64) {
        self.problem_mut().update_radius_multiplier(rad);
    }

    /// Update the solid meshes with the given points.
    pub fn update_vertices(&mut self, pts: &PointCloud) -> Result<(), Error> {
        self.problem_mut().update_vertices(pts)
    }

    /// Returns the solved positions of the vertices in original order.
    pub fn vertex_positions(&self) -> Vec<[T; 3]> {
        self.problem().vertex_positions()
    }

    /// Update the `mesh` and `prev_pos` with the current solution.
    fn commit_solution(&mut self, relax_max_step: bool) {
        {
            let Self {
                solver, solution, ..
            } = self;

            // Advance internal state (positions and velocities) of the problem.
            solver.problem_mut().advance(&solution);
        }

        // Reduce max_step for next iteration if the solution was a good one.
        if relax_max_step {
            let dt = self.time_step();
            if let Some(radius) = self.problem().min_contact_radius() {
                let step =
                    inf_norm(self.solution.iter().cloned()) * if dt > 0.0 { dt } else { 1.0 };
                let new_max_step = (step.to_f64().unwrap() - radius).max(self.max_step * 0.5);
                if self.max_step != new_max_step {
                    log::info!(
                        "Relaxing max step from {} to {}",
                        self.max_step,
                        new_max_step
                    );
                    self.max_step = new_max_step;
                    self.problem_mut().update_max_step(new_max_step);
                }
            }
        }
    }

    ///// Revert previously committed solution. We just advance in the opposite direction.
    //fn revert_solution(&mut self) {
    //    self.problem_mut().retreat();
    //}

    //fn initial_residual_error(&self) -> f64 {
    //    self.problem().initial_residual_error
    //}

    //fn save_current_active_constraint_set(&mut self) {
    //    let Solver {
    //        ref solver,
    //        ref mut old_active_constraint_set,
    //        ..
    //    } = self;
    //    old_active_constraint_set.clear();
    //    solver
    //        .solver_data()
    //        .problem
    //        .compute_active_constraint_set(old_active_constraint_set);
    //}

    //fn remap_warm_start(&mut self) {
    //    let Solver {
    //        solver,
    //        old_active_constraint_set,
    //        ..
    //    } = self;

    //    solver
    //        .solver_data_mut()
    //        .problem
    //        .remap_warm_start(old_active_constraint_set.view());
    //}

    //fn all_contacts_linear(&self) -> bool {
    //    self.problem().all_contacts_linear()
    //}

    /// Run the non-linear solver on one time step.
    pub fn step(&mut self) -> Result<SolveResult, Error> {
        let dt = self.time_step();
        self.iteration_count += 1;
        let Self {
            sim_params,
            solver,
            solution,
            ..
        } = self;

        log::trace!("Updating constraint set...");
        solver.problem_mut().update_constraint_set();
        solver.update_jacobian_indices();

        if sim_params.jacobian_test {
            solver.problem().check_jacobian(true)?;
        }

        // Update the current vertex data using the current dof state.
        solver.problem_mut().update_cur_vertices();

        let mut contact_iterations = 5i32;

        let velocity_clear_steps = if sim_params.velocity_clear_frequency > 0.0 {
            (1.0 / (f64::from(sim_params.velocity_clear_frequency) * dt))
                .round()
                .to_u32()
                .unwrap()
        } else {
            u32::MAX
        }
        .max(1);

        // Loop to resolve all contacts.
        loop {
            /***     Main solve step     ***/
            log::trace!("Begin main nonlinear solve.");
            let result = solver.solve_with(solution.as_mut_slice());
            /*******************************/

            log::trace!("Solve Result: {}", &result);
            match result.status {
                Status::Success | Status::MaximumIterationsExceeded => {
                    // Compute contact violation.
                    let constraint = solver
                        .problem_mut()
                        .contact_constraint(solution.as_slice())
                        .into_storage();
                    //let mut orig_lambda = constraint.clone();
                    //let mut shifted_lambda = constraint.clone();
                    let smallest = constraint
                        .iter()
                        .min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Less))
                        .copied()
                        .unwrap_or(T::zero())
                        .to_f64()
                        .unwrap();
                    //shifted_lambda.iter_mut().for_each(|x| *x -= smallest);
                    //compute_contact_penalty(
                    //    shifted_lambda.as_mut_slice(),
                    //    sim_params.contact_tolerance,
                    //);
                    //compute_contact_penalty(
                    //    orig_lambda.as_mut_slice(),
                    //    sim_params.contact_tolerance,
                    //);

                    let delta = sim_params.contact_tolerance as f64;
                    let denom: f64 = (compute_contact_penalty(0.0, delta as f32)
                        - sim_params.residual_tolerance.unwrap_or(0.0) as f64
                            / solver.problem().kappa)
                        .min(0.5 * delta)
                        .max(f64::EPSILON);

                    let bump_ratio: f64 =
                        compute_contact_penalty(smallest, sim_params.contact_tolerance) / denom;
                    log::trace!("Bump ratio: {}", bump_ratio);
                    log::trace!("Kappa: {}", solver.problem().kappa);
                    let violation = solver
                        .problem_mut()
                        .contact_violation(constraint.storage().as_slice());
                    log::trace!("Contact violation: {}", violation);

                    contact_iterations -= 1;

                    if contact_iterations < 0 {
                        break Err(Error::NLSolveError {
                            result: SolveResult {
                                status: Status::MaximumContactIterationsExceeded,
                                ..result
                            },
                        });
                    }

                    if violation > T::zero() {
                        solver.problem_mut().kappa *= bump_ratio.max(2.0);
                        continue;
                    } else {
                        // Relax kappa
                        if solver.problem().kappa > 1.0 / sim_params.contact_tolerance as f64 {
                            solver.problem_mut().kappa /= 2.0;
                        }
                        self.commit_solution(false);
                        // Kill velocities if needed.
                        if self.iteration_count % velocity_clear_steps == 0 {
                            self.solver.problem_mut().clear_velocities();
                        }
                        break Ok(result);
                    }
                }
                _ => {
                    break Err(Error::NLSolveError { result });
                }
            }
        }
    }
}
