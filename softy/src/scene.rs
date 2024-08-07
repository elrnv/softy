//! This module describes a convenient scene struct describing a simulatable configuration.

// TODO: Save only fixed vertices.
// TODO: Enable more sophisticated interpolation.

use std::cmp::Ordering;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};

use geo::attrib::Attrib;
use geo::topology::{CellIndex, VertexIndex};
use serde::{Deserialize, Serialize};
use serde_json as json;
use thiserror::Error;

use flatk::IntoStorage;

use crate::constraints::penalty_point_contact::FrictionalContactParams;
use crate::nl_fem::{SimParams, SolverBuilder, StepResult, ZoneParams};
use crate::{Material, Mesh};

#[derive(Debug, Error)]
pub enum SerializeError {
    #[error("RON deserialize: {}", .0)]
    RonDeserialize(#[from] ron::error::SpannedError),
    #[error("RON serialize: {}", .0)]
    RonSerialize(#[from] ron::Error),
    #[error("Bincode: {}", .0)]
    Bincode(#[from] bincode::Error),
    #[error("JSON serialize: {}", .0)]
    JsonSerialize(#[from] json::Error),
    #[error("IO: {}", .0)]
    Io(#[from] std::io::Error),
    #[error("Missing a newline between RON file and binary scene data")]
    MissingNewlineBeforeBinaryData,
}

#[derive(Debug, Error)]
pub enum SceneError {
    #[error("IO Error: {}", .0)]
    IO(#[from] std::io::Error),
    #[error("(De)Serialization error: {}", .0)]
    Serialize(#[from] SerializeError),
    #[error("Attribute transfer error: {}", .0)]
    Attribute(#[from] geo::attrib::Error),
    #[error("Solver error: {}", .0)]
    Solver(#[from] Box<crate::Error>),
    #[error("This library was compiled without JSON support.")]
    JSONUnsupported,
    #[error("Interrupted by callback")]
    Interrupted,
}

impl From<crate::Error> for SceneError {
    fn from(err: crate::Error) -> SceneError {
        SceneError::Solver(Box::new(err))
    }
}

/// An enum defining all attribute types supported by the solver.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum Attribute {
    F64(Vec<f64>),
    F32(Vec<f32>),
    I32(Vec<i32>),
    I8(Vec<i8>),
    Usize(Vec<usize>),
    F32x3(Vec<[f32; 3]>),
    F64x3(Vec<[f64; 3]>),
}

impl Attribute {
    fn from_attrib_data(attrib: &geo::attrib::AttributeData) -> Option<Self> {
        Some(if let Ok(v) = attrib.direct_clone_into_vec::<f64>() {
            Attribute::F64(v)
        } else if let Ok(v) = attrib.direct_clone_into_vec::<f32>() {
            Attribute::F32(v)
        } else if let Ok(v) = attrib.direct_clone_into_vec::<i32>() {
            Attribute::I32(v)
        } else if let Ok(v) = attrib.direct_clone_into_vec::<i8>() {
            Attribute::I8(v)
        } else if let Ok(v) = attrib.direct_clone_into_vec::<usize>() {
            Attribute::Usize(v)
        } else if let Ok(v) = attrib.direct_clone_into_vec::<[f32; 3]>() {
            Attribute::F32x3(v)
        } else if let Ok(v) = attrib.direct_clone_into_vec::<[f64; 3]>() {
            Attribute::F64x3(v)
        } else {
            return None;
        })
    }
    fn into_geo_attrib<I>(self) -> geo::attrib::Attribute<I> {
        match self {
            Attribute::F64(data) => geo::attrib::Attribute::direct_from_vec(data),
            Attribute::F32(data) => geo::attrib::Attribute::direct_from_vec(data),
            Attribute::I32(data) => geo::attrib::Attribute::direct_from_vec(data),
            Attribute::I8(data) => geo::attrib::Attribute::direct_from_vec(data),
            Attribute::Usize(data) => geo::attrib::Attribute::direct_from_vec(data),
            Attribute::F32x3(data) => geo::attrib::Attribute::direct_from_vec(data),
            Attribute::F64x3(data) => geo::attrib::Attribute::direct_from_vec(data),
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum CellType {
    Tetrahedron,
    Triangle,
}

impl From<geo::mesh::CellType> for CellType {
    fn from(t: geo::mesh::CellType) -> Self {
        match t {
            geo::mesh::CellType::Tetrahedron => CellType::Tetrahedron,
            geo::mesh::CellType::Triangle => CellType::Triangle,
        }
    }
}

impl From<CellType> for geo::mesh::CellType {
    fn from(t: CellType) -> Self {
        match t {
            CellType::Tetrahedron => geo::mesh::CellType::Tetrahedron,
            CellType::Triangle => geo::mesh::CellType::Triangle,
        }
    }
}

/// A simplified mesh structure suitable for serialization and deserialization.
#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct MeshTopo {
    pub indices: Vec<Vec<usize>>,
    pub types: Vec<CellType>,
    pub vertex_attributes: HashMap<String, Attribute>,
    pub cell_attributes: HashMap<String, Attribute>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct KeyframedVertexPositions {
    pub frame: u64,
    pub positions: Vec<[f64; 3]>,
}

/// Scene description.
#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct SceneData {
    // Mesh topology.
    mesh_topo: MeshTopo,
    /// A set of time stamped vertex positions.
    animation: Vec<KeyframedVertexPositions>,
}

impl SceneData {
    /// Set the simulation mesh representing all objects in the scene.
    ///
    /// This function sets the initial keyframe to be the mesh vertex positions.
    pub fn new(mesh: Mesh) -> Self {
        let vertex_attributes = mesh
            .vertex_attributes
            .iter()
            .filter_map(|(name, attrib)| {
                Attribute::from_attrib_data(&attrib.data).map(|attrib| (name.clone(), attrib))
            })
            .collect();
        let cell_attributes = mesh
            .cell_attributes
            .iter()
            .filter_map(|(name, attrib)| {
                Attribute::from_attrib_data(&attrib.data).map(|attrib| (name.clone(), attrib))
            })
            .collect();
        let index_blocks: Vec<_> = mesh
            .indices
            .clump_iter()
            .map(|chunked| chunked.into_storage().to_vec())
            .collect();
        let mesh_topo = MeshTopo {
            indices: index_blocks,
            types: mesh.types.iter().map(|&t| CellType::from(t)).collect(),
            vertex_attributes,
            cell_attributes,
        };
        let animation = vec![KeyframedVertexPositions {
            frame: 0,
            positions: mesh.vertex_positions.into_vec(),
        }];
        SceneData {
            mesh_topo,
            animation,
        }
    }

    /// Set the keyframed animation of the mesh representing all objects in the scene.
    ///
    /// This function will overwrite all other keyframes if the first element in
    /// `frames` is zero. If the first element of `frames` is not zero, then the first keyframe
    /// will remain unchanged.
    ///
    /// # Panics
    ///
    /// This function will panic if
    ///   - `frames` and `positions` have different sizes,
    ///   - they are empty, or
    ///   - `frames` is not monotonically increasing.
    pub fn set_keyframes(
        &mut self,
        frames: impl AsRef<[u64]>,
        positions: impl Into<Vec<Vec<[f64; 3]>>>,
    ) -> &mut Self {
        let frames = frames.as_ref();
        let positions = positions.into();

        // Ensure preconditions.
        assert_eq!(frames.len(), positions.len());
        assert!(!frames.is_empty());
        assert!(frames.windows(2).all(|ts| ts[0] <= ts[1]));

        let first_frame = *frames.first().unwrap();
        if first_frame == 0 {
            // Overwrite the initial configuration.
            self.animation.clear();
        } else {
            // Keep initial configuration.
            self.animation.truncate(1);
        }

        self.animation.extend(
            frames
                .iter()
                .zip(positions.into_iter())
                .map(|(&frame, positions)| KeyframedVertexPositions { frame, positions }),
        );
        self
    }

    /// Add a single keyframe to the animation.
    pub fn add_keyframe(&mut self, frame: u64, positions: Vec<[f64; 3]>) -> &mut Self {
        // No panic since animation should always be non-empty to contain at least one set of vertex
        // positions.
        let last = self.animation.last_mut().unwrap();
        match frame.cmp(&last.frame) {
            Ordering::Greater => {
                // Insert at the end. Presumably this is the most common scenario.
                self.animation
                    .push(KeyframedVertexPositions { frame, positions });
            }
            Ordering::Equal => {
                // Times coincide, overwrite the last one
                last.positions = positions;
            }
            Ordering::Less => {
                // Insert in the middle, or overwrite previous keyframe.
                match self.animation.binary_search_by_key(&frame, |tp| tp.frame) {
                    Ok(pos) => self.animation[pos].positions = positions,
                    Err(pos) => self
                        .animation
                        .insert(pos, KeyframedVertexPositions { frame, positions }),
                }
            }
        }
        self
    }

    /// Constructs a mesh from the stored topology and vertex position data for the first frame.
    fn build_mesh(&self) -> Result<Mesh, SceneError> {
        let types: Vec<_> = self
            .mesh_topo
            .types
            .iter()
            .map(|&x| geo::mesh::CellType::from(x))
            .collect();
        let mut mesh = Mesh::from_cells_and_types(
            self.animation.first().unwrap().positions.clone(),
            self.mesh_topo.indices.clone(),
            types,
        );

        for (name, attrib) in self.mesh_topo.vertex_attributes.iter() {
            mesh.insert_attrib::<VertexIndex>(
                name,
                attrib.clone().into_geo_attrib::<VertexIndex>(),
            )?;
        }
        for (name, attrib) in self.mesh_topo.cell_attributes.iter() {
            mesh.insert_attrib::<CellIndex>(name, attrib.clone().into_geo_attrib::<CellIndex>())?;
        }
        Ok(mesh)
    }
}

fn default_use_fixed() -> bool {
    true
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FrictionalContactConfig {
    /// Parameters controlling the contact and friction models.
    #[serde(default)]
    pub params: FrictionalContactParams,
    /// Ids of colliding objects.
    ///
    /// the
    pub object_ids: (usize, usize),
    /// Also uses fixed vertices for collision.
    #[serde(default = "default_use_fixed")]
    pub use_fixed: bool,
}

/// Configuration of the scene excluding mesh data.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SceneConfig {
    pub sim_params: SimParams,
    pub materials: Vec<Material>,
    pub frictional_contacts: Vec<FrictionalContactConfig>,
    pub volume_zones: ZoneParams,
}

impl SceneConfig {
    /// Construct a new scene configuration.
    pub fn new(sim_params: SimParams) -> Self {
        SceneConfig {
            sim_params,
            materials: Vec::new(),
            frictional_contacts: Vec::new(),
            volume_zones: ZoneParams::default(),
        }
    }

    /// Set the set materials used by the elements in this solver.
    pub fn set_materials(&mut self, materials: impl Into<Vec<Material>>) -> &mut Self {
        self.materials = materials.into();
        self
    }

    /// Set the set materials used by the elements in this solver.
    pub fn set_volume_zones_from_params(
        &mut self,
        zone_pressurizations: impl Into<Vec<f32>>,
        compression_coefficients: impl Into<Vec<f32>>,
        hessian_approximation: impl Into<Vec<bool>>,
    ) -> &mut Self {
        self.volume_zones.zone_pressurizations = zone_pressurizations.into();
        self.volume_zones.compression_coefficients = compression_coefficients.into();
        self.volume_zones.hessian_approximation = hessian_approximation.into();
        self
    }

    /// Set the set materials used by the elements in this solver.
    pub fn set_volume_zones(&mut self, volume_zones: impl Into<ZoneParams>) -> &mut Self {
        self.volume_zones = volume_zones.into();
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
        object_ids: (usize, usize),
        use_fixed: bool,
    ) -> &mut Self {
        // We can already weed out frictional contacts for pure static sims
        // since we already have the `SimParams`.
        if params.friction_params.is_none() || self.sim_params.time_step.is_some() {
            self.frictional_contacts.push(FrictionalContactConfig {
                params,
                object_ids,
                use_fixed,
            });
        }
        self
    }

    /// Writes this scene configuration to the given writer interpreted as a RON file.
    pub fn write_as_ron<W: std::io::Write>(&self, w: W) -> Result<(), SceneError> {
        Ok(
            ron::ser::to_writer_pretty(w, self, ron::ser::PrettyConfig::new())
                .map_err(SerializeError::from)?,
        )
    }

    /// Reads `SceneConfig` as a `RON` file.
    pub fn read_as_ron<R: Read>(mut reader: R) -> Result<(Self, Vec<u8>), SerializeError> {
        let mut bytes = Vec::new();
        reader.read_to_end(&mut bytes)?;
        let mut deserializer = ron::de::Deserializer::from_bytes(&bytes)?;
        let config = Self::deserialize(&mut deserializer)?;
        Ok((config, deserializer.remainder_bytes().to_vec()))
        // Don't check for trailing characters (which ron::de::from_reader does) since
        // we concatenate two file types together in `sfrb`.
    }
}

/// Complete scene description.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Scene {
    pub config: SceneConfig,
    pub scene: SceneData,
}

impl Scene {
    /// Construct a new scene.
    pub fn new(sim_params: SimParams, mesh: Mesh) -> Self {
        Scene {
            config: SceneConfig::new(sim_params),
            scene: SceneData::new(mesh),
        }
    }

    /// Set the animation of the mesh representing all objects in the scene.
    ///
    /// This function will overwrite all other keyframes if the first element in
    /// `frames` is zero. If the first element of `frames` is not zero, then the first keyframe
    /// will remain unchanged.
    ///
    /// # Panics
    ///
    /// This function will panic if
    ///   - `frames` and `positions` have different sizes,
    ///   - they are empty, or
    ///   - `frames` is not monotonically increasing.
    pub fn set_animation(
        &mut self,
        frames: impl AsRef<[u64]>,
        positions: impl Into<Vec<Vec<[f64; 3]>>>,
    ) -> &mut Self {
        self.scene.set_keyframes(frames, positions);
        self
    }

    /// Add a single keyframe to the animation.
    pub fn add_keyframe(&mut self, frame: u64, positions: impl Into<Vec<[f64; 3]>>) -> &mut Self {
        self.scene.add_keyframe(frame, positions.into());
        self
    }

    /// Set the set materials used by the elements in this solver.
    pub fn set_materials(&mut self, materials: impl Into<Vec<Material>>) -> &mut Self {
        self.config.set_materials(materials.into());
        self
    }

    /// Set the set materials used by the elements in this solver.
    pub fn set_volume_zones_from_params(
        &mut self,
        zone_pressurizations: impl Into<Vec<f32>>,
        compression_coefficients: impl Into<Vec<f32>>,
        hessian_approximation: impl Into<Vec<bool>>,
    ) -> &mut Self {
        self.config.set_volume_zones_from_params(
            zone_pressurizations.into(),
            compression_coefficients.into(),
            hessian_approximation.into(),
        );
        self
    }

    /// Set the set materials used by the elements in this solver.
    pub fn set_volume_zones(&mut self, volume_zones: impl Into<ZoneParams>) -> &mut Self {
        self.config.set_volume_zones(volume_zones.into());
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
        object_ids: (usize, usize),
        use_fixed: bool,
    ) -> &mut Self {
        self.config
            .add_frictional_contact(params, object_ids, use_fixed);
        self
    }

    /// Saves this scene as in `sfrb` format.
    ///
    /// The `sfrb` format is `ron` for `SceneConfig` followed by `bincode` for `SceneData`.
    pub fn save_as_sfrb(&self, path: impl AsRef<std::path::Path>) -> Result<(), SceneError> {
        let path = path.as_ref();
        File::create(path).map_err(SceneError::from).and_then(|f| {
            let mut writer = BufWriter::new(f);
            self.config.write_as_ron(&mut writer)?;

            writeln!(&mut writer).map_err(SerializeError::from)?;

            // Bincode for the data
            Ok(bincode::serialize_into(&mut writer, &self.scene).map_err(SerializeError::from)?)
        })
    }

    /// Loads the scene from an `sfrb` file.
    ///
    /// The `sfrb` format is `ron` for `SceneConfig` followed by `bincode` for `SceneData`.
    pub fn load_from_sfrb(path: impl AsRef<std::path::Path>) -> Result<Self, SceneError> {
        let path = path.as_ref();
        File::open(path).map_err(SceneError::from).and_then(|f| {
            let reader = BufReader::new(f);
            let (config, bytes) = SceneConfig::read_as_ron(reader)?;
            if bytes[0] != b'\n' {
                return Err(SceneError::Serialize(
                    SerializeError::MissingNewlineBeforeBinaryData,
                ));
            }
            let scene = bincode::deserialize(&bytes[1..]).map_err(SerializeError::from)?;
            Ok(Scene { config, scene })
        })
    }

    /// Saves this scene entirely as `ron`.
    ///
    /// This can be very space inefficient since `SceneData` can be very large.
    pub fn save_as_ron(&self, path: impl AsRef<std::path::Path>) -> Result<(), SceneError> {
        let path = path.as_ref();
        File::create(path).map_err(SceneError::from).and_then(|f| {
            let mut writer = BufWriter::new(f);
            Ok(
                ron::ser::to_writer_pretty(&mut writer, self, ron::ser::PrettyConfig::new())
                    .map_err(SerializeError::from)?,
            )
        })
    }

    /// Saves this scene entirely as `json`.
    ///
    /// This can be very space inefficient since `SceneData` can be very large.
    #[cfg(feature = "json")]
    pub fn save_as_json(&self, path: impl AsRef<std::path::Path>) -> Result<(), SceneError> {
        let path = path.as_ref();
        File::create(path).map_err(SceneError::from).and_then(|f| {
            let mut writer = BufWriter::new(f);
            Ok(json::to_writer_pretty(&mut writer, self).map_err(SerializeError::from)?)
        })
    }

    /// Loads the scene from a `RON` file.
    pub fn load_from_ron(path: impl AsRef<std::path::Path>) -> Result<Self, SceneError> {
        let path = path.as_ref();
        File::open(path).map_err(SceneError::from).and_then(|f| {
            let reader = BufReader::new(f);
            Ok(ron::de::from_reader(reader).map_err(SerializeError::from)?)
        })
    }

    /// Loads the scene from a `JSON` file.
    #[cfg(feature = "json")]
    pub fn load_from_json(path: impl AsRef<std::path::Path>) -> Result<Self, SceneError> {
        let path = path.as_ref();
        File::open(path).map_err(SceneError::from).and_then(|f| {
            let reader = BufReader::new(f);
            Ok(json::from_reader(reader).map_err(SerializeError::from)?)
        })
    }

    /// Runs a simulation on this scene.
    pub fn run(&self, steps: u64) -> Result<(), SceneError> {
        self.run_with(steps, |_, _, _| true)
    }

    /// Runs a simulation on this scene.
    ///
    /// If callback returns `false`, the simulation is interrupted.
    pub fn run_with(
        &self,
        steps: u64,
        mut callback: impl FnMut(u64, Mesh, Option<StepResult>) -> bool,
    ) -> Result<(), SceneError> {
        let mesh = self.scene.build_mesh()?;
        let mut solver_builder = SolverBuilder::new(self.config.sim_params.clone());
        solver_builder
            .set_mesh(mesh)
            .set_materials(self.config.materials.clone())
            .set_volume_zones(self.config.volume_zones.clone());
        for FrictionalContactConfig {
            params,
            object_ids: (obj, col),
            use_fixed,
        } in self.config.frictional_contacts.iter()
        {
            solver_builder.add_frictional_contact_with_fixed(*params, (*obj, *col), *use_fixed);
        }
        let mut solver = solver_builder.build::<f64>()?;

        let mut animated_positions = self.scene.animation.first().unwrap().positions.clone();

        let mut keyframe_index = 1;

        if !callback(0, solver.mesh(), None) {
            return Err(SceneError::Interrupted);
        }

        for frame in 1..=steps {
            let res = solver.step()?;
            if !callback(frame, solver.mesh(), Some(res)) {
                return Err(SceneError::Interrupted);
            }

            if keyframe_index < self.scene.animation.len() {
                let next_keyframe = self.scene.animation[keyframe_index].frame;

                if next_keyframe <= frame + 1 {
                    let next_pos = &self.scene.animation[keyframe_index].positions;

                    // Copy next keyframe positions to temp array.
                    for (out_p, next) in animated_positions.iter_mut().zip(next_pos.iter()) {
                        *out_p = *next;
                    }

                    keyframe_index += 1;

                    solver.update_vertex_positions(&animated_positions)?;
                }
            }
        }

        Ok(())
    }
}
