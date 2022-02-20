//! This module describes a convenient scene struct describing a simulatable configuration.

// TODO: Save only fixed vertices.
// TODO: Enable more sophisticated interpolation.

use std::collections::HashMap;
use flatk::IntoStorage;

use serde::{Serialize, Deserialize};
use thiserror::Error;
use geo::topology::{CellIndex, VertexIndex};
use geo::attrib::Attrib;

use crate::{FrictionalContactParams, Material, Mesh};
use crate::nl_fem::{SimParams, SolverBuilder, SolveResult};

#[derive(Debug, Error)]
pub enum SceneError {
    #[error("IO Error")]
    IO(#[from] std::io::Error),
    #[error("Serialization error")]
    Serialize,
    #[error("Attribute transfer error")]
    Attribute(#[from] geo::attrib::Error),
    #[error("Solver error")]
    Solver(#[from] crate::Error),
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
        } else if let Ok(v) = attrib.direct_clone_into_vec::<[f32;3]>() {
            Attribute::F32x3(v)
        } else if let Ok(v) = attrib.direct_clone_into_vec::<[f64;3]>() {
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
            CellType::Triangle =>    geo::mesh::CellType::Triangle,
        }
    }
}

/// A simplified mesh structure suitable for serialization and deserialization.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MeshTopo {
    pub indices: Vec<Vec<usize>>,
    pub types: Vec<CellType>,
    pub vertex_attributes: HashMap<String, Attribute>,
    pub cell_attributes: HashMap<String, Attribute>,
}

impl Default for MeshTopo {
    fn default() -> Self {
        MeshTopo {
            indices: Vec::new(),
            types: Vec::new(),
            vertex_attributes: HashMap::new(),
            cell_attributes: HashMap::new(),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct KeyframedVertexPositions {
    pub frame: u64,
    pub positions: Vec<[f64; 3]>,
}

/// Scene description.
#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct Scene {
    // Mesh topology.
    mesh_topo: MeshTopo,
    /// A set of time stamped vertex positions.
    animation: Vec<KeyframedVertexPositions>,
}

impl Scene {
    /// Set the simulation mesh representing all objects in the scene.
    ///
    /// This function sets the initial keyframe to be the mesh vertex positions.
    pub fn new(mesh: Mesh) -> Self {
        let vertex_attributes = mesh.vertex_attributes.iter().filter_map(|(name, attrib)| {
            Attribute::from_attrib_data(&attrib.data).map(|attrib| (name.clone(), attrib))
        }).collect();
        let cell_attributes = mesh.cell_attributes.iter().filter_map(|(name, attrib)| {
            Attribute::from_attrib_data(&attrib.data).map(|attrib| (name.clone(), attrib))
        }).collect();
        let index_blocks: Vec<_> = mesh.indices.clump_iter().map(|chunked| chunked.into_storage().to_vec()).collect();
        let mesh_topo = MeshTopo {
            indices: index_blocks,
            types: mesh.types.iter().map(|&t| CellType::from(t)).collect(),
            vertex_attributes,
            cell_attributes,
        };
        let mut animation = Vec::new();
        animation.push(KeyframedVertexPositions {
            frame: 0,
            positions: mesh.vertex_positions.into_vec()
        });
        Scene {
            mesh_topo,
            animation
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
    pub fn set_keyframes(&mut self, frames: impl AsRef<[u64]>, positions: impl Into<Vec<Vec<[f64; 3]>>>) -> &mut Self {
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

        self.animation.extend(frames.iter().zip(positions.into_iter()).map(|(&frame, positions)|
            KeyframedVertexPositions {
                frame,
                positions
            }
        ));
        self
    }

    /// Add a single keyframe to the animation.
    pub fn add_keyframe(&mut self, frame: u64, positions: Vec<[f64; 3]>) -> &mut Self {
        // No panic since animation should always be non-empty to contain at least one set of vertex
        // positions.
        let last = self.animation.last_mut().unwrap();
        if frame > last.frame {
            // Insert at the end. Presumably this is the most common scenario.
            self.animation.push(KeyframedVertexPositions { frame, positions });
        } else if frame == last.frame {
            // Times coincide, overwrite the last one
            last.positions = positions;
        } else {
            // Insert in the middle, or overwrite previous keyframe.
            match self.animation.binary_search_by_key(&frame, |tp| tp.frame) {
                Ok(pos) => self.animation[pos].positions = positions,
                Err(pos) => self.animation.insert(pos, KeyframedVertexPositions { frame, positions }),
            }
        }
        self
    }

    /// Constructs a mesh from the stored topology and vertex position data for the first frame.
    fn build_mesh(&self) -> Result<Mesh, SceneError> {
        let types: Vec<_> = self.mesh_topo.types.iter().map(|&x| geo::mesh::CellType::from(x)).collect();
        let mut mesh = Mesh::from_cells_and_types(self.animation.first().unwrap().positions.clone(), self.mesh_topo.indices.clone(), types);

        for (name, attrib) in self.mesh_topo.vertex_attributes.iter() {
            mesh.insert_attrib::<VertexIndex>(name, attrib.clone().into_geo_attrib::<VertexIndex>())?;
        }
        for (name, attrib) in self.mesh_topo.cell_attributes.iter() {
            mesh.insert_attrib::<CellIndex>(name, attrib.clone().into_geo_attrib::<CellIndex>())?;
        }
        Ok(mesh)
    }
}

/// Scene configuration.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SceneConfig {
    pub sim_params: SimParams,
    pub materials: Vec<Material>,
    pub frictional_contacts: Vec<(FrictionalContactParams, (usize, usize))>,
    pub scene: Scene,
}

impl SceneConfig {
    /// Construct a new scene configuration.
    pub fn new(sim_params: SimParams, mesh: Mesh) -> Self {
        SceneConfig {
            sim_params,
            materials: Vec::new(),
            frictional_contacts: Vec::new(),
            scene: Scene::new(mesh),
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
    pub fn set_animation(&mut self, frames: impl AsRef<[u64]>, positions: impl Into<Vec<Vec<[f64; 3]>>>) -> &mut Self {
        self.scene.set_keyframes(frames, positions);
        self
    }

    /// Add a single keyframe to the animation.
    pub fn add_keyframe(&mut self, frame: u64, positions: impl Into<Vec<[f64; 3]>>) -> &mut Self {
        self.scene.add_keyframe(frame, positions.into());
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

    /// Saves this scene configuration to the given path interpreted as a RON file.
    pub fn save_as_ron(&self, path: impl AsRef<std::path::Path>) -> Result<(), SceneError> {
        let path = path.as_ref();
        std::fs::File::create(path).map_err(SceneError::from).and_then(|f| {
            ron::ser::to_writer_pretty(f, self, ron::ser::PrettyConfig::new()).map_err(|_| SceneError::Serialize)
        }).into()
    }

    /// Saves this scene configuration to the given path interpreted as a binary file using `bincode`.
    #[cfg(feature="bincode")]
    pub fn save_as_bin(&self, path: impl AsRef<std::path::Path>) -> Result<(), SceneError> {
        let path = path.as_ref();
        std::fs::File::create(path).map_err(SceneError::from).and_then(|f| {
            bincode::serialize_into(f, self).map_err(|_| SceneError::Serialize)
        }).into()
    }

    /// Loads the scene from a `bincode` binary file.
    #[cfg(feature="bincode")]
    pub fn load_from_bin(path: impl AsRef<std::path::Path>) -> Result<Self, SceneError> {
        let path = path.as_ref();
        std::fs::File::open(path).map_err(SceneError::from).and_then(|f| {
            bincode::deserialize_from(f).map_err(|_| SceneError::Serialize)
        }).into()
    }

    /// Runs a simulation on this scene.
    pub fn run(&self, steps: u64) -> Result<(), SceneError> {
        self.run_with(steps, |_,_,_| { true })
    }

    /// Runs a simulation on this scene.
    ///
    /// If callback returns `false`, the simulation is interrupted.
    pub fn run_with(&self, steps: u64, callback: impl Fn(u64, SolveResult, Mesh) -> bool) -> Result<(), SceneError> {
        let mesh = self.scene.build_mesh()?;
        let mut solver_builder = SolverBuilder::new(self.sim_params.clone());
        solver_builder
            .set_mesh(mesh)
            .set_materials(self.materials.clone());
        for (fc_params, (obj, col)) in self.frictional_contacts.iter() {
            solver_builder.add_frictional_contact(fc_params.clone(), (*obj, *col));
        }
        let mut solver = solver_builder.build::<f64>()?;

        let mut animated_positions = self.scene.animation.first().unwrap().positions.clone();

        let mut keyframe_index = 1;
        for frame in 0..steps {
            let res = solver.step()?;
            if !callback(frame, res, solver.mesh()) {
                break;
            }

            if keyframe_index < self.scene.animation.len() {
                let prev_keyframe = self.scene.animation[keyframe_index - 1].frame;

                // Skip until the frame is in the range of keyframed animation.
                if prev_keyframe >= frame {
                    continue;
                }

                // Interpolate keyframed positions.
                let next_keyframe = self.scene.animation[keyframe_index].frame;
                let next_pos = &self.scene.animation[keyframe_index].positions;

                // Copy next keyframe positions to temp array.
                for (out_p, next) in animated_positions.iter_mut().zip(next_pos.iter()) {
                    *out_p = *next;
                }

                if next_keyframe > frame {
                    let prev_pos = &self.scene.animation[keyframe_index - 1].positions;
                    let t = (frame - prev_keyframe) as f64 / (next_keyframe - prev_keyframe) as f64;
                    for (out_p, prev, ) in animated_positions.iter_mut().zip(prev_pos.iter()) {
                        for i in 0..3 {
                            out_p[i] *= t;
                            out_p[i] += prev[i] * (t - 1.0);
                        }
                    }
                } else {
                    keyframe_index += 1;
                }

                solver.update_vertex_positions(&animated_positions)?;
            }
        }

        Ok(())
    }
}