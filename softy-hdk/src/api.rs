use std::collections::HashMap;
use std::sync::{Arc, Mutex, RwLock};

use thiserror::Error;

#[cfg(feature = "optsolver")]
use geo::attrib::*;
#[cfg(feature = "optsolver")]
use geo::mesh::topology::*;
use geo::topology::NumVertices;
use hdkrs::interop::CookResult;
use softy::nl_fem::LinearSolver;
use softy::{self, fem, Mesh, PointCloud};
#[cfg(feature = "optsolver")]
use softy::{PolyMesh, TetMesh, TetMeshExt};

use crate::{ElasticityModel, MaterialProperties, SimParams};

mod solver;

pub use self::solver::*;
use super::*;

impl From<ElasticityModel> for softy::ElasticityModel {
    fn from(model: ElasticityModel) -> Self {
        match model {
            ElasticityModel::StableNeoHookean => softy::ElasticityModel::StableNeoHookean,
            ElasticityModel::NeoHookean => softy::ElasticityModel::NeoHookean,
            i => panic!("Unrecognized elasticity model: {:?}", i),
        }
    }
}

/// A registry for solvers to be used as a global instance. We use a `HashMap` instead of `Vec` to
/// avoid dealing with fragmentation.
struct SolverRegistry {
    /// Solver key counter. Having this is not strictly necessary, but it helps identify an
    /// unoccupied key in the registry quickly. We essentially increment this key indefinitely to
    /// generate new keys.
    pub key_counter: u32,

    pub solver_table: HashMap<u32, Arc<Mutex<dyn Solver>>>,
}

lazy_static! {
    static ref SOLVER_REGISTRY: RwLock<SolverRegistry> = {
        RwLock::new(SolverRegistry {
            key_counter: 0,
            solver_table: HashMap::new(),
        })
    };
}

#[derive(Debug, Error)]
pub(crate) enum Error {
    #[error("Solver registry is full")]
    RegistryFull,
    #[error("Missing solver and mesh")]
    MissingSolverAndMesh,
    #[error("Object type ({object_type:?}) does not match material {material_id}")]
    #[cfg(feature = "optsolver")]
    MaterialObjectMismatch {
        material_id: u32,
        object_type: ObjectType,
    },
    #[error("Failed to create solver: {0}")]
    SolverCreate(#[from] softy::Error),
    #[error("Missing required mesh attribute: {0}")]
    RequiredMeshAttribute(#[from] geo::attrib::Error),
    #[error("Specified solver is unsupported")]
    UnsupportedSolver,
}

pub(crate) fn get_solver(
    solver_id: Option<u32>,
    mesh: Option<Mesh>,
    params: SimParams,
) -> Result<(u32, Arc<Mutex<dyn Solver>>), Error> {
    // Verify that the given id points to a valid solver.
    let reg_solver = solver_id.and_then(|id| {
        SOLVER_REGISTRY
            .read()
            .unwrap()
            .solver_table
            .get(&id)
            .map(|x| (id, Arc::clone(x)))
    });

    if let Some((id, solver)) = reg_solver {
        Ok((id, solver))
    } else {
        // Given solver id is invalid, need to create a new solver.
        register_new_solver(mesh, params)
    }
}

impl From<TimeIntegration> for softy::nl_fem::TimeIntegration {
    fn from(ti: TimeIntegration) -> softy::nl_fem::TimeIntegration {
        match ti {
            TimeIntegration::TR => softy::nl_fem::TimeIntegration::TR,
            TimeIntegration::BDF2 => softy::nl_fem::TimeIntegration::BDF2,
            //TimeIntegration::TRBDF2=> softy::nl_fem::TimeIntegration::TRBDF2,
            _ => softy::nl_fem::TimeIntegration::BE,
        }
    }
}

impl<'a> Into<softy::nl_fem::SimParams> for &'a SimParams {
    fn into(self) -> softy::nl_fem::SimParams {
        let SimParams {
            solver_type,
            time_step,
            gravity,
            velocity_clear_frequency,
            tolerance,
            residual_criterion,
            residual_tolerance,
            acceleration_criterion,
            acceleration_tolerance,
            velocity_criterion,
            velocity_tolerance,
            max_iterations,
            max_outer_iterations,
            derivative_test,
            friction_tolerance,
            contact_tolerance,
            contact_iterations,
            time_integration,
            ..
        } = *self;
        let line_search = match solver_type {
            SolverType::NewtonAssistedBacktracking => {
                fem::nl::LineSearch::default_assisted_backtracking()
            }
            SolverType::NewtonBacktracking => fem::nl::LineSearch::default_backtracking(),
            _ => fem::nl::LineSearch::None,
        };
        log::debug!("{:#?}", line_search);

        fem::nl::SimParams {
            time_step: if time_step > 0.0 {
                Some(time_step)
            } else {
                None
            },
            gravity: [0.0, -gravity, 0.0],
            velocity_clear_frequency,
            residual_tolerance: if residual_criterion {
                Some(residual_tolerance)
            } else {
                None
            },
            velocity_tolerance: if velocity_criterion {
                Some(velocity_tolerance)
            } else {
                None
            },
            acceleration_tolerance: if acceleration_criterion {
                Some(acceleration_tolerance)
            } else {
                None
            },
            max_iterations: max_outer_iterations,
            linsolve: if tolerance > 0.0 && max_iterations > 0 {
                LinearSolver::Iterative {
                    tolerance,
                    max_iterations,
                }
            } else {
                LinearSolver::Direct
            },
            line_search,
            derivative_test: derivative_test as u8,
            friction_tolerance,
            contact_tolerance,
            contact_iterations,
            time_integration: time_integration.into(),
        }
    }
}

#[cfg(feature = "optsolver")]
impl<'a> Into<fem::opt::SimParams> for &'a SimParams {
    fn into(self) -> fem::opt::SimParams {
        let SimParams {
            time_step,
            gravity,
            ref log_file,

            friction_iterations,

            velocity_clear_frequency,
            tolerance,
            max_iterations,
            outer_tolerance,
            max_outer_iterations,

            print_level,
            derivative_test,
            mu_strategy,
            max_gradient_scaling,
            ..
        } = *self;
        fem::opt::SimParams {
            time_step: if time_step > 0.0 {
                Some(time_step)
            } else {
                None
            },
            gravity: [0.0, -gravity, 0.0],
            clear_velocity: velocity_clear_frequency * time_step > 1.0,
            tolerance,
            max_iterations,
            outer_tolerance,
            max_outer_iterations,
            friction_iterations,
            print_level,
            derivative_test,
            mu_strategy: match mu_strategy {
                MuStrategy::Monotone => fem::opt::MuStrategy::Monotone,
                MuStrategy::Adaptive => fem::opt::MuStrategy::Adaptive,
                i => panic!("Unrecognized mu strategy: {:?}", i),
            },
            max_gradient_scaling,
            log_file: if log_file.is_empty() {
                None
            } else {
                Some(std::path::PathBuf::from(log_file))
            },
        }
    }
}

/// Build a material from the given parameters and set it to the specified id.
fn build_material_library(params: &SimParams) -> Vec<softy::Material> {
    let SimParams {
        ref materials,
        volume_constraint,
        time_step,
        ..
    } = *params;

    let mut material_library = Vec::new();

    // Add a default fixed material.
    material_library.push(softy::Material::Fixed(softy::FixedMaterial::new(0)));

    // Add the rest of the materials as specified in the UI.
    material_library.extend(
        materials
            .iter()
            .enumerate()
            .filter_map(|(idx, material_props)| {
                let material_id = idx + 1;

                let MaterialProperties {
                    object_type,
                    bending_stiffness,
                    elasticity_model,
                    bulk_modulus,
                    shear_modulus,
                    density,
                    damping,
                } = *material_props;

                match object_type {
                    ObjectType::Solid => Some(softy::Material::Solid(
                        softy::SolidMaterial::new(material_id as usize)
                            .with_elasticity(softy::Elasticity::from_bulk_shear_with_model(
                                bulk_modulus,
                                shear_modulus,
                                elasticity_model.into(),
                            ))
                            .with_volume_preservation(volume_constraint)
                            .with_density(density)
                            .with_damping(damping, time_step),
                    )),
                    ObjectType::Shell => Some(softy::Material::SoftShell(
                        softy::SoftShellMaterial::new(material_id as usize)
                            .with_elasticity(softy::Elasticity::from_bulk_shear(
                                bulk_modulus,
                                shear_modulus,
                            ))
                            .with_bending_stiffness(bending_stiffness)
                            .with_density(density)
                            .with_damping(damping, time_step),
                    )),
                    ObjectType::Rigid => Some(softy::Material::Rigid(softy::RigidMaterial::new(
                        material_id as usize,
                        density,
                    ))),

                    _ => None,
                }
            }),
    );

    material_library
}

/// Build a material from the given parameters and set it to the specified id.
#[cfg(feature = "optsolver")]
fn get_solid_material(params: &SimParams, material_id: u32) -> Result<softy::SolidMaterial, Error> {
    let SimParams {
        ref materials,
        volume_constraint,
        time_step,
        ..
    } = *params;

    // Material 0 is reserved for default
    if material_id <= 0 {
        return Ok(softy::SolidMaterial::new(0));
    }

    match materials.as_slice().get((material_id - 1) as usize) {
        Some(&MaterialProperties {
            object_type,
            elasticity_model,
            bulk_modulus,
            shear_modulus,
            density,
            damping,
            .. // bending stiffness is ignored
        }) => {
            if object_type != ObjectType::Solid {
                return Err(Error::MaterialObjectMismatch {
                    material_id: material_id as u32,
                    object_type,
                });
            }
            Ok(softy::SolidMaterial::new(material_id as usize)
                .with_elasticity(softy::ElasticityParameters::from_bulk_shear_with_model(
                    bulk_modulus,
                    shear_modulus,
                    elasticity_model.into(),
                ))
                .with_volume_preservation(volume_constraint)
                .with_density(density)
                .with_damping(damping, time_step))
        }
        None => Ok(softy::SolidMaterial::new(material_id as usize)),
    }
}

/// Build a shell material from the given parameters and set it to the specified id.
#[cfg(feature = "optsolver")]
fn get_shell_material(params: &SimParams, material_id: u32) -> Result<softy::ShellMaterial, Error> {
    let SimParams {
        ref materials,
        time_step,
        ..
    } = *params;

    // Material 0 is reserved for default
    if material_id <= 0 {
        return Ok(softy::ShellMaterial::new(0));
    }

    match materials.as_slice().get((material_id - 1) as usize) {
        Some(&MaterialProperties {
            object_type,
            bending_stiffness,
            bulk_modulus,
            shear_modulus,
            density,
            damping,
            ..
        }) => match object_type {
            ObjectType::Shell => Ok(softy::ShellMaterial::new(material_id as usize)
                .with_elasticity(softy::ElasticityParameters::from_bulk_shear(
                    bulk_modulus,
                    shear_modulus,
                ))
                .with_bending_stiffness(bending_stiffness)
                .with_density(density)
                .with_damping(damping, time_step)),
            ObjectType::Rigid => {
                Ok(softy::ShellMaterial::new(material_id as usize).with_density(density))
            }
            _ => Err(Error::MaterialObjectMismatch {
                material_id: material_id as u32,
                object_type,
            }),
        },
        None => Ok(softy::ShellMaterial::new(material_id as usize)),
    }
}

fn get_frictional_contacts<'a>(
    params: &'a SimParams,
) -> Vec<(softy::FrictionalContactParams, (usize, &'a [u32]))> {
    params
        .frictional_contacts
        .as_slice()
        .iter()
        .map(|frictional_contact| {
            let FrictionalContactParams {
                object_material_id,
                ref collider_material_ids,
                kernel,
                contact_type,
                radius_multiplier,
                smoothness_tolerance,
                contact_offset,
                use_fixed,
                smoothing_weight,
                friction_forwarding,
                dynamic_cof,
                friction_profile,
                friction_tolerance,
                friction_inner_iterations,
            } = *frictional_contact;
            let radius_multiplier = f64::from(radius_multiplier);
            let tolerance = f64::from(smoothness_tolerance);
            (
                softy::FrictionalContactParams {
                    kernel: match kernel {
                        Kernel::Interpolating => {
                            softy::KernelType::Interpolating { radius_multiplier }
                        }
                        Kernel::Approximate => softy::KernelType::Approximate {
                            tolerance,
                            radius_multiplier,
                        },
                        Kernel::Cubic => softy::KernelType::Cubic { radius_multiplier },
                        Kernel::Global => softy::KernelType::Global { tolerance },
                        i => panic!("Unrecognized kernel: {:?}", i),
                    },
                    contact_type: match contact_type {
                        ContactType::LinearizedPoint => softy::ContactType::LinearizedPoint,
                        ContactType::Point => softy::ContactType::Point,
                        i => panic!("Unrecognized contact type: {:?}", i),
                    },
                    contact_offset: f64::from(contact_offset),
                    use_fixed,
                    friction_params: if dynamic_cof == 0.0 || friction_inner_iterations == 0 {
                        None
                    } else {
                        Some(softy::FrictionParams {
                            smoothing_weight: f64::from(smoothing_weight),
                            friction_forwarding: f64::from(friction_forwarding),
                            dynamic_friction: f64::from(dynamic_cof),
                            inner_iterations: friction_inner_iterations as usize,
                            tolerance: f64::from(friction_tolerance),
                            print_level: 0,
                            friction_profile: match friction_profile {
                                FrictionProfile::Quadratic => softy::FrictionProfile::Quadratic,
                                _ => softy::FrictionProfile::Stabilized,
                            },
                        })
                    },
                },
                (
                    object_material_id as usize,
                    collider_material_ids.as_slice(),
                ),
            )
        })
        .collect()
}

trait SolverBuilder {
    fn set_mesh(&mut self, mesh: Mesh, params: &SimParams) -> Result<(), Error>;
    fn set_volume_zone_coefficients(
        &mut self,
        zone_pressurizations: Vec<f32>,
        compression_coefficients: Vec<f32>,
        hessian_approximation: Vec<bool>,
    );
    fn add_frictional_contact(
        &mut self,
        fc: softy::FrictionalContactParams,
        indices: (usize, usize),
    );
    fn build(&mut self) -> Result<Arc<Mutex<dyn Solver>>, Error>;
}
#[cfg(feature = "optsolver")]
impl SolverBuilder for fem::opt::SolverBuilder {
    fn set_mesh(&mut self, mesh: Mesh, params: &SimParams) -> Result<(), Error> {
        use geo::algo::{split::TypedMesh, SplitIntoConnectedComponents};
        let meshes = mesh.split_into_typed_meshes();
        for mesh in meshes.into_iter() {
            match mesh {
                TypedMesh::Tet(mut tetmesh) => {
                    softy::init_mesh_source_index_attribute(&mut tetmesh)?;
                    for mesh in TetMeshExt::from(tetmesh).split_into_connected_components() {
                        let material_id = mesh
                            .attrib_as_slice::<i32, CellIndex>("mtl_id")
                            .map(|slice| {
                                utils::mode_u32(slice.iter().map(|&x| {
                                    if x < 0 {
                                        0u32
                                    } else {
                                        x as u32
                                    }
                                }))
                                .0
                            })
                            .unwrap_or(0);
                        let solid_material = get_solid_material(params, material_id)?;
                        fem::opt::SolverBuilder::add_solid(
                            self,
                            TetMesh::from(mesh),
                            solid_material,
                        );
                    }
                }
                TypedMesh::Tri(mut trimesh) => {
                    let mut mesh = PolyMesh::from(trimesh);
                    softy::init_mesh_source_index_attribute(&mut mesh)?;
                    for mesh in mesh.reversed().split_into_connected_components() {
                        let material_id = mesh
                            .attrib_as_slice::<i32, FaceIndex>("mtl_id")
                            .map(|slice| {
                                utils::mode_u32(slice.iter().map(|&x| {
                                    if x < 0 {
                                        0u32
                                    } else {
                                        x as u32
                                    }
                                }))
                                .0
                            })
                            .unwrap_or(0);
                        let shell_material = get_shell_material(params, material_id)?;
                        fem::opt::SolverBuilder::add_shell(self, mesh, shell_material);
                    }
                }
            }
        }
        Ok(())
    }
    fn set_volume_zone_coefficients(
        &mut self,
        zone_pressurizations: Vec<f32>,
        compression_coefficients: Vec<f32>,
        hessian_approximation: Vec<bool>,
    ) {
    }
    fn add_frictional_contact(
        &mut self,
        fc: softy::FrictionalContactParams,
        indices: (usize, usize),
    ) {
        fem::opt::SolverBuilder::add_frictional_contact(self, fc, indices);
    }
    fn build(&mut self) -> Result<Arc<Mutex<dyn Solver>>, Error> {
        Ok(Arc::new(Mutex::new(fem::opt::SolverBuilder::build(self)?)))
    }
}

impl SolverBuilder for fem::nl::SolverBuilder {
    fn set_mesh(&mut self, mut mesh: Mesh, _: &SimParams) -> Result<(), Error> {
        softy::init_mesh_source_index_attribute(&mut mesh)?;
        mesh.reverse_if(|_, cell_type| matches!(cell_type, geo::mesh::CellType::Triangle));
        fem::nl::SolverBuilder::set_mesh(self, mesh);
        Ok(())
    }
    fn set_volume_zone_coefficients(
        &mut self,
        zone_pressurizations: Vec<f32>,
        compression_coefficients: Vec<f32>,
        hessian_approximation: Vec<bool>,
    ) {
        fem::nl::SolverBuilder::set_volume_penalty_params(
            self,
            zone_pressurizations,
            compression_coefficients,
            hessian_approximation,
        );
    }
    fn add_frictional_contact(
        &mut self,
        fc: softy::FrictionalContactParams,
        indices: (usize, usize),
    ) {
        fem::nl::SolverBuilder::add_frictional_contact(self, fc, indices);
    }
    fn build(&mut self) -> Result<Arc<Mutex<dyn Solver>>, Error> {
        Ok(Arc::new(Mutex::new(fem::nl::SolverBuilder::build(self)?)))
    }
}

/// Register a new solver in the registry. (Rust side)
//#[allow(clippy::needless_pass_by_value)]
#[inline]
pub(crate) fn register_new_solver(
    mesh: Option<Mesh>,
    params: SimParams,
) -> Result<(u32, Arc<Mutex<dyn Solver>>), Error> {
    if mesh.is_none() {
        return Err(Error::MissingSolverAndMesh);
    }

    // Build a basic solver with a solid material.
    let mut solver_builder: Box<dyn SolverBuilder> = match params.solver_type {
        #[cfg(feature = "optsolver")]
        SolverType::Ipopt => Box::new(fem::opt::SolverBuilder::new((&params).into())),
        #[cfg(not(feature = "optsolver"))]
        SolverType::Ipopt => return Err(Error::UnsupportedSolver),
        // All other solvers are custom nonlinear system solvers.
        _ => {
            let mut builder = Box::new(fem::nl::SolverBuilder::new((&params).into()));
            builder.set_materials(build_material_library(&params));
            builder
        }
    };

    if let Some(mesh) = mesh {
        solver_builder.set_mesh(mesh, &params)?;
    }

    for (frictional_contact, indices) in get_frictional_contacts(&params) {
        for &collider_index in indices.1.iter() {
            solver_builder
                .add_frictional_contact(frictional_contact, (indices.0, collider_index as usize));
        }
    }

    solver_builder.set_volume_zone_coefficients(
        params.zone_pressurizations,
        params.compression_coefficients,
        params
            .hessian_approximation
            .iter()
            .map(|&x| x != 0)
            .collect(),
    );

    let solver = solver_builder.build()?;

    // Get a mutable reference to the solver registry.
    let SolverRegistry {
        ref mut key_counter,
        ref mut solver_table,
    } = *SOLVER_REGISTRY.write().unwrap();

    // Find a place in the registry to insert the new solver
    let mut counter = 0_u32; // use this to catch the case when registry is full.
    let vacant_id = loop {
        *key_counter += 1;
        counter += 1;
        if !solver_table.contains_key(key_counter) {
            break Some(*key_counter);
        } else if counter == 0 {
            break None;
        }
    };

    if let Some(id) = vacant_id {
        // Insert a new solver at the location we determined is vacant.
        let new_solver = solver_table.entry(id).or_insert_with(|| solver);
        Ok((id, Arc::clone(new_solver)))
    } else {
        Err(Error::RegistryFull)
    }
}

/// Perform one solve step given a solver.
#[inline]
pub(crate) fn step<F>(
    solver: &mut dyn Solver,
    mesh_points: Option<PointCloud>,
    check_interrupt: F,
) -> (Option<Mesh>, CookResult)
where
    F: FnMut() -> bool + Sync + Send + 'static,
{
    solver.set_interrupter(Box::new(check_interrupt));

    // Update mesh points
    if let Some(pts) = mesh_points {
        match solver.update_vertices(&pts) {
            Err(softy::Error::SizeMismatch) =>
                return (None, CookResult::Error(
                        format!("Input points ({}) don't coincide with solver Mesh ({}).",
                        pts.num_vertices(), solver.num_vertices()))),
            Err(softy::Error::AttribError { source }) =>
                return (None, CookResult::Warning(
                        format!("Failed to find 8-bit integer attribute \"fixed\", which marks animated vertices. ({})", source))),
            Err(err) =>
                return (None, CookResult::Error(
                        format!("Error updating mesh vertices. ({})", err))),
            _ => {}
        }
    }

    let cook_result = convert_to_cookresult(solver.solve().into());
    (Some(solver.mesh()), cook_result)
}

/// Clear all solvers from the registry and reset the counter.
#[inline]
pub(crate) fn clear_solver_registry() {
    let SolverRegistry {
        ref mut key_counter,
        ref mut solver_table,
    } = *SOLVER_REGISTRY.write().unwrap();
    *key_counter = 0;
    solver_table.clear();
}

/// Get a solver given the parameters and compute one step.
#[inline]
pub(crate) fn cook<F>(
    solver_id: Option<u32>,
    mesh: Option<Mesh>,
    params: SimParams,
    check_interrupt: F,
) -> (Option<(u32, Mesh)>, CookResult)
where
    F: FnMut() -> bool + Sync + Send + 'static,
{
    match get_solver(solver_id, mesh, params) {
        Ok((solver_id, solver)) => {
            let mut solver = solver.lock().unwrap();
            solver.set_interrupter(Box::new(check_interrupt));
            let cook_result = convert_to_cookresult(solver.solve());
            let solver_mesh = solver.mesh();
            (Some((solver_id, solver_mesh)), cook_result)
        }
        Err(err) => (
            None,
            CookResult::Error(format!("Couldn't find or create a valid solver: {:?}", err)),
        ),
    }
}

fn convert_to_cookresult(res: softy::SimResult) -> CookResult {
    match res {
        softy::SimResult::Success(msg) => CookResult::Success(msg),
        softy::SimResult::Warning(msg) => CookResult::Warning(msg),
        softy::SimResult::Error(msg) => CookResult::Error(msg),
    }
}
