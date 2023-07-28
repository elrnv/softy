use std::collections::HashMap;
use std::sync::{Arc, Mutex, RwLock};

use thiserror::Error;

use geo::{attrib::*, topology::*, VertexPositions};
use hdkrs::interop::CookResult;
use softy::nl_fem::LinearSolver;
use softy::{self, fem, Mesh, PointCloud, Pos64Type, POSITION64_ATTRIB};

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
    #[error("Failed to create solver: {0}")]
    SolverCreate(#[from] softy::Error),
    #[error("Missing required mesh attribute: {0}")]
    RequiredMeshAttribute(#[from] geo::attrib::Error),
    // #[error("Specified solver is unsupported")]
    // UnsupportedSolver,
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
            TimeIntegration::TRBDF2 => softy::nl_fem::TimeIntegration::TRBDF2(0.5),
            TimeIntegration::TRBDF2U => {
                softy::nl_fem::TimeIntegration::TRBDF2(2.0 - 2.0_f32.sqrt())
            }
            TimeIntegration::SDIRK2 => softy::nl_fem::TimeIntegration::SDIRK2,
            _ => softy::nl_fem::TimeIntegration::BE,
        }
    }
}

impl From<Preconditioner> for softy::nl_fem::Preconditioner {
    fn from(p: Preconditioner) -> softy::nl_fem::Preconditioner {
        match p {
            Preconditioner::IncompleteJacobi => softy::nl_fem::Preconditioner::IncompleteJacobi,
            Preconditioner::ApproximateJacobi => softy::nl_fem::Preconditioner::ApproximateJacobi,
            _ => softy::nl_fem::Preconditioner::None,
        }
    }
}

impl<'a> Into<softy::nl_fem::SimParams> for &'a SimParams {
    fn into(self) -> softy::nl_fem::SimParams {
        let SimParams {
            solver_type,
            line_search,
            backtracking_coeff,
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
            contact_iterations,
            time_integration,
            preconditioner,
            project_element_hessians,
            ref log_file,
            ..
        } = *self;
        let solver_type = match solver_type {
            SolverType::AdaptiveNewton => fem::nl::SolverType::AdaptiveNewton,
            _ => fem::nl::SolverType::Newton,
        };
        let line_search = match line_search {
            LineSearch::AssistedBacktracking => {
                fem::nl::LineSearch::default_assisted_backtracking()
            }
            LineSearch::ContactAssistedBacktracking => {
                fem::nl::LineSearch::default_contact_assisted_backtracking()
            }
            LineSearch::Backtracking => fem::nl::LineSearch::default_backtracking(),
            _ => fem::nl::LineSearch::None,
        }
        .with_step_factor(backtracking_coeff);
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
            solver_type,
            derivative_test: derivative_test as u8,
            contact_iterations,
            time_integration: time_integration.into(),
            preconditioner: preconditioner.into(),
            log_file: if log_file.is_empty() {
                None
            } else {
                Some(std::path::PathBuf::from(log_file))
            },
            project_element_hessians,
        }
    }
}

/// Build a material from the given parameters and set it to the specified id.
fn build_material_library(params: &SimParams) -> Vec<softy::Material> {
    let SimParams {
        ref materials,
        volume_constraint,
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
                            .with_damping(damping),
                    )),
                    ObjectType::Shell => Some(softy::Material::SoftShell(
                        softy::SoftShellMaterial::new(material_id as usize)
                            .with_elasticity(softy::Elasticity::from_bulk_shear(
                                bulk_modulus,
                                shear_modulus,
                            ))
                            .with_bending_stiffness(bending_stiffness)
                            .with_density(density)
                            .with_damping(damping),
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

#[derive(Copy, Clone, PartialEq)]
enum GenericFrictionalContactParams {
    NL(softy::constraints::penalty_point_contact::FrictionalContactParams),
}

fn get_frictional_contacts<'a>(
    params: &'a SimParams,
) -> Vec<(GenericFrictionalContactParams, (usize, &'a [u32]), bool)> {
    params
        .frictional_contacts
        .as_slice()
        .iter()
        .map(|frictional_contact| {
            let FrictionalContactParams {
                object_material_id,
                ref collider_material_ids,
                kernel,
                radius_multiplier,
                smoothness_tolerance,
                contact_offset,
                use_fixed,
                dynamic_cof,
                static_cof,
                viscous_friction,
                stribeck_velocity,
                friction_profile,
                lagged_friction,
                incomplete_friction_jacobian,
                friction_tolerance,
                contact_tolerance,
                ..
            } = *frictional_contact;
            let radius_multiplier = f64::from(radius_multiplier);
            let tolerance = f64::from(smoothness_tolerance);
            (
                GenericFrictionalContactParams::NL(
                    softy::constraints::penalty_point_contact::FrictionalContactParams {
                        kernel: match kernel {
                            Kernel::Smooth => softy::KernelType::Smooth {
                                tolerance,
                                radius_multiplier,
                            },
                            Kernel::Approximate => softy::KernelType::Approximate {
                                tolerance,
                                radius_multiplier,
                            },
                            Kernel::Cubic => softy::KernelType::Cubic { radius_multiplier },
                            Kernel::Global => softy::KernelType::Global { tolerance },
                            i => panic!("Unrecognized kernel: {:?}", i),
                        },
                        contact_offset: f64::from(contact_offset),
                        tolerance: contact_tolerance,
                        stiffness: 1.0 / contact_tolerance,
                        friction_params:
                            softy::constraints::penalty_point_contact::FrictionParams {
                                dynamic_friction: f64::from(dynamic_cof),
                                static_friction: f64::from(static_cof),
                                viscous_friction: f64::from(viscous_friction),
                                stribeck_velocity: f64::from(stribeck_velocity),
                                friction_profile: match friction_profile {
                                    FrictionProfile::Quadratic => softy::FrictionProfile::Quadratic,
                                    _ => softy::FrictionProfile::Stabilized,
                                },
                                epsilon: friction_tolerance as f64,
                                lagged: lagged_friction,
                                incomplete_jacobian: incomplete_friction_jacobian,
                            },
                    },
                ),
                (
                    object_material_id as usize,
                    collider_material_ids.as_slice(),
                ),
                use_fixed,
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
        fc: GenericFrictionalContactParams,
        indices: (usize, usize),
        use_fixed: bool,
    );
    fn build(&mut self) -> Result<Arc<Mutex<dyn Solver>>, Error>;
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
        fc: GenericFrictionalContactParams,
        indices: (usize, usize),
        use_fixed: bool,
    ) {
        let GenericFrictionalContactParams::NL(fc) = fc;
        fem::nl::SolverBuilder::add_frictional_contact_with_fixed(self, fc, indices, use_fixed);
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
    let mut solver_builder: Box<dyn SolverBuilder> = {
        let mut builder = Box::new(fem::nl::SolverBuilder::new((&params).into()));
        builder.set_materials(build_material_library(&params));
        builder
    };

    if let Some(mut mesh) = mesh {
        // Get 64bit positions from the attribute if any, since houdini uses 32bit positions.
        // This is useful when 64 bit precision is needed for analysis when restarting the solver.
        if let Ok(pos64) = mesh.remove_attrib::<VertexIndex>(POSITION64_ATTRIB) {
            if let Ok(pos64_iter) = pos64.direct_iter::<Pos64Type>() {
                mesh.vertex_positions_mut()
                    .iter_mut()
                    .zip(pos64_iter)
                    .for_each(|(out_pos, &pos64)| {
                        *out_pos = pos64;
                    });
            } else {
                // Wrong attribute, put it back.
                // No panic since we just removed that attribute, that space is guaranteed to be vacant.
                mesh.insert_attrib(POSITION64_ATTRIB, pos64).unwrap();
            }
        }
        solver_builder.set_mesh(mesh, &params)?;
    }

    let frictional_contact_params = get_frictional_contacts(&params);

    for (frictional_contact, indices, use_fixed) in frictional_contact_params.into_iter() {
        for &collider_index in indices.1.iter() {
            solver_builder.add_frictional_contact(
                frictional_contact,
                (indices.0, collider_index as usize),
                use_fixed,
            );
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
