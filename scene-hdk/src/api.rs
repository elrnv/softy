use thiserror::Error;

use hdkrs::interop::CookResult;
use softy::nl_fem::LinearSolver;
use softy::{self, fem, Mesh, PointCloud};

use crate::{ElasticityModel, MaterialProperties, SimParams};

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
#[derive(Debug, Error)]
pub(crate) enum Error {
    #[error("Missing mesh")]
    MissingMesh,
    #[error("Missing required mesh attribute: {0}")]
    RequiredMeshAttribute(#[from] geo::attrib::Error),
}

impl From<TimeIntegration> for softy::nl_fem::TimeIntegration {
    fn from(ti: TimeIntegration) -> softy::nl_fem::TimeIntegration {
        match ti {
            TimeIntegration::TR => softy::nl_fem::TimeIntegration::TR,
            //TimeIntegration::BDF2 => softy::nl_fem::TimeIntegration::BDF2,
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
            jacobian_test: derivative_test > 0,
            friction_tolerance,
            contact_tolerance,
            contact_iterations,
            time_integration: time_integration.into(),
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
                // contact_type,
                radius_multiplier,
                smoothness_tolerance,
                contact_offset,
                use_fixed,
                dynamic_cof,
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
                    contact_type: softy::ContactType::Point,
                    contact_offset: f64::from(contact_offset),
                    use_fixed,
                    friction_params: if dynamic_cof == 0.0 {
                        None
                    } else {
                        Some(softy::FrictionParams {
                            smoothing_weight: 0.0,
                            friction_forwarding: 1.0,
                            dynamic_friction: f64::from(dynamic_cof),
                            inner_iterations: 0,
                            tolerance: 0.0,
                            print_level: 0,
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

#[inline]
pub(crate) fn new_scene(mesh: Option<Mesh>, params: SimParams) -> Result<SceneConfig, Error> {
    // Build a basic solver with a solid material.
    let mut scene = if let Some(mesh) = mesh {
        SceneConfig::new((&params).into(), mesh)
    } else {
        return Err(Error::MissingMesh);
    };

    scene.set_materials(build_material_library(&params));

    for (frictional_contact, indices) in get_frictional_contacts(&params) {
        for &collider_index in indices.1.iter() {
            scene.add_frictional_contact(frictional_contact, (indices.0, collider_index as usize));
        }
    }

    Ok(scene)
}

/// Add a keyframe to the scene configuration.
#[inline]
pub(crate) fn add_keyframe(
    scene: &mut SceneConfig,
    frame: u64,
    mesh_points: PointCloud,
) -> CookResult {
    scene.add_keyframe(frame, mesh_points.vertex_positions.into_vec());
    CookResult::Success(String::new())
}

#[inline]
pub(crate) fn save(scene: &SceneConfig, path: impl AsRef<std::path::Path>) -> CookResult {
    match scene.save_as_bin(path) {
        Ok(()) => CookResult::Success(String::new()),
        Err(err) => CookResult::Error(format!("Failed to save scene file: {}", err)),
    }
}
