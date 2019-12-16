//use rayon::prelude::*;

mod rigid_shell;
mod soft_shell;
mod soft_solid;

pub(crate) use rigid_shell::*;
pub(crate) use soft_shell::*;
pub(crate) use soft_solid::*;

pub trait Inertia<'a, E> {
    fn inertia(&'a self) -> E;
}

#[cfg(test)]
mod tests {
    use geo::mesh::VertexPositions;
    use crate::objects::solid::*;
    use crate::objects::shell::*;

    use crate::energy_models::{Either, test_utils::*};
    use crate::fem::SolverBuilder;
    use crate::objects::*;

    use super::*;

    mod solid {
        use super::*;

        fn solid_material() -> SolidMaterial {
            SolidMaterial::new(0).with_density(1000.0)
        }

        fn test_solids() -> Vec<TetMeshSolid> {
            let material = solid_material();

            test_tetmeshes()
                .into_iter()
                .map(|mut tetmesh| {
                    // Prepare attributes relevant for elasticity computations.
                    SolverBuilder::prepare_deformable_tetmesh_attributes(&mut tetmesh).unwrap();
                    let mut solid = TetMeshSolid::new(tetmesh, material);
                    solid.init_density_attribute().unwrap();
                    solid
                })
                .collect()
        }

        fn build_energies(solids: &[TetMeshSolid]) -> Vec<(TetMeshInertia, Vec<[f64; 3]>)> {
            solids
                .iter()
                .map(|solid| (solid.inertia(), solid.tetmesh.vertex_positions().to_vec()))
                .collect()
        }

        #[test]
        fn gradient() {
            let solids = test_solids();
            gradient_tester(build_energies(&solids), EnergyType::Velocity);
        }

        #[test]
        fn hessian() {
            let solids = test_solids();
            hessian_tester(build_energies(&solids), EnergyType::Velocity);
        }
    }

    mod shell {
        use super::*;

        fn soft_shell_material() -> SoftShellMaterial {
            SoftShellMaterial::new(0).with_density(1000.0)
        }

        fn test_shells() -> Vec<TriMeshShell> {
            test_trimeshes()
                .into_iter()
                .map(|trimesh| {
                    // Prepare attributes relevant for elasticity computations.
                    let mut shell = TriMeshShell::soft(trimesh, soft_shell_material());
                    shell.init_deformable_attributes().unwrap();
                    shell.init_density_attribute().unwrap();
                    shell
                })
                .collect()
        }

        fn build_energies(shells: &[TriMeshShell]) -> Vec<(SoftShellInertia, Vec<[f64; 3]>)> {
            shells
                .iter()
                .map(|shell| {
                    match shell.inertia().unwrap() {
                        Either::Left(inertia) =>
                            (inertia, shell.trimesh.vertex_positions().to_vec()),
                        Either::Right(_) => unreachable!(),
                    }
                })
                .collect()
        }

        #[test]
        fn gradient() {
            let shells = test_shells();
            gradient_tester(build_energies(&shells), EnergyType::Velocity);
        }

        #[test]
        fn hessian() {
            let shells = test_shells();
            hessian_tester(build_energies(&shells), EnergyType::Velocity);
        }
    }
}
