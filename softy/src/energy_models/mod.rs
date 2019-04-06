pub mod gravity;
pub mod momentum;
pub mod volumetric_neohookean;

#[cfg(test)]
pub(crate) mod test_utils {
    use crate::energy::*;
    use crate::fem::SolverBuilder;
    use crate::test_utils::*;
    use crate::TetMesh;
    use approx::*;
    use autodiff::F;
    use geo::mesh::VertexPositions;
    use num_traits::Zero;
    use reinterpret::*;

    /// Prepare test meshes
    pub(crate) fn test_meshes() -> Vec<TetMesh> {
        let mut meshes = vec![
            make_one_tet_mesh(),
            make_one_deformed_tet_mesh(),
            make_three_tet_mesh(),
        ];
        for mesh in meshes.iter_mut() {
            SolverBuilder::prepare_mesh_attributes(mesh).unwrap();
        }
        meshes
    }

    fn random_displacement(n: usize) -> Vec<F> {
        use rand::{distributions::Uniform, Rng, SeedableRng, StdRng};
        let mut rng: StdRng = SeedableRng::from_seed([3; 32]);
        let range = Uniform::new(-0.1, 0.1);
        (0..n).map(move |_| F::cst(rng.sample(range))).collect()
    }

    /// The variable type giving previous state in the energy.
    pub(crate) enum EnergyType {
        Position,
        Velocity(f64), // requires timestep
    }

    pub(crate) fn gradient_tester<B, E>(build_energy: B, ty: EnergyType)
    where
        B: Fn(TetMesh) -> E,
        E: Energy<F> + EnergyGradient<F>,
    {
        for mesh in test_meshes().into_iter() {
            let pos = mesh.vertex_positions().to_vec();
            let energy_model = build_energy(mesh);

            let mut dx = random_displacement(3 * pos.len());
            let x: Vec<F> = match ty {
                EnergyType::Position => reinterpret_vec(pos)
                    .into_iter()
                    .map(|x: f64| F::cst(x))
                    .collect(),
                EnergyType::Velocity(dt) => dx.iter().map(|&disp| disp / dt).collect(),
            };
            let mut grad = vec![F::zero(); x.len()];
            energy_model.add_energy_gradient(&x, &dx, &mut grad);

            for i in 0..x.len() {
                dx[i] = F::var(dx[i]);
                let energy = energy_model.energy(&x, &dx);
                assert_relative_eq!(
                    grad[i].value(),
                    energy.deriv(),
                    max_relative = 1e-6,
                    epsilon = 1e-10
                );
                dx[i] = F::cst(dx[i]);
            }
        }
    }

    pub(crate) fn hessian_tester<B, E>(build_energy: B, ty: EnergyType)
    where
        B: Fn(TetMesh) -> E,
        E: EnergyGradient<F> + EnergyHessian,
    {
        use crate::matrix::{MatrixElementIndex as Index, MatrixElementTriplet as Triplet};

        for mesh in test_meshes().into_iter() {
            let pos = mesh.vertex_positions().to_vec();
            let energy_model = build_energy(mesh);

            let mut dx = random_displacement(3 * pos.len());
            let x: Vec<F> = match ty {
                EnergyType::Position => reinterpret_vec(pos)
                    .into_iter()
                    .map(|x: f64| F::cst(x))
                    .collect(),
                EnergyType::Velocity(dt) => dx.iter().map(|&disp| disp / dt).collect(),
            };
            let mut hess_triplets =
                vec![Triplet::new(0, 0, F::zero()); energy_model.energy_hessian_size()];
            energy_model.energy_hessian(&x, &dx, &mut hess_triplets);

            // Build a dense hessian
            let mut hess = vec![vec![F::zero(); x.len()]; x.len()];
            for Triplet {
                idx: Index { row, col },
                val,
            } in hess_triplets.into_iter()
            {
                hess[row][col] += val;
                if row != col {
                    hess[col][row] += val;
                }
            }

            for i in 0..x.len() {
                dx[i] = F::var(dx[i]);
                let mut grad = vec![F::zero(); x.len()];
                energy_model.add_energy_gradient(&x, &dx, &mut grad);
                for j in 0..x.len() {
                    assert_relative_eq!(
                        hess[i][j].value(),
                        grad[j].deriv(),
                        max_relative = 1e-6,
                        epsilon = 1e-10
                    );
                }
                dx[i] = F::cst(dx[i]);
            }
        }
    }
}
