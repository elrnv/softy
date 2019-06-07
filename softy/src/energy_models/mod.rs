pub mod gravity;
pub mod momentum;
pub mod tet_nh;

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
    #[derive(Copy, Clone, Debug, PartialEq)]
    pub(crate) enum EnergyType {
        Position,
        Velocity,
    }

    /// Construct a step in autodiff format given a set of initial positions. This function returns
    /// initial and current values for the independent variable determined by `ty`.
    fn autodiff_step(pos: &[f64], ty: EnergyType) -> (Vec<F>, Vec<F>) {
        let v0 = vec![F::cst(0.0); pos.len()]; // previous vel
        let v1 = random_displacement(pos.len()); // current vel

        let p0: Vec<F> = pos.iter().map(|&x| F::cst(x)).collect();

        let p1: Vec<F> = pos
            .iter()
            .zip(v1.iter())
            .map(|(&x, &v)| F::cst(x + v))
            .collect();

        match ty {
            EnergyType::Position => (p0, p1),
            EnergyType::Velocity => (v0, v1),
        }
    }

    pub(crate) fn gradient_tester<B, E>(build_energy: B, ty: EnergyType)
    where
        B: Fn(TetMesh) -> E,
        E: Energy<F> + EnergyGradient<F>,
    {
        for mesh in test_meshes().into_iter() {
            let pos = mesh.vertex_positions().to_vec();
            let (x0, mut x1) = autodiff_step(reinterpret_slice(&pos), ty);
            let energy_model = build_energy(mesh);

            let mut grad = vec![F::zero(); 3 * pos.len()];
            energy_model.add_energy_gradient(&x0, &x1, &mut grad);

            for i in 0..x0.len() {
                x1[i] = F::var(x1[i]);
                let energy = energy_model.energy(&x0, &x1);
                assert_relative_eq!(
                    grad[i].value(),
                    energy.deriv(),
                    max_relative = 1e-6,
                    epsilon = 1e-10
                );
                x1[i] = F::cst(x1[i]);
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
            let (x0, mut x1) = autodiff_step(reinterpret_slice(&pos), ty);
            let energy_model = build_energy(mesh);

            let mut hess_triplets =
                vec![Triplet::new(0, 0, F::zero()); energy_model.energy_hessian_size()];
            energy_model.energy_hessian(&x0, &x1, F::cst(1.0), &mut hess_triplets);

            // Build a dense hessian
            let mut hess = vec![vec![F::zero(); x0.len()]; x0.len()];
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

            for i in 0..x0.len() {
                x1[i] = F::var(x1[i]);
                let mut grad = vec![F::zero(); x0.len()];
                energy_model.add_energy_gradient(&x0, &x1, &mut grad);
                for j in 0..x0.len() {
                    assert_relative_eq!(
                        hess[i][j].value(),
                        grad[j].deriv(),
                        max_relative = 1e-6,
                        epsilon = 1e-10
                    );
                }
                x1[i] = F::cst(x1[i]);
            }
        }
    }
}
