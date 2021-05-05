pub mod elasticity;
pub mod gravity;
pub mod inertia;

use crate::energy::*;
use crate::matrix::MatrixElementIndex;
use num_traits::FromPrimitive;
use tensr::Real;

/// Define a nullable energy, which is used to represent zero energies. For example
/// a fixed mesh can use this energy in place of elasticity, gravity or inertia.

impl<T: Real, E: Energy<T>> Energy<T> for Option<E> {
    fn energy(&self, x0: &[T], x1: &[T]) -> T {
        self.as_ref().map_or(T::zero(), |e| e.energy(x0, x1))
    }
}

impl<X: Real, T: Real, E: EnergyGradient<X, T>> EnergyGradient<X, T> for Option<E> {
    fn add_energy_gradient(&self, x0: &[X], x1: &[T], g: &mut [T]) {
        match self {
            Some(e) => e.add_energy_gradient(x0, x1, g),
            None => {}
        }
    }
}

impl<E: EnergyHessianTopology> EnergyHessianTopology for Option<E> {
    fn energy_hessian_size(&self) -> usize {
        self.as_ref().map_or(0, |e| e.energy_hessian_size())
    }
    fn num_hessian_diagonal_nnz(&self) -> usize {
        self.as_ref().map_or(0, |e| e.num_hessian_diagonal_nnz())
    }
    fn energy_hessian_indices_offset(
        &self,
        off: MatrixElementIndex,
        indices: &mut [MatrixElementIndex],
    ) {
        match self {
            Some(e) => e.energy_hessian_indices_offset(off, indices),
            None => {}
        }
    }
    // This method is often overloaded so we forward it here explicitly for efficiency.
    fn energy_hessian_rows_cols_offset<I: FromPrimitive + Send + bytemuck::Pod>(
        &self,
        off: MatrixElementIndex,
        rows: &mut [I],
        cols: &mut [I],
    ) {
        match self {
            Some(e) => e.energy_hessian_rows_cols_offset(off, rows, cols),
            None => {}
        }
    }
}

impl<T: Real + Send + Sync, E: EnergyHessian<T>> EnergyHessian<T> for Option<E> {
    fn energy_hessian_values(&self, x0: &[T], x1: &[T], scale: T, vals: &mut [T]) {
        match self {
            Some(e) => e.energy_hessian_values(x0, x1, scale, vals),
            None => {}
        }
    }
}

/// Another energy adapter for combining two different energies.
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum Either<A, B> {
    Left(A),
    Right(B),
}

impl<T: Real, A: Energy<T>, B: Energy<T>> Energy<T> for Either<A, B> {
    fn energy(&self, x0: &[T], x1: &[T]) -> T {
        match self {
            Either::Left(e) => e.energy(x0, x1),
            Either::Right(e) => e.energy(x0, x1),
        }
    }
}

impl<X: Real, T: Real, A: EnergyGradient<X, T>, B: EnergyGradient<X, T>> EnergyGradient<X, T>
    for Either<A, B>
{
    fn add_energy_gradient(&self, x0: &[X], x1: &[T], g: &mut [T]) {
        match self {
            Either::Left(e) => e.add_energy_gradient(x0, x1, g),
            Either::Right(e) => e.add_energy_gradient(x0, x1, g),
        }
    }
}

impl<A: EnergyHessianTopology, B: EnergyHessianTopology> EnergyHessianTopology for Either<A, B> {
    fn energy_hessian_size(&self) -> usize {
        match self {
            Either::Left(e) => e.energy_hessian_size(),
            Either::Right(e) => e.energy_hessian_size(),
        }
    }
    fn num_hessian_diagonal_nnz(&self) -> usize {
        match self {
            Either::Left(e) => e.num_hessian_diagonal_nnz(),
            Either::Right(e) => e.num_hessian_diagonal_nnz(),
        }
    }
    fn energy_hessian_indices_offset(
        &self,
        off: MatrixElementIndex,
        indices: &mut [MatrixElementIndex],
    ) {
        match self {
            Either::Left(e) => e.energy_hessian_indices_offset(off, indices),
            Either::Right(e) => e.energy_hessian_indices_offset(off, indices),
        }
    }
    // This method is often overloaded so we forward it here explicitly for efficiency.
    fn energy_hessian_rows_cols_offset<I: FromPrimitive + Send + bytemuck::Pod>(
        &self,
        off: MatrixElementIndex,
        rows: &mut [I],
        cols: &mut [I],
    ) {
        match self {
            Either::Left(e) => e.energy_hessian_rows_cols_offset(off, rows, cols),
            Either::Right(e) => e.energy_hessian_rows_cols_offset(off, rows, cols),
        }
    }
}

impl<T: Real + Send + Sync, A: EnergyHessian<T>, B: EnergyHessian<T>> EnergyHessian<T>
    for Either<A, B>
{
    fn energy_hessian_values(&self, x0: &[T], x1: &[T], scale: T, vals: &mut [T]) {
        match self {
            Either::Left(e) => e.energy_hessian_values(x0, x1, scale, vals),
            Either::Right(e) => e.energy_hessian_values(x0, x1, scale, vals),
        }
    }
}

#[cfg(test)]
pub(crate) mod test_utils {
    use crate::energy::*;
    use crate::test_utils::*;
    use crate::{TetMesh, TriMesh};
    use approx::*;
    use autodiff::F1 as F;
    use num_traits::Zero;

    pub(crate) fn test_tetmeshes() -> Vec<TetMesh> {
        vec![
            make_one_tet_mesh(),
            make_one_deformed_tet_mesh(),
            make_three_tet_mesh(),
        ]
    }

    pub(crate) fn test_trimeshes() -> Vec<TriMesh> {
        vec![
            make_one_tri_mesh(),
            make_one_deformed_tri_mesh(),
            make_two_tri_mesh(),
            make_three_tri_mesh(),
            make_four_tri_mesh(),
            make_four_tri_mesh_unoriented(),
        ]
    }

    pub(crate) fn test_rigid_trimeshes() -> Vec<TriMesh> {
        vec![make_one_tet_trimesh()]
    }

    fn random_displacement(n: usize) -> Vec<F> {
        use rand::distributions::Uniform;
        use rand::prelude::*;
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

    pub(crate) fn gradient_tester<E>(configurations: Vec<(E, Vec<[f64; 3]>)>, ty: EnergyType)
    where
        E: Energy<F> + EnergyGradient<F, F>,
    {
        for (energy, pos) in configurations.iter() {
            let (x0, mut x1) = autodiff_step(bytemuck::cast_slice(&pos), ty);

            let mut grad = vec![F::zero(); 3 * pos.len()];
            energy.add_energy_gradient(&x0, &x1, &mut grad);

            for i in 0..x0.len() {
                x1[i] = F::var(x1[i]);
                let energy = energy.energy(&x0, &x1);
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

    pub(crate) fn hessian_tester<E>(configurations: Vec<(E, Vec<[f64; 3]>)>, ty: EnergyType)
    where
        E: EnergyGradient<F, F> + EnergyHessian<F>,
    {
        // An arbitrary scale (!=1.0) that will ensure that Hessians are scaled correctly.
        let scale = 0.2;
        use crate::matrix::{MatrixElementIndex as Index, MatrixElementTriplet as Triplet};

        for (energy, pos) in configurations.iter() {
            let (x0, mut x1) = autodiff_step(bytemuck::cast_slice(&pos), ty);

            let mut hess_triplets =
                vec![Triplet::new(0, 0, F::zero()); energy.energy_hessian_size()];
            energy.energy_hessian(&x0, &x1, F::cst(scale), &mut hess_triplets);

            // Build a dense hessian
            let mut hess_ad = vec![vec![0.0; x0.len()]; x0.len()];
            let mut hess = vec![vec![0.0; x0.len()]; x0.len()];
            for Triplet {
                idx: Index { row, col },
                val,
            } in hess_triplets.into_iter()
            {
                hess[row][col] += val.value();
                if row != col {
                    hess[col][row] += val.value();
                }
            }

            let mut success = true;
            for i in 0..x0.len() {
                x1[i] = F::var(x1[i]);
                let mut grad = vec![F::zero(); x0.len()];
                energy.add_energy_gradient(&x0, &x1, &mut grad);
                grad.iter_mut().for_each(|g| *g *= scale);
                for j in 0..x0.len() {
                    let res = relative_eq!(
                        hess[i][j],
                        grad[j].deriv(),
                        max_relative = 1e-6,
                        epsilon = 1e-10
                    );
                    hess_ad[i][j] = grad[j].deriv();
                    if !res {
                        success = false;
                        eprintln!("({}, {}): {} vs. {}", i, j, hess[i][j], grad[j].deriv());
                    }
                }
                x1[i] = F::cst(x1[i]);
            }

            if !success && x0.len() < 15 {
                // Print dense hessian if its small
                eprintln!("Actual:");
                for row in 0..x0.len() {
                    for col in 0..=row {
                        eprint!("{:10.2e}", hess[row][col]);
                    }
                    eprintln!("");
                }

                eprintln!("Expected:");
                for row in 0..x0.len() {
                    for col in 0..=row {
                        eprint!("{:10.2e}", hess_ad[row][col]);
                    }
                    eprintln!("");
                }
            }
            assert!(success);
        }
    }
}
