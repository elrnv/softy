pub mod elasticity;
pub mod gravity;
pub mod inertia;

use crate::energy::*;
use crate::matrix::MatrixElementIndex;
use crate::Real;
use num_traits::FromPrimitive;

/// Define a nullable energy, which is used to represent zero energies. For example
/// a fixed mesh can use this energy in place of elasticity, gravity or inertia.

impl<T: Real, E: Energy<T>> Energy<T> for Option<E> {
    fn energy(&self, x: &[T], v: &[T]) -> T {
        self.as_ref().map_or(T::zero(), |e| e.energy(x, v))
    }
}

impl<X: Real, T: Real, E: EnergyGradient<X, T>> EnergyGradient<X, T> for Option<E> {
    fn add_energy_gradient(&self, x: &[X], v: &[T], g: &mut [T], dqdv: T) {
        match self {
            Some(e) => e.add_energy_gradient(x, v, g, dqdv),
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
    fn energy_hessian_values(&self, x0: &[T], x1: &[T], scale: T, vals: &mut [T], dqdv: T) {
        match self {
            Some(e) => e.energy_hessian_values(x0, x1, scale, vals, dqdv),
            None => {}
        }
    }
    fn add_energy_hessian_diagonal(&self, x0: &[T], x1: &[T], scale: T, diag: &mut [T], dqdv: T) {
        match self {
            Some(e) => e.add_energy_hessian_diagonal(x0, x1, scale, diag, dqdv),
            None => {}
        }
    }
}

/// Either energy adapter for combining two different energies.
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum Either<A, B> {
    Left(A),
    Right(B),
}

impl<T: Real, A: Energy<T>, B: Energy<T>> Energy<T> for Either<A, B> {
    fn energy(&self, x: &[T], v: &[T]) -> T {
        match self {
            Either::Left(e) => e.energy(x, v),
            Either::Right(e) => e.energy(x, v),
        }
    }
}

impl<X: Real, T: Real, A: EnergyGradient<X, T>, B: EnergyGradient<X, T>> EnergyGradient<X, T>
    for Either<A, B>
{
    fn add_energy_gradient(&self, x: &[X], v: &[T], g: &mut [T], dqdv: T) {
        match self {
            Either::Left(e) => e.add_energy_gradient(x, v, g, dqdv),
            Either::Right(e) => e.add_energy_gradient(x, v, g, dqdv),
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
    fn energy_hessian_values(&self, x: &[T], v: &[T], scale: T, vals: &mut [T], dqdv: T) {
        match self {
            Either::Left(e) => e.energy_hessian_values(x, v, scale, vals, dqdv),
            Either::Right(e) => e.energy_hessian_values(x, v, scale, vals, dqdv),
        }
    }
    fn add_energy_hessian_diagonal(&self, x: &[T], v: &[T], scale: T, diag: &mut [T], dqdv: T) {
        match self {
            Either::Left(e) => e.add_energy_hessian_diagonal(x, v, scale, diag, dqdv),
            Either::Right(e) => e.add_energy_hessian_diagonal(x, v, scale, diag, dqdv),
        }
    }
}

/*
 * Energy implementation for a tuple of energies.
 */

impl<T: Real, A: Energy<T>, B: Energy<T>> Energy<T> for (A, B) {
    fn energy(&self, x: &[T], v: &[T]) -> T {
        self.0.energy(x, v) + self.1.energy(x, v)
    }
}

impl<X: Real, T: Real, A: EnergyGradient<X, T>, B: EnergyGradient<X, T>> EnergyGradient<X, T>
    for (A, B)
{
    fn add_energy_gradient(&self, x: &[X], v: &[T], g: &mut [T], dqdv: T) {
        self.0.add_energy_gradient(x, v, g, dqdv);
        self.1.add_energy_gradient(x, v, g, dqdv);
    }
}

impl<A: EnergyHessianTopology, B: EnergyHessianTopology> EnergyHessianTopology for (A, B) {
    fn energy_hessian_size(&self) -> usize {
        self.0.energy_hessian_size() + self.1.energy_hessian_size()
    }
    fn num_hessian_diagonal_nnz(&self) -> usize {
        self.0.num_hessian_diagonal_nnz() + self.1.num_hessian_diagonal_nnz()
    }
    fn energy_hessian_indices_offset(
        &self,
        off: MatrixElementIndex,
        indices: &mut [MatrixElementIndex],
    ) {
        self.0
            .energy_hessian_indices_offset(off, &mut indices[..self.0.energy_hessian_size()]);
        self.1
            .energy_hessian_indices_offset(off, &mut indices[self.0.energy_hessian_size()..]);
    }
    // This method is often overloaded so we forward it here explicitly for efficiency.
    fn energy_hessian_rows_cols_offset<I: FromPrimitive + Send + bytemuck::Pod>(
        &self,
        off: MatrixElementIndex,
        rows: &mut [I],
        cols: &mut [I],
    ) {
        self.0.energy_hessian_rows_cols_offset(
            off,
            &mut rows[..self.0.energy_hessian_size()],
            &mut cols[..self.0.energy_hessian_size()],
        );
        self.1.energy_hessian_rows_cols_offset(
            off,
            &mut rows[self.0.energy_hessian_size()..],
            &mut cols[self.0.energy_hessian_size()..],
        );
    }
}

impl<T: Real + Send + Sync, A: EnergyHessian<T>, B: EnergyHessian<T>> EnergyHessian<T> for (A, B) {
    fn energy_hessian_values(&self, x: &[T], v: &[T], scale: T, vals: &mut [T], dqdv: T) {
        self.0
            .energy_hessian_values(x, v, scale, &mut vals[..self.0.energy_hessian_size()], dqdv);
        self.1
            .energy_hessian_values(x, v, scale, &mut vals[self.0.energy_hessian_size()..], dqdv);
    }
    fn add_energy_hessian_diagonal(&self, x: &[T], v: &[T], scale: T, diag: &mut [T], dqdv: T) {
        self.0.add_energy_hessian_diagonal(x, v, scale, diag, dqdv);
        self.1.add_energy_hessian_diagonal(x, v, scale, diag, dqdv);
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

    #[cfg(feature = "optsolver")]
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
        #[cfg(feature = "optsolver")]
        Velocity,
    }

    /// Construct a step in autodiff format given a set of initial positions. This function returns
    /// initial and current values for the independent variable determined by `ty`.
    fn autodiff_step(pos: &[f64], ty: EnergyType, dt: f64) -> (Vec<F>, Vec<F>, Vec<F>) {
        #[cfg(feature = "optsolver")]
        let dx0 = vec![F::cst(0.0); pos.len()]; // previous disp
        let dx = random_displacement(pos.len()); // current disp
        let x0: Vec<F> = pos.iter().map(|&x| F::cst(x)).collect();
        let x: Vec<F> = x0
            .iter()
            .zip(dx.iter())
            .map(|(&x, &dx)| F::cst(x + dt * dx))
            .collect();

        match ty {
            EnergyType::Position => (x0, x, dx),
            #[cfg(feature = "optsolver")]
            EnergyType::Velocity => (x0, dx0, dx),
        }
    }

    pub(crate) fn gradient_tester<E>(configurations: Vec<(E, &[[f64; 3]])>, ty: EnergyType)
    where
        E: Energy<F> + EnergyGradient<F, F>,
    {
        let dqdv = 1.0;
        for (energy, pos) in configurations.iter() {
            let (x0, mut x, mut dx) = autodiff_step(bytemuck::cast_slice(&pos), ty, dqdv);

            let mut grad = vec![F::zero(); 3 * pos.len()];
            energy.add_energy_gradient(&x, &dx, &mut grad, F::cst(dqdv));

            for i in 0..x.len() {
                dx[i] = F::var(dx[i]);
                x[i] = x0[i] + dx[i] * F::cst(dqdv);
                let energy = energy.energy(&x, &dx);
                assert_relative_eq!(
                    grad[i].value(),
                    energy.deriv(),
                    max_relative = 1e-6,
                    epsilon = 1e-10
                );

                dx[i] = F::cst(dx[i]);
                x[i] = F::cst(x[i]);
            }
        }
    }

    pub(crate) fn hessian_tester<E>(configurations: Vec<(E, &[[f64; 3]])>, ty: EnergyType)
    where
        E: EnergyGradient<F, F> + EnergyHessian<F>,
    {
        let dt = 1.0;

        // An arbitrary scale (!=1.0) that will ensure that Hessians are scaled correctly.
        let scale = 0.2;
        use crate::matrix::{MatrixElementIndex as Index, MatrixElementTriplet as Triplet};

        for (energy, pos) in configurations.iter() {
            let (x0, mut x, mut dx) = autodiff_step(bytemuck::cast_slice(&pos), ty, dt);

            let mut hess_triplets =
                vec![Triplet::new(0, 0, F::zero()); energy.energy_hessian_size()];
            energy.energy_hessian(&x, &dx, F::cst(scale), &mut hess_triplets, F::cst(dt));

            // Tests the energy hessian diagonal which is used for preconditioners.
            let mut hess_diagonal = vec![F::zero(); dx.len()];
            energy.add_energy_hessian_diagonal(
                &x,
                &dx,
                F::cst(scale),
                &mut hess_diagonal,
                F::cst(dt),
            );

            // Build a dense hessian
            let mut hess_ad = vec![vec![0.0; x.len()]; x.len()];
            let mut hess = vec![vec![0.0; x.len()]; x.len()];
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

            // Test diagonal
            for idx in 0..x.len() {
                assert_relative_eq!(
                    hess_diagonal[idx].value(),
                    hess[idx][idx],
                    max_relative = 1e-6,
                    epsilon = 1e-10
                );
            }

            let mut success = true;
            for i in 0..x.len() {
                dx[i] = F::var(dx[i]);
                x[i] = x0[i] + dx[i] * dt;
                let mut grad = vec![F::zero(); x.len()];
                energy.add_energy_gradient(&x, &dx, &mut grad, F::cst(1.0));
                grad.iter_mut().for_each(|g| *g *= scale);
                for j in 0..x.len() {
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
                dx[i] = F::cst(dx[i]);
                x[i] = F::cst(x[i]);
            }

            if !success && x.len() < 15 {
                // Print dense hessian if its small
                eprintln!("Actual:");
                for row in 0..x.len() {
                    for col in 0..=row {
                        eprint!("{:10.2e}", hess[row][col]);
                    }
                    eprintln!();
                }

                eprintln!("Expected:");
                for row in 0..x.len() {
                    for col in 0..=row {
                        eprint!("{:10.2e}", hess_ad[row][col]);
                    }
                    eprintln!();
                }
            }
            assert!(success);
        }
    }
}
