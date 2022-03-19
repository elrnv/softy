use crate::Real;

use super::{SolveResult, Status};

/// Implementation of the BiConjugate Gradient STABilized (BiCGSTAB) algorithm for
/// non-symmetric indefinite linear systems.
///
/// The BiCGSTAB method solves the system `Ax = b` where `A` is Hermitian, and `b` is non-zero.
/// https://en.wikipedia.org/wiki/Biconjugate_gradient_stabilized_method
#[allow(non_snake_case)]
pub struct BiCGSTAB<T: Real> {
    pub max_iter: u32,
    pub tol: f32,
    r0s: na::DVector<T>,
    /// Used to restart the algorithm when the r0s becomes too orthogonal to r.
    rhs: na::DVector<T>,
    p: na::DVector<T>,
    Ar: na::DVector<T>,
    Ap: na::DVector<T>,
}

impl<T> BiCGSTAB<T>
where
    T: Real,
{
    #[allow(non_snake_case)]
    #[inline]
    pub fn new(size: usize, max_iter: u32, tol: f32) -> Self {
        let r0s = na::DVector::zeros(size);
        let rhs = na::DVector::zeros(size);
        let p = na::DVector::zeros(size);
        let Ar = na::DVector::zeros(size);
        let Ap = Ar.clone();

        BiCGSTAB {
            max_iter: u32::MAX.min(max_iter),
            tol: f32::EPSILON.max(tol),
            r0s,
            rhs,
            p,
            Ar,
            Ap,
        }
    }

    /// Solves `Ax = b` where the product `Ax` is provided by the function `matvec`.
    ///
    /// `matvec` takes in the vector `x` and the output mutable slice. If `matvec` returns
    /// false, then the computation stops and the `SolveResult` will report an interrupted status.
    #[inline]
    pub fn solve<F>(&mut self, matvec: F, x: &mut [T], b: &mut [T]) -> SolveResult
    where
        F: FnMut(&[T], &mut [T]) -> bool,
    {
        self.solve_precond(matvec, x, b, |_| true)
    }

    /// Solves `Ax = b` where the product `Ax` is provided by the function `matvec`.
    ///
    /// `matvec` takes in the vector `x` and the output mutable slice. If `matvec` returns
    /// false, then the computation stops and the `SolveResult` will report an interrupted status.
    #[allow(non_snake_case)]
    pub fn solve_precond<F, P>(
        &mut self,
        mut matvec: F,
        x: &mut [T],
        b: &mut [T],
        mut precond_solve: P,
    ) -> SolveResult
    where
        F: FnMut(&[T], &mut [T]) -> bool,
        P: FnMut(&mut [T]) -> bool,
    {
        let BiCGSTAB {
            max_iter,
            tol,
            ref mut r0s,
            ref mut p,
            ref mut rhs,
            ref mut Ar,
            ref mut Ap,
            ..
        } = *self;

        debug_assert_eq!(b.len(), x.len());
        debug_assert_eq!(p.len(), x.len());

        rhs.as_mut_slice().copy_from_slice(b);
        let b_norm_sq = rhs.norm_squared().to_f64().unwrap();

        let mut x: na::DVectorSliceMut<T> = x.into();
        let mut r: na::DVectorSliceMut<T> = b.into();

        // Return if b is zero --- the solution is trivial.
        if b_norm_sq == 0.0 {
            x.fill(T::zero());
            return SolveResult {
                iterations: 0,
                residual: 0.0,
                error: 0.0,
                status: Status::Success,
            };
        }

        let tol_sq = b_norm_sq * (tol * tol) as f64;
        let eps_sq = f64::EPSILON * f64::EPSILON;

        // Compute p = A*x0
        if !matvec(x.as_slice(), p.as_mut_slice()) {
            return SolveResult {
                iterations: 0,
                residual: 0.0, // Residual is unknown at this point.
                error: 0.0,
                status: Status::Interrupted,
            };
        }

        // r0 = b - p
        r.axpy(-T::one(), p, T::one());

        // Choose r0s arbitrary such that (r0s, r) != 0. We chose r0s = r.
        r0s.as_mut_slice().copy_from_slice(r.as_slice());

        let mut r0s_norm_sq = r0s.norm_squared().to_f64().unwrap();

        // p0 = r0
        p.as_mut_slice().copy_from_slice(r.as_slice());

        // Initialize Ap0
        matvec(p.as_slice(), Ap.as_mut_slice());

        // Initialize temporaries.
        let mut rho = T::one();
        let mut alpha = T::one();
        let mut w = T::one();

        let mut iterations = 0;
        loop {
            // Stop if |r_new| / |r_init| < tol
            // or if the iterations reaches max_iter.
            let r_norm_sq = r.norm_squared().to_f64().unwrap();
            log::trace!("r norm sq ratio: {:?}", r_norm_sq / b_norm_sq);
            if r_norm_sq <= tol_sq {
                let residual = r_norm_sq.sqrt();
                break SolveResult {
                    iterations,
                    residual,
                    error: residual / b_norm_sq.sqrt(),
                    status: Status::Success,
                };
            } else if iterations >= max_iter {
                let residual = r_norm_sq.sqrt();
                break SolveResult {
                    iterations,
                    residual,
                    error: residual / b_norm_sq.sqrt(),
                    status: Status::MaximumIterationsExceeded,
                };
            }

            // β = (r0sr_new/r0s'r) * (α/w)
            let mut rho_new = r0s.dot(&r);
            log::trace!("rho = {:?}", rho_new);

            // Restart with a different r0s if r becomes orthogonal to r0s.
            if rho_new.to_f64().unwrap().abs() <= eps_sq * r0s_norm_sq {
                // Compute r = A*x
                if !matvec(x.as_slice(), r.as_mut_slice()) {
                    let residual = r_norm_sq.sqrt();
                    return SolveResult {
                        iterations,
                        residual,
                        error: residual / b_norm_sq.sqrt(),
                        status: Status::Interrupted,
                    };
                }

                // r = b - A*x
                r.axpy(T::one(), rhs, -T::one());
                r0s.as_mut_slice().copy_from_slice(r.as_slice());
                rho_new = r0s.norm_squared();
                r0s_norm_sq = rho_new.to_f64().unwrap();
                log::trace!(
                    "r too orthogonal to r0s, restarting with rho = {:?}",
                    rho_new
                );
            }

            let beta = (rho_new / rho) * (alpha / w);
            log::trace!("beta = {:?}", beta);
            rho = rho_new;

            // p = r + β(p - wAp)
            p.axpy(-w, Ap, T::one());
            p.axpy(T::one(), &r, beta);

            if !precond_solve(p.as_mut_slice()) {
                let residual = r_norm_sq.sqrt();
                break SolveResult {
                    iterations,
                    residual,
                    error: residual / b_norm_sq.sqrt(),
                    status: Status::InterruptedPreconditionerSolve,
                };
            }

            // Compute new Ap
            if !matvec(p.as_slice(), Ap.as_mut_slice()) {
                let residual = r_norm_sq.sqrt();
                break SolveResult {
                    iterations,
                    residual,
                    error: residual / b_norm_sq.sqrt(),
                    status: Status::Interrupted,
                };
            }

            let mut r0sAp = r0s.dot(Ap);
            log::trace!("r0sAp = {:?}", r0sAp);

            // Restart with a different r0s if Ap becomes orthogonal to r0s.
            if r0sAp.to_f64().unwrap().abs() <= eps_sq * r0s_norm_sq {
                // Compute r = A*x
                if !matvec(x.as_slice(), r.as_mut_slice()) {
                    let residual = r_norm_sq.sqrt();
                    return SolveResult {
                        iterations,
                        residual,
                        error: residual / b_norm_sq.sqrt(),
                        status: Status::Interrupted,
                    };
                }

                // r = b - A*x
                r.axpy(T::one(), rhs, -T::one());
                r0s.as_mut_slice().copy_from_slice(r.as_slice());
                r0sAp = r0s.norm_squared();
                r0s_norm_sq = r0sAp.to_f64().unwrap();
                log::trace!(
                    "Ap too orthogonal to r0s, restarting with r0sAp = {:?}",
                    r0sAp
                );
            }

            // α = r0s'r/ r0s'Ap
            alpha = rho / r0sAp;
            log::trace!("alpha = {:?}", alpha);

            // s = r - α * Ap
            r.axpy(-alpha, Ap, T::one());

            if !precond_solve(r.as_mut_slice()) {
                let residual = r_norm_sq.sqrt();
                break SolveResult {
                    iterations,
                    residual,
                    error: residual / b_norm_sq.sqrt(),
                    status: Status::InterruptedPreconditionerSolve,
                };
            }

            // Compute new As
            if !matvec(r.as_slice(), Ar.as_mut_slice()) {
                let residual = r.norm().to_f64().unwrap();
                break SolveResult {
                    iterations,
                    residual,
                    error: residual / b_norm_sq.sqrt(),
                    status: Status::Interrupted,
                };
            }

            // ω = sAs/s'A'As

            let Ar_norm_sq = Ar.norm_squared();

            // Gracefully handle degenerate case.
            w = if Ar_norm_sq > T::zero() {
                r.dot(&Ar) / Ar_norm_sq
            } else {
                T::zero()
            };

            log::trace!("w = {:?}", w);

            // x = x + αp + ωs
            x.axpy(alpha, p, T::one());
            x.axpy(w, &r, T::one());

            // r = s - w * Ar
            r.axpy(-w, Ar, T::one());

            iterations += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn bicgstab_simple() {
        // Test that BiCGSTAB works with a simple 2x2 system.
        let mtx = vec![1.0, 2.0, 3.0, 4.0];
        let mut b = vec![5.0, 6.0];
        let mut x = vec![0.0, 0.0];

        let mut cr = BiCGSTAB::new(2, 1000, 1e-4);
        let result = cr.solve(
            |x, out| {
                out[0] = mtx[0] * x[0] + mtx[1] * x[1];
                out[1] = mtx[2] * x[0] + mtx[3] * x[1];
                true
            },
            x.as_mut_slice(),
            b.as_mut_slice(),
        );

        std::dbg!(result);

        assert!(
            f64::abs(x[0] + 4.0) < 1e-3,
            "expected: {}; actual: {}",
            -4.0,
            x[0]
        );
        assert!(
            f64::abs(x[1] - 4.5) < 1e-3,
            "expected: {}; actual: {}",
            4.5,
            x[1]
        );
    }
}
