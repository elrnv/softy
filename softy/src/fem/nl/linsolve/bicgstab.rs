use crate::Real;

use super::{SolveResult, Status};

trait Axpy<T: Real, R> {
    fn axpy(&mut self, x: &R, a: T);
    fn axbzpy(&mut self, x: &R, a: T, z: &R, b: T) {
        self.axpy(x, a);
        self.axpy(z, b);
    }
}

impl<'a, T: Real> Axpy<T, na::DVectorSlice<'a, T>> for na::DVector<T> {
    fn axpy(&mut self, x: &na::DVectorSlice<'a, T>, a: T) {
        self.axpy(a, x, T::one());
    }
}

impl<'a, T: Real> Axpy<T, na::DVectorSlice<'a, T>> for na::DVectorSliceMut<'_, T> {
    fn axpy(&mut self, x: &na::DVectorSlice<'a, T>, a: T) {
        self.axpy(a, x, T::one());
    }
}

#[cfg(feature = "arrayfire")]
impl<T: Real> Axpy<T, af::Array<T>> for af::Array<T> {
    fn axpy(&mut self, x: &af::Array<T>, a: T) {
        self += a * x;
    }
}

/// Implementation of the BiConjugate Gradient Stabilized (BiCGStab) algorithm for
/// non-symmetric indefinite linear systems.
///
/// The BiCGSTAB method solves the system `Ax = b` where `A` is Hermitian, and `b` is non-zero.
/// https://en.wikipedia.org/wiki/Biconjugate_gradient_stabilized_method
#[allow(non_snake_case)]
pub struct BiCGSTAB<V> {
    pub max_iter: u32,
    pub tol: f32,
    r0s: V,
    /// Used to restart the algorithm when the r0s becomes too orthogonal to r.
    rhs: V,
    p: V,
    t: V,
    v: V,
    y: V,
    z: V,
}

impl<T> BiCGSTAB<na::DVector<T>>
where
    T: Real,
{
    #[allow(non_snake_case)]
    #[inline]
    pub fn new(size: usize, max_iter: u32, tol: f32) -> Self {
        let r0s = na::DVector::zeros(size);
        let rhs = r0s.clone();
        let p = r0s.clone();
        let t = r0s.clone();
        let v = r0s.clone();
        let y = r0s.clone();
        let z = r0s.clone();

        BiCGSTAB {
            max_iter: u32::MAX.min(max_iter),
            tol: f32::EPSILON.max(tol),
            r0s,
            rhs,
            p,
            t,
            v,
            y,
            z,
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
        self.solve_precond(matvec, x, b, |input, _| input)
    }

    /// Solves `Ax = b` where the product `Ax` is provided by the function `matvec`.
    ///
    /// `matvec` takes in the vector `x` and the output mutable slice. If `matvec` returns
    /// false, then the computation stops and the `SolveResult` will report an interrupted status.
    ///
    /// `precond_solve` applies the right preconditioner inverse to the given vector, i.e. it computes
    /// `K^{-1}_2 p` where `p` is the given vector and `K_2` is the right
    /// preconditioner.
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
        P: for<'a> FnMut(&'a [T], &'a mut [T]) -> &'a [T],
    {
        let BiCGSTAB {
            max_iter,
            tol,
            ref mut r0s,
            ref mut p,
            ref mut rhs,
            ref mut t,
            ref mut v,
            ref mut y,
            ref mut z,
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

        // r0 = b - p = b - Ax0
        r.axpy(-T::one(), p, T::one());

        // Choose r0s arbitrary such that (r0s, r) != 0. We chose r0s = r.
        r0s.as_mut_slice().copy_from_slice(r.as_slice());

        let mut r0s_norm_sq = r0s.norm_squared().to_f64().unwrap();
        log::trace!("r0s norm sq: {:?}", r0s_norm_sq);

        // p0 = r0
        p.as_mut_slice().copy_from_slice(r.as_slice());

        // v0 = p0 = 0
        // p.as_mut_slice().fill(T::zero());
        // v.as_mut_slice().fill(T::zero());

        let mut y_slice = precond_solve(p.as_slice(), y.as_mut_slice());

        // Initialize v0 = Ay
        matvec(y_slice, v.as_mut_slice());

        // Initialize temporaries.
        let mut rho = T::one();
        let mut alpha; // = T::one();
        let mut w; // = T::one();

        let mut iterations = 1;

        // Check residual size, if small return result, otherwise return nothing.
        let check_residual =
            |r_norm_sq: f64, b_norm_sq: f64, iterations: u32| -> Option<SolveResult> {
                if r_norm_sq <= tol_sq {
                    let residual = r_norm_sq.sqrt();
                    Some(SolveResult {
                        iterations,
                        residual,
                        error: residual / b_norm_sq.sqrt(),
                        status: Status::Success,
                    })
                } else if r_norm_sq.is_nan() || !r_norm_sq.is_finite() {
                    Some(SolveResult {
                        iterations,
                        residual: num_traits::Float::infinity(),
                        error: num_traits::Float::infinity(),
                        status: Status::NanDetected,
                    })
                } else {
                    None
                }
            };

        // Reset r0s if needed.
        let reset_r0s = |x: &na::DVectorSliceMut<T>,
                         r_norm_sq: f64,
                         r: &mut na::DVectorSliceMut<T>,
                         r0s_norm_sq: &mut f64,
                         r0s: &mut na::DVector<T>,
                         rho_new: &mut T,
                         iterations: u32,
                         matvec: &mut F|
         -> Option<SolveResult> {
            if rho_new.to_f64().unwrap().abs() <= eps_sq * *r0s_norm_sq {
                // Compute r = A*x
                if !matvec(x.as_slice(), r.as_mut_slice()) {
                    let residual = r_norm_sq.sqrt();
                    return Some(SolveResult {
                        iterations,
                        residual,
                        error: residual / b_norm_sq.sqrt(),
                        status: Status::Interrupted,
                    });
                }

                // r = b - A*x
                r.axpy(T::one(), rhs, -T::one());
                r0s.as_mut_slice().copy_from_slice(r.as_slice());
                *rho_new = r0s.norm_squared();
                *r0s_norm_sq = rho_new.to_f64().unwrap();
                log::trace!(
                    "r too orthogonal to r0s, restarting with rho = {:?}",
                    rho_new
                );
            }
            None
        };

        let mut r_norm_sq = r.norm_squared().to_f64().unwrap();

        loop {
            let mut r0sTv = r0s.dot(v);
            log::trace!("r0sTv = {:?}", r0sTv);

            // Restart with a different r0s if v becomes orthogonal to r0s.
            if r0sTv.to_f64().unwrap().abs() <= eps_sq * r0s_norm_sq {
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
                //r.axpy(T::one(), rhs, -T::one());
                r0s.as_mut_slice().copy_from_slice(v.as_slice());
                r0sTv = r0s.norm_squared();
                r0s_norm_sq = r0sTv.to_f64().unwrap();
                log::trace!(
                    "v too orthogonal to r0s, restarting with r0sTv = {:?}",
                    r0sTv
                );
            }

            // α = rho / r0s'v
            alpha = rho / r0sTv;
            log::trace!("alpha = {:?}", alpha);

            // s = r - α * v
            r.axpy(-alpha, v, T::one());

            r_norm_sq = r.norm_squared().to_f64().unwrap();
            if let Some(result) = check_residual(r_norm_sq, b_norm_sq, iterations) {
                x.axpy(alpha, &y_slice.into(), T::one());
                break result;
            }

            // z = K^{-1} s (reusing r for s)
            let zn: na::DVectorSlice<T> = precond_solve(r.as_slice(), z.as_mut_slice()).into();

            // Compute new t = Az
            if !matvec(zn.as_slice(), t.as_mut_slice()) {
                let residual = r.norm().to_f64().unwrap();
                break SolveResult {
                    iterations,
                    residual,
                    error: residual / b_norm_sq.sqrt(),
                    status: Status::Interrupted,
                };
            }

            // ω = s't/t't

            let t_norm_sq = t.dot(&t);

            // Gracefully handle degenerate case.
            w = if t_norm_sq > T::zero() {
                r.dot(&t) / t_norm_sq
            } else {
                let residual = r_norm_sq.sqrt();
                break SolveResult {
                    iterations,
                    residual,
                    error: residual / b_norm_sq.sqrt(),
                    status: Status::SingularMatrix,
                };
            };

            log::trace!("w = {:?}", w);

            // x = x + αy + ωz
            x.axpy(alpha, &y_slice.into(), T::one());
            x.axpy(w, &zn, T::one());

            // r = s - w * t
            r.axpy(-w, t, T::one());

            // Stop if |r_new| / |r_init| < tol
            // or if the iterations reaches max_iter.
            r_norm_sq = r.norm_squared().to_f64().unwrap();
            log::trace!("r norm sq ratio: {:?}", r_norm_sq / b_norm_sq);

            if let Some(result) = check_residual(r_norm_sq, b_norm_sq, iterations) {
                break result;
            }

            if iterations >= max_iter {
                let residual = r_norm_sq.sqrt();
                break SolveResult {
                    iterations,
                    residual,
                    error: residual / b_norm_sq.sqrt(),
                    status: Status::MaximumIterationsExceeded,
                };
            }

            let mut rho_new = r0s.dot(&r);
            log::trace!(
                "rho = {:?} (eps_sq * r0s^2 = {:?})",
                rho_new,
                eps_sq * r0s_norm_sq
            );

            // Restart with a different r0s if r becomes orthogonal to r0s.
            if let Some(result) = reset_r0s(
                &x,
                r_norm_sq,
                &mut r,
                &mut r0s_norm_sq,
                &mut *r0s,
                &mut rho_new,
                iterations,
                &mut matvec,
            ) {
                break result;
            }

            // β = (rho_i / rho_{i-1})(α/w_{i-1})
            let beta = (rho_new / rho) * (alpha / w);
            log::trace!("beta = {:?}", beta);
            rho = rho_new;

            // p = r_{i-1} + β(p_{i-1} - w_{i-1} v_{i-1})
            p.axpy(-w, v, T::one());
            p.axpy(T::one(), &r, beta);

            // y = K^{-1} p_i where K^{-1} = K^{-1}_2 K^{-1}_1 is the inverse preconditioner product
            // of both right and left inverse preconditioners respectively.
            y_slice = precond_solve(p.as_slice(), y.as_mut_slice());

            // Compute new v
            if !matvec(y_slice, v.as_mut_slice()) {
                let residual = r_norm_sq.sqrt();
                break SolveResult {
                    iterations,
                    residual,
                    error: residual / b_norm_sq.sqrt(),
                    status: Status::Interrupted,
                };
            }

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
