use crate::Real;

use super::{SolveResult, Status};

/// Implementation of the conjugate residual method.
///
/// The Conjugate Residual method solves the system `Ax = b` where `A` is Hermitian, and `b` is non-zero.
/// https://en.wikipedia.org/wiki/Conjugate_residual_method
#[allow(non_snake_case)]
pub struct ConjugateResidual<T: Real> {
    max_iter: u32,
    tol: f32,
    p: na::DVector<T>,
    Ar: na::DVector<T>,
    Ap: na::DVector<T>,
    ///// Preconditioner.
    //M: Option<tensr::DSMatrix<T>>,
}

impl<T> ConjugateResidual<T>
where
    T: Real,
{
    #[allow(non_snake_case)]
    pub fn new(size: usize, max_iter: u32, tol: f32) -> Self {
        // Allocate p, Ap and Ar.
        let p = na::DVector::zeros(size);
        let Ar = na::DVector::zeros(size);
        let Ap = Ar.clone();

        ConjugateResidual {
            max_iter: u32::MAX.min(max_iter),
            tol: f32::EPSILON.max(tol),
            p,
            Ar,
            Ap,
            //M: None,
        }
    }

    /// Solves `Ax = b` where the product `Ax` is provided by the function `matvec`.
    ///
    /// `matvec` takes in the vector `x` and the output mutable slice. If `matvec` returns
    /// false, then the computation stops and the `SolveResult` will report an interrupted status.
    #[allow(non_snake_case)]
    pub fn solve<F>(&mut self, mut matvec: F, x: &mut [T], b: &mut [T]) -> SolveResult
    where
        F: FnMut(&[T], &mut [T]) -> bool,
    {
        let ConjugateResidual {
            max_iter,
            tol,
            ref mut p,
            ref mut Ar,
            ref mut Ap,
            ..
        } = *self;

        let tol = tol as f64;

        debug_assert_eq!(b.len(), x.len());
        debug_assert_eq!(p.len(), x.len());

        let mut x: na::DVectorViewMut<T> = x.into();
        let mut r: na::DVectorViewMut<T> = b.into();

        let b_norm_sq = r.norm_squared().to_f64().unwrap();

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

        let mut r_norm_sq = r.norm_squared().to_f64().unwrap();

        // Return if the residual norm is already zero.
        if r_norm_sq <= f64::EPSILON * f64::EPSILON {
            let residual = r_norm_sq.sqrt();
            return SolveResult {
                iterations: 0,
                residual,
                error: residual / b_norm_sq.sqrt(),
                status: Status::Success,
            };
        }

        // p0 = r0
        p.as_mut_slice().copy_from_slice(r.as_slice());

        // Initialize Ap0 and Ar0
        matvec(r.as_slice(), Ar.as_mut_slice());
        Ap.as_mut_slice().copy_from_slice(Ar.as_slice());

        let mut rAr = r.dot(Ar);

        let mut iterations = 0;
        loop {
            // α = rAr/ p'A'Ap
            let alpha = rAr / Ap.dot(Ap);

            // x = x + α * p
            x.axpy(alpha, p, T::one());

            // r = r - α * Ap
            r.axpy(-alpha, Ap, T::one());

            iterations += 1;

            // Stop if |r_new| / |r_init| < tol
            // or if the iterations reaches max_iter.
            r_norm_sq = r.norm_squared().to_f64().unwrap();
            if r_norm_sq < tol * tol * b_norm_sq {
                let residual = r_norm_sq.sqrt();
                return SolveResult {
                    iterations,
                    residual,
                    error: residual / b_norm_sq.sqrt(),
                    status: Status::Success,
                };
            } else if iterations >= max_iter {
                let residual = r_norm_sq.sqrt();
                return SolveResult {
                    iterations,
                    residual,
                    error: residual / b_norm_sq.sqrt(),
                    status: Status::MaximumIterationsExceeded,
                };
            }

            // Compute new Ar
            if !matvec(r.as_slice(), Ar.as_mut_slice()) {
                let residual = r_norm_sq.sqrt();
                return SolveResult {
                    iterations,
                    residual,
                    error: residual / b_norm_sq.sqrt(),
                    status: Status::Interrupted,
                };
            }

            let rAr_new = r.dot(Ar);
            let beta = rAr_new / rAr;
            rAr = rAr_new;

            // p = r + βp
            p.axpy(T::one(), &r, beta);

            // Ap = Ar + βAp
            Ap.axpy(T::one(), Ar, beta);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn cr_simple() {
        // Test that CR works with a simple symmetric 2x2 system.
        let mtx = vec![1.0, 2.0, 2.0, 3.0];
        let mut b = vec![5.0, 6.0];
        let mut x = vec![0.0, 0.0];

        let mut cr = ConjugateResidual::new(2, 1000, 1e-4);
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
            f64::abs(x[0] + 3.0) < 1e-3,
            "expected: {}; actual: {}",
            -4.0,
            x[0]
        );
        assert!(
            f64::abs(x[1] - 4.0) < 1e-3,
            "expected: {}; actual: {}",
            4.5,
            x[1]
        );
    }
}
