use crate::geo::Real;
use crate::geo::math::{Vector3, Matrix3};

/// Enumerate all implemented kernels. This is useful for switching between kernels dynamically.
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum KernelType {
    Interpolating { radius: f64 },
    Approximate { radius: f64, tolerance: f64 },
    Cubic { radius: f64 },
    Global { tolerance: f64 },
    Hrbf,
}

/// This kernel trait defines a 1D basis kernel interface.
pub trait Kernel<T: Real> {
    /// Main kernel function evaluated at `x`.
    fn f(&self, x: T) -> T;
    /// First derivative of the kernel evaluated at `x`.
    fn df(&self, x: T) -> T;
    /// Second derivative of the kernel evaluated at `x`.
    fn ddf(&self, x: T) -> T;
}

/// Global kernel with falloff proportional to inverse distance squared.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct GlobalInvDistance2 {
    epsilon: f64,
}

impl GlobalInvDistance2 {
    pub fn new(tolerance: f64) -> Self {
        GlobalInvDistance2 { epsilon: tolerance }
    }
}

impl<T: Real> Kernel<T> for GlobalInvDistance2 {
    fn f(&self, x: T) -> T {
        let eps = T::from(self.epsilon).unwrap();
        let w = T::one() / (x * x + eps * eps);
        w * w
    }
    fn df(&self, x: T) -> T {
        T::from(self.f(autodiff::F::var(x)).deriv()).unwrap()
    }

    fn ddf(&self, x: T) -> T {
        T::from(self.df(autodiff::F::var(x)).deriv()).unwrap()
    }
}

/// Local cubic kernel with compact support. This kernel is non-interpolating but is very simple
/// and fast to compute.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct LocalCubic {
    radius: f64,
}

impl LocalCubic {
    pub fn new(radius: f64) -> Self {
        LocalCubic { radius }
    }
}

impl<T: Real> Kernel<T> for LocalCubic {
    fn f(&self, x: T) -> T {
        let r = T::from(self.radius).unwrap();
        if x > r {
            return T::zero();
        }

        let _2 = T::from(2.0).unwrap();
        let _3 = T::from(3.0).unwrap();

        T::one() - _3 * x * x / (r * r) + _2 * x * x * x / (r * r * r)
    }
    fn df(&self, x: T) -> T {
        let r = T::from(self.radius).unwrap();

        if x > r {
            return T::zero();
        }

        let _6 = T::from(6.0).unwrap();

         _6*(x*x / ( r * r * r ) - x / ( r * r ))
    }

    fn ddf(&self, x: T) -> T {
        let r = T::from(self.radius).unwrap();

        if x > r {
            return T::zero();
        }

        let _12 = T::from(12.0).unwrap();
        let _6 = T::from(6.0).unwrap();

         _12 * x / ( r * r * r ) - _6 / ( r * r )
    }
}

/// Local interpolating kernel. This kernel is exactly interpolating but it suffers from smoothness
/// artifacts. Note that `closest_d` represents the distance to the closest neighbour, which means
/// it must be manually updated before evaluating the kernel.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct LocalInterpolating {
    radius: f64,
    closest_d: f64,
}

impl LocalInterpolating {
    pub fn new(radius: f64) -> Self {
        LocalInterpolating {
            radius,
            closest_d: radius,
        }
    }
}

impl<T: Real> Kernel<T> for LocalInterpolating {
    fn f(&self, x: T) -> T {
        let r = T::from(self.radius).unwrap();
        let xc = T::from(self.closest_d).unwrap();
        if x > r {
            return T::zero();
        }

        let envelope = LocalCubic::new(self.radius).f(x);

        let s = x / r;
        let sc = xc / r;
        envelope * sc * sc * (T::one() / (s * s) - T::one())
    }
    fn df(&self, x: T) -> T {
        T::from(self.f(autodiff::F::var(x)).deriv()).unwrap()
    }

    fn ddf(&self, x: T) -> T {
        T::from(self.df(autodiff::F::var(x)).deriv()).unwrap()
    }
}

/// This kernel is a compromise between the cubic and interpolating kernels. This kernel is fairly
/// cheap to compute and has flexible smoothness properties, which are controllable using the
/// `tolerance` parameter.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct LocalApproximate {
    radius: f64,
    tolerance: f64,
}

impl LocalApproximate {
    pub fn new(radius: f64, tolerance: f64) -> Self {
        LocalApproximate { radius, tolerance }
    }
}

impl<T: Real> Kernel<T> for LocalApproximate {
    fn f(&self, x: T) -> T {
        let r = T::from(self.radius).unwrap();

        if x > r {
            return T::zero();
        }

        let eps = T::from(self.tolerance).unwrap();

        let _2 = T::from(2.0).unwrap();

        let d = x / r;
        let ddeps = T::one() / (d * d + eps);
        let eps1 = T::one() + eps;
        let eps1_2 = eps1*eps1;
        let factor = eps*eps*eps1_2 / ( T::one() + _2 * eps );
        factor * (ddeps * ddeps - T::one() / eps1_2)
    }
    fn df(&self, x: T) -> T {
        let r = T::from(self.radius).unwrap();

        if x > r {
            return T::zero();
        }

        let eps = T::from(self.tolerance).unwrap();

        let _2 = T::from(2.0).unwrap();
        let _4 = T::from(4.0).unwrap();

        let d = x/r;
        let eps1 = T::one() + eps;
        let eps1_2 = eps1*eps1;
        let factor = eps*eps*eps1_2 / ( T::one() + _2 * eps );
        let d2_eps = d * d + eps;
        - factor * _4 * d / ( d2_eps * d2_eps * d2_eps )
    }

    fn ddf(&self, x: T) -> T {
        let r = T::from(self.radius).unwrap();

        if x > r {
            return T::zero();
        }

        let eps = T::from(self.tolerance).unwrap();

        let _2 = T::from(2.0).unwrap();
        let _4 = T::from(4.0).unwrap();
        let _6 = T::from(6.0).unwrap();

        let d = x/r;
        let eps1 = T::one() + eps;
        let eps1_2 = eps1*eps1;
        let factor = eps*eps*eps1_2 / ( T::one() + _2 * eps );
        let d2_eps = d * d + eps;
        let d2_eps4 = d2_eps * d2_eps * d2_eps * d2_eps;
        factor * _4 * ( _6 * d * d - d2_eps ) / d2_eps4
    }
}

/// This kernel trait defines a radial basis kernel interface.
pub trait SphericalKernel<T: Real>: Kernel<T> {
    /// Main kernel function evaluated at `x` with center at `p`.
    #[inline]
    fn eval(&self, x: Vector3<T>, p: Vector3<T>) -> T {
        Kernel::f(self, (x-p).norm())
    }
    /// First derivative wrt `x` of the kernel evaluated at `x` with center at `p`.
    /// To compute the derivatives wrt `p`, simply negate this derivative.
    #[inline]
    fn grad(&self, x: Vector3<T>, p: Vector3<T>) -> Vector3<T> {
        let diff: Vector3<T> = x - p;
        let norm = diff.norm();
        if norm > T::zero() {
            diff * (Kernel::df(self, norm) / norm)
        } else {
            Vector3::zeros()
        }
    }
    /// Second derivative wrt `x` of the kernel evaluated at `x` with center at `p`.
    /// To compute the derivatives wrt `p`, simply negate this derivative.
    #[inline]
    fn hess(&self, x: Vector3<T>, p: Vector3<T>) -> Matrix3<T> {
        let diff = x - p;
        let dot = diff.dot(diff);
        let norm = dot.sqrt();
        let norm_inv = T::one() / norm;
        let norm_inv3 = norm_inv*norm_inv*norm_inv;
        let identity = Matrix3::identity();
        (identity * norm_inv -  diff * (diff.transpose() * norm_inv3)) * Kernel::ddf(self, norm)
    }

    /// Set the distance to the closest point. Some kernels with background weights can use this
    /// information. Because kernels are lightweight, this function makes a new kernel instead of
    /// modifying the existing one. This decision makes parallel code using kernels easier to manage.
    fn with_closest_dist(self, dist: f64) -> Self;
}

//
// Implement Spherical kernel for all kernels defined above
//

impl<T: Real> SphericalKernel<T> for GlobalInvDistance2 {
    #[inline]
    fn with_closest_dist( self, _: f64) -> Self { self }
}

impl<T: Real> SphericalKernel<T> for LocalCubic {
    #[inline]
    fn with_closest_dist(self, _: f64) -> Self { self }
}

impl<T: Real> SphericalKernel<T> for LocalInterpolating {
    #[inline]
    fn with_closest_dist(mut self, dist: f64) -> Self {
        self.closest_d = dist;
        self
    }
}

impl<T: Real> SphericalKernel<T> for LocalApproximate {
    #[inline]
    fn with_closest_dist(self, _: f64) -> Self { self }
}


#[cfg(test)]
mod tests {
    use super::*;
    use autodiff::F;

    /// Test first derivative
    fn test_derivatives<K: Kernel<F>>(kern: &K, start: usize) {
        for i in start..50 {
            // Autodiff test
            let x = F::var(0.1 * i as f64);
            let f = kern.f(x);
            let df = kern.df(x);
            let ddf = kern.ddf(x);
            assert_relative_eq!(df.value(), f.deriv(), max_relative=1e-8);
            assert_relative_eq!(ddf.value(), df.deriv(), max_relative=1e-8);
        }
    }

    /// Tests kernel locality.
    fn test_locality<K: Kernel<f64>>(kern: &K, radius: f64) {
        for i in 0..10 {
            let off = 0.5 * i as f64;
            assert_eq!(kern.f(radius + off * off), 0.0);
        }
    }

    #[test]
    fn global_inv_dist2_kernel_test() {
        // Test the properties of the local approximate kernel and check its derivatives.
        let tol = 0.01;
        let kern = GlobalInvDistance2::new(tol);

        test_derivatives(&kern, 0);
    }

    #[test]
    fn local_cubic_kernel_test() {
        // Test the properties of the local approximate kernel and check its derivatives.
        let radius = 1.0;
        let kern = LocalCubic::new(radius);

        // Check that the kernel has compact support: it's zero outside the radius
        test_locality(&kern, radius);

        test_derivatives(&kern, 0);
    }

    #[test]
    fn local_approximate_kernel_test() {
        // Test the properties of the local approximate kernel and check its derivatives.
        let radius = 1.0;
        let tolerance = 0.01;
        let kern = LocalApproximate::new(radius, tolerance);

        // Check that the kernel has compact support: it's zero outside the radius
        test_locality(&kern, radius);

        test_derivatives(&kern, 0);
    }

    #[test]
    fn local_interpolating_kernel_test() {
        // Test the properties of the local approximate kernel and check its derivatives.
        let radius = 1.0;
        let kern = LocalInterpolating::new(radius);
        let kern = SphericalKernel::<f64>::with_closest_dist(kern, 0.1);

        // Check that the kernel has compact support: it's zero outside the radius
        test_locality(&kern, radius);

        // Note: interpolating kernel is degenerate at 0, so start with 1.
        test_derivatives(&kern, 1);
    }
    
}

#[cfg(all(feature = "unstable", test))]
mod bench {
    extern crate test;
    use super::*;
    use autodiff::F;
    use self::test::Bencher;

    #[bench]
    fn autodiff_kernel_derivative(b: &mut Bencher) {
        let radius = 1.0;
        let tolerance = 0.01;
        let kern = LocalApproximate::new(radius, tolerance);
        b.iter(|| {
            let mut total = 0.0;
            for i in 0..9 {
                let f = kern.f(F::var(0.1 * i as f64));
                total += f.deriv()
            }
            total
        });
    }

    #[bench]
    fn manual_kernel_derivative(b: &mut Bencher) {
        let radius = 1.0;
        let tolerance = 0.01;
        let kern = LocalApproximate::new(radius, tolerance);
        b.iter(|| {
            let mut total = 0.0;
            for i in 0..9 {
                total += kern.df(0.1 * i as f64);
            }
            total
        });
    }
}
