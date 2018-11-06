use crate::geo::Real;

/// Enumerate all implemented kernels. This is useful for switching between kernels dynamically.
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum KernelType {
    Interpolating { radius: f64 },
    Approximate { radius: f64, tolerance: f64 },
    Cubic { radius: f64 },
    Global { tolerance: f64 },
    Hrbf,
}

/// This kernel trait defines a radial basis kernel interface.
pub trait Kernel<T: Real> {
    /// Main kernel function evaluated at `x`.
    fn f(&self, x: T) -> T;
    /// First derivative of the kernel evaluated at `x`.
    fn df(&self, x: T) -> T;
    /// Second derivative of the kernel evaluated at `x`.
    fn ddf(&self, x: T) -> T;
}

/// Global kernel with falloff proportional to inverse distance squared.
pub struct GlobalInvDistance2 {
    epsilon: f64,
}

impl GlobalInvDistance2 {
    pub fn new(tolerance: f64) -> Self {
        GlobalInvDistance2 { epsilon: tolerance }
    }
}

impl Kernel<f64> for GlobalInvDistance2 {
    fn f(&self, x: f64) -> f64 {
        let w = 1.0 / (x * x + self.epsilon * self.epsilon);
        w * w
    }
    fn df(&self, _x: f64) -> f64 {
        0.0
    }

    fn ddf(&self, _x: f64) -> f64 {
        0.0
    }
}

/// Local cubic kernel with compact support. This kernel is non-interpolating but is very simple
/// and fast to compute.
pub struct LocalCubic {
    radius: f64,
}

impl LocalCubic {
    pub fn new(radius: f64) -> Self {
        LocalCubic { radius }
    }
}

impl Kernel<f64> for LocalCubic {
    fn f(&self, x: f64) -> f64 {
        let r = self.radius;
        if x > r {
            return 0.0;
        }

        1.0 - 3.0 * x * x / (r * r) + 2.0 * x * x * x / (r * r * r)
    }
    fn df(&self, _x: f64) -> f64 {
        0.0
    }

    fn ddf(&self, _x: f64) -> f64 {
        0.0
    }
}

/// Local interpolating kernel. This kernel is exactly interpolating but it suffers from smoothness
/// artifacts. Note that `closest_d` represents the distance to the closest neighbour, which means
/// it must be manually updated before evaluating the kernel.
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
    pub fn update_closest(&mut self, closest: f64) -> &mut Self {
        self.closest_d = closest;
        self
    }
}

impl Kernel<f64> for LocalInterpolating {
    fn f(&self, x: f64) -> f64 {
        let r = self.radius;
        let xc = self.closest_d;
        if x > r {
            return 0.0;
        }

        let envelope = LocalCubic::new(r).f(x);

        let s = x / r;
        let sc = xc / r;
        envelope * sc * sc * (1.0 / (s * s) - 1.0)
    }
    fn df(&self, _x: f64) -> f64 {
        0.0
    }

    fn ddf(&self, _x: f64) -> f64 {
        0.0
    }
}

/// This kernel is a compromise between the cubic and interpolating kernels. This kernel is fairly
/// cheap to compute and has flexible smoothness properties, which are controllable using the
/// `tolerance` parameter.
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

#[cfg(test)]
mod tests {
    use super::*;
    use autodiff::F;

    /// Test first derivative
    fn test_derivatives<K: Kernel<F>>(kern: &K) {
        for i in 0..50 {
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
    fn local_approximate_kernel_test() {
        // Test the properties of the local approximate kernel and check its derivatives.
        let radius = 1.0;
        let tolerance = 0.01;
        let kern = LocalApproximate::new(radius, tolerance);

        // Check that the kernel has compact support: it's zero outside the radius
        test_locality(&kern, radius);

        test_derivatives(&kern);
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
