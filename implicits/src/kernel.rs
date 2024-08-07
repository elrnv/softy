#![allow(clippy::just_underscores_and_digits)]

use super::Error;
use num_traits::{Float, Zero};
use std::ops::Neg;
use tensr::{Matrix, Matrix3, Scalar, Vector3};

/// Enumerate all implemented kernels. This is useful for switching between kernels dynamically.
#[derive(Copy, Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum KernelType {
    Interpolating {
        radius_multiplier: f64,
    },
    Smooth {
        radius_multiplier: f64,
        tolerance: f64,
    },
    Approximate {
        radius_multiplier: f64,
        tolerance: f64,
    },
    Cubic {
        radius_multiplier: f64,
    },
    Global {
        tolerance: f64,
    },
    Hrbf,
}

#[derive(Copy, Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum LocalKernel {
    Interpolating {
        radius_multiplier: f64,
    },
    Smooth {
        tolerance: f64,
        radius_multiplier: f64,
    },
    Approximate {
        radius_multiplier: f64,
        tolerance: f64,
    },
    Cubic {
        radius_multiplier: f64,
    },
}

#[derive(Copy, Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum GlobalKernel {
    InvDistance2 { tolerance: f64 },
}

impl LocalKernel {
    pub fn radius_multiplier(self) -> f64 {
        match self {
            LocalKernel::Interpolating { radius_multiplier }
            | LocalKernel::Smooth {
                radius_multiplier, ..
            }
            | LocalKernel::Approximate {
                radius_multiplier, ..
            }
            | LocalKernel::Cubic { radius_multiplier } => radius_multiplier,
        }
    }

    pub fn with_radius_multiplier(self, radius_multiplier: f64) -> Self {
        match self {
            LocalKernel::Interpolating { .. } => LocalKernel::Interpolating { radius_multiplier },
            LocalKernel::Smooth { tolerance, .. } => LocalKernel::Smooth {
                radius_multiplier,
                tolerance,
            },
            LocalKernel::Approximate { tolerance, .. } => LocalKernel::Approximate {
                radius_multiplier,
                tolerance,
            },
            LocalKernel::Cubic { .. } => LocalKernel::Cubic { radius_multiplier },
        }
    }
}

/// Apply a function with an instantiated kernel. This function allows users to branch on the
/// kernel type outside of inner loops, which can be costly.
/// `base_radius` gives the absolute radius of the kernel which is then scaled by the
/// corresponding `radius_multiplier` (if any) provided by the specific kernel type.
macro_rules! apply_as_spherical {
    ($kernel:expr, $base_radius:expr, $f:expr) => {
        match $kernel {
            LocalKernel::Interpolating { radius_multiplier } => $f(
                $crate::kernel::LocalInterpolating::new($base_radius * radius_multiplier),
            ),
            LocalKernel::Smooth {
                radius_multiplier,
                tolerance,
            } => $f($crate::kernel::LocalSmooth::new(
                $base_radius * radius_multiplier,
                tolerance,
            )),
            LocalKernel::Approximate {
                radius_multiplier,
                tolerance,
            } => $f($crate::kernel::LocalApproximate::new(
                $base_radius * radius_multiplier,
                tolerance,
            )),
            LocalKernel::Cubic { radius_multiplier } => $f($crate::kernel::LocalCubic::new(
                $base_radius * radius_multiplier,
            )),
        }
    };
    ($kernel:expr, $f:expr) => {
        match $kernel {
            GlobalKernel::InvDistance2 { tolerance } => {
                $f($crate::kernel::GlobalInvDistance2::new(tolerance))
            }
        }
    };
}

/// Same as above but with matching match arms producing an iterator.
macro_rules! apply_as_spherical_impl_iter {
    // Fallible version
    ($kernel:expr, $base_radius:expr, $f:expr, ?) => {{
        match $kernel {
            LocalKernel::Interpolating { radius_multiplier } => Either::Left(Either::Left($f(
                $crate::kernel::LocalInterpolating::new($base_radius * radius_multiplier),
            )?)),
            LocalKernel::Smooth {
                radius_multiplier,
                tolerance,
            } => Either::Left(Either::Right($f($crate::kernel::LocalSmooth::new(
                $base_radius * radius_multiplier,
                tolerance,
            ))?)),
            LocalKernel::Approximate {
                radius_multiplier,
                tolerance,
            } => Either::Right(Either::Left($f($crate::kernel::LocalApproximate::new(
                $base_radius * radius_multiplier,
                tolerance,
            ))?)),
            LocalKernel::Cubic { radius_multiplier } => Either::Right(Either::Right($f(
                $crate::kernel::LocalCubic::new($base_radius * radius_multiplier),
            )?)),
        }
    }};
    ($kernel:expr, $f:expr, ?) => {
        match $kernel {
            GlobalKernel::InvDistance2 { tolerance } => {
                $f($crate::kernel::GlobalInvDistance2::new(tolerance))?
            }
        }
    };
    ($kernel:expr, $base_radius:expr, $f:expr) => {{
        match $kernel {
            LocalKernel::Interpolating { radius_multiplier } => Either::Left(Either::Left($f(
                $crate::kernel::LocalInterpolating::new($base_radius * radius_multiplier),
            ))),
            LocalKernel::Smooth {
                radius_multiplier,
                tolerance,
            } => Either::Left(Either::Right($f($crate::kernel::LocalSmooth::new(
                $base_radius * radius_multiplier,
                tolerance,
            )))),
            LocalKernel::Approximate {
                radius_multiplier,
                tolerance,
            } => Either::Right(Either::Left($f($crate::kernel::LocalApproximate::new(
                $base_radius * radius_multiplier,
                tolerance,
            )))),
            LocalKernel::Cubic { radius_multiplier } => Either::Right(Either::Right($f(
                $crate::kernel::LocalCubic::new($base_radius * radius_multiplier),
            ))),
        }
    }};
    ($kernel:expr, $f:expr) => {
        match $kernel {
            GlobalKernel::InvDistance2 { tolerance } => {
                $f($crate::kernel::GlobalInvDistance2::new(tolerance))
            }
        }
    };
}

impl From<KernelType> for LocalKernel {
    fn from(kernel: KernelType) -> Self {
        match kernel {
            KernelType::Interpolating { radius_multiplier } => {
                LocalKernel::Interpolating { radius_multiplier }
            }
            KernelType::Cubic { radius_multiplier } => LocalKernel::Cubic { radius_multiplier },
            KernelType::Smooth {
                radius_multiplier,
                tolerance,
            } => LocalKernel::Smooth {
                radius_multiplier,
                tolerance,
            },
            KernelType::Approximate {
                radius_multiplier,
                tolerance,
            } => LocalKernel::Approximate {
                radius_multiplier,
                tolerance,
            },
            _ => panic!("Incorrect kernel type conversion"),
        }
    }
}

impl From<KernelType> for GlobalKernel {
    fn from(kernel: KernelType) -> Self {
        match kernel {
            KernelType::Global { tolerance } => GlobalKernel::InvDistance2 { tolerance },
            _ => panic!("Incorrect kernel type conversion"),
        }
    }
}

impl KernelType {
    /// Same as `apply_as_spherical` but without the need to specify the kernel, which allows us to
    /// write this as a function.
    pub fn apply_fns<MlsF, HrbfF, O>(self, mut mls: MlsF, mut hrbf: HrbfF) -> Result<O, Error>
    where
        MlsF: FnMut() -> Result<O, Error>,
        HrbfF: FnMut() -> Result<O, Error>,
    {
        match self {
            KernelType::Interpolating { .. }
            | KernelType::Approximate { .. }
            | KernelType::Smooth { .. }
            | KernelType::Cubic { .. }
            | KernelType::Global { .. } => mls(),
            KernelType::Hrbf => hrbf(),
        }
    }
}

/// This kernel trait defines a 1D basis kernel interface.
pub trait Kernel<T> {
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

impl<T: Scalar + Float> Kernel<T> for GlobalInvDistance2 {
    fn f(&self, x: T) -> T {
        let eps = T::from(self.epsilon).unwrap();
        let w = T::one() / (x * x + eps * eps);
        w * w
    }

    fn df(&self, x: T) -> T {
        self.f(autodiff::F::<T, T>::var(x)).deriv()
    }

    fn ddf(&self, x: T) -> T {
        self.df(autodiff::F::<T, T>::var(x)).deriv()
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

impl<T: Scalar> Kernel<T> for LocalCubic {
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

        _6 * (x * x / (r * r * r) - x / (r * r))
    }

    fn ddf(&self, x: T) -> T {
        let r = T::from(self.radius).unwrap();

        if x > r {
            return T::zero();
        }

        let _12 = T::from(12.0).unwrap();
        let _6 = T::from(6.0).unwrap();

        _12 * x / (r * r * r) - _6 / (r * r)
    }
}

/// Local interpolating kernel. This kernel is exactly interpolating but it suffers from smoothness
/// artifacts. Note that `closest_d` represents the distance to the closest neighbor, which means
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

impl<T: Scalar + Float> Kernel<T> for LocalInterpolating {
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
        self.f(autodiff::F::var(x)).deriv()
    }

    fn ddf(&self, x: T) -> T {
        self.df(autodiff::F::var(x)).deriv()
    }
}

/// This kernel is a compromise between the cubic and interpolating kernels.
///
/// This kernel is fairly cheap to compute and has flexible smoothness properties, which are controllable using the
/// `tolerance` parameter.
///
/// In contrast to `LocalApproximate`, this kernel has a zero derivative at `radius`, which is important for generating ontinuous forces.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct LocalSmooth {
    pub radius: f64,
    pub tolerance: f64,
}

impl LocalSmooth {
    pub fn new(radius: f64, tolerance: f64) -> Self {
        LocalSmooth { radius, tolerance }
    }
}

impl<T: Scalar + num_traits::Float> Kernel<T> for LocalSmooth {
    fn f(&self, x: T) -> T {
        let r = T::from(self.radius).unwrap();
        if x > r {
            return T::zero();
        }

        let t = T::from(self.tolerance).unwrap();

        let _2 = T::from(2.0).unwrap();
        let _3 = T::from(3.0).unwrap();

        let d = x / r;
        let d2 = d * d;
        let d3 = d2 * d;

        // T::exp(-x2/eps)*(T::one() - _3 * x2 / r2 + _2 * x2 * x / (r2 * r))
        t * (_2 * d3 - _3 * d2 + T::one()) / (t + d2)
    }
    fn df(&self, x: T) -> T {
        let r = T::from(self.radius).unwrap();

        if x > r {
            return T::zero();
        }

        let t = T::from(self.tolerance).unwrap();

        let _2 = T::from(2.0).unwrap();
        let _3 = T::from(3.0).unwrap();

        let d = x / r;
        let d2 = d * d;
        let d3 = d2 * d;
        let d4 = d3 * d;

        // _2*(x2-xr)*T::exp(-x2/eps) *(r2  + xr +_3*eps - _2*x2)/(r2*r*eps)
        _2 * t * d * (_3 * t * d - _3 * t + d3 - T::one()) / (r * (t * t + _2 * t * d2 + d4))
    }

    fn ddf(&self, x: T) -> T {
        let r = T::from(self.radius).unwrap();

        if x > r {
            return T::zero();
        }

        let t = T::from(self.tolerance).unwrap();

        let _2 = T::from(2.0).unwrap();
        let _3 = T::from(3.0).unwrap();
        // let _4 = T::from(4.0).unwrap();
        // let _5 = T::from(5.0).unwrap();
        let _6 = T::from(6.0).unwrap();
        let _9 = T::from(9.0).unwrap();

        let d = x / r;
        let d2 = d * d;
        let d3 = d2 * d;
        let d4 = d3 * d;
        let d6 = d3 * d3;
        let t2 = t * t;
        let t3 = t2 * t;

        // let exp = T::exp(-x2/eps);
        // -_2 * exp *(r3*(eps - _2*x2) + _3 * r * (eps2 - _5*eps*x2 + _2*x2*x2) - _6*eps2*x + _14*eps*x3 - _4*x3*x2)/(r3*eps2)
        _2 * t * (_6 * t2 * d - _3 * t2 - _2 * t * d3 + _9 * t * d2 - t + _3 * d2)
            / (r * r * (t3 + _3 * t2 * d2 + _3 * t * d4 + d6))
    }
}

/// This kernel is a compromise between the cubic and interpolating kernels. This kernel is fairly
/// cheap to compute and has flexible smoothness properties, which are controllable using the
/// `tolerance` parameter.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct LocalApproximate {
    pub radius: f64,
    pub tolerance: f64,
}

impl LocalApproximate {
    pub fn new(radius: f64, tolerance: f64) -> Self {
        LocalApproximate { radius, tolerance }
    }
}

impl<T: Scalar + Neg<Output = T>> Kernel<T> for LocalApproximate {
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
        let eps1_2 = eps1 * eps1;
        let factor = eps * eps * eps1_2 / (T::one() + _2 * eps);
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

        let d = x / r;
        let eps1 = T::one() + eps;
        let eps1_2 = eps1 * eps1;
        let factor = eps * eps * eps1_2 / (T::one() + _2 * eps);
        let d2_eps = d * d + eps;
        -factor * _4 * d / (r * d2_eps * d2_eps * d2_eps)
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

        let d = x / r;
        let eps1 = T::one() + eps;
        let eps1_2 = eps1 * eps1;
        let factor = eps * eps * eps1_2 / (T::one() + _2 * eps);
        let d2_eps = d * d + eps;
        let d2_eps4 = d2_eps * d2_eps * d2_eps * d2_eps;
        factor * _4 * (_6 * d * d - d2_eps) / (d2_eps4 * r * r)
    }
}

/// This kernel trait defines a radial basis kernel interface.
pub trait RadialKernel<T: Scalar + Float>: Kernel<T> {
    /// Main kernel function evaluated at `x` with center at `p`.
    #[inline]
    fn eval(&self, x: Vector3<T>, p: Vector3<T>) -> T {
        self.f((x - p).norm())
    }
    /// First derivative wrt `x` of the kernel evaluated at `x` with center at `p`.
    /// To compute the derivatives wrt `p`, simply negate this derivative.
    #[inline]
    fn grad(&self, x: Vector3<T>, p: Vector3<T>) -> Vector3<T> {
        let diff: Vector3<T> = x - p;
        let norm = diff.norm();
        if norm > T::zero() {
            diff * (self.df(norm) / norm)
        } else {
            Vector3::zero()
        }
    }
    /// Second derivative wrt `x` of the kernel evaluated at `x` with center at `p`.
    /// The hessian wrt `p` is identical.
    #[inline]
    fn hess(&self, x: Vector3<T>, p: Vector3<T>) -> Matrix3<T> {
        let diff = x - p;
        let norm = diff.norm();
        if norm > T::zero() {
            let norm_inv = T::one() / norm;
            let norm_inv2 = norm_inv * norm_inv;
            let identity = Matrix3::identity();
            let proj_diff = diff * diff.transpose() * norm_inv2;
            (identity - proj_diff) * (self.df(norm) * norm_inv) + proj_diff * self.ddf(norm)
        } else {
            Matrix3::zero()
        }
    }

    /// Set the distance to the closest point. Some kernels with background weights can use this
    /// information. Because kernels are lightweight, this function makes a new kernel instead of
    /// modifying the existing one. This decision makes parallel code using kernels easier to manage.
    fn with_closest_dist(self, dist: T) -> Self;
}

/// A spherical kernel with a well defined radius of influence. This means that there is a
/// radius beyond which the kernel value is always zero. If radius is finite, it typically means
/// that the kernel has compact support, although it is still valid to set radius to a finite
/// number for something like the Gaussian kernel, which will just mean that the background
/// potential will be mixed in full past the the radius, which can represent a standard
/// deviation.
pub trait SphericalKernel<T: Scalar + Float>: RadialKernel<T> {
    /// Produce the radius of influence of this kernel.
    fn radius(&self) -> T;
}

//
// Implement Radial kernel for all kernels defined above
//

impl<T: Scalar + Float> RadialKernel<T> for GlobalInvDistance2 {
    #[inline]
    fn with_closest_dist(self, _: T) -> Self {
        self
    }
}

impl<T: Scalar + Float> RadialKernel<T> for LocalCubic {
    #[inline]
    fn with_closest_dist(self, _: T) -> Self {
        self
    }
}

impl<T: Scalar + Float> RadialKernel<T> for LocalInterpolating {
    #[inline]
    fn with_closest_dist(mut self, dist: T) -> Self {
        self.closest_d = dist.to_f64().unwrap();
        self
    }
}

impl<T: Scalar + Float> RadialKernel<T> for LocalApproximate {
    #[inline]
    fn with_closest_dist(self, _: T) -> Self {
        self
    }
}

impl<T: Scalar + Float> RadialKernel<T> for LocalSmooth {
    #[inline]
    fn with_closest_dist(self, _: T) -> Self {
        self
    }
}

//
// Implement Spherical kernel for all kernels defined above
//

impl<T: Scalar + Float> SphericalKernel<T> for LocalCubic {
    #[inline]
    fn radius(&self) -> T {
        T::from(self.radius).unwrap()
    }
}

impl<T: Scalar + Float> SphericalKernel<T> for LocalInterpolating {
    #[inline]
    fn radius(&self) -> T {
        T::from(self.radius).unwrap()
    }
}

impl<T: Scalar + Float> SphericalKernel<T> for LocalApproximate {
    #[inline]
    fn radius(&self) -> T {
        T::from(self.radius).unwrap()
    }
}

impl<T: Scalar + Float> SphericalKernel<T> for LocalSmooth {
    #[inline]
    fn radius(&self) -> T {
        T::from(self.radius).unwrap()
    }
}

impl<T: Scalar + Float> SphericalKernel<T> for GlobalInvDistance2 {
    #[inline]
    fn radius(&self) -> T {
        T::infinity()
    }
}

// TODO: This kernel is a half-arsed attempt at refactoring, and is not actually used properly.
//       The reason is that the gradient is tricky to compute for implicit surface vertices and the
//       two terms need to be computed separately. There should be a better way to abstract this
//       and it should be done in tandem with abstracting the contact jacobian neighbourhood weight
//       derivatives.
/// This kernel is separate from the rest and weighs contributions based on how far from parallel and in the same direction they are.
pub struct NormalKernel<T> {
    unit_nml: Vector3<T>,
    grad_phi: Vector3<T>,
}

impl<T: Scalar + Float> NormalKernel<T> {
    #[inline]
    pub fn new(unit_nml: Vector3<T>, grad_phi: Vector3<T>) -> Self {
        NormalKernel { unit_nml, grad_phi }
    }
    /// Main kernel function evaluated at `x` with center at `p`.
    #[inline]
    pub fn eval(&self) -> T {
        let half = T::from(0.5).unwrap();
        let unit_nml = self.unit_nml; //.cast::<f64>().cast::<T>();
        let grad_phi = self.grad_phi; //.cast::<f64>().cast::<T>();
        let nml_dot_grad = unit_nml.dot(grad_phi);
        let w = half * (T::one() + nml_dot_grad);
        w * w
    }
    /// First derivative wrt `x` of the kernel evaluated at `x` with center at `p`.
    /// To compute the derivatives wrt `p`, simply negate this derivative.
    #[inline]
    pub fn grad(&self, unit_nml_grad: Matrix3<T>, jac_grad_phi_t: Matrix3<T>) -> Vector3<T> {
        let half = T::from(0.5).unwrap();
        let nml_dot_grad = self.unit_nml.dot(self.grad_phi);
        let w = half * (T::one() + nml_dot_grad);
        (jac_grad_phi_t * self.unit_nml + unit_nml_grad * self.grad_phi) * w
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use autodiff::F1;

    /// Test derivatives.
    fn test_derivatives<K: Kernel<F1>>(kern: &K, start: usize) {
        for i in start..55 {
            // Autodiff test
            let x = F1::var(0.1 * i as f64);
            let f = kern.f(x);
            let df = kern.df(x);
            let ddf = kern.ddf(x);
            assert_relative_eq!(df.value(), f.deriv(), max_relative = 1e-8);
            assert_relative_eq!(ddf.value(), df.deriv(), max_relative = 1e-8);
        }
    }

    /// Test radial derivative.
    fn test_radial_derivatives<K: RadialKernel<F1>>(kern: &K, start: usize) {
        for j in 0..3 {
            // for each component
            for i in start..55 {
                // Autodiff test
                let x = F1::cst(0.1 * i as f64);
                let mut q = Vector3::new([x, x, x]);
                q[j] = F1::var(x);
                let f = kern.eval(q, Vector3::zero());
                let df = kern.grad(q, Vector3::zero());
                let ddf = kern.hess(q, Vector3::zero());
                assert_relative_eq!(df[j].value(), f.deriv(), max_relative = 1e-8);
                for k in 0..3 {
                    assert_relative_eq!(ddf[k][j].value(), df[k].deriv(), max_relative = 1e-8);
                }
            }
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
        test_radial_derivatives(&kern, 1);
    }

    #[test]
    fn local_compact_kernel_test() {
        // Test the properties of the local approximate kernel and check its derivatives.
        let radius = 5.0;
        let tolerance = 0.01;
        let kern = LocalSmooth::new(radius, tolerance);

        // Check that the kernel has compact support: it's zero outside the radius
        test_locality(&kern, radius);

        test_derivatives(&kern, 0);
        test_radial_derivatives(&kern, 1);
    }

    #[test]
    fn local_cubic_kernel_test() {
        // Test the properties of the local approximate kernel and check its derivatives.
        let radius = 5.0;
        let kern = LocalCubic::new(radius);

        // Check that the kernel has compact support: it's zero outside the radius
        test_locality(&kern, radius);

        test_derivatives(&kern, 0);
        test_radial_derivatives(&kern, 1);
    }

    #[test]
    fn local_approximate_kernel_test() {
        // Test the properties of the local approximate kernel and check its derivatives.
        let radius = 5.0;
        let tolerance = 0.01;
        let kern = LocalApproximate::new(radius, tolerance);

        // Check that the kernel has compact support: it's zero outside the radius
        test_locality(&kern, radius);

        test_derivatives(&kern, 0);
        test_radial_derivatives(&kern, 1);
    }

    #[test]
    fn local_interpolating_kernel_test() {
        // Test the properties of the local approximate kernel and check its derivatives.
        let radius = 5.0;
        let kern = LocalInterpolating::new(radius);
        let kern = RadialKernel::<f64>::with_closest_dist(kern, 0.1);

        // Check that the kernel has compact support: it's zero outside the radius
        test_locality(&kern, radius);

        // Note: interpolating kernel is degenerate at 0, so start with 1.
        test_derivatives(&kern, 1);
        test_radial_derivatives(&kern, 1);
    }
}

#[cfg(all(feature = "unstable", test))]
mod bench {
    extern crate test;
    use self::test::Bencher;
    use super::*;
    use autodiff::F1;

    #[bench]
    fn autodiff_kernel_derivative(b: &mut Bencher) {
        let radius = 1.0;
        let tolerance = 0.01;
        let kern = LocalApproximate::new(radius, tolerance);
        b.iter(|| {
            let mut total = 0.0;
            for i in 0..9 {
                let f = kern.f(F1::var(0.1 * i as f64));
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
