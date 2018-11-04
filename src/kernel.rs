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
        LocalApproximate {
            radius,
            tolerance,
        }
    }
}

impl Kernel<f64> for LocalApproximate {
    fn f(&self, x: f64) -> f64 {
        let r = self.radius;

        if x > r {
            return 0.0;
        }

        let eps = self.tolerance;

        let w = |d| {
            let ddeps = 1.0 / (d * d + eps);
            let epsp1 = 1.0 / (1.0 + eps);
            (ddeps * ddeps - epsp1 * epsp1) / (1.0 / (eps * eps) - epsp1 * epsp1)
        };

        w(x / r)
    }
    fn df(&self, _x: f64) -> f64 {
        0.0
    }

    fn ddf(&self, _x: f64) -> f64 {
        0.0
    }
}
