/// Enumerate all implemented kernels. This is useful for switching between kernels dynamically.
#[derive(Copy, Clone, Debug)]
pub enum Kernel {
    Interpolating { radius: f64 },
    Approximate { radius: f64, tolerance: f64 },
    Cubic { radius: f64 },
    Global { tolerance: f64 },
    Hrbf,
}

pub(crate) fn global_inv_dist2_kernel(r: f64, epsilon: f64) -> f64 {
    let w = 1.0 / (r * r + epsilon * epsilon);
    w * w
}

pub(crate) fn local_cubic_kernel(r: f64, radius: f64) -> f64 {
    if r > radius {
        return 0.0;
    }

    1.0 - 3.0 * r * r / (radius * radius) + 2.0 * r * r * r / (radius * radius * radius)
}

pub(crate) fn local_interpolating_kernel(r: f64, radius: f64, closest_d: f64) -> f64 {
    if r > radius {
        return 0.0;
    }

    let envelope = local_cubic_kernel(r, radius);

    let s = r / radius;
    let sc = closest_d / radius;
    envelope * sc * sc * (1.0 / (s * s) - 1.0)
}

pub(crate) fn local_approximate_kernel(r: f64, radius: f64, tolerance: f64) -> f64 {
    if r > radius {
        return 0.0;
    }

    let eps = tolerance;

    let w = |d| {
        let ddeps = 1.0 / (d * d + eps);
        let epsp1 = 1.0 / (1.0 + eps);
        (ddeps * ddeps - epsp1 * epsp1) / (1.0 / (eps * eps) - epsp1 * epsp1)
    };

    w(r / radius) // /denom
}
