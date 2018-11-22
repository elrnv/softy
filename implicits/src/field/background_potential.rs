use crate::geo::math::Vector3;
use crate::geo::Real;
use crate::kernel::{SphericalKernel};
use crate::field::samples::{Sample, SamplesView};

/// Precomputed data used for background potential computation.
#[derive(Copy, Clone, Debug)]
pub(crate) enum BackgroundPotentialValue<T: Real> {
    /// The value of the background potential at the query point.
    Constant(T),
    /// A Dynamic potential is computed based on the distance to the closest sample point.
    ClosestSampleDistance,
}

/// This struct represents the data needed to compute a background potential at a local query
/// point. This struct also conviently computes useful information about the neighbourhood (like
/// closest distance to a sample point) that can be reused elsewhere.
#[derive(Clone, Debug)]
pub(crate) struct BackgroundPotential<'a, T: Real, K: SphericalKernel<T> + Clone + std::fmt::Debug>
{
    /// Position of the point at which we should evaluate the potential field.
    pub query_pos: Vector3<T>,
    /// Samples that influence the potential field.
    pub samples: SamplesView<'a, 'a, T>,
    /// Data needed to compute the background potential value and its derivative.
    pub bg_potential_value: BackgroundPotentialValue<T>,
    /// The reciprocal of the sum of all the weights in the neighbourhood of the query point.
    pub weight_sum_inv: T,
    /// The spherical kernel used to compute the weights.
    pub kernel: K,
    /// The distance to the closest sample.
    pub closest_sample_dist: T,
    /// Displacement vector to the query point from the closest sample point.
    pub closest_sample_disp: Vector3<T>,
    /// The index of the closest sample point.
    pub closest_sample_index: usize,
    /// Radius of the neighbourhood of the query point.
    pub radius: T,
}

impl<'a, T: Real, K: SphericalKernel<T> + Copy + std::fmt::Debug + Send + Sync + 'a>
    BackgroundPotential<'a, T, K>
{
    /// Pass in the unnormalized weight sum excluding the weight for the background potential.
    pub(crate) fn new(
        q: Vector3<T>,
        samples: SamplesView<'a, 'a, T>,
        radius: T,
        kernel: K,
        dynamic_bg: bool,
    ) -> Self {
        // Precompute data about the closest sample point (displacement, index and distance).
        let min_sample = samples
            .iter()
            .map(|Sample { index, pos, .. }| {
                let disp = q - pos;
                (index, disp, disp.norm_squared())
            })
            .min_by(|(_, _, d0), (_, _, d1)| {
                d0.partial_cmp(d1)
                    .expect("Detected NaN. Please report this bug.")
            });

        if min_sample.is_none() {
            panic!("No surface samples found. Please report this bug.");
        }

        let (closest_sample_index, closest_sample_disp, mut closest_sample_dist) =
            min_sample.unwrap();
        closest_sample_dist = closest_sample_dist.sqrt();

        // Compute the weight sum here. This will be available to the usesr of bg data.
        let mut weight_sum = T::zero();
        for Sample { pos, .. } in samples.iter() {
            let w = kernel.with_closest_dist(closest_sample_dist).eval(q, pos);
            weight_sum += w;
        }

        // Initialize the background potential struct.
        let mut bg = BackgroundPotential {
            query_pos: q,
            samples,
            bg_potential_value: if dynamic_bg {
                BackgroundPotentialValue::ClosestSampleDistance
            } else {
                BackgroundPotentialValue::Constant(T::zero()) // TODO: Support non-zero constant potentials
            },
            weight_sum_inv: T::zero(), // temporarily set the weight_sum_inv
            kernel,
            closest_sample_dist,
            closest_sample_disp,
            closest_sample_index,
            radius,
        };

        // Finalize the weight sum reciprocal.
        bg.weight_sum_inv = T::one() / (weight_sum + bg.background_weight());

        bg
    }

    pub(crate) fn closest_sample_dist(&self) -> T {
        self.closest_sample_dist
    }

    pub(crate) fn weight_sum_inv(&self) -> T {
        self.weight_sum_inv
    }

    pub(crate) fn background_weight(&self) -> T {
        self.kernel.f(self.radius - self.closest_sample_dist)
    }

    pub(crate) fn background_weight_gradient(&self, index: usize) -> Vector3<T> {
        if index == self.closest_sample_index {
            self.closest_sample_disp
                * (self.kernel.df(self.radius - self.closest_sample_dist)
                    / self.closest_sample_dist)
        } else {
            Vector3::zeros()
        }
    }

    /// Compute the unnormalized weighted background potential value. This is typically very
    /// simple, but the caller must remember to multiply it by the `weight_sum_inv` to get the true
    /// background potential contribution.
    pub(crate) fn compute_unnormalized_weighted_potential(&self) -> T {
        // Unpack background data.
        let BackgroundPotential {
            bg_potential_value,
            closest_sample_dist: dist,
            ..
        } = *self;

        self.background_weight()
            * match bg_potential_value {
                BackgroundPotentialValue::Constant(potential) => potential,
                BackgroundPotentialValue::ClosestSampleDistance => dist,
            }
    }

    /// Compute background potential derivative contribution.
    /// Compute derivative if the closest point is in the neighbourhood. Otherwise we
    /// assume the background potential is constant.
    pub(crate) fn compute_jacobian(&self) -> impl Iterator<Item = Vector3<T>> + 'a {
        // Unpack background data.
        let BackgroundPotential {
            query_pos: q,
            samples,
            bg_potential_value,
            weight_sum_inv,
            kernel,
            closest_sample_dist: dist,
            closest_sample_disp: disp,
            closest_sample_index,
            radius,
        } = *self;

        // The unnormalized weight evaluated at the distance to the boundary of the
        // neighbourhood.
        let wb = kernel.f(radius - dist);

        // Gradient of the unnormalized weight evaluated at the distance to the
        // boundary of the neighbourhood.
        let dwbdp = self.background_weight_gradient(closest_sample_index);

        samples.into_iter().map(move |Sample { index, pos, .. }| {
            // Gradient of the unnormalized weight for the current sample point.
            let dwdp = -kernel.with_closest_dist(dist).grad(q, pos);

            // This term is valid for constant or dynamic background potentials.
            let constant_term =
                |potential: T| dwdp * (-potential * wb * weight_sum_inv * weight_sum_inv);

            match bg_potential_value {
                BackgroundPotentialValue::Constant(potential) => constant_term(potential),
                BackgroundPotentialValue::ClosestSampleDistance => {
                    let mut grad = constant_term(dist);

                    if index == closest_sample_index {
                        //grad += dwbdp * dist + disp * (wb / dist)
                        grad += dwbdp * (dist * weight_sum_inv * (T::one() - weight_sum_inv * wb))
                            - disp * (wb * weight_sum_inv / dist)
                    }

                    grad
                }
            }
        })
    }
}

