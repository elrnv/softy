use crate::field::samples::{Sample, SamplesView};
use crate::geo::math::Vector3;
use crate::geo::Real;
use crate::kernel::SphericalKernel;

/// Different types of background fields supported.
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum BackgroundFieldType {
    /// Don't use a background field at all, no weights are included for that.
    None,
    /// Use a zero background field.
    Zero,
    /// Use the background field given in the input.
    FromInput,
    /// Distance to the closest point.
    DistanceBased,
    /// Normal displacement dot product to the closest polygon.
    NormalBased,
}

/// Precomputed data used for background field computation.
#[derive(Copy, Clone, Debug, PartialEq)]
pub(crate) enum BackgroundFieldValue<T> {
    /// No value, nothing to mix in, background weight is always zero.
    None,
    /// The value of the background field at the query point.
    Constant(T),
    /// A Dynamic field is computed based on the distance to the closest sample point.
    ClosestSampleDistance,
    /// A Dynamic field is computed based on the normal displacement. In other words, this is
    /// determined by the dot product of the sample displacement and a normal.
    ClosestSampleNormalDisp,
}

impl<V: num_traits::Zero> BackgroundFieldValue<V> {
    /// Use this constructor for computing the Jacobian of the field.
    pub fn jac(ty: BackgroundFieldType) -> Self {
        match ty {
            BackgroundFieldType::None => BackgroundFieldValue::None,
            BackgroundFieldType::Zero => BackgroundFieldValue::Constant(V::zero()),
            BackgroundFieldType::FromInput => BackgroundFieldValue::Constant(V::zero()),
            BackgroundFieldType::DistanceBased => {
                BackgroundFieldValue::ClosestSampleDistance
            }
            BackgroundFieldType::NormalBased => {
                BackgroundFieldValue::ClosestSampleNormalDisp
            }
        }
    }

    /// Use this constructor for computing the field.
    pub fn val(ty: BackgroundFieldType, field_value: V) -> Self {
        match ty {
            BackgroundFieldType::None => BackgroundFieldValue::None,
            BackgroundFieldType::Zero => BackgroundFieldValue::Constant(V::zero()),
            BackgroundFieldType::FromInput => {
                BackgroundFieldValue::Constant(field_value)
            }
            BackgroundFieldType::DistanceBased => {
                BackgroundFieldValue::ClosestSampleDistance
            }
            BackgroundFieldType::NormalBased => {
                BackgroundFieldValue::ClosestSampleNormalDisp
            }
        }
    }
}

/// This struct represents the data needed to compute a background field at a local query
/// point. This struct also conviently computes useful information about the neighbourhood (like
/// closest distance to a sample point) that can be reused elsewhere.
#[derive(Clone, Debug)]
pub(crate) struct BackgroundField<'a,T,V,K>
where T: Real,
      K: SphericalKernel<T> + Clone + std::fmt::Debug
{
    /// Position of the point at which we should evaluate the field.
    pub query_pos: Vector3<T>,
    /// Samples that influence the field.
    pub samples: SamplesView<'a, 'a, T>,
    /// Data needed to compute the background field value and its derivative.
    pub bg_field_value: BackgroundFieldValue<V>,
    /// The sum of all the weights in the neighbourhood of the query point.
    pub weight_sum: T,
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

impl<'a, T, V, K>
    BackgroundField<'a, T, V, K>
    where T: Real,
          V: Copy + Clone + std::fmt::Debug + PartialEq,
          K: SphericalKernel<T> + Copy + std::fmt::Debug + Send + Sync + 'a
{
    /// Pass in the unnormalized weight sum excluding the weight for the background field.
    pub(crate) fn new(
        q: Vector3<T>,
        samples: SamplesView<'a, 'a, T>,
        radius: T,
        kernel: K,
        bg_value: BackgroundFieldValue<V>,
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

        // Initialize the background field struct.
        let mut bg = BackgroundField {
            query_pos: q,
            samples,
            bg_field_value: bg_value,
            weight_sum,
            kernel,
            closest_sample_dist,
            closest_sample_disp,
            closest_sample_index,
            radius,
        };

        // Finalize the weight sum.
        bg.weight_sum += bg.background_weight();

        bg
    }

    pub(crate) fn closest_sample_dist(&self) -> T {
        self.closest_sample_dist
    }

    pub(crate) fn weight_sum_inv(&self) -> T {
        if self.weight_sum == T::zero() { T::zero() } else { T::one() / self.weight_sum }
    }

    pub(crate) fn background_weight(&self) -> T {
        match self.bg_field_value {
            BackgroundFieldValue::None => T::zero(),
            _ => self.kernel.f(self.radius - self.closest_sample_dist),
        }
    }

    pub(crate) fn background_weight_gradient(&self, index: usize) -> Vector3<T> {
        if index == self.closest_sample_index
            && self.bg_field_value != BackgroundFieldValue::None
        {
            self.closest_sample_disp
                * (self.kernel.df(self.radius - self.closest_sample_dist)
                    / self.closest_sample_dist)
        } else {
            Vector3::zeros()
        }
    }
}

impl<'a, T, V, K> BackgroundField<'a, T, V, K>
    where T: Real,
          V: Copy + Clone + std::fmt::Debug + PartialEq + std::ops::Mul<T, Output=V> + num_traits::Zero,
          K: SphericalKernel<T> + Copy + std::fmt::Debug + Send + Sync + 'a
{
    /// Compute the unnormalized weighted background field value. This is typically very
    /// simple, but the caller must remember to multiply it by the `weight_sum_inv` to get the true
    /// background field contribution.
    pub(crate) fn compute_unnormalized_weighted_vector_field(&self) -> V {
        // Unpack background data.
        let BackgroundField {
            bg_field_value,
            ..
        } = *self;

        let field_val = match bg_field_value {
            BackgroundFieldValue::None => V::zero(),
            BackgroundFieldValue::Constant(val) => val,
            BackgroundFieldValue::ClosestSampleDistance => V::zero(),
            BackgroundFieldValue::ClosestSampleNormalDisp => V::zero(),
        };

        field_val * self.background_weight()
    }
}

impl<'a, T, K> BackgroundField<'a, T, T, K>
    where T: Real,
          K: SphericalKernel<T> + Copy + std::fmt::Debug + Send + Sync + 'a
{
    /// Compute the unnormalized weighted background field value. This is typically very
    /// simple, but the caller must remember to multiply it by the `weight_sum_inv` to get the true
    /// background field contribution.
    pub(crate) fn compute_unnormalized_weighted_scalar_field(&self) -> T {
        // Unpack background data.
        let BackgroundField {
            bg_field_value,
            closest_sample_dist: dist,
            ..
        } = *self;

        let field_val = match bg_field_value {
            BackgroundFieldValue::None => T::zero(),
            BackgroundFieldValue::Constant(val) => val,
            BackgroundFieldValue::ClosestSampleDistance => dist,
            BackgroundFieldValue::ClosestSampleNormalDisp => dist,
        };
        field_val * self.background_weight()
    }

    /// Compute background field derivative contribution.
    /// Compute derivative if the closest point is in the neighbourhood. Otherwise we
    /// assume the background field is constant.
    pub(crate) fn compute_jacobian(&self) -> impl Iterator<Item = Vector3<T>> + 'a {
        // Unpack background data.
        let BackgroundField {
            query_pos: q,
            samples,
            bg_field_value,
            kernel,
            closest_sample_dist: dist,
            closest_sample_disp: disp,
            closest_sample_index,
            ..
        } = *self;

        let weight_sum_inv = self.weight_sum_inv();

        // The unnormalized weight evaluated at the distance to the boundary of the
        // neighbourhood.
        let wb = self.background_weight();

        // Gradient of the unnormalized weight evaluated at the distance to the
        // boundary of the neighbourhood.
        let dwbdp = self.background_weight_gradient(closest_sample_index);

        samples.into_iter().map(move |Sample { index, pos, .. }| {
            // This term is valid for constant or dynamic background potentials.
            let constant_term = |field: T| {
                // Gradient of the unnormalized weight for the current sample point.
                let dwdp = -kernel.with_closest_dist(dist).grad(q, pos);
                dwdp * (-field * wb * weight_sum_inv * weight_sum_inv)
            };

            match bg_field_value {
                BackgroundFieldValue::None => Vector3::zeros(),
                BackgroundFieldValue::Constant(field) => constant_term(field),
                BackgroundFieldValue::ClosestSampleNormalDisp
                | BackgroundFieldValue::ClosestSampleDistance => {
                    let mut grad = constant_term(dist);

                    if index == closest_sample_index {
                        grad += dwbdp * (dist * weight_sum_inv * (T::one() - weight_sum_inv * wb))
                            - disp * (wb * weight_sum_inv / dist)
                    }

                    grad
                }
            }
        })
    }
}
