use crate::field::samples::{Sample, SamplesView};
use crate::kernel::SphericalKernel;
use geo::math::Vector3;
use geo::Real;

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
pub(crate) enum BackgroundFieldValue<V> {
    /// No value, nothing to mix in, background weight is always zero.
    None,
    /// The value of the background field at the query point.
    Constant(V),
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
            BackgroundFieldType::DistanceBased => BackgroundFieldValue::ClosestSampleDistance,
            BackgroundFieldType::NormalBased => BackgroundFieldValue::ClosestSampleNormalDisp,
        }
    }

    /// Use this constructor for computing the field.
    pub fn val(ty: BackgroundFieldType, field_value: V) -> Self {
        match ty {
            BackgroundFieldType::None => BackgroundFieldValue::None,
            BackgroundFieldType::Zero => BackgroundFieldValue::Constant(V::zero()),
            BackgroundFieldType::FromInput => BackgroundFieldValue::Constant(field_value),
            BackgroundFieldType::DistanceBased => BackgroundFieldValue::ClosestSampleDistance,
            BackgroundFieldType::NormalBased => BackgroundFieldValue::ClosestSampleNormalDisp,
        }
    }
}

/// This struct represents the data needed to compute a background field at a local query
/// point. This struct also conviently computes useful information about the neighbourhood (like
/// closest distance to a sample point) that can be reused elsewhere.
#[derive(Clone, Debug)]
pub(crate) struct BackgroundField<'a, T, V, K>
where
    T: Real,
    K: SphericalKernel<T> + Clone + std::fmt::Debug,
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

impl<'a, T, V, K> BackgroundField<'a, T, V, K>
where
    T: Real,
    V: Copy + Clone + std::fmt::Debug + PartialEq,
    K: SphericalKernel<T> + Copy + std::fmt::Debug + Send + Sync + 'a,
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
        if self.weight_sum == T::zero() {
            T::zero()
        } else {
            T::one() / self.weight_sum
        }
    }

    /// The background weight is given by `w(b)` where `w` is the kernel function,
    /// `b = r - |q - p|` is the distance from the closest point to the edge of the kernel
    /// boundary. Note that this weight is unnormalized.
    pub(crate) fn background_weight(&self) -> T {
        match self.bg_field_value {
            BackgroundFieldValue::None => T::zero(),
            _ => self.kernel.f(self.radius - self.closest_sample_dist),
        }
    }

    /// Unnormalized background wieght derivative with respect to the given sample point index.
    /// If the given sample is not the closest sample, then the drivative is zero.
    /// If the given index is `None`, then the derivative is with respect to the query point.
    #[inline]
    pub(crate) fn background_weight_gradient(&self, index: Option<usize>) -> Vector3<T> {
        if self.bg_field_value == BackgroundFieldValue::None {
            return Vector3::zeros();
        }

        if let Some(index) = index {
            // Derivative with respect to the sample at the given index
            if index == self.closest_sample_index {
                self.closest_sample_disp
                    * (self.kernel.df(self.radius - self.closest_sample_dist)
                        / self.closest_sample_dist)
            } else {
                Vector3::zeros()
            }
        } else {
            // Derivative with respect to the query position
            -self.closest_sample_disp
                * (self.kernel.df(self.radius - self.closest_sample_dist)
                    / self.closest_sample_dist)
        }
    }
}

impl<'a, T, V, K> BackgroundField<'a, T, V, K>
where
    T: Real,
    V: Copy + Clone + std::fmt::Debug + PartialEq + std::ops::Mul<T, Output = V> + num_traits::Zero,
    K: SphericalKernel<T> + Copy + std::fmt::Debug + Send + Sync + 'a,
{
    /// Compute the unnormalized weighted background field value. This is typically very
    /// simple, but the caller must remember to multiply it by the `weight_sum_inv` to get the true
    /// background field contribution.
    pub(crate) fn compute_unnormalized_weighted_vector_field(&self) -> V {
        // Unpack background data.
        let BackgroundField { bg_field_value, .. } = *self;

        let field_val = match bg_field_value {
            BackgroundFieldValue::None => V::zero(),
            BackgroundFieldValue::Constant(val) => val,
            BackgroundFieldValue::ClosestSampleDistance => V::zero(),
            BackgroundFieldValue::ClosestSampleNormalDisp => V::zero(),
        };

        field_val * self.background_weight()
    }
}

// The following functions are value for scalar fields of type T only.
impl<'a, T, K> BackgroundField<'a, T, T, K>
where
    T: Real,
    K: SphericalKernel<T> + Copy + std::fmt::Debug + Send + Sync + 'a,
{
    /// Return the background field value.
    pub(crate) fn field_value(&self) -> T {
        // Unpack background data.
        match self.bg_field_value {
            BackgroundFieldValue::None => T::zero(),
            BackgroundFieldValue::Constant(val) => val,
            BackgroundFieldValue::ClosestSampleDistance
            | BackgroundFieldValue::ClosestSampleNormalDisp => self.closest_sample_dist,
        }
    }

    /// Return the background field gradient.
    pub(crate) fn field_gradient(&self) -> Vector3<T> {
        // Unpack background data.
        let BackgroundField {
            bg_field_value,
            closest_sample_dist: dist,
            closest_sample_disp: disp,
            ..
        } = *self;

        match bg_field_value {
            BackgroundFieldValue::None | BackgroundFieldValue::Constant(_) => Vector3::zeros(),
            BackgroundFieldValue::ClosestSampleDistance
            | BackgroundFieldValue::ClosestSampleNormalDisp => disp * (T::one() / dist),
        }
    }

    /// Compute the unnormalized weighted background field value. This is typically very
    /// simple, but the caller must remember to multiply it by the `weight_sum_inv` to get the true
    /// background field contribution.
    pub(crate) fn compute_unnormalized_weighted_scalar_field(&self) -> T {
        self.field_value() * self.background_weight()
    }

    /// Compute background field derivative contribution.
    /// Compute derivative if the closest point is in the neighbourhood. Otherwise we
    /// assume the background field is constant. This Jacobian is taken with respect to the
    /// sample points.
    pub(crate) fn compute_jacobian(&self) -> impl Iterator<Item = Vector3<T>> + 'a {
        // Unpack background data.
        let BackgroundField {
            query_pos: q,
            samples,
            bg_field_value,
            kernel,
            closest_sample_dist: dist,
            closest_sample_index,
            ..
        } = *self;

        let weight_sum_inv = self.weight_sum_inv();

        // The normalized weight evaluated at the distance to the boundary of the
        // neighbourhood.
        let wb = self.background_weight() * weight_sum_inv;

        // Gradient of the unnormalized weight evaluated at the distance to the
        // boundary of the neighbourhood.
        let dwbdp = self.background_weight_gradient(Some(closest_sample_index));

        let bg_grad = self.field_gradient();

        samples.into_iter().map(move |Sample { index, pos, .. }| {
            // This term is valid for constant or dynamic background potentials.
            let constant_term = |field: T| {
                // Gradient of the unnormalized weight for the current sample point.
                let dwdp = kernel.with_closest_dist(dist).grad(q, pos);
                dwdp * (field * wb * weight_sum_inv)
            };

            match bg_field_value {
                BackgroundFieldValue::None => Vector3::zeros(),
                BackgroundFieldValue::Constant(field) => constant_term(T::from(field).unwrap()),
                BackgroundFieldValue::ClosestSampleNormalDisp
                | BackgroundFieldValue::ClosestSampleDistance => {
                    let mut grad = constant_term(dist);

                    if index == closest_sample_index {
                        grad += dwbdp * (dist * weight_sum_inv * (T::one() - wb)) - bg_grad * wb
                    }

                    grad
                }
            }
        })
    }

    /// Compute background field derivative contribution.
    /// Compute derivative if the closest point is in the neighbourhood. Otherwise we
    /// assume the background field is constant. This Jacobian is taken with respect to the
    /// query position.
    pub(crate) fn compute_query_jacobian(&self) -> Vector3<T> {
        // Unpack background data.
        let BackgroundField {
            query_pos: q,
            samples,
            kernel,
            bg_field_value,
            closest_sample_dist: dist,
            ..
        } = *self;

        let weight_sum_inv = self.weight_sum_inv();

        // The normalized weight evaluated at the distance to the boundary of the
        // neighbourhood.
        let wb = self.background_weight() * weight_sum_inv;

        let mut dw_total = Vector3::zeros();
        for grad in samples
            .into_iter()
            .map(move |Sample { pos, .. }| kernel.with_closest_dist(dist).grad(q, pos))
        {
            dw_total += grad;
        }

        let dwb = self.background_weight_gradient(None);
        dw_total += dwb;

        let bg_val = self.field_value();

        // The term without the background field derivative.
        let constant_term = (dwb - dw_total * wb) * (bg_val * weight_sum_inv);

        let grad = match bg_field_value {
            BackgroundFieldValue::None => Vector3::zeros(),
            _ => constant_term,
        };

        grad + self.field_gradient() * wb
    }
}
