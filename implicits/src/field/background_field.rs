use crate::field::samples::{Sample, SamplesView};
use crate::kernel::{RadialKernel, SphericalKernel};
use geo::math::{Matrix3, Vector3};
use geo::Real;
use serde::{Deserialize, Serialize};

/// Different types of background fields supported.
#[derive(Copy, Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum BackgroundFieldType {
    /// Use a zero background field.
    Zero,
    /// Use the background field given in the input.
    FromInput,
    /// Signed distance to the closest sample.
    DistanceBased,
}

/// Parameters used to pick which type of background field should be used.
#[derive(Copy, Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BackgroundFieldParams {
    pub field_type: BackgroundFieldType,
    pub weighted: bool,
}

/// Precomputed data used for background field computation.
#[derive(Copy, Clone, Debug, PartialEq)]
pub(crate) enum BackgroundFieldValue<V> {
    /// The value of the background field at the query point.
    Constant(V),
    /// A Dynamic field is computed based on the signed distance to the closest sample point. The
    /// sign is determined by the orienatation with respect to sample normals.
    ClosestSampleSignedDistance,
}

impl<V: num_traits::Zero> BackgroundFieldValue<V> {
    /// Use this constructor for computing the field or its derivatives.
    /// For computing derivatives, the field value is not actually needed, so we default it to zero
    /// if it's not provided.
    pub fn new(field_type: BackgroundFieldType, field_value: Option<V>) -> Self {
        let field_value = field_value.unwrap_or_else(V::zero);
        match field_type {
            BackgroundFieldType::Zero => BackgroundFieldValue::Constant(V::zero()),
            BackgroundFieldType::FromInput => BackgroundFieldValue::Constant(field_value),
            BackgroundFieldType::DistanceBased => BackgroundFieldValue::ClosestSampleSignedDistance,
        }
    }
}

/// An enum identifying the index of the closest sample to a given query point. A local index
/// indicates that the closest point is within some radius (e.g. kernel radius) of the query point.
/// A `Global` index is outside this radius, which indicates that the query point is outside the
/// local radius of the samples, and the field is purely a background field.
#[derive(Copy, Clone, Debug, Serialize, Deserialize)]
pub enum ClosestIndex {
    Local(usize),
    Global(usize),
}

impl ClosestIndex {
    /// Get the index ignoring whether it is local or global.
    pub fn get(self) -> usize {
        match self {
            ClosestIndex::Local(index) | ClosestIndex::Global(index) => index,
        }
    }
}

/// This `struct` represents the data needed to compute a background field at a local query
/// point. This `struct` also conveniently computes useful information about the neighbourhood
/// (like closest distance to a sample point) that can be reused elsewhere.
#[derive(Clone, Debug)]
pub(crate) struct BackgroundField<'a, T, V, K>
where
    T: Real,
    K: RadialKernel<T> + Clone + std::fmt::Debug,
{
    /// Position of the point at which we should evaluate the field.
    pub query_pos: Vector3<T>,
    /// Samples that influence the field.
    pub samples: SamplesView<'a, 'a, T>,
    /// Data needed to compute the background field value and its derivative.
    pub bg_field_value: BackgroundFieldValue<V>,
    /// Determines if this background field should be mixed in with the local field.
    pub weighted: bool,
    /// The sum of all the weights in the neighbourhood of the query point.
    pub weight_sum: T,
    /// The radial kernel used to compute the weights.
    pub kernel: K,
    /// The distance to the closest sample.
    pub closest_sample_dist: T,
    /// Displacement vector to the query point from the closest sample point.
    pub closest_sample_disp: Vector3<T>,
    /// The index of the closest sample point in the samples neighbourhood.
    pub closest_sample_index: ClosestIndex,
}

impl<'a, T, V, K> BackgroundField<'a, T, V, K>
where
    T: Real,
    V: Copy + Clone + std::fmt::Debug + PartialEq + num_traits::Zero,
    K: SphericalKernel<T> + Copy + std::fmt::Debug + Send + Sync + 'a,
{
    /// Build a background field.
    /// This function returns the `InvalidBackgroundConstruction` error when `local_samples_view`
    /// is empty and `global_closest` is not given. This means that there is not enough information
    /// in the input to actually build the desired background field.
    /// This function is the most general for constructing Background fields. It is useful in code
    /// that may optionally build local and global fields. In other words, passing in the
    /// `global_closest` index guarantees that this function wont fail.
    pub(crate) fn new(
        q: Vector3<T>,
        local_samples_view: SamplesView<'a, 'a, T>,
        global_closest: Option<usize>,
        kernel: K,
        bg_params: BackgroundFieldParams,
        bg_value: Option<V>,
    ) -> Result<Self, crate::Error> {
        if let Some(closest) = global_closest {
            Ok(Self::global(
                q,
                local_samples_view,
                closest,
                kernel,
                bg_params,
                bg_value,
            ))
        } else {
            Self::local(q, local_samples_view, kernel, bg_params, bg_value)
        }
    }

    /// A Helper function to compute the closest sample to the query point `q` in the given set of
    /// samples. It is important to use this same function when determining closest points (as in
    /// local and global field constructors below) because this ensures that in the precence of
    /// discontinuities in the global field, the derivatives will be computed using the same
    /// closest points as the potential field itself.
    fn closest_sample(q: Vector3<T>, samples: SamplesView<'a, 'a, T>) -> Option<usize> {
        samples
            .iter()
            .map(|Sample { index, pos, .. }| (index, (q - pos).norm_squared()))
            .min_by(|(_, d0), (_, d1)| {
                d0.partial_cmp(d1)
                    .unwrap_or_else(|| panic!("Detected NaN. Please report this bug. Failed to compare distances {:?} and {:?}", d0, d1))
            }).map(|(index, _)| index)
    }

    /// Build a local background field that is valid only when `local_samples_view` is non empty.
    /// This function returns the `InvalidBackgroundConstruction` error when `local_samples_view`
    /// is empty. This means that there is not enough information in the input to actually build
    /// the desired background field.
    pub(crate) fn local(
        q: Vector3<T>,
        local_samples_view: SamplesView<'a, 'a, T>,
        kernel: K,
        bg_params: BackgroundFieldParams,
        bg_value: Option<V>,
    ) -> Result<Self, crate::Error> {
        let closest_sample_index = {
            if let Some(index) = Self::closest_sample(q, local_samples_view) {
                ClosestIndex::Local(index)
            } else {
                return Err(crate::Error::InvalidBackgroundConstruction);
            }
        };

        Ok(Self::new_impl(
            q,
            local_samples_view,
            closest_sample_index,
            kernel,
            bg_params,
            bg_value,
        ))
    }

    /// Build a global background field that is valid even when `local_samples_view` is empty. This
    /// is useful for visualization, but ultimately in simulation we wouldn't do this to avoid
    /// unnecessary computations. This function requires the index of the closest sample to be
    /// passed in as `global_closest`. This function doesn't fail.
    pub(crate) fn global(
        q: Vector3<T>,
        local_samples_view: SamplesView<'a, 'a, T>,
        global_closest: usize,
        kernel: K,
        bg_params: BackgroundFieldParams,
        bg_value: Option<V>,
    ) -> Self {
        // Check that the closest sample is in our neighbourhood of samples.

        // Note that the following assertion may sometimes fail. An issue has been filed with the
        // author of rstar: https://github.com/Stoeoef/rstar/issues/13
        // As such it shall remain a debug_assert.
        debug_assert!(
            local_samples_view.is_empty()
                || local_samples_view
                    .iter()
                    .find(|&sample| sample.index == global_closest)
                    .is_some()
        );
        let closest_sample_index = {
            if let Some(index) = Self::closest_sample(q, local_samples_view) {
                ClosestIndex::Local(index)
            } else {
                ClosestIndex::Global(global_closest)
            }
        };

        Self::new_impl(
            q,
            local_samples_view,
            closest_sample_index,
            kernel,
            bg_params,
            bg_value,
        )
    }

    /// Internal constructor that is guaranteed to build a background field from the given
    /// parameters where the given `closest_sample_index` is already determined.
    fn new_impl(
        q: Vector3<T>,
        local_samples_view: SamplesView<'a, 'a, T>,
        closest_sample_index: ClosestIndex,
        kernel: K,
        bg_params: BackgroundFieldParams,
        bg_value: Option<V>,
    ) -> Self {
        let (closest_sample_disp, closest_sample_dist) = {
            let Sample { pos, .. } = local_samples_view.at_index(closest_sample_index.get());
            let disp = q - pos;
            (disp, disp.norm())
        };

        // Compute the weight sum here. This will be available to the users of bg data.
        let mut weight_sum = T::zero();
        for Sample { pos, .. } in local_samples_view.iter() {
            let w = kernel.with_closest_dist(closest_sample_dist).eval(q, pos);
            weight_sum += w;
        }

        // Initialize the background field struct.
        let mut bg = BackgroundField {
            query_pos: q,
            samples: local_samples_view,
            bg_field_value: BackgroundFieldValue::new(bg_params.field_type, bg_value),
            weighted: bg_params.weighted,
            weight_sum,
            kernel,
            closest_sample_dist,
            closest_sample_disp,
            closest_sample_index,
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
        if self.kernel.radius() < self.closest_sample_dist {
            // Closest point is not inside the local neighbourhood.
            T::one()
        } else if self.weighted {
            self.kernel
                .f(self.kernel.radius() - self.closest_sample_dist)
        } else {
            T::zero()
        }
    }

    /// Unnormalized background weight derivative with respect to the given sample point index.
    /// If the given sample is not the closest sample, then the derivative is zero.
    /// If the given index is `None`, then the derivative is with respect to the query point.
    #[inline]
    pub(crate) fn background_weight_gradient(&self, index: Option<usize>) -> Vector3<T> {
        if self.kernel.radius() < self.closest_sample_dist || !self.weighted {
            return Vector3::zeros();
        }

        if let Some(index) = index {
            // Derivative with respect to the sample at the given index
            if let ClosestIndex::Local(local_sample_index) = self.closest_sample_index {
                if index == local_sample_index {
                    return self.closest_sample_disp
                        * (self
                            .kernel
                            .df(self.kernel.radius() - self.closest_sample_dist)
                            / self.closest_sample_dist);
                }
            }
            Vector3::zeros()
        } else {
            // Derivative with respect to the query position
            -self.closest_sample_disp
                * (self
                    .kernel
                    .df(self.kernel.radius() - self.closest_sample_dist)
                    / self.closest_sample_dist)
        }
    }

    /// Unnormalized background wieght hessian with respect to the given sample point index.
    /// If the given sample is not the closest sample, then the drivative is zero.
    /// If the given index is `None`, then the derivative is with respect to the query point.
    #[inline]
    pub(crate) fn background_weight_hessian(&self, index: Option<usize>) -> Matrix3<T> {
        if self.kernel.radius() < self.closest_sample_dist || !self.weighted {
            return Matrix3::zeros();
        }

        if let Some(index) = index {
            match self.closest_sample_index {
                ClosestIndex::Local(local_sample_index) => {
                    if index != local_sample_index {
                        // requested point is not the closest one, but it's in the local neighbourhood.
                        return Matrix3::zeros();
                    }
                }
                ClosestIndex::Global(_) =>
                // No local neighbourhood, this hessian is for background only.
                {
                    return Matrix3::zeros();
                }
            }
        }

        // Derivative with respect to the sample at the given index
        // or Derivative with respect to the query position ( same thing )
        let disp = self.closest_sample_disp / self.closest_sample_dist;
        let dwb = self
            .kernel
            .df(self.kernel.radius() - self.closest_sample_dist);
        let ddwb = self
            .kernel
            .ddf(self.kernel.radius() - self.closest_sample_dist);
        disp * ddwb * disp.transpose()
            - ((Matrix3::identity() - disp * disp.transpose()) * (dwb / self.closest_sample_dist))
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
        let field_val = match self.bg_field_value {
            BackgroundFieldValue::Constant(val) => val,
            BackgroundFieldValue::ClosestSampleSignedDistance => V::zero(), // This is non sensical for vector fields
        };

        field_val * self.background_weight()
    }
}

/// Free function to compute the indices corresponding to the [`hessian_block`] function.
pub(crate) fn hessian_block_indices<'a>(
    weighted: bool,
    nbrs: impl Iterator<Item = usize> + Clone + 'a,
) -> impl Iterator<Item = (usize, usize)> + 'a {
    let diag_iter = nbrs.clone().map(move |i| (i, i));

    let off_diag_iter = if weighted {
        Some(
            nbrs.clone()
                .flat_map(move |i| nbrs.clone().filter(move |&j| i < j).map(move |j| (j, i))),
        )
    } else {
        None
    };

    diag_iter.chain(off_diag_iter.into_iter().flatten())
}

// The following functions are value for scalar fields of type T only.
impl<'a, T, K> BackgroundField<'a, T, T, K>
where
    T: Real,
    K: SphericalKernel<T> + Copy + std::fmt::Debug + Send + Sync + 'a,
{
    /// Return `1.0` if the query point is outside and `-1.0` if it's inside.
    fn orientation(&self) -> T {
        let nml = self.samples.all_normals()[self.closest_sample_index.get()];
        nml.dot(self.closest_sample_disp).signum()
    }

    /// Return the background field value.
    pub(crate) fn field_value(&self) -> T {
        // Unpack background data.
        match self.bg_field_value {
            BackgroundFieldValue::Constant(val) => val,
            BackgroundFieldValue::ClosestSampleSignedDistance => {
                self.closest_sample_dist * self.orientation()
            }
        }
    }

    /// Return the background field gradient. With respect to the query position.
    pub(crate) fn field_gradient(&self) -> Vector3<T> {
        // Unpack background data.
        let BackgroundField {
            bg_field_value,
            closest_sample_dist: dist,
            closest_sample_disp: disp,
            ..
        } = *self;

        match bg_field_value {
            BackgroundFieldValue::Constant(_) => Vector3::zeros(),
            BackgroundFieldValue::ClosestSampleSignedDistance => {
                // The strategy for removing NaNs here is that degenerate cases may produce NaNs
                // that may otherwise be zeroed out elsewhere. This check ensures that this
                // cancellation happens.
                if dist != T::zero() {
                    // Remove possibility of NaN
                    disp * (self.orientation() / dist)
                } else {
                    Vector3::zeros()
                }
            }
        }
    }

    /// Return the background field Hessian. Note that the Hessian is the same regardless of
    /// whether the independent variable considered is sample point position or query point
    /// position.
    pub(crate) fn field_hessian(&self) -> Matrix3<T> {
        // Unpack background data.
        let BackgroundField {
            bg_field_value,
            closest_sample_dist: dist,
            closest_sample_disp: disp,
            ..
        } = *self;

        match bg_field_value {
            BackgroundFieldValue::Constant(_) => Matrix3::zeros(),
            BackgroundFieldValue::ClosestSampleSignedDistance => {
                if dist != T::zero() {
                    // Remove possibility of NaN
                    let dist_inv = self.orientation() / dist;
                    let dir = disp * dist_inv;
                    Matrix3::diag([dist_inv; 3]) - dir * (dir.transpose() * dist_inv)
                } else {
                    Matrix3::zeros()
                }
            }
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
            weighted,
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
        let dwbdp = self.background_weight_gradient(Some(closest_sample_index.get()));

        let field = self.field_value();

        let bg_grad = self.field_gradient();

        samples.into_iter().map(move |Sample { index, pos, .. }| {
            let mut grad = Vector3::zeros();

            if bg_field_value == BackgroundFieldValue::ClosestSampleSignedDistance
                && index == closest_sample_index.get()
            {
                grad -= bg_grad * wb;
            }

            if !weighted {
                return grad;
            }

            // This term is valid for constant or dynamic background potentials.
            grad += {
                // Gradient of the unnormalized weight for the current sample point.
                let dwdp = kernel.with_closest_dist(dist).grad(q, pos);
                dwdp * (field * wb * weight_sum_inv)
            };

            if bg_field_value == BackgroundFieldValue::ClosestSampleSignedDistance
                && index == closest_sample_index.get()
            {
                grad += dwbdp * (field * weight_sum_inv * (T::one() - wb))
            }

            grad
        })
    }

    ///// Outside the local region.
    //pub(crate) fn compute_global_jacobian(&self) -> Vector3<T> {
    //    let weight_sum_inv = self.weight_sum_inv();

    //    // The normalized weight evaluated at the distance to the boundary of the
    //    // neighbourhood.
    //    let wb = self.background_weight() * weight_sum_inv;

    //    let bg_grad = self.field_gradient();

    //    bg_grad * (-wb)
    //}

    /// Compute background field second derivative contribution.
    /// Compute Hessian if the closest point is in the neighbourhood. Otherwise we
    /// assume the background field is constant. This Hessian is taken with respect to the
    /// sample points.
    pub(crate) fn hessian_blocks(&self) -> impl Iterator<Item = Matrix3<T>> + 'a {
        // Unpack background data.
        let BackgroundField {
            query_pos: q,
            samples,
            bg_field_value,
            weighted,
            kernel,
            closest_sample_dist: dist,
            closest_sample_index,
            ..
        } = *self;

        let weight_sum_inv = self.weight_sum_inv();

        let wb = self.background_weight();
        let dwb = self.background_weight_gradient(Some(closest_sample_index.get()));
        let ddwb = self.background_weight_hessian(Some(closest_sample_index.get()));
        let f = self.field_value();
        let df = -self.field_gradient(); // negative since it's wrt sample points.
        let ddf = self.field_hessian();

        use crate::hessian::sym_outer;

        // Diagonal entries.
        let diag_iter = samples.into_iter().map(move |Sample { index, pos, .. }| {
            let mut hess = Matrix3::zeros();

            if bg_field_value == BackgroundFieldValue::ClosestSampleSignedDistance
                && index == closest_sample_index.get()
            {
                hess += ddf * wb;
            }

            if !weighted {
                return hess * weight_sum_inv;
            }

            let dw = kernel.with_closest_dist(dist).grad(q, pos);
            let _2 = T::from(2.0).unwrap();

            hess += {
                let ddw = kernel.with_closest_dist(dist).hess(q, pos);
                (dw * (dw.transpose() * (_2 * weight_sum_inv)) - ddw) * (f * wb * weight_sum_inv)
            };

            if bg_field_value == BackgroundFieldValue::ClosestSampleSignedDistance
                && index == closest_sample_index.get()
            {
                let factor = T::one() - wb * weight_sum_inv;
                hess += ddwb * (factor * f)
                    - dwb * dwb.transpose() * (factor * _2 * f * weight_sum_inv)
                    + sym_outer(dwb, dw)
                        * ((T::one() - _2 * wb * weight_sum_inv) * f * weight_sum_inv)
                    + sym_outer(df, dwb) * factor
                    + sym_outer(df, dw) * wb * weight_sum_inv
            }

            hess * weight_sum_inv
        });

        // Off-diagonal entries. These are only present when the background is weighted in the
        // local region.
        let off_diag_iter = if weighted {
            Some(samples.into_iter().flat_map(move |sample_i| {
                let dw_i = kernel.with_closest_dist(dist).grad(q, sample_i.pos);
                samples
                    .into_iter()
                    .filter(move |sample_j| sample_i.index < sample_j.index)
                    .map(move |sample_j| {
                        let _2 = T::from(2.0).unwrap();
                        let dw_j = kernel.with_closest_dist(dist).grad(q, sample_j.pos);

                        let dwdw_factor = _2 * f * weight_sum_inv;
                        let dwdwb_factor = f * (T::one() - _2 * wb * weight_sum_inv);

                        let hess = if sample_i.index == closest_sample_index.get() {
                            dw_j * ((dw_i * dwdw_factor + df) * wb + dwb * dwdwb_factor).transpose()
                        } else if sample_j.index == closest_sample_index.get() {
                            ((df + dw_j * dwdw_factor) * wb + dwb * dwdwb_factor) * dw_i.transpose()
                        } else {
                            (dw_j * dw_i.transpose()) * (dwdw_factor * wb)
                        };

                        hess * weight_sum_inv * weight_sum_inv
                    })
            }))
        } else {
            None
        };

        diag_iter.chain(off_diag_iter.into_iter().flatten())
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
            weighted,
            closest_sample_dist: dist,
            ..
        } = *self;

        let weight_sum_inv = self.weight_sum_inv();

        // The normalized weight evaluated at the distance to the boundary of the
        // neighbourhood.
        let wb = self.background_weight() * weight_sum_inv;

        let grad = if weighted {
            let mut dw_total: Vector3<T> = samples
                .into_iter()
                .map(move |Sample { pos, .. }| kernel.with_closest_dist(dist).grad(q, pos))
                .sum();

            let dwb = self.background_weight_gradient(None);
            dw_total += dwb;

            let bg_val = self.field_value();

            (dwb - dw_total * wb) * (bg_val * weight_sum_inv)
        } else {
            Vector3::zeros()
        };

        grad + self.field_gradient() * wb
    }

    /// Compute background field query Hessian contribution.
    /// Compute second derivative if the closest point is in the neighbourhood. Otherwise we
    /// assume the background field is constant. This Hessian is taken with respect to the
    /// query position.
    pub(crate) fn compute_query_hessian(&self) -> Matrix3<T> {
        use crate::hessian::sym_outer;

        // Unpack background data.
        let BackgroundField {
            query_pos: q,
            samples,
            kernel,
            weighted,
            closest_sample_dist: dist,
            ..
        } = *self;

        let weight_sum_inv = self.weight_sum_inv();

        // The unnormalized weight evaluated at the distance to the boundary of the neighbourhood.
        let wb = self.background_weight();

        let mut hess = self.field_hessian() * wb;
        if weighted {
            let mut dw_total: Vector3<T> = samples
                .into_iter()
                .map(move |Sample { pos, .. }| kernel.with_closest_dist(dist).grad(q, pos))
                .sum();
            let dwb = self.background_weight_gradient(None);
            dw_total += dwb;

            let mut ddw_total: Matrix3<T> = samples
                .into_iter()
                .map(move |Sample { pos, .. }| kernel.with_closest_dist(dist).hess(q, pos))
                .sum();
            let ddwb = self.background_weight_hessian(None);
            ddw_total += ddwb;

            let f = self.field_value();
            let df = self.field_gradient();
            let _2 = T::from(2.0).unwrap();

            hess += sym_outer(df, dwb)
                - sym_outer(dw_total, df) * (wb * weight_sum_inv)
                - sym_outer(dw_total, dwb) * (f * weight_sum_inv)
                + dw_total
                    * (dw_total.transpose() * (_2 * f * wb * weight_sum_inv * weight_sum_inv))
                + ddwb * f
                - ddw_total * f * wb * weight_sum_inv
        }

        hess * weight_sum_inv
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hessian::print_full_hessian;
    use crate::{make_grid, mls_from_polymesh, Error, KernelType, Params, SampleType};
    use geo::mesh::VertexPositions;

    #[test]
    fn constant_unweighted_bg() -> Result<(), Error> {
        // Create a surface sample mesh.
        let octahedron_trimesh = utils::make_sample_octahedron();
        let mut sphere = geo::mesh::PolyMesh::from(octahedron_trimesh);

        // Translate the mesh up
        utils::translate(&mut sphere, [0.0, 0.0, 3.0]);

        // Construct the implicit surface.
        let mut surface = mls_from_polymesh(
            &sphere,
            Params {
                kernel: KernelType::Approximate {
                    tolerance: 0.00001,
                    radius_multiplier: 2.45,
                },
                background_field: BackgroundFieldParams {
                    field_type: BackgroundFieldType::FromInput,
                    weighted: false,
                },
                sample_type: SampleType::Face,
                max_step: 100.0,
            },
        )?;

        // Make a grid mesh to be tested against.
        let grid = make_grid(5, 5);

        // Compute potential.
        let mut potential = vec![-1.0; grid.vertex_positions().len()];
        surface.compute_neighbours(grid.vertex_positions());
        surface.potential(grid.vertex_positions(), &mut potential)?;

        // Potential should be all -1 (unchanged) because the radius is too small.
        for &pot in potential.iter() {
            assert_eq!(pot, -1.0);
        }
        Ok(())
    }

    #[test]
    fn constant_bg() -> Result<(), Error> {
        // Create a surface sample mesh.
        let octahedron_trimesh = utils::make_sample_octahedron();
        let mut sphere = geo::mesh::PolyMesh::from(octahedron_trimesh);

        // Translate the mesh up
        utils::translate(&mut sphere, [0.0, 0.0, 3.0]);

        // Construct the implicit surface.
        let mut surface = mls_from_polymesh(
            &sphere,
            Params {
                kernel: KernelType::Approximate {
                    tolerance: 0.00001,
                    radius_multiplier: 2.45,
                },
                background_field: BackgroundFieldParams {
                    field_type: BackgroundFieldType::FromInput,
                    weighted: true,
                },
                sample_type: SampleType::Face,
                max_step: 100.0,
            },
        )?;

        // Make a grid mesh to be tested against.
        let grid = make_grid(5, 5);

        // Compute potential.
        let mut potential = vec![-1.0; grid.vertex_positions().len()];
        surface.compute_neighbours(grid.vertex_positions());
        surface.potential(grid.vertex_positions(), &mut potential)?;

        // Potential should be all -1 (unchanged) because the radius is too small.
        for &pot in potential.iter() {
            assert_relative_eq!(pot, -1.0);
        }
        Ok(())
    }

    fn distance_based_bg_sample_hessian_tester(
        radius_multiplier: f64,
        mesh: &geo::mesh::TriMesh<f64>,
        qs: &[Vector3<f64>],
    ) -> Result<(), Error> {
        use autodiff::F;

        // Create a surface sample mesh.
        let sphere = geo::mesh::PolyMesh::from(mesh.clone());

        let bg_field_params = BackgroundFieldParams {
            field_type: BackgroundFieldType::DistanceBased,
            weighted: true,
        };

        let tolerance = 0.00001;
        let kernel_type = KernelType::Approximate {
            tolerance,
            radius_multiplier,
        };

        // Construct the implicit surface.
        let mut surface = mls_from_polymesh::<F>(
            &sphere,
            Params {
                kernel: kernel_type,
                background_field: bg_field_params,
                sample_type: SampleType::Face,
                max_step: 0.0,
            },
        )?;

        let query_points: Vec<_> = qs
            .iter()
            .map(|&q| q.map(|x| F::cst(x)).into_inner())
            .collect();

        surface.compute_neighbours(&query_points);
        let neighs: Vec<_> = surface.trivial_neighbourhood_seq()?.collect();
        let mut samples = surface.samples().clone();

        let radius = surface.radius();

        let kernel = crate::kernel::LocalApproximate::new(radius, tolerance);

        // Compute potential.
        for (&q, n) in query_points.iter().zip(neighs.iter()) {
            let mut hess = vec![vec![0.0; samples.len() * 3]; samples.len() * 3];

            {
                let view = SamplesView::new(n, &samples);
                if view.is_empty() {
                    continue;
                }
                let bg = BackgroundField::<F, F, _>::local(
                    Vector3(q),
                    view,
                    kernel,
                    bg_field_params,
                    None,
                )
                .unwrap();
                let hess_iter = bg
                    .hessian_blocks()
                    .zip(hessian_block_indices(bg.weighted, n.iter().cloned()));

                for (h, (j, i)) in hess_iter {
                    for r in 0..3 {
                        for c in 0..3 {
                            hess[3 * i + c][3 * j + r] = h[c][r].value();
                            if i != j {
                                hess[3 * j + r][3 * i + c] = h[c][r].value();
                            }
                        }
                    }
                }
            }

            let mut ad_hess = vec![vec![0.0; samples.len() * 3]; samples.len() * 3];

            let mut success = true;

            for wrt_sample in 0..samples.len() {
                for wrt in 0..3 {
                    samples.points[wrt_sample][wrt] = F::var(samples.points[wrt_sample][wrt]);
                    let jac: Vec<_> = {
                        let view = SamplesView::new(n, &samples);
                        let bg =
                            BackgroundField::local(Vector3(q), view, kernel, bg_field_params, None)
                                .unwrap();
                        let mut jac = vec![Vector3::zeros(); samples.len()];
                        for (sample, j) in view.iter().zip(bg.compute_jacobian()) {
                            jac[sample.index] = j;
                        }
                        jac
                    };
                    for sample in 0..samples.len() {
                        for k in 0..3 {
                            if !relative_eq!(
                                hess[3 * sample + k][3 * wrt_sample + wrt],
                                jac[sample][k].deriv(),
                                max_relative = 1e-5
                            ) {
                                println!(
                                    "{:.5} vs {:.5}",
                                    hess[3 * sample + k][3 * wrt_sample + wrt],
                                    jac[sample][k].deriv()
                                );
                                success = false;
                            }
                            ad_hess[3 * sample + k][3 * wrt_sample + wrt] += jac[sample][k].deriv();
                        }
                    }
                    samples.points[wrt_sample][wrt] = F::cst(samples.points[wrt_sample][wrt]);
                }
            }

            if !success {
                print_full_hessian(&hess, surface.num_samples() * 3, "BG Hess");
                print_full_hessian(&ad_hess, surface.num_samples() * 3, "BG AD Hess");
            }
            assert!(success);
        }
        Ok(())
    }

    fn distance_based_bg_query_hessian_tester(
        radius_multiplier: f64,
        mesh: &geo::mesh::TriMesh<f64>,
        qs: &[Vector3<f64>],
    ) -> Result<(), Error> {
        use autodiff::F;

        // Create a surface sample mesh.
        let sphere = geo::mesh::PolyMesh::from(mesh.clone());

        let bg_field_params = BackgroundFieldParams {
            field_type: BackgroundFieldType::DistanceBased,
            weighted: true,
        };

        let tolerance = 0.00001;
        let kernel_type = KernelType::Approximate {
            tolerance,
            radius_multiplier,
        };

        // Construct the implicit surface.
        let mut surface = mls_from_polymesh::<F>(
            &sphere,
            Params {
                kernel: kernel_type,
                background_field: bg_field_params,
                sample_type: SampleType::Face,
                max_step: 0.0,
            },
        )?;

        let mut query_points: Vec<_> = qs
            .iter()
            .map(|&q| q.map(|x| F::cst(x)).into_inner())
            .collect();

        surface.compute_neighbours(&query_points);
        let neighs: Vec<_> = surface.trivial_neighbourhood_seq()?.collect();
        let samples = surface.samples().clone();

        let radius = surface.radius();

        let kernel = crate::kernel::LocalApproximate::new(radius, tolerance);

        // Compute potential.
        for (q, n) in query_points.iter_mut().zip(neighs.iter()) {
            let view = SamplesView::new(n, &samples);
            if view.is_empty() {
                continue;
            }

            let hess = {
                let bg = BackgroundField::<F, F, _>::local(
                    Vector3(*q),
                    view,
                    kernel,
                    bg_field_params,
                    None,
                )
                .unwrap();
                bg.compute_query_hessian()
            };

            let mut success = true;

            for wrt in 0..3 {
                q[wrt] = F::var(q[wrt]);
                let jac = {
                    let bg =
                        BackgroundField::local(Vector3(*q), view, kernel, bg_field_params, None)
                            .unwrap();
                    bg.compute_query_jacobian()
                };

                for k in 0..3 {
                    if !relative_eq!(hess[k][wrt].value(), jac[k].deriv(), max_relative = 1e-5) {
                        println!("{:.5} vs {:.5}", hess[k][wrt].value(), jac[k].deriv());
                        success = false;
                    }
                }
                q[wrt] = F::cst(q[wrt]);
            }

            assert!(success);
        }
        Ok(())
    }

    #[test]
    fn two_triangles_distance_based_bg() -> Result<(), Error> {
        let (verts, indices) =
            crate::jacobian::make_two_test_triangles(0.0, &mut || Vector3::zeros());
        let mesh = geo::mesh::TriMesh::new(
            reinterpret::reinterpret_vec(verts),
            reinterpret::reinterpret_vec(indices),
        );

        let query_points = vec![
            Vector3([0.0, 0.2, 0.0]),
            Vector3([0.0, 0.0001, 0.0]),
            Vector3([0.0, -0.4, 0.0]),
        ];

        for i in 1..50 {
            let radius_mult = 1.0 + 0.1 * i as f64;
            distance_based_bg_sample_hessian_tester(radius_mult, &mesh, &query_points)?;
            distance_based_bg_query_hessian_tester(radius_mult, &mesh, &query_points)?;
        }
        Ok(())
    }

    #[test]
    fn octahedron_distance_based_bg() -> Result<(), Error> {
        let mesh = utils::make_sample_octahedron();
        let grid = make_grid(5, 5);

        let query_points: Vec<_> = grid.vertex_position_iter().map(|&q| Vector3(q)).collect();

        for i in 1..50 {
            let radius_mult = 1.0 + 0.1 * i as f64;
            distance_based_bg_sample_hessian_tester(radius_mult, &mesh, &query_points)?;
            distance_based_bg_query_hessian_tester(radius_mult, &mesh, &query_points)?;
        }
        Ok(())
    }
}
