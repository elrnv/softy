use crate::field::samples::{Sample, SamplesView};
use crate::kernel::{RadialKernel, SphericalKernel};
use geo::math::{Matrix3, Vector3};
use geo::Real;

/// Different types of background fields supported.
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum BackgroundFieldType {
    /// Use a zero background field.
    Zero,
    /// Use the background field given in the input.
    FromInput,
    /// Signed distance to the closest sample.
    DistanceBased,
}

/// Parameters used to pick which type of background field should be used.
#[derive(Copy, Clone, Debug, PartialEq)]
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
        let field_value = field_value.unwrap_or(V::zero());
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
#[derive(Copy, Clone, Debug)]
pub enum ClosestIndex {
    Local(usize),
    Global(usize),
}

impl ClosestIndex {
    /// Get the index ignoring whether it is local or global.
    pub fn get(self) -> usize {
        match self {
            ClosestIndex::Local(index) | ClosestIndex::Global(index) => index
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
            let min_sample = local_samples_view.iter()
                .map(|Sample { index, pos, .. }| (index, (q - pos).norm_squared()))
                .min_by(|(_, d0), (_, d1)| {
                    d0.partial_cmp(d1)
                        .expect("Detected NaN. Please report this bug.")
                });
            if let Some((index, _)) = min_sample {
                ClosestIndex::Local(index)
            } else {
                return Err(crate::Error::InvalidBackgroundConstruction);
            }
        };

        Ok(Self::new(q, local_samples_view, closest_sample_index, kernel, bg_params, bg_value))
    }

    /// Build a global background field that is valid even when `local_samples_view` is empty. This
    /// is useful for visualization, but ultimately in simulation we wouldn't do this to avoid
    /// unnecessary computations. This function doesn't fail.
    pub(crate) fn global(
        q: Vector3<T>,
        local_samples_view: SamplesView<'a, 'a, T>,
        global_closest: usize,
        kernel: K,
        bg_params: BackgroundFieldParams,
        bg_value: Option<V>,
    ) -> Self {

        // Determine if the closest sample is in our neighbourhood of samples.
        let closest_sample_index = 
            if let Some(sample) = local_samples_view.iter().find(|&sample| sample.index == global_closest) {
                ClosestIndex::Local(sample.index)
            } else {
                ClosestIndex::Global(global_closest)
            };

        Self::new(q, local_samples_view, closest_sample_index, kernel, bg_params, bg_value)
    }

    /// Internal constructor that is guaranteed to build a background field from the given
    /// parameters where the give `closest_sample_index` is already determined.
    fn new(
        q: Vector3<T>,
        local_samples_view: SamplesView<'a, 'a, T>,
        closest_sample_index: ClosestIndex,
        kernel: K,
        bg_params: BackgroundFieldParams,
        bg_value: Option<V>,
    ) -> Self
    {
        let (closest_sample_disp, closest_sample_dist) = {
            let Sample { pos, .. } = local_samples_view.at_index(closest_sample_index.get());
            let disp = q - pos;
            (disp, disp.norm())
        };
        
        // Compute the weight sum here. This will be available to the usesr of bg data.
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
        } else {
            if self.weighted {
                self.kernel.f(self.kernel.radius() - self.closest_sample_dist)
            } else {
                T::zero()
            }
        }
    }

    /// Unnormalized background wieght derivative with respect to the given sample point index.
    /// If the given sample is not the closest sample, then the drivative is zero.
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
                ClosestIndex::Local(local_sample_index) =>
                    if index != local_sample_index {
                        // requested point is not the closest one, but it's in the local neighbourhood.
                        return Matrix3::zeros();
                    },
                ClosestIndex::Global(_) =>
                    // No local neighbourhood, this hessian is for background only.
                    return Matrix3::zeros(),
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
            BackgroundFieldValue::ClosestSampleSignedDistance =>
                self.closest_sample_dist * self.orientation(),
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
            BackgroundFieldValue::Constant(_) => Vector3::zeros(),
            BackgroundFieldValue::ClosestSampleSignedDistance =>
                disp * (self.orientation() / dist),
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

            if bg_field_value == BackgroundFieldValue::ClosestSampleSignedDistance {
                if index == closest_sample_index.get() {
                    grad -= bg_grad * wb;
                }
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

            if bg_field_value == BackgroundFieldValue::ClosestSampleSignedDistance {
                if index == closest_sample_index.get() {
                    grad += dwbdp * (field * weight_sum_inv * (T::one() - wb))
                }
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
        let grad = if weighted {
            (dwb - dw_total * wb) * (bg_val * weight_sum_inv)
        } else {
            Vector3::zeros()
        };

        grad + self.field_gradient() * wb
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use geo::mesh::VertexPositions;
    use crate::{surface_from_polymesh, make_grid, Error, Params, KernelType, SampleType};

    #[test]
    fn constant_unweighted_bg() -> Result<(), Error>{
        // Create a surface sample mesh.
        let octahedron_trimesh = utils::make_sample_octahedron();
        let mut sphere = geo::mesh::PolyMesh::from(octahedron_trimesh);

        // Translate the mesh up
        utils::translate(&mut sphere, [0.0, 0.0, 3.0]);

        // Construct the implicit surface.
        let surface = surface_from_polymesh(
            &sphere,
            Params {
                kernel: KernelType::Approximate {
                    tolerance: 0.00001,
                    radius: 1.0,
                },
                background_field: BackgroundFieldParams { field_type: BackgroundFieldType::FromInput, weighted: false },
                sample_type: SampleType::Face,
                max_step: 100.0,
            },
        )?;

        // Make a grid mesh to be tested against.
        let grid = make_grid(5, 5);

        // Compute potential.
        let mut potential = vec![-1.0; grid.vertex_positions().len()];
        surface.potential(grid.vertex_positions(), &mut potential)?;

        // Potential should be all -1 (unchanged) because the radius is too small.
        for &pot in potential.iter() {
            assert_eq!(pot, -1.0);
        }
        Ok(())
    }

    #[test]
    fn constant_bg() -> Result<(), Error>{
        // Create a surface sample mesh.
        let octahedron_trimesh = utils::make_sample_octahedron();
        let mut sphere = geo::mesh::PolyMesh::from(octahedron_trimesh);

        // Translate the mesh up
        utils::translate(&mut sphere, [0.0, 0.0, 3.0]);

        // Construct the implicit surface.
        let surface = surface_from_polymesh(
            &sphere,
            Params {
                kernel: KernelType::Approximate {
                    tolerance: 0.00001,
                    radius: 1.0,
                },
                background_field: BackgroundFieldParams { field_type: BackgroundFieldType::FromInput, weighted: true },
                sample_type: SampleType::Face,
                max_step: 100.0,
            },
        )?;

        // Make a grid mesh to be tested against.
        let grid = make_grid(5, 5);

        // Compute potential.
        let mut potential = vec![-1.0; grid.vertex_positions().len()];
        surface.potential(grid.vertex_positions(), &mut potential)?;

        // Potential should be all -1 (unchanged) because the radius is too small.
        for &pot in potential.iter() {
            assert_relative_eq!(pot, -1.0);
        }
        Ok(())
    }
}
