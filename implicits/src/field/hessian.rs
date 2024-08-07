use super::*;
use crate::jacobian::{
    normalized_neighbor_weight_gradient, normalized_neighbor_weight_nml_gradient,
    normalized_neighbor_weight_nml_vertex_gradients, normalized_neighbor_weight_vertex_gradients,
    query_jacobian_at, rodrigues_rotation, weight_nml_sum_inv,
};
use crate::Error;
use ahash::{AHashSet as HashSet, RandomState};
use rayon::iter::Either;
use std::cmp::Ordering;
use tensr::IntoTensor;

/// Symmetric outer product of two vectors: a*b' + b*a'
pub(crate) fn sym_outer<T: Scalar>(a: Vector3<T>, b: Vector3<T>) -> Matrix3<T> {
    let two = T::from(2.0).unwrap();
    let m00 = two * a[0] * b[0];
    let m11 = two * a[1] * b[1];
    let m22 = two * a[2] * b[2];
    let m01 = a[0] * b[1] + a[1] * b[0];
    let m02 = a[0] * b[2] + a[2] * b[0];
    let m12 = a[1] * b[2] + a[2] * b[1];
    [[m00, m01, m02], [m01, m11, m12], [m02, m12, m22]].into_tensor()
}

impl<T: Scalar> ImplicitSurface<T> {
    /// Get the number of Hessian non-zeros for the face unit normal Hessian.
    /// This is essentially the number of items returned by
    /// `compute_face_unit_normals_hessian_products`.
    pub fn num_face_unit_normals_hessian_entries(num_samples: usize) -> usize {
        num_samples * 6 * 6
    }
}

/// Hessian indices with respect to samples.
fn sample_hessian_indices<'a>(
    nbrs: impl Iterator<Item = usize> + Clone + 'a,
) -> impl Iterator<Item = (usize, usize)> + 'a {
    nbrs.clone().map(move |i| (i, i)).chain(
        nbrs.clone()
            .flat_map(move |i| nbrs.clone().filter(move |&j| i < j).map(move |j| (j, i))),
    )
}

/// Indices for the surface Hessian product for a single face (meaning that samples are located
/// on faces). This code mimics what's generated in `face_hessian_at`.
fn face_hessian_indices_iter<'a>(
    nbrs: impl Iterator<Item = usize> + Clone + 'a,
    surface_topo: &'a [[usize; 3]],
    weighted: bool,
) -> impl Iterator<Item = (usize, usize)> + 'a {
    let samples_to_vertices = move |row: usize, col: usize| {
        surface_topo[row].iter().flat_map(move |&r| {
            surface_topo[col]
                .iter()
                .filter(move |&&c| c <= r)
                .map(move |&c| (r, c))
        })
    };

    // For each surface vertex contribution
    let main_hess_indices = sample_hessian_indices(nbrs.clone())
        .chain(background_field::hessian_block_indices(
            weighted,
            nbrs.clone(),
        ))
        .flat_map(move |(row, col)| {
            let upper_triangular_entries = if row > col {
                Some(samples_to_vertices(col, row))
            } else {
                None
            };
            samples_to_vertices(row, col).chain(upper_triangular_entries.into_iter().flatten())
        });

    // Add in the normal gradient multiplied by a vector of given Vector3 values.
    let nml_hess_indices = nbrs
        .clone()
        .flat_map(move |sample_i| {
            surface_topo[sample_i].iter().flat_map(move |&i| {
                surface_topo[sample_i]
                    .iter()
                    .filter(move |&&j| j <= i)
                    .map(move |&j| (i, j))
            })
        })
        .chain(nbrs.clone().flat_map(move |sample_i| {
            surface_topo[sample_i]
                .iter()
                .flat_map(move |&i| {
                    surface_topo[sample_i].iter().map(
                        move |&j| {
                            if i >= j {
                                (i, j)
                            } else {
                                (j, i)
                            }
                        },
                    )
                })
                .chain(nbrs.clone().flat_map(move |sample_j| {
                    surface_topo[sample_j].iter().flat_map(move |&j| {
                        surface_topo[sample_i].iter().map(
                            move |&i| {
                                if j >= i {
                                    (j, i)
                                } else {
                                    (i, j)
                                }
                            },
                        )
                    })
                }))
        }));

    // There are 3 contributions from each sample to each vertex.
    nml_hess_indices.chain(main_hess_indices)
}

impl<T: Real> QueryTopo<T> {
    /*
     * Query Hessian
     */

    pub fn num_query_hessian_product_entries(&self) -> usize {
        self.num_neighborhoods() * 6
    }

    pub fn query_hessian_product_indices_iter(&self) -> impl Iterator<Item = (usize, usize)> + '_ {
        self.query_hessian_product_block_indices_iter()
            .flat_map(move |i| {
                (0..3).flat_map(move |r| (0..=r).map(move |c| (3 * i + r, 3 * i + c)))
            })
    }

    /// Returns an iterator over all query hessian blocks.
    ///
    /// The query hessian is diagonal so the iterator returns the index of each diagonal element
    /// in the sparse matrix.
    pub fn query_hessian_product_block_indices_iter(&self) -> impl Iterator<Item = usize> + '_ {
        self.trivial_neighborhood_seq()
            .enumerate()
            .filter(move |(_, nbrs)| !nbrs.is_empty())
            .map(move |(i, _)| i)
    }

    pub fn query_hessian_product_values_iter<'a>(
        &'a self,
        query_points: &'a [[T; 3]],
        multipliers: &'a [T],
    ) -> impl Iterator<Item = T> + 'a {
        apply_kernel_query_fn_impl_iter!(self, |kernel| self
            .query_hessian_product_values_iter_impl(query_points, multipliers, kernel))
    }

    pub fn query_hessian_product_indexed_blocks_iter<'a>(
        &'a self,
        query_points: &'a [[T; 3]],
        multipliers: impl IntoIterator<Item = T> + 'a,
    ) -> impl Iterator<Item = (usize, [[T; 3]; 3])> + 'a {
        apply_kernel_query_fn_impl_iter!(self, |kernel| self
            .query_hessian_product_indexed_blocks_iter_impl(query_points, multipliers, kernel))
    }

    pub fn query_hessian_product_values(
        &self,
        query_points: &[[T; 3]],
        multipliers: &[T],
        values: &mut [T],
    ) {
        self.query_hessian_product_scaled_values(query_points, multipliers, T::one(), values)
    }

    /// Compute the Hessian product of this implicit surface function with respect to query points.
    /// The product is with the given multipliers: one per query point.
    pub fn query_hessian_product_scaled_values(
        &self,
        query_points: &[[T; 3]],
        multipliers: &[T],
        scale: T,
        values: &mut [T],
    ) {
        apply_kernel_query_fn!(self, |kernel| self.query_hessian_product_values_impl(
            query_points,
            multipliers,
            kernel,
            scale,
            values
        ))
    }

    /// This function returns an iterator over the values of the hessian product matrix with 6 (lower triangular)
    /// entries per diagonal 3x3 block of the hessian product.
    pub(crate) fn query_hessian_product_values_iter_impl<'a, K: 'a>(
        &'a self,
        query_points: &'a [[T; 3]],
        multipliers: &'a [T],
        kernel: K,
    ) -> impl Iterator<Item = T> + 'a
    where
        T: Real,
        K: SphericalKernel<T> + std::fmt::Debug + Copy + Sync + Send,
    {
        self.query_hessian_product_indexed_blocks_iter_impl(
            query_points,
            multipliers.iter().cloned(),
            kernel,
        )
        .flat_map(|(_, mtx)| (0..3).flat_map(move |r| (0..=r).map(move |c| mtx[r][c])))
    }

    /// This function returns an iterator over the row-major blocks of the hessian product matrix.
    ///
    /// This matrix is diagonal so the returned index represents both row and column index of each
    /// block.
    pub(crate) fn query_hessian_product_indexed_blocks_iter_impl<'a, K: 'a, I>(
        &'a self,
        query_points: &'a [[T; 3]],
        multipliers: I,
        kernel: K,
    ) -> impl Iterator<Item = (usize, [[T; 3]; 3])> + 'a
    where
        I: IntoIterator<Item = T> + 'a,
        T: Real,
        K: SphericalKernel<T> + std::fmt::Debug + Copy + Sync + Send,
    {
        let neigh_points = self.trivial_neighborhood_seq();

        let ImplicitSurfaceBase {
            ref samples,
            bg_field_params,
            ..
        } = *self.base();

        // For each row (query point)
        zip!(query_points.iter(), neigh_points)
            .enumerate()
            .filter(|(_, (_, nbrs))| !nbrs.is_empty())
            .zip(multipliers)
            .map(move |((idx, (q, nbr_points)), mult)| {
                let view = SamplesView::new(nbr_points, samples);
                let mtx = query_hessian_at(Vector3::new(*q), view, kernel, bg_field_params) * mult;
                (idx, mtx.into_data())
            })
    }

    /// This function populates the values of the hessian product matrix with 6 (lower triangular)
    /// entries per diagonal 3x3 block of the hessian product.
    pub(crate) fn query_hessian_product_values_impl<K>(
        &self,
        query_points: &[[T; 3]],
        multipliers: &[T],
        kernel: K,
        scale: T,
        values: &mut [T],
    ) where
        T: Real,
        K: SphericalKernel<T> + std::fmt::Debug + Copy + Sync + Send,
    {
        let mut count = 0;
        values
            .iter_mut()
            .zip(self.query_hessian_product_values_iter_impl(query_points, multipliers, kernel))
            .for_each(|(v, new_v)| {
                *v = new_v * scale;
                count += 1;
            });

        // Ensure that all values are filled
        debug_assert_eq!(values.len(), count);
    }

    /*
     * Surface Hessian
     */

    /// Get the total number of entries for the sparse Hessian non-zeros. The Hessian is taken with
    /// respect to sample points. This estimate is based on the current neighbor data, which
    /// gives the number of query points, if the neighborhood was not precomputed this function
    /// returns `None`.
    pub fn num_surface_hessian_product_entries(&self) -> Option<usize> {
        // TODO: Figure out how to do this more efficiently.
        Some(self.surface_hessian_product_indices_iter().ok()?.count())
    }

    pub fn surface_hessian_product_values(
        &self,
        query_points: &[[T; 3]],
        multipliers: &[T],
        values: &mut [T],
    ) -> Result<(), Error> {
        self.surface_hessian_product_scaled_values(query_points, multipliers, T::one(), values)
    }

    pub fn surface_hessian_product_values_iter<'a>(
        &'a self,
        query_points: &'a [[T; 3]],
        multipliers: &'a [T],
    ) -> Result<impl Iterator<Item = T> + 'a, Error> {
        Ok(apply_kernel_query_fn_impl_iter!(self, |kernel| {
            self.surface_hessian_product_values_iter_impl(query_points, multipliers, kernel)
        }, ?))
    }

    /// Returns row-major indexed blocks.
    ///
    /// The diagonal blocks are non-truncated full matrices, but otherwise the upper triangular part
    /// is omitted.
    pub fn surface_hessian_product_indexed_blocks_iter<'a>(
        &'a self,
        query_points: &'a [[T; 3]],
        multipliers: &'a [T],
    ) -> Result<impl Iterator<Item = (usize, usize, [[T; 3]; 3])> + 'a, Error> {
        Ok(apply_kernel_query_fn_impl_iter!(self, |kernel| {
            self.surface_hessian_product_indexed_blocks_iter_impl(query_points, multipliers, kernel)
        }, ?))
    }

    /// Compute the Hessian of this implicit surface function with respect to surface
    /// points multiplied by a vector of multipliers (one for each query point).
    pub fn surface_hessian_product_scaled_values(
        &self,
        query_points: &[[T; 3]],
        multipliers: &[T],
        scale: T,
        values: &mut [T],
    ) -> Result<(), Error> {
        apply_kernel_query_fn!(self, |kernel| {
            self.surface_hessian_product_values_impl(
                query_points,
                multipliers,
                kernel,
                scale,
                values,
            )
        })
    }

    /// Compute the indices for the implicit surface potential Hessian with respect to surface
    /// points. This returns an iterator over all the hessian product indices.
    pub fn surface_hessian_product_indices_iter(
        &self,
    ) -> Result<impl Iterator<Item = (usize, usize)> + '_, Error> {
        Ok(self
            .surface_hessian_product_block_indices_iter()?
            .flat_map(move |(row, col)| {
                (0..3).flat_map(move |r| (0..3).map(move |c| (3 * row + r, 3 * col + c)))
            })
            .filter(move |(row, col)| row >= col))
    }

    /// Compute the indices for the implicit surface potential Hessian with respect to surface
    /// points.
    ///
    /// This returns an iterator over all the block hessian product indices.
    pub fn surface_hessian_product_block_indices_iter(
        &self,
    ) -> Result<impl Iterator<Item = (usize, usize)> + '_, Error> {
        let neigh_points = self.trivial_neighborhood_seq();

        let ImplicitSurfaceBase {
            ref surface_topo,
            sample_type,
            bg_field_params,
            ..
        } = *self.base();

        match sample_type {
            SampleType::Vertex => Err(Error::UnsupportedSampleType),
            SampleType::Face => Ok(neigh_points.flat_map(move |nbr_points| {
                face_hessian_indices_iter(
                    nbr_points.iter().cloned(),
                    surface_topo,
                    bg_field_params.weighted,
                )
            })),
        }
    }

    pub fn surface_hessian_product_indices(
        &self,
        rows: &mut [usize],
        cols: &mut [usize],
    ) -> Result<(), Error> {
        for (i, (r, c)) in self.surface_hessian_product_indices_iter()?.enumerate() {
            rows[i] = r;
            cols[i] = c;
        }
        Ok(())
    }

    pub(crate) fn surface_hessian_product_values_impl<K>(
        &self,
        query_points: &[[T; 3]],
        multipliers: &[T],
        kernel: K,
        scale: T,
        values: &mut [T],
    ) -> Result<(), Error>
    where
        T: Real,
        K: SphericalKernel<T> + std::fmt::Debug + Copy + Sync + Send,
    {
        values
            .iter_mut()
            .zip(self.surface_hessian_product_values_iter_impl(
                query_points,
                multipliers,
                kernel,
            )?)
            .for_each(|(val, new_val)| *val = new_val * scale);
        Ok(())
    }

    pub(crate) fn surface_hessian_product_values_iter_impl<'a, K: 'a>(
        &'a self,
        query_points: &'a [[T; 3]],
        multipliers: &'a [T],
        kernel: K,
    ) -> Result<Box<dyn Iterator<Item = T> + 'a>, Error>
    where
        T: Real,
        K: SphericalKernel<T> + std::fmt::Debug + Copy + Sync + Send,
    {
        Ok(Box::new(
            self.surface_hessian_product_indexed_blocks_iter_impl(
                query_points,
                multipliers,
                kernel,
            )?
            .flat_map(move |(row, col, mtx)| {
                (0..3).flat_map(move |r| {
                    (0..3)
                        .filter(move |c| 3 * row + r >= 3 * col + c)
                        .map(move |c| mtx[r][c])
                })
            }),
        ))
    }

    // 3x3 row-major matrix blocks in the lower triangular part.
    // Blocks on the diagonal are full 3x3 matrices.
    pub(crate) fn surface_hessian_product_indexed_blocks_iter_impl<'a, K: 'a>(
        &'a self,
        query_points: &'a [[T; 3]],
        multipliers: &'a [T],
        kernel: K,
    ) -> Result<impl Iterator<Item = (usize, usize, [[T; 3]; 3])> + 'a, Error>
    where
        T: Real,
        K: SphericalKernel<T> + std::fmt::Debug + Copy + Sync + Send,
    {
        let neigh_points = self.trivial_neighborhood_seq();

        let ImplicitSurfaceBase {
            ref samples,
            ref surface_topo,
            ref surface_vertex_positions,
            bg_field_params,
            sample_type,
            ..
        } = *self.base();

        match sample_type {
            SampleType::Vertex => Err(Error::UnsupportedSampleType),
            SampleType::Face => {
                let face_hess = zip!(query_points.iter(), neigh_points)
                    .filter(|(_, nbrs)| !nbrs.is_empty())
                    .zip(multipliers.iter())
                    .flat_map(move |((q, nbr_points), lambda)| {
                        let view = SamplesView::new(nbr_points, samples);
                        face_hessian_at(
                            Vector3::new(*q),
                            view,
                            kernel,
                            surface_topo,
                            surface_vertex_positions,
                            bg_field_params,
                            *lambda,
                        )
                        .map(|(i, j, m)| (i, j, m.into_data()))
                    });

                Ok(face_hess)
            }
        }
    }

    /// Computes `d/dq J b`.
    ///
    /// Here `J` is the contact Jacobian and `b` is a vector of constant multipliers.
    /// Returned blocks are column-major.
    pub fn contact_jacobian_jacobian_product_indexed_blocks_iter<'a>(
        &'a self,
        query_points: &'a [[T; 3]],
        multipliers: &'a [[T; 3]],
    ) -> Result<impl Iterator<Item = (usize, usize, [[T; 3]; 3])> + 'a, Error> {
        Ok(apply_kernel_query_fn_impl_iter!(self, |kernel| {
            self.contact_jacobian_jacobian_product_indexed_blocks_iter_impl(query_points, multipliers, kernel)
        }, ?))
    }

    // Returns column major blocks.
    pub(crate) fn contact_jacobian_jacobian_product_indexed_blocks_iter_impl<'a, K: 'a>(
        &'a self,
        query_points: &'a [[T; 3]],
        multipliers: &'a [[T; 3]],
        kernel: K,
    ) -> Result<impl Iterator<Item = (usize, usize, [[T; 3]; 3])> + 'a, Error>
    where
        T: Real,
        K: SphericalKernel<T> + std::fmt::Debug + Copy + Sync + Send,
    {
        let neigh_points = self.trivial_neighborhood_seq();

        let ImplicitSurfaceBase {
            ref samples,
            ref surface_topo,
            ref surface_vertex_positions,
            bg_field_params,
            sample_type,
            ..
        } = *self.base();

        match sample_type {
            SampleType::Vertex => Err(Error::UnsupportedSampleType),
            SampleType::Face => {
                let hess = zip!(query_points.iter(), neigh_points)
                    .enumerate()
                    .filter(|(_, (_, nbrs))| !nbrs.is_empty())
                    .flat_map(move |(row_idx, (q, nbr_points))| {
                        let view = SamplesView::new(nbr_points, samples);
                        contact_jacobian_jacobian_at(
                            Vector3::new(*q),
                            view,
                            kernel,
                            surface_vertex_positions,
                            surface_topo,
                            bg_field_params,
                            multipliers,
                        )
                        .map(move |(col_idx, m)| (row_idx, col_idx, m.into_data()))
                    });

                Ok(hess)
            }
        }
    }

    /// Computes `d/dq J b` with respect to query position.
    ///
    /// Here `J` is the contact Jacobian and `b` is a vector of constant multipliers.
    /// Returned blocks are column-major.
    pub fn contact_query_jacobian_jacobian_product_indexed_blocks_iter<'a>(
        &'a self,
        query_points: &'a [[T; 3]],
        multipliers: &'a [[T; 3]],
    ) -> Result<impl Iterator<Item = (usize, usize, [[T; 3]; 3])> + 'a, Error> {
        Ok(apply_kernel_query_fn_impl_iter!(self, |kernel| {
            self.contact_query_jacobian_jacobian_product_indexed_blocks_iter_impl(query_points, multipliers, kernel)
        }, ?))
    }

    // Returns column major blocks.
    pub(crate) fn contact_query_jacobian_jacobian_product_indexed_blocks_iter_impl<'a, K: 'a>(
        &'a self,
        query_points: &'a [[T; 3]],
        multipliers: &'a [[T; 3]],
        kernel: K,
    ) -> Result<impl Iterator<Item = (usize, usize, [[T; 3]; 3])> + 'a, Error>
    where
        T: Real,
        K: SphericalKernel<T> + std::fmt::Debug + Copy + Sync + Send,
    {
        let neigh_points = self.trivial_neighborhood_seq();

        let ImplicitSurfaceBase {
            ref samples,
            ref surface_topo,
            bg_field_params,
            sample_type,
            ..
        } = *self.base();

        let third = T::one() / T::from(3.0_f64).unwrap();

        match sample_type {
            SampleType::Vertex => Err(Error::UnsupportedSampleType),
            SampleType::Face => {
                let hess = zip!(query_points.iter(), neigh_points)
                    .enumerate()
                    .filter(|(_, (_, nbrs))| !nbrs.is_empty())
                    .flat_map(move |(vtx_idx, (q, nbr_points))| {
                        let q = Vector3::from(*q);
                        let view = SamplesView::new(nbr_points, samples);
                        let bg: BackgroundField<T, T, K> =
                            BackgroundField::local(q, view, kernel, bg_field_params, None).unwrap();

                        // let weight_sum_inv = bg.weight_sum_inv();
                        let closest_d = bg.closest_sample_dist();
                        let grad_phi = query_jacobian_at(q, view, None, kernel, bg_field_params);
                        let weight_sum_inv =
                            weight_nml_sum_inv(q, view, kernel, closest_d, grad_phi);
                        // let dw_neigh =
                        //     normalized_neighbor_weight_gradient(q, view, kernel, bg.clone());
                        let ddpsi = query_hessian_at(q, view, kernel, bg_field_params);

                        let dw_neigh = normalized_neighbor_weight_nml_gradient(
                            q,
                            view,
                            kernel,
                            bg.clone(),
                            grad_phi,
                            ddpsi.transpose(),
                        );

                        view.into_iter().map(move |sample| {
                            let mut lambda = Vector3::zero();
                            surface_topo[sample.index].iter().for_each(|&vtx_idx| {
                                lambda += multipliers[vtx_idx].into_tensor();
                            });
                            lambda *= third;
                            let mtx = query_contact_jacobian_gradient_product_at(
                                q,
                                sample,
                                kernel,
                                grad_phi,
                                dw_neigh,
                                ddpsi,
                                weight_sum_inv,
                                closest_d,
                                lambda,
                                false,
                            );
                            (vtx_idx, vtx_idx, mtx.into_data())
                        })
                    });
                Ok(hess)
            }
        }
    }

    /// Computes `d/dq J^T b` where q are sample indices and returns column-major indexed blocks.
    ///
    /// Here `J` is the contact Jacobian and `b` is a vector of constant multipliers.
    /// Returned blocks are column-major.
    pub fn contact_hessian_product_indexed_blocks_iter<'a>(
        &'a self,
        query_points: &'a [[T; 3]],
        multipliers: &'a [[T; 3]],
    ) -> Result<impl Iterator<Item = (usize, usize, [[T; 3]; 3])> + 'a, Error> {
        Ok(apply_kernel_query_fn_impl_iter!(self, |kernel| {
            self.contact_hessian_product_indexed_blocks_iter_impl(query_points, multipliers, kernel)
        }, ?))
    }

    // Returns column major blocks
    pub(crate) fn contact_hessian_product_indexed_blocks_iter_impl<'a, K: 'a>(
        &'a self,
        query_points: &'a [[T; 3]],
        multipliers: &'a [[T; 3]],
        kernel: K,
    ) -> Result<impl Iterator<Item = (usize, usize, [[T; 3]; 3])> + 'a, Error>
    where
        T: Real,
        K: SphericalKernel<T> + std::fmt::Debug + Copy + Sync + Send,
    {
        debug_assert_eq!(
            self.trivial_neighborhood_seq()
                .filter(|nbrs| !nbrs.is_empty())
                .count(),
            multipliers.len()
        );

        let neigh_points = self.trivial_neighborhood_seq();

        let ImplicitSurfaceBase {
            ref samples,
            ref surface_topo,
            ref surface_vertex_positions,
            bg_field_params,
            sample_type,
            ..
        } = *self.base();

        match sample_type {
            SampleType::Vertex => Err(Error::UnsupportedSampleType),
            SampleType::Face => {
                // Contraction between multipliers and contact points.
                // Each of these is a "Hessian" at a particular contact point.
                let hess = zip!(query_points.iter(), neigh_points)
                    .filter(|(_, nbrs)| !nbrs.is_empty())
                    .zip(multipliers.iter())
                    .flat_map(move |((q, nbr_points), lambda)| {
                        let view = SamplesView::new(nbr_points, samples);
                        contact_hessian_at(
                            Vector3::new(*q),
                            view,
                            kernel,
                            surface_vertex_positions,
                            surface_topo,
                            bg_field_params,
                            Vector3::new(*lambda),
                        )
                        .map(|(i, j, m)| (i, j, m.into_data()))
                    });
                Ok(hess)
            }
        }
    }

    /// Computes `d/dq J^T b` where q are query positions and returns column-major indexed blocks.
    ///
    /// Here `J` is the contact Jacobian and `b` is a vector of constant multipliers.
    /// Returned blocks are column-major.
    pub fn contact_query_hessian_product_indexed_blocks_iter<'a>(
        &'a self,
        query_points: &'a [[T; 3]],
        multipliers: &'a [[T; 3]],
    ) -> Result<impl Iterator<Item = (usize, usize, [[T; 3]; 3])> + 'a, Error> {
        Ok(apply_kernel_query_fn_impl_iter!(self, |kernel| {
            self.contact_query_hessian_product_indexed_blocks_iter_impl(query_points, multipliers, kernel)
        }, ?))
    }

    /// Computes `d/dq J^T b` where `q` are query points and returns column-major indexed blocks.
    ///
    /// Here `J` is the contact Jacobian and `b` is a vector of constant multipliers.
    /// Returned blocks are column-major.
    pub(crate) fn contact_query_hessian_product_indexed_blocks_iter_impl<'a, K: 'a>(
        &'a self,
        query_points: &'a [[T; 3]],
        multipliers: &'a [[T; 3]],
        kernel: K,
    ) -> Result<impl Iterator<Item = (usize, usize, [[T; 3]; 3])> + 'a, Error>
    where
        T: Real,
        K: SphericalKernel<T> + std::fmt::Debug + Copy + Sync + Send,
    {
        debug_assert_eq!(
            self.trivial_neighborhood_seq()
                .filter(|nbrs| !nbrs.is_empty())
                .count(),
            multipliers.len()
        );

        let neigh_points = self.trivial_neighborhood_seq();

        let ImplicitSurfaceBase {
            ref samples,
            ref surface_topo,
            bg_field_params,
            sample_type,
            ..
        } = *self.base();

        let third = T::one() / T::from(3.0_f64).unwrap();

        match sample_type {
            SampleType::Vertex => Err(Error::UnsupportedSampleType),
            SampleType::Face => {
                // Contraction between multipliers and contact points.
                // Each of these is a "Hessian" at a particular contact point.
                Ok(zip!(query_points.iter(), neigh_points)
                    .enumerate()
                    .filter(|(_, (_, nbrs))| !nbrs.is_empty())
                    .zip(multipliers.iter())
                    .flat_map(move |((query_index, (q, nbr_points)), lambda)| {
                        let q = Vector3::from(*q);
                        let view = SamplesView::new(nbr_points, samples);
                        let bg: BackgroundField<T, T, K> =
                            BackgroundField::local(q, view, kernel, bg_field_params, None).unwrap();

                        // let weight_sum_inv = bg.weight_sum_inv();
                        let closest_d = bg.closest_sample_dist();
                        let grad_phi = query_jacobian_at(q, view, None, kernel, bg_field_params);
                        let weight_sum_inv =
                            weight_nml_sum_inv(q, view, kernel, closest_d, grad_phi);
                        // let dw_neigh =
                        //     normalized_neighbor_weight_gradient(q, view, kernel, bg.clone());
                        let ddpsi = query_hessian_at(q, view, kernel, bg_field_params);
                        let dw_neigh = normalized_neighbor_weight_nml_gradient(
                            q,
                            view,
                            kernel,
                            bg.clone(),
                            grad_phi,
                            ddpsi.transpose(),
                        );

                        view.into_iter().flat_map(move |sample| {
                            let mtx = query_contact_jacobian_gradient_product_at(
                                q,
                                sample,
                                kernel,
                                grad_phi,
                                dw_neigh,
                                ddpsi,
                                weight_sum_inv,
                                closest_d,
                                Vector3::from(*lambda),
                                true,
                            );
                            IntoIterator::into_iter(surface_topo[sample.index]).map(
                                move |vtx_idx| (vtx_idx, query_index, (mtx * third).into_data()),
                            )
                        })
                    }))
            }
        }
    }

    /// Computes `d/dq grad_x Psi(x)` where `q` are sample adjacent vertices.
    ///
    /// Here `Psi` is the implicit function.
    /// Returned blocks are row-major.
    pub fn sample_query_hessian_indexed_blocks_iter<'a>(
        &'a self,
        query_points: &'a [[T; 3]],
    ) -> Result<impl Iterator<Item = (usize, usize, [[T; 3]; 3])> + 'a, Error> {
        Ok(apply_kernel_query_fn_impl_iter!(self, |kernel| {
            self.sample_query_hessian_product_indexed_blocks_iter_impl(query_points, None, kernel)
        }, ?))
    }

    /// Computes `d/dq (grad_x Psi(x) b)` where `q` are sample adjacent vertices.
    ///
    /// Here `Psi` is the implicit function and `b` are the multipliers.
    /// Returned blocks are row-major.
    pub fn sample_query_hessian_product_indexed_blocks_iter<'a>(
        &'a self,
        query_points: &'a [[T; 3]],
        multipliers: &'a [T],
    ) -> Result<impl Iterator<Item = (usize, usize, [[T; 3]; 3])> + 'a, Error> {
        Ok(apply_kernel_query_fn_impl_iter!(self, |kernel| {
            self.sample_query_hessian_product_indexed_blocks_iter_impl(query_points, Some(multipliers), kernel)
        }, ?))
    }

    // Returns row major blocks
    pub(crate) fn sample_query_hessian_product_indexed_blocks_iter_impl<'a, K: 'a>(
        &'a self,
        query_points: &'a [[T; 3]],
        multipliers: Option<&'a [T]>,
        kernel: K,
    ) -> Result<impl Iterator<Item = (usize, usize, [[T; 3]; 3])> + 'a, Error>
    where
        T: Real,
        K: SphericalKernel<T> + std::fmt::Debug + Copy + Sync + Send,
    {
        let neigh_points = self.trivial_neighborhood_seq();

        let ImplicitSurfaceBase {
            ref samples,
            ref surface_topo,
            ref surface_vertex_positions,
            bg_field_params,
            sample_type,
            ..
        } = *self.base();

        let mult_iter = if let Some(mult) = multipliers {
            Either::Left(mult.iter().cloned())
        } else {
            Either::Right(std::iter::repeat(T::one()))
        };

        match sample_type {
            SampleType::Vertex => Err(Error::UnsupportedSampleType),
            SampleType::Face => {
                let hess = query_points
                    .iter()
                    .zip(neigh_points)
                    .enumerate()
                    .filter(|(_, (_, nbrs))| !nbrs.is_empty())
                    .zip(mult_iter)
                    .flat_map(move |((query_idx, (&q, nbr_points)), mult)| {
                        let view = SamplesView::new(nbr_points, samples);
                        let q = Vector3::new(q);

                        let bg: BackgroundField<T, T, K> =
                            BackgroundField::local(q, view, kernel, bg_field_params, None).unwrap();

                        let weight_sum_inv = bg.weight_sum_inv();
                        let closest_d = bg.closest_sample_dist();

                        let dwdq_neigh = normalized_neighbor_weight_vertex_gradients(
                            q,
                            view,
                            kernel,
                            surface_vertex_positions.len(),
                            surface_topo,
                            bg.clone(),
                        );

                        let dw_neigh =
                            normalized_neighbor_weight_gradient(q, view, kernel, bg.clone());
                        let ddwdq_neigh = normalized_neighbor_weight_vertex_hessian(
                            q,
                            view,
                            kernel,
                            surface_vertex_positions.len(),
                            surface_topo,
                            bg,
                        );

                        let unit_nml_grads = view
                            .into_iter()
                            .map(move |sample| {
                                let mut sample_unit_nml = sample.nml;
                                let sample_area = sample_unit_nml.normalize();
                                face_unit_normal_gradient_iter(
                                    sample.index,
                                    sample_unit_nml,
                                    T::one() / sample_area,
                                    surface_vertex_positions,
                                    surface_topo,
                                )
                                .collect::<Vec<_>>()
                            })
                            .collect::<Vec<_>>();

                        let seen =
                            std::cell::RefCell::new(vec![false; surface_vertex_positions.len()]);

                        let entries = view
                            .into_iter()
                            .flat_map(|col_sample| {
                                surface_topo[col_sample.index]
                                    .iter()
                                    .filter_map(|&tri_vtx| {
                                        if seen.borrow()[tri_vtx] {
                                            return None;
                                        } else {
                                            seen.borrow_mut()[tri_vtx] = true;
                                        }
                                        let jac_grad_phi = view
                                            .iter()
                                            .enumerate()
                                            .map(|(which_sample, sample)| {
                                                (
                                                    surface_topo[sample.index]
                                                        .iter()
                                                        .position(|&x| x == tri_vtx),
                                                    which_sample,
                                                    sample,
                                                )
                                            })
                                            .map(|(which_vtx, which_sample, sample)| {
                                                let sample_unit_nml = sample.nml.normalized();
                                                sample_query_hessian_component_at(
                                                    q,
                                                    sample,
                                                    sample_unit_nml,
                                                    kernel,
                                                    &unit_nml_grads[which_sample],
                                                    which_vtx,
                                                    dw_neigh,
                                                    ddwdq_neigh[tri_vtx],
                                                    dwdq_neigh[tri_vtx],
                                                    weight_sum_inv,
                                                    closest_d,
                                                )
                                            })
                                            .sum::<Matrix3<T>>();
                                        Some((
                                            query_idx,
                                            tri_vtx,
                                            (jac_grad_phi * mult).into_data(),
                                        ))
                                    })
                            })
                            .collect::<Vec<_>>();
                        entries.into_iter()
                    });
                Ok(hess)
            }
        }
    }

    /// Computes indices for `d/dq (grad_x Psi(x) b)` where `q` are sample adjacent vertices.
    ///
    /// Here `Psi` is the implicit function and `b` are the multipliers.
    /// Returned blocks are row-major.
    pub fn sample_query_hessian_product_block_indices_iter(
        &self,
    ) -> Result<impl Iterator<Item = (usize, usize)> + '_, Error> {
        let neigh_points = self.trivial_neighborhood_seq();

        let ImplicitSurfaceBase {
            ref samples,
            ref surface_topo,
            ref surface_vertex_positions,
            sample_type,
            ..
        } = *self.base();

        match sample_type {
            SampleType::Vertex => Err(Error::UnsupportedSampleType),
            SampleType::Face => {
                let hess = neigh_points
                    .enumerate()
                    .filter(|(_, nbrs)| !nbrs.is_empty())
                    .flat_map(move |(query_idx, nbr_points)| {
                        let view = SamplesView::new(nbr_points, samples);
                        let seen =
                            std::cell::RefCell::new(vec![false; surface_vertex_positions.len()]);
                        let indices = view
                            .into_iter()
                            .flat_map(|sample| {
                                surface_topo[sample.index].iter().filter_map(|&tri_vtx| {
                                    if seen.borrow()[tri_vtx] {
                                        return None;
                                    } else {
                                        seen.borrow_mut()[tri_vtx] = true;
                                    }
                                    Some((query_idx, tri_vtx))
                                })
                            })
                            .collect::<Vec<_>>();
                        indices.into_iter()
                    });
                Ok(hess)
            }
        }
    }
}

/*
 * Hessian components
 *
 * The following functions compute components of various Hessians.
 */

/*
 * Query hessian components
 */
/// Compute the Hessian of the potential field with respect to the given query point.
pub(crate) fn query_hessian_at<'a, T, K: 'a>(
    q: Vector3<T>,
    view: SamplesView<'a, 'a, T>,
    kernel: K,
    bg_field_params: BackgroundFieldParams,
) -> Matrix3<T>
where
    T: Real,
    K: SphericalKernel<T> + std::fmt::Debug + Copy + Sync + Send,
{
    let bg: BackgroundField<T, T, K> =
        BackgroundField::local(q, view, kernel, bg_field_params, None).unwrap();

    // Background potential Jacobian.
    let closest_d = bg.closest_sample_dist();
    let weight_sum_inv = bg.weight_sum_inv();

    let bg_hess = bg.compute_query_hessian();

    // For each surface vertex contribution
    let dw_neigh = jacobian::normalized_neighbor_weight_gradient(q, view, kernel, bg.clone());
    let ddw_neigh = normalized_neighbor_weight_hessian(q, view, kernel, bg);

    view.into_iter()
        .map(
            move |Sample {
                      pos, nml, value, ..
                  }| {
                let unit_nml = nml * (T::one() / nml.norm());
                let w = kernel.with_closest_dist(closest_d).eval(q, pos);
                let dw = kernel.with_closest_dist(closest_d).grad(q, pos);
                let ddw = kernel.with_closest_dist(closest_d).hess(q, pos);
                let psi = T::from(value).unwrap() + unit_nml.dot(q - pos);
                sym_outer(unit_nml, dw)
                    - sym_outer(unit_nml, dw_neigh) * w
                    - sym_outer(dw_neigh, dw) * psi
                    - ddw_neigh * psi * w
                    + (dw_neigh * (dw_neigh.transpose() * (T::from(2.0).unwrap() * w)) + ddw) * psi
            },
        )
        .sum::<Matrix3<T>>()
        * weight_sum_inv
        + bg_hess
}

/// Compute the normalized sum of all sample weight gradients.
pub(crate) fn normalized_neighbor_weight_hessian<'a, T, K, V>(
    q: Vector3<T>,
    samples: SamplesView<'a, 'a, T>,
    kernel: K,
    bg: BackgroundField<'a, T, V, K>,
) -> Matrix3<T>
where
    T: Real,
    K: SphericalKernel<T> + std::fmt::Debug + Copy + Sync + Send + 'a,
    V: Copy + Clone + std::fmt::Debug + PartialEq + num_traits::Zero,
{
    let closest_d = bg.closest_sample_dist();

    let weight_sum_inv = bg.weight_sum_inv();

    let mut ddw_neigh: Matrix3<T> = samples
        .iter()
        .map(|s| kernel.with_closest_dist(closest_d).hess(q, s.pos))
        .sum();

    // Contribution from the background potential
    ddw_neigh += bg.background_weight_hessian(None);

    ddw_neigh * weight_sum_inv // normalize the neighborhood derivative
}

/// Compute the normalized sum of all sample weight gradients.
pub(crate) fn normalized_neighbor_weight_vertex_hessian<'a, T, K, V>(
    q: Vector3<T>,
    samples: SamplesView<'a, 'a, T>,
    kernel: K,
    num_surface_vertices: usize,
    surface_topo: &'a [[usize; 3]],
    bg: BackgroundField<'a, T, V, K>,
) -> Vec<Matrix3<T>>
where
    T: Real,
    K: SphericalKernel<T> + std::fmt::Debug + Copy + Sync + Send + 'a,
    V: Copy + Clone + std::fmt::Debug + PartialEq + num_traits::Zero,
{
    let closest_d = bg.closest_sample_dist();

    let weight_sum_inv = bg.weight_sum_inv();
    let third = T::from(1.0 / 3.0).unwrap();
    let factor = third * weight_sum_inv;

    let mut ddw_neigh = vec![Matrix3::zero(); num_surface_vertices];

    samples.iter().for_each(|s| {
        let bg_hess = bg.background_weight_hessian(Some(s.index));
        let ddw = (kernel.with_closest_dist(closest_d).hess(q, s.pos) + bg_hess) * factor;
        surface_topo[s.index]
            .iter()
            .for_each(|&v| ddw_neigh[v] -= ddw);
    });

    ddw_neigh // normalize the neighborhood derivative
}

/*
 * Surface hessian components
 */

pub(crate) fn face_hessian_at<'a, T, K: 'a>(
    q: Vector3<T>,
    view: SamplesView<'a, 'a, T>,
    kernel: K,
    surface_topo: &'a [[usize; 3]],
    surface_vertex_positions: &'a [[T; 3]],
    bg_field_params: BackgroundFieldParams,
    multiplier: T,
) -> impl Iterator<Item = (usize, usize, Matrix3<T>)> + 'a
where
    T: Real,
    K: SphericalKernel<T> + std::fmt::Debug + Copy + Sync + Send,
{
    let bg = BackgroundField::local(q, view, kernel, bg_field_params, None).unwrap();

    let ninth = T::one() / T::from(9.0).unwrap();

    let samples_to_vertices = move |row: usize, col: usize, hess: Matrix3<T>| {
        surface_topo[row].iter().flat_map(move |&i| {
            surface_topo[col]
                .iter()
                .filter(move |&&j| j <= i)
                .map(move |&j| (i, j, hess * ninth))
        })
    };

    // For each surface vertex contribution
    let main_hess = sample_hessian_at(q, view, kernel, bg.clone()).flat_map(move |hess| {
        let upper_triangular_entries = if hess.0 > hess.1 {
            Some(samples_to_vertices(hess.1, hess.0, hess.2.transpose()))
        } else {
            None
        };
        samples_to_vertices(hess.0, hess.1, hess.2)
            .chain(upper_triangular_entries.into_iter().flatten())
    });

    // Add in the normal gradient multiplied by a vector of given Vector3 values.
    let nml_hess = normal_hessian_at(q, view, kernel, surface_topo, surface_vertex_positions, bg);

    nml_hess
        .chain(main_hess)
        .map(move |hess| (hess.0, hess.1, hess.2 * multiplier))
}

/// Hessian part with respect to samples.
pub(crate) fn sample_hessian_at<'a, T, K: 'a>(
    q: Vector3<T>,
    samples: SamplesView<'a, 'a, T>,
    kernel: K,
    bg: BackgroundField<'a, T, T, K>,
) -> impl Iterator<Item = (usize, usize, Matrix3<T>)> + 'a
where
    T: Real,
    K: SphericalKernel<T> + std::fmt::Debug + Copy + Sync + Send,
{
    let csd = bg.closest_sample_dist();
    let ws_inv = bg.weight_sum_inv();
    let ws_inv2 = ws_inv * ws_inv;

    let local_pot = compute_local_potential_at(q, samples, kernel, ws_inv, csd);

    let bg_hess = bg
        .hessian_blocks()
        .zip(background_field::hessian_block_indices(
            bg.weighted,
            samples.indices().iter().cloned(),
        ))
        .filter(|(_, (j, i))| i <= j)
        .map(|(h, (j, i))| (j, i, h));

    let _2 = T::from(2.0).unwrap();

    let bg_diag = bg.clone();
    let diag_iter = samples.into_iter().map(
        move |Sample {
                  index: i,
                  pos,
                  nml,
                  value: phi,
                  ..
              }| {
            let unit_nml = nml * (T::one() / nml.norm());
            let psi = phi + (q - pos).dot(unit_nml);
            let w = kernel.with_closest_dist(csd).eval(q, pos);
            let dw = -kernel.with_closest_dist(csd).grad(q, pos);
            let ddw = kernel.with_closest_dist(csd).hess(q, pos);
            let dwb = bg_diag.background_weight_gradient(Some(i));
            let ddwb = bg_diag.background_weight_hessian(Some(i));
            let dws = dwb + dw;

            let mut h = Matrix3::zeros();

            h += ddw * psi;
            h -= (ddw + ddwb) * local_pot;
            h -= sym_outer(dw, unit_nml);
            h -= sym_outer(dws, dw) * (psi * ws_inv);
            h += dws * (dws.transpose() * (_2 * local_pot * ws_inv));
            h += sym_outer(unit_nml, dws) * (w * ws_inv);

            (i, i, h * ws_inv)
        },
    );

    let off_diag_iter = samples.into_iter().flat_map(
        move |Sample {
                  index: i,
                  pos: pos_i,
                  nml: nml_i,
                  value: phi_i,
                  ..
              }| {
            let bg = bg.clone();
            let unit_nml_i = nml_i * (T::one() / nml_i.norm());
            let psi_i = phi_i + (q - pos_i).dot(unit_nml_i);
            let w_i = kernel.with_closest_dist(csd).eval(q, pos_i);
            let dw_i = -kernel.with_closest_dist(csd).grad(q, pos_i);
            let dwb_i = bg.background_weight_gradient(Some(i));
            let dws_i = dwb_i + dw_i;
            samples
                .into_iter()
                .filter(move |sample_j| i < sample_j.index)
                .map(
                    move |Sample {
                              index: j,
                              pos: pos_j,
                              nml: nml_j,
                              value: phi_j,
                              ..
                          }| {
                        let unit_nml_j = nml_j * (T::one() / nml_j.norm());
                        let psi_j = phi_j + (q - pos_j).dot(unit_nml_j);

                        let mut h = Matrix3::zeros();

                        let w_j = kernel.with_closest_dist(csd).eval(q, pos_j);
                        let dw_j = -kernel.with_closest_dist(csd).grad(q, pos_j);
                        let dwb_j = bg.background_weight_gradient(Some(j));
                        let dws_j = dwb_j + dw_j;

                        h -= dws_j * (dw_i.transpose() * psi_i);
                        h -= dw_j * (dws_i.transpose() * psi_j);
                        h += dws_j * (dws_i.transpose() * (_2 * local_pot));

                        h += unit_nml_j * (dws_i.transpose() * w_j);
                        h += dws_j * (unit_nml_i.transpose() * w_i);

                        (j, i, h * ws_inv2)
                    },
                )
        },
    );

    diag_iter.chain(off_diag_iter).chain(bg_hess)
}

/// A helper function to compute the normal part of the hessian for this field.
fn normal_hessian_at<'a, T, K: 'a>(
    q: Vector3<T>,
    samples: SamplesView<'a, 'a, T>,
    kernel: K,
    surface_topo: &'a [[usize; 3]],
    surface_vertex_positions: &'a [[T; 3]],
    bg: BackgroundField<'a, T, T, K>,
) -> impl Iterator<Item = (usize, usize, Matrix3<T>)> + 'a
where
    T: Real,
    K: SphericalKernel<T> + std::fmt::Debug + Copy + Sync + Send,
{
    let csd = bg.closest_sample_dist();
    let ws_inv = bg.weight_sum_inv();

    let sym_mult = move |Sample { pos, .. }| kernel.with_closest_dist(csd).eval(q, pos) * ws_inv;

    let sym = face_unit_normals_symmetric_jacobian(
        samples,
        surface_vertex_positions,
        surface_topo,
        sym_mult,
    );

    let nml_hess_multiplier = move |Sample { pos, .. }| {
        let w = kernel.with_closest_dist(csd).eval(q, pos);
        (q - pos) * (w * ws_inv)
    };

    // Compute the unit normal hessian product.
    let nml_hess_iter = compute_face_unit_normals_hessian_products(
        samples,
        surface_vertex_positions,
        surface_topo,
        nml_hess_multiplier,
    );

    let third = T::one() / T::from(3.0).unwrap();

    // Remaining hessian terms
    let coupling_nml_hess_iter = samples.into_iter().flat_map(move |sample_l| {
        let Sample {
            index: index_l,
            pos: pos_l,
            ..
        } = sample_l;

        let dwb_l = bg.background_weight_gradient(Some(index_l));
        let dw_l = -kernel.with_closest_dist(csd).grad(q, pos_l);
        let dws_l = dw_l + dwb_l;

        face_unit_normal_gradient_iter_from_sample(sample_l, surface_vertex_positions, surface_topo)
            .enumerate()
            .flat_map(move |(i, dn_li)| {
                let row_vtx = surface_topo[index_l][i];
                (0..3).map(move |j| {
                    let col_vtx = surface_topo[index_l][j];

                    let mtx = (dn_li * (q - pos_l) * dw_l.transpose()) * (ws_inv * third);

                    match row_vtx.cmp(&col_vtx) {
                        Ordering::Greater => (row_vtx, col_vtx, mtx),
                        Ordering::Less => (col_vtx, row_vtx, mtx.transpose()),
                        Ordering::Equal => (row_vtx, col_vtx, mtx + mtx.transpose()),
                    }
                })
            })
            .chain(samples.into_iter().flat_map(move |sample_k| {
                let Sample {
                    index: index_k,
                    pos: pos_k,
                    ..
                } = sample_k;
                let wk = kernel.with_closest_dist(csd).eval(q, pos_k);
                face_unit_normal_gradient_iter_from_sample(
                    sample_k,
                    surface_vertex_positions,
                    surface_topo,
                )
                .enumerate()
                .flat_map(move |(i, dn_ki)| {
                    let row_vtx = surface_topo[index_k][i];
                    (0..3).map(move |j| {
                        let col_vtx = surface_topo[index_l][j];

                        let mtx = -dn_ki
                            * (q - pos_k)
                            * dws_l.transpose()
                            * (wk * ws_inv * ws_inv * third);

                        match row_vtx.cmp(&col_vtx) {
                            Ordering::Greater => (row_vtx, col_vtx, mtx),
                            Ordering::Less => (col_vtx, row_vtx, mtx.transpose()),
                            Ordering::Equal => (row_vtx, col_vtx, mtx + mtx.transpose()),
                        }
                    })
                })
            }))
    });

    zip!(sym, nml_hess_iter)
        .map(move |(s, n)| {
            debug_assert_eq!(s.0, n.0);
            debug_assert_eq!(s.1, n.1);
            (s.0, s.1, n.2 - s.2)
        })
        .chain(coupling_nml_hess_iter)
}

/// Compute the symmetric jacobian of the face normals with respect to
/// surface vertices. This is the Jacobian plus its transpose.
/// This function is needed to compute the Hessian, which means we are only interested in the
/// lower triangular part.
pub(crate) fn face_unit_normals_symmetric_jacobian<'a, T, F>(
    samples: SamplesView<'a, 'a, T>,
    surface_vertices: &'a [[T; 3]],
    surface_topo: &'a [[usize; 3]],
    mut multiplier: F,
) -> impl Iterator<Item = (usize, usize, Matrix3<T>)> + 'a
where
    T: Real,
    F: FnMut(Sample<T>) -> T + 'a,
{
    let third = T::one() / T::from(3.0).unwrap();
    samples.into_iter().flat_map(move |sample| {
        let tri_indices = &surface_topo[sample.index];
        let nml_proj = scaled_tangent_projection_from_sample(sample); // symmetric
        let lambda = multiplier(sample);
        (0..3).flat_map(move |k| {
            let vtx_row = tri_indices[k];
            let tri = Triangle::from_indexed_slice(tri_indices, surface_vertices);
            (0..3)
                .filter(move |&l| tri_indices[l] <= vtx_row)
                .map(move |l| {
                    let vtx_col = tri_indices[l];
                    // TODO: The following matrix probably has a simpler form (possible
                    // diagonal?) Rewrite in terms of this form.
                    // Note: area_normal_gradient is in column-major form, we convert to row-major
                    // here. Since it is skew symmetric, we simply take the negative.
                    let mtx = nml_proj * Matrix3::new(tri.area_normal_gradient(l))
                        - Matrix3::new(tri.area_normal_gradient(k)) * nml_proj;
                    (vtx_row, vtx_col, mtx * (lambda * third))
                })
        })
    })
}

/// Block lower triangular part of the unit normal Hessian multiplied by the given multiplier.
pub(crate) fn compute_face_unit_normals_hessian_products<'a, T, F>(
    samples: SamplesView<'a, 'a, T>,
    surface_vertices: &'a [[T; 3]],
    surface_topo: &'a [[usize; 3]],
    mut multiplier: F,
) -> impl Iterator<Item = (usize, usize, Matrix3<T>)> + 'a
where
    T: Real,
    F: FnMut(Sample<T>) -> Vector3<T> + 'a,
{
    // For each triangle contribution (one element in a sum)
    samples.into_iter().flat_map(move |sample| {
        let tri_indices = &surface_topo[sample.index];
        let mut nml = sample.nml;
        let norm_inv = T::one() / nml.normalize();
        let nml_proj = scaled_tangent_projection(nml, norm_inv);
        let mult = multiplier(sample);
        let tri = Triangle::from_indexed_slice(tri_indices, surface_vertices);
        // Converting to row-major here causes sign change.
        let grad = [
            Matrix3::new(tri.area_normal_gradient(0)),
            Matrix3::new(tri.area_normal_gradient(1)),
            Matrix3::new(tri.area_normal_gradient(2)),
        ];

        // row >= col
        // For each row
        (0..3).flat_map(move |j| {
            let vtx_row = tri_indices[j];
            (0..3)
                .filter(move |&i| tri_indices[i] <= vtx_row)
                .map(move |i| {
                    let vtx_col = tri_indices[i];
                    let proj_mult = nml_proj * mult; // projected multiplier
                    let nml_mult_prod = nml_proj * nml.dot(mult)
                        + proj_mult * nml.transpose()
                        + nml * proj_mult.transpose();
                    // converting to row-major causes sign change
                    let m = (grad[j] * nml_mult_prod * grad[i]) * norm_inv
                        - Matrix3::new(Triangle::area_normal_hessian_product(
                            j,
                            i,
                            proj_mult.into(),
                        ));
                    (vtx_row, vtx_col, m)
                })
        })
    })
}

/*
 * Contact hessian components
 */

pub(crate) fn jacobian_gradient_phi<'a, T, K>(
    q: Vector3<T>,
    view: SamplesView<'a, 'a, T>,
    kernel: K,
    neigh_verts: &'a [usize],
    surface_topo: &'a [[usize; 3]],
    unit_nml_grads: &[Vec<Matrix3<T>>],
    dw_neigh_normalized: Vector3<T>,
    ddwdq_neigh_normalized: &[Matrix3<T>],
    dwdq_neigh_normalized: &[Vector3<T>],
    weight_sum_inv: T,
    closest_d: T,
) -> Vec<Matrix3<T>>
where
    T: Real,
    K: SphericalKernel<T> + std::fmt::Debug + Copy + Sync + Send,
{
    neigh_verts
        .iter()
        .map(|&vtx_idx| {
            view.iter()
                .enumerate()
                .map(|(which_sample, sample)| {
                    (
                        surface_topo[sample.index]
                            .iter()
                            .position(|&x| x == vtx_idx),
                        which_sample,
                        sample,
                    )
                })
                .map(|(which_vtx, which_sample, sample)| {
                    let sample_unit_nml = sample.nml.normalized();
                    sample_query_hessian_component_at(
                        q,
                        sample,
                        sample_unit_nml,
                        kernel,
                        &unit_nml_grads[which_sample],
                        which_vtx,
                        dw_neigh_normalized,
                        ddwdq_neigh_normalized[vtx_idx],
                        dwdq_neigh_normalized[vtx_idx],
                        weight_sum_inv,
                        closest_d,
                    )
                })
                .sum::<Matrix3<T>>()
        })
        .collect::<Vec<_>>()
}

pub(crate) fn neighborhood_verts<'a, T>(
    view: SamplesView<'a, 'a, T>,
    surface_topo: &'a [[usize; 3]],
) -> Vec<usize>
where
    T: Real,
{
    let mut neigh_verts: HashSet<_> = {
        // Deterministic hashset.
        let hash_builder = RandomState::with_seeds(7, 47, 2377, 719);
        HashSet::with_capacity_and_hasher(view.len() * 3, hash_builder)
    };
    neigh_verts.extend(
        view.into_iter()
            .flat_map(|sample| IntoIterator::into_iter(surface_topo[sample.index])),
    );
    neigh_verts.into_iter().collect()
}

/// Computes the jacobian of the contact jacobian for a single contact
/// `d/dq J_i b` where `b` are the multipliers.
///
/// Blocks are column-major.
pub(crate) fn contact_jacobian_jacobian_at<'a, T, K: 'a>(
    q: Vector3<T>,
    view: SamplesView<'a, 'a, T>,
    kernel: K,
    surface_vertices: &'a [[T; 3]],
    surface_topo: &'a [[usize; 3]],
    bg_field_params: BackgroundFieldParams,
    multipliers: &'a [[T; 3]],
) -> impl Iterator<Item = (usize, [[T; 3]; 3])> + 'a
where
    T: Real,
    K: SphericalKernel<T> + std::fmt::Debug + Copy + Sync + Send,
{
    let bg: BackgroundField<T, T, K> =
        BackgroundField::local(q, view, kernel, bg_field_params, None).unwrap();

    let closest_d = bg.closest_sample_dist();

    let grad_phi = query_jacobian_at(q, view, None, kernel, bg_field_params);

    let weight_sum_inv = bg.weight_sum_inv();
    let weight_nml_sum_inv = weight_nml_sum_inv(q, view, kernel, closest_d, grad_phi);

    let dwdq_neigh = normalized_neighbor_weight_vertex_gradients(
        q,
        view,
        kernel,
        surface_vertices.len(),
        surface_topo,
        bg.clone(),
    );
    let dw_neigh = normalized_neighbor_weight_gradient(q, view, kernel, bg.clone());
    let ddwdq_neigh = normalized_neighbor_weight_vertex_hessian(
        q,
        view,
        kernel,
        surface_vertices.len(),
        surface_topo,
        bg.clone(),
    );

    let third = T::from(1.0 / 3.0).unwrap();

    let unit_nml_grads = view
        .into_iter()
        .map(move |sample| {
            let mut sample_unit_nml = sample.nml;
            let sample_area = sample_unit_nml.normalize();
            face_unit_normal_gradient_iter(
                sample.index,
                sample_unit_nml,
                T::one() / sample_area,
                surface_vertices,
                surface_topo,
            )
            .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();

    // A set of unique surface vertex indices in neighbourhood of the query point.
    let neigh_verts = neighborhood_verts(view, surface_topo);

    let jac_grad_phi = jacobian_gradient_phi(
        q,
        view,
        kernel,
        &neigh_verts,
        &surface_topo,
        &unit_nml_grads,
        dw_neigh,
        &ddwdq_neigh,
        &dwdq_neigh,
        weight_sum_inv,
        closest_d,
    );

    let dwnmldq_neigh = normalized_neighbor_weight_nml_vertex_gradients(
        q,
        view,
        kernel,
        surface_vertices.len(),
        surface_topo,
        bg,
        grad_phi,
        &unit_nml_grads,
        &jac_grad_phi,
        &neigh_verts,
    );

    // For each sample
    view.into_iter()
        .enumerate()
        .flat_map(move |(which_sample, sample)| {
            // For each triangle vertex
            let mut lambda = Vector3::zero();
            surface_topo[sample.index].iter().for_each(|&vtx_idx| {
                lambda += multipliers[vtx_idx].into_tensor();
            });
            lambda *= third;

            let hess = sample_contact_jacobian_gradient_product_at(
                q,
                sample,
                which_sample,
                surface_topo,
                kernel,
                grad_phi,
                // dw_neigh,
                // ddwdq_neigh.clone(),
                // dwdq_neigh.clone(),
                dwnmldq_neigh.clone(),
                unit_nml_grads.clone(),
                jac_grad_phi.clone(),
                neigh_verts.clone(),
                // weight_sum_inv,
                weight_nml_sum_inv,
                closest_d,
                lambda,
                false,
            );
            hess.into_iter()
                .map(move |(col_idx, hess)| (col_idx, hess.into_data()))
        })
}

/// Computes the hessian for a single contact
/// `d/dq J_i^T b` where b is the multiplier
///
/// Note that this "Hessian" is not symmetric since the contact "Jacobian" is not really a
/// Jacobian of anything, it is an approximation of dx/dq with some physical meaning.
///
/// The returned `Matrix3` are interpreted as column major (implicit transpose of a row-major matrix)
pub(crate) fn contact_hessian_at<'a, T, K: 'a>(
    q: Vector3<T>,
    view: SamplesView<'a, 'a, T>,
    kernel: K,
    surface_vertices: &'a [[T; 3]],
    surface_topo: &'a [[usize; 3]],
    bg_field_params: BackgroundFieldParams,
    multiplier: Vector3<T>,
) -> impl Iterator<Item = (usize, usize, [[T; 3]; 3])> + 'a
where
    T: Real,
    K: SphericalKernel<T> + std::fmt::Debug + Copy + Sync + Send,
{
    let bg: BackgroundField<T, T, K> =
        BackgroundField::local(q, view, kernel, bg_field_params, None).unwrap();

    let closest_d = bg.closest_sample_dist();

    let grad_phi = query_jacobian_at(q, view, None, kernel, bg_field_params);

    let weight_sum_inv = bg.weight_sum_inv();
    let weight_nml_sum_inv = weight_nml_sum_inv(q, view, kernel, closest_d, grad_phi);

    let dwdq_neigh = normalized_neighbor_weight_vertex_gradients(
        q,
        view,
        kernel,
        surface_vertices.len(),
        surface_topo,
        bg.clone(),
    );
    let dw_neigh = normalized_neighbor_weight_gradient(q, view, kernel, bg.clone());
    let ddwdq_neigh = normalized_neighbor_weight_vertex_hessian(
        q,
        view,
        kernel,
        surface_vertices.len(),
        surface_topo,
        bg.clone(),
    );

    let third = T::one() / T::from(3.0).unwrap();

    let unit_nml_grads = view
        .into_iter()
        .map(move |sample| {
            let mut sample_unit_nml = sample.nml;
            let sample_area = sample_unit_nml.normalize();
            face_unit_normal_gradient_iter(
                sample.index,
                sample_unit_nml,
                T::one() / sample_area,
                surface_vertices,
                surface_topo,
            )
            .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();

    let neigh_verts = neighborhood_verts(view, surface_topo);

    let jac_grad_phi = jacobian_gradient_phi(
        q,
        view,
        kernel,
        &neigh_verts,
        &surface_topo,
        &unit_nml_grads,
        dw_neigh,
        &ddwdq_neigh,
        &dwdq_neigh,
        weight_sum_inv,
        closest_d,
    );

    let dwnmldq_neigh = normalized_neighbor_weight_nml_vertex_gradients(
        q,
        view,
        kernel,
        surface_vertices.len(),
        surface_topo,
        bg.clone(),
        grad_phi,
        &unit_nml_grads,
        &jac_grad_phi,
        &neigh_verts,
    );

    // For each sample
    view.into_iter()
        .enumerate()
        .flat_map(move |(which_sample, sample)| {
            // For each triangle vertex
            // let dwdq_neigh = dwdq_neigh.clone();
            let dwnmldq_neigh = dwnmldq_neigh.clone();
            // let ddwdq_neigh = ddwdq_neigh.clone();
            let unit_nml_grads = unit_nml_grads.clone();
            let jac_grad_phi = jac_grad_phi.clone();
            let neigh_verts = neigh_verts.clone();
            surface_topo[sample.index].iter().flat_map(move |&row_vtx| {
                let hess = sample_contact_jacobian_gradient_product_at(
                    q,
                    sample,
                    which_sample,
                    surface_topo,
                    kernel,
                    grad_phi,
                    // dw_neigh,
                    // ddwdq_neigh.clone(),
                    // dwdq_neigh.clone(),
                    dwnmldq_neigh.clone(),
                    unit_nml_grads.clone(),
                    jac_grad_phi.clone(),
                    neigh_verts.clone(),
                    weight_nml_sum_inv,
                    // weight_sum_inv,
                    closest_d,
                    multiplier,
                    true,
                );
                hess.into_iter()
                    .map(move |(col_idx, hess)| (row_vtx, col_idx, (hess * third).into_data()))
            })
        })
}

/// Contact Jacobian derivative with respect to query position.
///
/// Returns coulumn major blocks.
pub(crate) fn query_contact_jacobian_gradient_product_at<'a, T, K: 'a>(
    q: Vector3<T>,
    sample: Sample<T>,
    kernel: K,
    grad_phi: Vector3<T>,
    dw_neigh_normalized: Vector3<T>,
    ddpsi: Matrix3<T>,
    weight_sum_inv: T,
    closest_d: T,
    multiplier: Vector3<T>,
    transpose: bool,
) -> Matrix3<T>
where
    T: Real,
    K: SphericalKernel<T> + std::fmt::Debug + Copy + Sync + Send,
{
    let sample_unit_nml = sample.nml.normalized();
    let nml_dot_grad = sample_unit_nml.dot(grad_phi);
    let mut u = sample_unit_nml.cross(grad_phi);

    if transpose {
        // The negative makes rodrigues rotation transpose.
        u *= -T::one();
    }
    let ux = u.skew();

    // Qs (or Qs^T) matrix in the paper.
    let rot = rodrigues_rotation(ux, nml_dot_grad);

    let kernel = kernel.with_closest_dist(closest_d);
    let nml_kern = NormalKernel::new(sample_unit_nml, grad_phi);
    let w_nml = nml_kern.eval();
    let w = kernel.eval(q, sample.pos);
    let w_normalized = w_nml * w * weight_sum_inv;
    let dw = kernel.grad(q, sample.pos);
    let dw_normalized =
        (dw * w_nml + nml_kern.grad(Matrix3::zero(), ddpsi.transpose()) * w) * weight_sum_inv;
    let dw_normalized_term =
        (dw_normalized - dw_neigh_normalized * w_normalized) * (rot * multiplier).transpose();

    let rot_jac = rodrigues_rotation_jacobian_product_at(
        sample_unit_nml,
        grad_phi,
        ddpsi.transpose(),
        Matrix3::zero(),
        nml_dot_grad,
        multiplier,
        transpose,
    );

    dw_normalized_term + rot_jac * w_normalized
}

/// Contact Jacobian derivative with respect to the three adjacent vertex positions.
///
/// Returns column major blocks.
pub(crate) fn sample_contact_jacobian_gradient_product_at<'a, T, K: 'a>(
    q: Vector3<T>,
    sample: Sample<T>,
    which_sample: usize,
    surface_topo: &'a [[usize; 3]],
    kernel: K,
    grad_phi: Vector3<T>,
    dwdq_neigh_normalized: Vec<Vector3<T>>,
    unit_nml_grads: Vec<Vec<Matrix3<T>>>,
    jac_grad_phi: Vec<Matrix3<T>>,
    neigh_verts: Vec<usize>,
    weight_sum_inv: T,
    closest_d: T,
    multiplier: Vector3<T>,
    transpose: bool,
) -> impl Iterator<Item = (usize, Matrix3<T>)> + 'a
where
    T: Real,
    K: SphericalKernel<T> + std::fmt::Debug + Copy + Sync + Send,
{
    // Jacobian values
    let kernel = kernel.with_closest_dist(closest_d);
    let w = kernel.eval(q, sample.pos);
    let w_normalized = w * weight_sum_inv;
    let dw = kernel.grad(q, sample.pos);
    //grad_phi.normalize();
    let sample_unit_nml = sample.nml.normalized();
    let nml_dot_grad = sample_unit_nml.dot(grad_phi);
    let mut u = sample_unit_nml.cross(grad_phi);

    // TODO: Figure out why the transpose is wrong here
    if transpose {
        // The negative makes rodrigues rotation transpose.
        u *= -T::one();
    }
    let ux = u.skew();

    // Qs (or Qs^T) matrix in the paper.
    let rot = rodrigues_rotation(ux, nml_dot_grad);
    let rot_b_t = (rot * multiplier).transpose();

    let third = T::from(1.0 / 3.0).unwrap();

    let nml_kern = NormalKernel::new(sample_unit_nml, grad_phi);
    let w_nml = nml_kern.eval();

    // Hessian specific values.
    // Negative in front of kernel grad is because grad is wrt sample pos not q.
    // let dw_normalized = dw * weight_sum_inv;
    let dw_normalized = (dw * w_nml) * weight_sum_inv;
    // This term is transposed to make it column-major.
    let dw_normalized_term = dw_normalized * (rot_b_t * (-third));

    let dw_normalized_entries = surface_topo[sample.index]
        .iter()
        .map(move |&tri_vtx| (tri_vtx, dw_normalized_term));

    let sqrt_w_nml_factor =
        T::from(0.5).unwrap() * (T::one() + sample_unit_nml.dot(grad_phi)) * w_normalized;

    let unit_nml_grad_entries = surface_topo[sample.index]
        .iter()
        .zip(unit_nml_grads[which_sample].clone().into_iter())
        .map(move |(&tri_vtx, unit_nml_grad)| {
            (
                tri_vtx,
                (unit_nml_grad * grad_phi * sqrt_w_nml_factor) * rot_b_t,
            )
        });

    // Compute Jacobian of the rotation matrix Qs multiplied by multiplier.
    let second_term_entries = neigh_verts
        .iter()
        .zip(jac_grad_phi.iter())
        .map(|(&tri_vtx, jac_grad_phi)| {
            // let dw_normalized_denom_term = -dwdq_neigh_normalized[tri_vtx]
            //     * w
            //     * weight_sum_inv
            //     * (rot * multiplier).transpose();
            let dw_normalized_denom_term =
                -dwdq_neigh_normalized[tri_vtx] * w_normalized * w_nml * rot_b_t;

            let jac_grad_phi_t = jac_grad_phi.transpose();

            let jac_grad_phi_term =
                (jac_grad_phi_t * sample_unit_nml * sqrt_w_nml_factor) * rot_b_t;

            let unit_nml_grad = surface_topo[sample.index]
                .iter()
                .position(|&x| x == tri_vtx)
                .map(|which_vtx| unit_nml_grads[which_sample][which_vtx])
                .unwrap_or_else(Vector3::zero);

            let rot_jac = rodrigues_rotation_jacobian_product_at(
                sample_unit_nml,
                grad_phi,
                jac_grad_phi_t,
                unit_nml_grad,
                nml_dot_grad,
                multiplier,
                transpose,
            );
            (
                tri_vtx,
                dw_normalized_denom_term + rot_jac * w_normalized * w_nml + jac_grad_phi_term,
            )
        })
        .collect::<Vec<_>>();
    dw_normalized_entries
        .chain(second_term_entries.into_iter())
        .chain(unit_nml_grad_entries.into_iter())
}

/// Jacobian of `Qs^T b` for some multiplier `b` with respect to sample position.
///
/// Returns a column-major matrix.
pub(crate) fn rodrigues_rotation_jacobian_product_at<T>(
    sample_unit_nml: Vector3<T>,
    grad_phi: Vector3<T>,
    jac_grad_phi_t: Matrix3<T>,
    unit_nml_grad: Matrix3<T>,
    nml_dot_grad: T,
    multiplier: Vector3<T>,
    transpose: bool,
) -> Matrix3<T>
where
    T: Real,
{
    // First term (-jac n x grad_phi x b)
    let d_n_cross_grad_phi = [
        unit_nml_grad[0].cross(grad_phi) + sample_unit_nml.cross(jac_grad_phi_t[0]),
        unit_nml_grad[1].cross(grad_phi) + sample_unit_nml.cross(jac_grad_phi_t[1]),
        unit_nml_grad[2].cross(grad_phi) + sample_unit_nml.cross(jac_grad_phi_t[2]),
    ];

    let first = [
        d_n_cross_grad_phi[0].cross(multiplier).into_data(),
        d_n_cross_grad_phi[1].cross(multiplier).into_data(),
        d_n_cross_grad_phi[2].cross(multiplier).into_data(),
    ]
    .into_tensor();

    let n_cross_grad_phi = sample_unit_nml.cross(grad_phi);

    let n_cross_grad_phi_cross_b = n_cross_grad_phi.cross(multiplier);

    let second = [
        (d_n_cross_grad_phi[0].cross(n_cross_grad_phi_cross_b)
            + n_cross_grad_phi.cross(d_n_cross_grad_phi[0].cross(multiplier)))
        .into_data(),
        (d_n_cross_grad_phi[1].cross(n_cross_grad_phi_cross_b)
            + n_cross_grad_phi.cross(d_n_cross_grad_phi[1].cross(multiplier)))
        .into_data(),
        (d_n_cross_grad_phi[2].cross(n_cross_grad_phi_cross_b)
            + n_cross_grad_phi.cross(d_n_cross_grad_phi[2].cross(multiplier)))
        .into_data(),
    ]
    .into_tensor();

    let denom = T::one() + nml_dot_grad;
    let factor = if denom == T::zero() {
        T::zero()
    } else {
        T::one() / denom
    }; // heuristic

    let jac_scalar_factor =
        (unit_nml_grad * grad_phi + jac_grad_phi_t * sample_unit_nml) * (-factor * factor);

    // Note that this is a transpose of what is written in the notes since we are returning a column-major matrix here.
    let third = jac_scalar_factor * n_cross_grad_phi.cross(n_cross_grad_phi_cross_b).transpose();

    (if transpose { -first } else { first }) + second * factor + third
}

/// Computes the off-diagonal elements of the second derivative of the implicit function.
///
/// This is `jac_q grad_x Psi` where the Jacobian is with respect to vertices of triangle
/// corresponding to sample s.
/// Note that this function computes only a component of the whole derivative.
pub(crate) fn sample_query_hessian_component_at<'a, T, K: 'a>(
    q: Vector3<T>,
    sample: Sample<T>,
    sample_unit_nml: Vector3<T>,
    kernel: K,
    unit_nml_grads: &[Matrix3<T>],
    which_vtx: Option<usize>,
    dw_neigh_normalized: Vector3<T>,
    ddwdq_neigh_normalized: Matrix3<T>,
    dwdq_neigh_normalized: Vector3<T>,
    weight_sum_inv: T,
    closest_d: T,
) -> Matrix3<T>
where
    T: Real,
    K: SphericalKernel<T> + std::fmt::Debug + Copy + Sync + Send,
{
    let w = kernel.with_closest_dist(closest_d).eval(q, sample.pos);
    let dw = kernel.with_closest_dist(closest_d).grad(q, sample.pos);
    let ddw = kernel.with_closest_dist(closest_d).hess(q, sample.pos);
    let psi = T::from(sample.value).unwrap() + sample_unit_nml.dot(q - sample.pos);
    let dw_normalized = (dw - dw_neigh_normalized * w) * weight_sum_inv;
    let grad_psi = dw_normalized * psi + (sample_unit_nml * w) * weight_sum_inv;

    let third = T::from(1.0 / 3.0).unwrap();

    // 1
    let mut res = -grad_psi * (dwdq_neigh_normalized.transpose()); // Note a 1/3 factor is already in dwdq_neigh_normalized
                                                                   // last part of 3
    res += (dw_neigh_normalized * (dwdq_neigh_normalized.transpose()) - ddwdq_neigh_normalized)
        * w
        * psi
        * weight_sum_inv;
    if let Some(vtx_idx) = which_vtx {
        // 2
        res += dw_normalized
            * (unit_nml_grads[vtx_idx] * (q - sample.pos) - sample_unit_nml * third).transpose();
        res += (
            // first part of 3
            (-ddw + dw_neigh_normalized * dw.transpose()) * psi * third
            // 4
            - sample_unit_nml * (dw.transpose() * third)
                + unit_nml_grads[vtx_idx].transpose() * w
        ) * weight_sum_inv
    }
    res
}

/// Helper function to print the dense Hessian given by a vector of vectors in column-major format.
/// This can also be used elsewhere for testing that involves hessians (e.g. background field).
#[cfg(test)]
#[allow(dead_code)]
pub(crate) fn print_full_hessian(hess: &[Vec<f64>], size: usize, name: &str) {
    println!("{} = ", name);
    for r in 0..size {
        for c in 0..size {
            if relative_eq!(hess[c][r], 0.0, max_relative = 1e-6, epsilon = 1e-12) {
                print!("     .    ",);
            } else {
                print!("{:9.5} ", hess[c][r]);
            }
        }
        println!();
    }
    println!();
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::jacobian::consolidate_contact_jacobian;
    use crate::kernel;
    use autodiff::F1;
    use geo::mesh::builder::*;
    use geo::mesh::{TriMesh, VertexPositions};
    use jacobian::{
        consolidate_face_jacobian, make_perturb_fn, make_test_triangle, make_three_test_triangles,
        make_two_test_triangles, new_test_samples,
    };

    /// High level test of the Hessian as the derivative of the Jacobian.
    fn query_hessian_tester(
        query_points: &[[f64; 3]],
        surf_mesh: &TriMesh<f64>,
        kernel: KernelType,
        max_step: f64,
        bg_field_params: BackgroundFieldParams,
    ) -> Result<(), Error> {
        let params = crate::Params {
            kernel,
            background_field: bg_field_params,
            sample_type: SampleType::Face,
            max_step,
            base_radius: None,
        };

        let surf = crate::mls_from_trimesh::<F1>(&surf_mesh, params)
            .expect("Failed to construct an implicit surface.");

        let ad_tri_verts: Vec<_> = surf_mesh
            .vertex_position_iter()
            .map(|&x| Vector3::new(x).mapd(|x| F1::cst(x)).into_data())
            .collect();
        let mut ad_query_points: Vec<_> = query_points
            .iter()
            .map(|&a| Vector3::new(a).mapd(|x| F1::cst(x)).into_data())
            .collect();
        let num_query_points = query_points.len();

        let mut query_surf = surf.query_topo(&ad_query_points);
        let num_hess_entries = query_surf.num_query_hessian_product_entries();
        let num_jac_entries = query_surf.num_query_jacobian_entries();

        // Compute the complete Hessian.
        let mut hess_rows = vec![0; num_hess_entries];
        let mut hess_cols = vec![0; num_hess_entries];
        let mut hess_values = vec![F1::cst(0.0); num_hess_entries];
        let mut jac_rows = vec![0; num_jac_entries];
        let mut jac_cols = vec![0; num_jac_entries];
        let mut jac_values = vec![F1::cst(0.0); num_jac_entries];

        let num_neighs = query_surf.num_neighborhoods();
        let mut multipliers = vec![F1::cst(0.0); num_neighs];
        query_surf
            .query_hessian_product_indices_iter()
            .zip(hess_rows.iter_mut().zip(hess_cols.iter_mut()))
            .for_each(|((r, c), (out_r, out_c))| {
                *out_r = r;
                *out_c = c;
            });

        let mut hess_full = vec![vec![0.0; 3 * num_query_points]; 3 * num_query_points];
        let mut ad_hess_full = vec![vec![0.0; 3 * num_query_points]; 3 * num_query_points];

        let query_neighborhood_sizes = query_surf.neighborhood_sizes();
        dbg!(&query_neighborhood_sizes);

        // We use the multipliers to isolate the hessian for each query point.
        for (mult_idx, q_idx) in (0..num_query_points)
            .filter(|&q_idx| query_neighborhood_sizes[q_idx] != 0)
            .enumerate()
        {
            multipliers[mult_idx] = F1::cst(1.0);

            query_surf.query_hessian_product_values(
                &ad_query_points,
                &multipliers,
                &mut hess_values,
            );

            let mut success = true;

            // Test the accuracy of each component of the hessian against an autodiff version of the
            // second derivative.
            for vtx in 0..num_query_points {
                for i in 0..3 {
                    //eprintln!("DERIVATIVE WRT vtx {}, component {}", vtx, i);
                    // Set a variable to take the derivative with respect to, using autodiff.
                    ad_query_points[vtx][i] = F1::var(ad_query_points[vtx][i]);
                    query_surf.update_surface(ad_tri_verts.iter().cloned(), true);

                    query_surf.query_jacobian_values(&ad_query_points, &mut jac_values);
                    query_surf
                        .query_jacobian_indices_iter()
                        .zip(jac_rows.iter_mut().zip(jac_cols.iter_mut()))
                        .for_each(|((r, c), (out_r, out_c))| {
                            *out_r = r;
                            *out_c = c;
                        });

                    // Get the jacobian for the specific query point we are interested in.
                    let mut jac_q = vec![F1::cst(0.0); num_query_points * 3];
                    for (&r, &c, &jac) in zip!(jac_rows.iter(), jac_cols.iter(), jac_values.iter())
                    {
                        if r == q_idx {
                            jac_q[c] += jac;
                        }
                    }

                    // Consolidate the Hessian to the particular vertex and component we are
                    // interested in.
                    let mut hess_vtx = vec![F1::cst(0.0); num_query_points * 3];
                    for (&r, &c, &h) in zip!(hess_rows.iter(), hess_cols.iter(), hess_values.iter())
                    {
                        if r == 3 * vtx + i {
                            hess_vtx[c] += h;
                        } else if c == 3 * vtx + i {
                            hess_vtx[r] += h;
                        }
                    }

                    for (linear_j, (jac, hes)) in jac_q.iter().zip(hess_vtx).enumerate() {
                        // Check the derivative of the autodiff with our previously computed Jacobian.
                        if !relative_eq!(
                            hes.value(),
                            jac.deriv(),
                            max_relative = 1e-6,
                            epsilon = 1e-12
                        ) {
                            println!("{:.5} vs {:.5}", hes.value(), jac.deriv());
                            success = false;
                        }
                        ad_hess_full[3 * vtx + i][linear_j] += jac.deriv();
                        hess_full[3 * vtx + i][linear_j] += hes.value();
                    }

                    // Reset the variable back to being a constant.
                    ad_query_points[vtx][i] = F1::cst(ad_query_points[vtx][i]);
                }

                if !success {
                    eprintln!("Multiplier: {}", mult_idx);
                    print_full_hessian(&hess_full, 3 * num_query_points, "Full Hessian");
                    print_full_hessian(
                        &ad_hess_full,
                        3 * num_query_points,
                        "Full Autodiff Hessian",
                    );
                }
                assert!(success, "Hessian does not match its AutoDiff counterpart");
            }
            multipliers[mult_idx] = F1::cst(0.0); // reset multiplier
        }
        Ok(())
    }

    /// Test the highest level query Hessian functions with a tetrahedron.
    #[test]
    fn one_tet_query_hessian_test() -> Result<(), Error> {
        let qs: Vec<_> = (0..4).map(|i| [0.0, -0.5 + 0.25 * i as f64, 0.0]).collect();

        let trimesh = TriMesh::from(PlatonicSolidBuilder::new().build_tetrahedron());

        for i in 9..10 {
            let radius_multiplier = 1.0 + 0.5 * (i as f64);
            eprintln!("radius multiplier: {}", radius_multiplier);
            let bg_params = BackgroundFieldParams {
                field_type: BackgroundFieldType::Zero,
                weighted: false,
            };
            let run = |kernel| query_hessian_tester(&qs, &trimesh, kernel, 0.0, bg_params);

            eprintln!("Checking Interpolating Kernel");
            run(KernelType::Interpolating { radius_multiplier })?;
            eprintln!("Checking Approximate Kernel");
            run(KernelType::Approximate {
                tolerance: 0.00001,
                radius_multiplier,
            })?;
            eprintln!("Checking Cubic Kernel");
            run(KernelType::Cubic { radius_multiplier })?;
            eprintln!("Checking Global Kernel");
            run(KernelType::Global { tolerance: 0.00001 })?;
        }
        Ok(())
    }

    /// High level test of the Hessian as the derivative of the Jacobian.
    fn surface_hessian_tester(
        query_points: &[[f64; 3]],
        surf_mesh: &TriMesh<f64>,
        kernel: KernelType,
        max_step: f64,
        bg_field_params: BackgroundFieldParams,
    ) -> Result<(), Error> {
        let params = crate::Params {
            kernel,
            background_field: bg_field_params,
            sample_type: SampleType::Face,
            max_step,
            base_radius: None,
        };

        let surf = crate::mls_from_trimesh::<F1>(&surf_mesh, params)
            .expect("Failed to construct an implicit surface.");

        let mut ad_tri_verts: Vec<_> = surf_mesh
            .vertex_position_iter()
            .map(|&x| Vector3::new(x).mapd(|x| F1::cst(x)).into_data())
            .collect();
        let num_verts = ad_tri_verts.len();
        let ad_query_points: Vec<_> = query_points
            .iter()
            .map(|&a| Vector3::new(a).mapd(|x| F1::cst(x)).into_data())
            .collect();
        let num_query_points = query_points.len();

        let mut query_surf = surf.query_topo(&ad_query_points);
        let num_hess_entries = query_surf
            .num_surface_hessian_product_entries()
            .expect("Uncached query points.");
        let num_jac_entries = query_surf.num_surface_jacobian_entries();

        // Compute the complete Hessian.
        let mut hess_rows = vec![0; num_hess_entries];
        let mut hess_cols = vec![0; num_hess_entries];
        let mut hess_values = vec![F1::cst(0.0); num_hess_entries];
        let mut jac_rows = vec![0; num_jac_entries];
        let mut jac_cols = vec![0; num_jac_entries];
        let mut jac_values = vec![F1::cst(0.0); num_jac_entries];

        let num_neighs = query_surf.num_neighborhoods();
        let mut multipliers = vec![F1::cst(0.0); num_neighs];
        query_surf
            .surface_hessian_product_indices(&mut hess_rows, &mut hess_cols)
            .expect("Failed to compute hessian indices");

        let mut hess_full = vec![vec![0.0; 3 * num_verts]; 3 * num_verts];
        let mut ad_hess_full = vec![vec![0.0; 3 * num_verts]; 3 * num_verts];

        let query_neighborhood_sizes = query_surf.neighborhood_sizes();
        dbg!(&query_neighborhood_sizes);

        // We use the multipliers to isolate the hessian for each query point.
        for (mult_idx, q_idx) in (0..num_query_points)
            .filter(|&q_idx| query_neighborhood_sizes[q_idx] != 0)
            .enumerate()
        {
            multipliers[mult_idx] = F1::cst(1.0);

            query_surf
                .surface_hessian_product_values(&ad_query_points, &multipliers, &mut hess_values)
                .expect("Failed to compute hessian product");
            let hess_values2: Vec<_> = query_surf
                .surface_hessian_product_values_iter(&ad_query_points, &multipliers)
                .expect("Failed to compute hessian product using iterators")
                .collect();
            assert!(hess_values
                .iter()
                .zip(hess_values2.iter())
                .all(|(a, b)| a == b));

            let mut success = true;

            // Test the accuracy of each component of the hessian against an autodiff version of the
            // second derivative.
            for vtx in 0..num_verts {
                for i in 0..3 {
                    // Set a variable to take the derivative with respect to, using autodiff.
                    ad_tri_verts[vtx][i] = F1::var(ad_tri_verts[vtx][i]);
                    query_surf.update_surface(ad_tri_verts.iter().cloned(), true);

                    query_surf.surface_jacobian_values(&ad_query_points, &mut jac_values);
                    query_surf.surface_jacobian_indices(&mut jac_rows, &mut jac_cols);

                    // Get the jacobian for the specific query point we are interested in.
                    let mut jac_q = vec![F1::cst(0.0); num_verts * 3];
                    for (&r, &c, &jac) in zip!(jac_rows.iter(), jac_cols.iter(), jac_values.iter())
                    {
                        if r == q_idx {
                            jac_q[c] += jac;
                        }
                    }

                    // Consolidate the Hessian to the particular vertex and component we are
                    // interested in.
                    let mut hess_vtx = vec![F1::cst(0.0); num_verts * 3];
                    for (&r, &c, &h) in zip!(hess_rows.iter(), hess_cols.iter(), hess_values.iter())
                    {
                        if r == 3 * vtx + i {
                            hess_vtx[c] += h;
                        } else if c == 3 * vtx + i {
                            hess_vtx[r] += h;
                        }
                    }

                    for (linear_j, (jac, hes)) in jac_q.iter().zip(hess_vtx).enumerate() {
                        // Check the derivative of the autodiff with our previously computed Jacobian.
                        if !relative_eq!(
                            hes.value(),
                            jac.deriv(),
                            max_relative = 1e-6,
                            epsilon = 1e-12
                        ) {
                            println!("{:.5} vs {:.5}", hes.value(), jac.deriv());
                            success = false;
                        }
                        ad_hess_full[3 * vtx + i][linear_j] += jac.deriv();
                        hess_full[3 * vtx + i][linear_j] += hes.value();
                    }

                    // Reset the variable back to being a constant.
                    ad_tri_verts[vtx][i] = F1::cst(ad_tri_verts[vtx][i]);
                }

                if !success {
                    print_full_hessian(&hess_full, 3 * num_verts, "Full Hessian");
                    print_full_hessian(&ad_hess_full, 3 * num_verts, "Full Autodiff Hessian");
                }
                assert!(success, "Hessian does not match its AutoDiff counterpart");
            }
            multipliers[mult_idx] = F1::cst(0.0); // reset multiplier
        }
        Ok(())
    }

    /// Test the highest level surface Hessian functions with a tetrahedron.
    #[test]
    fn one_tet_hessian_test() -> Result<(), Error> {
        let qs: Vec<_> = (0..4).map(|i| [0.0, -0.5 + 0.25 * i as f64, 0.0]).collect();

        let trimesh = TriMesh::from(PlatonicSolidBuilder::new().build_tetrahedron());

        for i in 1..10 {
            let radius_multiplier = 1.0 + 0.5 * (i as f64);
            let bg_params = BackgroundFieldParams {
                field_type: BackgroundFieldType::Zero,
                weighted: false,
            };
            let run = |kernel| surface_hessian_tester(&qs, &trimesh, kernel, 0.0, bg_params);

            run(KernelType::Interpolating { radius_multiplier })?;
            run(KernelType::Approximate {
                tolerance: 0.00001,
                radius_multiplier,
            })?;
            run(KernelType::Cubic { radius_multiplier })?;
            run(KernelType::Global { tolerance: 0.00001 })?;
        }
        Ok(())
    }

    /// Test the highest level surface Hessian functions with a single triangle
    #[test]
    fn one_triangle_hessian_test() -> Result<(), Error> {
        let qs = vec![[0.0, 0.4, 0.0], [0.0, 0.0, 0.0], [0.0, -0.4, 0.0]];
        let mut perturb = make_perturb_fn();
        let tri_verts = make_test_triangle(0.0, &mut perturb)
            .into_iter()
            .map(|x| x.into_data())
            .collect();
        let tri = geo::mesh::TriMesh::new(tri_verts, vec![[0, 1, 2]]);

        for i in 1..50 {
            let radius_multiplier = 1.0 + 0.1 * (i as f64);
            let bg_params = BackgroundFieldParams {
                field_type: BackgroundFieldType::Zero,
                weighted: false,
            };
            let run = |kernel| surface_hessian_tester(&qs, &tri, kernel, 0.0, bg_params);

            run(KernelType::Interpolating { radius_multiplier })?;
            run(KernelType::Approximate {
                tolerance: 0.00001,
                radius_multiplier,
            })?;
            run(KernelType::Cubic { radius_multiplier })?;
            run(KernelType::Global { tolerance: 0.00001 })?;
        }
        Ok(())
    }

    /// High level test of the contact Hessian as the derivative of the contact Jacobian.
    fn full_contact_hessian_tester(
        query_points: &[[f64; 3]],
        surf_mesh: &TriMesh<f64>,
        kernel: KernelType,
        max_step: f64,
        bg_field_params: BackgroundFieldParams,
    ) -> Result<(), Error> {
        let params = crate::Params {
            kernel,
            background_field: bg_field_params,
            sample_type: SampleType::Face,
            max_step,
            base_radius: None,
        };

        let surf = crate::mls_from_trimesh::<F1>(&surf_mesh, params)
            .expect("Failed to construct an implicit surface.");

        let mut ad_tri_verts: Vec<_> = surf_mesh
            .vertex_position_iter()
            .map(|&x| Vector3::new(x).mapd(|x| F1::cst(x)).into_data())
            .collect();
        let num_verts = ad_tri_verts.len();
        let ad_query_points: Vec<_> = query_points
            .iter()
            .map(|&a| Vector3::new(a).mapd(|x| F1::cst(x)).into_data())
            .collect();
        let num_query_points = query_points.len();

        let mut query_surf = surf.query_topo(&ad_query_points);
        let num_neighs = query_surf.num_neighborhoods();
        let mut multipliers = vec![[F1::cst(0.0); 3]; num_neighs];

        let num_jac_entries = query_surf.num_contact_jacobian_entries();

        // Compute the complete Hessian product.
        let mut jac_values = vec![F1::cst(0.0); num_jac_entries];

        let mut hess_full = vec![vec![0.0; 3 * num_verts]; 3 * num_verts];
        let mut ad_hess_full = vec![vec![0.0; 3 * num_verts]; 3 * num_verts];

        let query_neighborhood_sizes = query_surf.neighborhood_sizes();
        // dbg!(&query_neighborhood_sizes);

        // We use the multipliers to isolate the hessian for each query point.
        for (mult_idx, q_idx) in (0..num_query_points)
            .filter(|&q_idx| query_neighborhood_sizes[q_idx] != 0)
            .enumerate()
        {
            for k in 0..3 {
                multipliers[mult_idx][k] = F1::cst(1.0);

                let hess_entries: Vec<(usize, usize, [[F1; 3]; 3])> = query_surf
                    .contact_hessian_product_indexed_blocks_iter(&ad_query_points, &multipliers)
                    .expect("Failed to compute hessian product")
                    .collect();

                let mut success = true;

                // Test the accuracy of each component of the hessian against an autodiff version of the
                // second derivative.
                for vtx in 0..num_verts {
                    for i in 0..3 {
                        // Set a variable to take the derivative with respect to, using autodiff.
                        ad_tri_verts[vtx][i] = F1::var(ad_tri_verts[vtx][i]);
                        query_surf.update_surface(ad_tri_verts.iter().cloned(), true);

                        query_surf.contact_jacobian_values(&ad_query_points, &mut jac_values);
                        let (jac_rows, jac_cols): (Vec<_>, Vec<_>) =
                            query_surf.contact_jacobian_indices_iter().unzip();
                        assert_eq!(jac_rows.len(), jac_values.len());

                        let jac2: Vec<_> = query_surf
                            .contact_jacobian_matrices_par_iter(&ad_query_points)
                            .collect();
                        // let values: Vec<_> = jac2.iter().flat_map(|(_,_,b)| b.iter().flat_map(|r| r.iter().cloned())).collect();
                        // eprintln!("parvals: {values:?}");
                        // eprintln!("vals   : {:?}", &jac_values);

                        let mut djac_q2 = vec![0.0; num_verts * 3];
                        for (row, col, block) in jac2.iter() {
                            if *row == q_idx {
                                for ii in 0..3 {
                                    djac_q2[3 * *col + ii] += block[k][ii].deriv();
                                }
                            }
                        }

                        // Get the Jacobian for the specific query point we are interested in.
                        let mut djac_q = vec![0.0; num_verts * 3];
                        for (&r, &c, &jac) in
                            zip!(jac_rows.iter(), jac_cols.iter(), jac_values.iter())
                        {
                            if r == 3 * q_idx + k {
                                djac_q[c] += jac.deriv();
                            }
                        }

                        assert!(djac_q2.iter().zip(djac_q.iter()).all(|(a, b)| a == b));

                        // Consolidate the Hessian to the particular vertex and component we are
                        // interested in.
                        let mut hess_vtx = vec![F1::cst(0.0); num_verts * 3];
                        for (r, c, h) in hess_entries.iter() {
                            if *c == vtx {
                                for j in 0..3 {
                                    hess_vtx[3 * *r + j] += h[i][j];
                                }
                            }
                        }

                        for (linear_j, (jac, hes)) in djac_q.iter().zip(hess_vtx).enumerate() {
                            // Check the derivative of the autodiff with our previously computed Jacobian.
                            if !relative_eq!(hes.value(), jac, max_relative = 1e-6, epsilon = 1e-12)
                            {
                                println!("{:.5} vs {:.5}", hes.value(), jac);
                                success = false;
                            }
                            ad_hess_full[3 * vtx + i][linear_j] += jac;
                            hess_full[3 * vtx + i][linear_j] += hes.value();
                        }

                        // Reset the variable back to being a constant.
                        ad_tri_verts[vtx][i] = F1::cst(ad_tri_verts[vtx][i]);
                    }
                }
                if !success {
                    print_full_hessian(&hess_full, 3 * num_verts, "Full Hessian");
                    print_full_hessian(&ad_hess_full, 3 * num_verts, "Full Autodiff Hessian");
                }
                assert!(success, "Hessian does not match its AutoDiff counterpart");
                multipliers[mult_idx][k] = F1::cst(0.0); // reset multiplier
            }
        }
        Ok(())
    }

    /// Test the highest level surface Hessian functions with a tetrahedron.
    #[test]
    fn one_tet_full_contact_hessian_test() -> Result<(), Error> {
        let qs: Vec<_> = (0..4).map(|i| [0.0, -0.5 + 0.25 * i as f64, 0.0]).collect();

        let trimesh = TriMesh::from(PlatonicSolidBuilder::new().build_tetrahedron());

        for i in 1..10 {
            let radius_multiplier = 1.0 + 0.5 * (i as f64);
            let bg_params = BackgroundFieldParams {
                field_type: BackgroundFieldType::Zero,
                weighted: false,
            };
            let run = |kernel| full_contact_hessian_tester(&qs, &trimesh, kernel, 0.0, bg_params);

            run(KernelType::Interpolating { radius_multiplier })?;
            run(KernelType::Approximate {
                tolerance: 0.00001,
                radius_multiplier,
            })?;
            run(KernelType::Cubic { radius_multiplier })?;
            run(KernelType::Global { tolerance: 0.00001 })?;
        }
        Ok(())
    }

    fn one_tet_face_hessian<P: FnMut() -> Vector3<f64>>(
        radius: f64,
        bg_field_params: BackgroundFieldParams,
        perturb: &mut P,
    ) {
        let tet = TriMesh::from(PlatonicSolidBuilder::new().build_tetrahedron());
        let qs: Vec<_> = (0..4)
            .map(|i| Vector3::new([0.0, -0.5 + 0.25 * i as f64, 0.0]))
            .collect();

        for q in qs.into_iter() {
            face_hessian_tester(q, &tet, radius, 0.0, bg_field_params);
            face_hessian_tester(q + perturb(), &tet, radius, 0.0, bg_field_params);
            face_hessian_tester(q + perturb(), &tet, radius, 1.0, bg_field_params);
        }
    }

    fn two_triangle_face_hessian<P: FnMut() -> Vector3<f64>>(
        radius: f64,
        bg_field_params: BackgroundFieldParams,
        perturb: &mut P,
    ) {
        let (tri_verts, tri_indices) = make_two_test_triangles(0.0, perturb);
        let tri = TriMesh::new(tri_verts, tri_indices);
        let qs = vec![
            Vector3::new([0.0, 0.2, 0.0]),
            Vector3::new([0.0, 0.0001, 0.0]),
            Vector3::new([0.0, -0.4, 0.0]),
        ];

        for q in qs.into_iter() {
            face_hessian_tester(q, &tri, radius, 0.0, bg_field_params);
            face_hessian_tester(q + perturb(), &tri, radius, 0.0, bg_field_params);
            face_hessian_tester(q + perturb(), &tri, radius, 1.0, bg_field_params);
        }
    }

    fn one_triangle_face_hessian<P: FnMut() -> Vector3<f64>>(
        radius: f64,
        bg_field_params: BackgroundFieldParams,
        perturb: &mut P,
    ) {
        let tri_verts: Vec<_> = make_test_triangle(0.0, perturb)
            .into_iter()
            .map(|x| x.into_data())
            .collect();
        let tri = TriMesh::new(tri_verts, vec![[0, 1, 2]]);
        let qs = vec![
            Vector3::new([0.0, 0.2, 0.0]),
            Vector3::new([0.0, 0.0001, 0.0]),
            Vector3::new([0.0, -0.4, 0.0]),
        ];

        for q in qs.into_iter() {
            face_hessian_tester(q, &tri, radius, 0.0, bg_field_params);
            face_hessian_tester(q + perturb(), &tri, radius, 0.0, bg_field_params);
            face_hessian_tester(q + perturb(), &tri, radius, 1.0, bg_field_params);
        }
    }

    /// Test the `face_hessian_at` function. This verifies that it indeed produces the derivative
    /// of the `face_jacobian_at` function. This function also tests that `face_jacobian_at` is the
    /// derivative of `compute_potential_at` for good measure.
    fn face_hessian_tester(
        q: Vector3<f64>,
        mesh: &TriMesh<f64>,
        radius: f64,
        max_step: f64,
        bg_field_params: BackgroundFieldParams,
    ) {
        let q = q.mapd(|x| F1::cst(x)); // convert to autodiff
        let mut tri_verts: Vec<_> = mesh
            .vertex_position_iter()
            .map(|&x| Vector3::new(x).mapd(|x| F1::cst(x)).into())
            .collect();
        let tri_faces = mesh.indices.as_slice();
        let num_verts = tri_verts.len();

        let samples = new_test_samples(SampleType::Face, &tri_faces, &tri_verts);

        let neighbors: Vec<_> = samples
            .iter()
            .filter(|s| (q - s.pos).norm() < F1::cst(radius + max_step))
            .map(|sample| sample.index)
            .collect();

        if neighbors.is_empty() {
            return;
        }

        // Radius is such that samples are captured by the query point.
        let kernel = kernel::LocalApproximate::new(radius, 0.00001);

        let view = SamplesView::new(neighbors.as_ref(), &samples);

        // Compute the complete hessian.
        let hess: Vec<(usize, usize, Matrix3<F1>)> = face_hessian_at(
            q,
            view,
            kernel,
            tri_faces,
            &tri_verts,
            bg_field_params,
            F1::cst(1.0),
        )
        .collect();

        let mut hess_full = vec![vec![0.0; 3 * num_verts]; 3 * num_verts];
        let mut hess_full2 = vec![vec![0.0; 3 * num_verts]; 3 * num_verts];

        for &(r, c, h) in hess.iter() {
            for j in 0..3 {
                for i in 0..3 {
                    hess_full[3 * c + j][3 * r + i] += h[i][j].value();
                }
            }
        }

        let mut ad_hess_full = vec![vec![0.0; 3 * num_verts]; 3 * num_verts];

        let mut success = true;

        // Test the accuracy of each component of the hessian against an autodiff version of the
        // second derivative.
        for vtx in 0..num_verts {
            for i in 0..3 {
                // Set a variable to take the derivative with respect to, using autodiff.
                tri_verts[vtx][i] = F1::var(tri_verts[vtx][i]);
                //println!("row_vtx = {}; i = {}", vtx, i);

                // We need to update samples to make sure the normals and centroids are recomputed
                // using the correct wrt autodiff variable.
                let samples = new_test_samples(SampleType::Face, &tri_faces, &tri_verts);
                //for (p, tri_indices) in samples.positions.iter_mut().zip(tri_faces.iter()) {
                //    use geo::ops::Centroid;
                //    let tri = Triangle::from_indexed_slice(tri_indices, &tri_verts);
                //    *p = tri.centroid();
                //}
                let view = SamplesView::new(neighbors.as_ref(), &samples);

                // Compute the Jacobian. After calling this function, calling
                // `.deriv()` on the output will give us the second derivative.
                let jac: Vec<_> = jacobian::face_jacobian_at(
                    q,
                    view,
                    kernel,
                    tri_faces,
                    &tri_verts,
                    bg_field_params,
                )
                .collect();

                let vert_jac = consolidate_face_jacobian(&jac, &neighbors, tri_faces, num_verts);

                // Compute the potential and test the jacobian for good measure.
                let mut p = F1::cst(0.0);
                compute_potential_at(q, view, kernel, bg_field_params, &mut p);

                // Test the surface Jacobian against autodiff on the potential computation.
                //println!("jac {:9.5} vs {:9.5}", vert_jac[vtx][i].value(), p.deriv());
                if !p.deriv().is_nan() {
                    assert_relative_eq!(
                        vert_jac[vtx][i].value(),
                        p.deriv(),
                        max_relative = 1e-5,
                        epsilon = 1e-10
                    );
                }

                // Consolidate the hessian to this particular vertex and component.
                let mut hess_vtx = vec![Vector3::zero(); num_verts];
                for &(r, c, h) in hess.iter() {
                    assert!(r >= c, "Hessian is not block lower triangular.");
                    if r == vtx {
                        hess_vtx[c] += h[i];
                    } else if c == vtx {
                        hess_vtx[r] += h.transpose()[i];
                    }
                }

                for (vtx_idx, (jac, hes)) in vert_jac.iter().zip(hess_vtx).enumerate() {
                    for j in 0..3 {
                        // Check the derivative of the autodiff with our previously computed Jacobian.
                        if !relative_eq!(
                            hes[j].value(),
                            jac[j].deriv(),
                            max_relative = 1e-6,
                            epsilon = 1e-12
                        ) {
                            println!(
                                "col_vtx = {}; j = {}; {:14.10} vs {:14.10}",
                                vtx_idx,
                                j,
                                hes[j].value(),
                                jac[j].deriv()
                            );
                            success = false;
                        }
                        ad_hess_full[3 * vtx_idx + j][3 * vtx + i] += jac[j].deriv();
                        hess_full2[3 * vtx_idx + j][3 * vtx + i] += hes[j].value();
                    }
                }

                // Reset the variable back to being a constant.
                tri_verts[vtx][i] = F1::cst(tri_verts[vtx][i]);
            }
        }

        if !success {
            print_full_hessian(&hess_full, 3 * num_verts, "Block Lower Triangular Hessian");
            print_full_hessian(&hess_full2, 3 * num_verts, "Full Hessian");
            print_full_hessian(&ad_hess_full, 3 * num_verts, "Full Autodiff Hessian");
        }

        assert!(success, "Hessian does not match its AutoDiff counterpart");
    }

    #[test]
    fn two_triangle_face_hessian_test() {
        let mut no_perturb = || Vector3::zero();
        let mut perturb = make_perturb_fn();
        for i in 1..25 {
            let radius = 0.2 * (i as f64);

            let mut run_test = |field_type, weighted| {
                let bg_params = BackgroundFieldParams {
                    field_type,
                    weighted,
                };
                // Simple test with no perturbation. This derivative is easier to interpret.
                two_triangle_face_hessian(radius, bg_params, &mut no_perturb);
                // Less trivial test with perturbation.
                two_triangle_face_hessian(radius, bg_params, &mut perturb);
            };

            run_test(BackgroundFieldType::Zero, false);
            run_test(BackgroundFieldType::Zero, true);
            run_test(BackgroundFieldType::FromInput, false);
            run_test(BackgroundFieldType::FromInput, true);
            run_test(BackgroundFieldType::DistanceBased, false);
            run_test(BackgroundFieldType::DistanceBased, true);
        }
    }

    #[test]
    fn one_triangle_face_hessian_test() {
        let mut no_perturb = || Vector3::zero();
        let mut perturb = make_perturb_fn();
        for i in 1..25 {
            let radius = 0.2 * (i as f64);

            let mut run_test = |field_type, weighted| {
                let bg_params = BackgroundFieldParams {
                    field_type,
                    weighted,
                };
                // Unperturbed triangle. This test is easier to debug.
                one_triangle_face_hessian(radius, bg_params, &mut no_perturb);
                // Test with a perturbed triangle. This test verifies that the derivative is robust
                // against perturbation.
                one_triangle_face_hessian(radius, bg_params, &mut perturb);
            };

            run_test(BackgroundFieldType::Zero, false);
            run_test(BackgroundFieldType::Zero, true);
            run_test(BackgroundFieldType::FromInput, false);
            run_test(BackgroundFieldType::FromInput, true);
            run_test(BackgroundFieldType::DistanceBased, false);
            run_test(BackgroundFieldType::DistanceBased, true);
        }
    }

    #[test]
    fn one_tet_face_hessian_test() {
        let mut no_perturb = || Vector3::zero();
        let mut perturb = make_perturb_fn();
        for i in 1..10 {
            let radius = 0.5 * (i as f64);
            let mut run_test = |field_type, weighted| {
                let bg_params = BackgroundFieldParams {
                    field_type,
                    weighted,
                };
                // Unperturbed tet
                one_tet_face_hessian(radius, bg_params, &mut no_perturb);
                // Perturbed tet
                one_tet_face_hessian(radius, bg_params, &mut perturb);
            };

            run_test(BackgroundFieldType::Zero, false);
            run_test(BackgroundFieldType::Zero, true);
            run_test(BackgroundFieldType::FromInput, false);
            run_test(BackgroundFieldType::FromInput, true);
            run_test(BackgroundFieldType::DistanceBased, false);
            run_test(BackgroundFieldType::DistanceBased, true);
        }
    }

    fn one_tet_contact_hessian<P: FnMut() -> Vector3<f64>>(
        radius: f64,
        bg_field_params: BackgroundFieldParams,
        perturb: &mut P,
    ) {
        let tet = TriMesh::from(PlatonicSolidBuilder::new().build_tetrahedron());
        let qs: Vec<_> = (0..4)
            .map(|i| Vector3::new([0.1, -0.5 + 0.25 * i as f64, 0.0]))
            .collect();

        for q in qs.into_iter() {
            contact_hessian_at_tester(q, &tet, radius, 0.0, bg_field_params);
            contact_hessian_at_tester(q + perturb(), &tet, radius, 0.0, bg_field_params);
            contact_hessian_at_tester(q + perturb(), &tet, radius, 1.0, bg_field_params);
        }
    }

    fn one_triangle_contact_hessian<P: FnMut() -> Vector3<f64>>(
        radius: f64,
        bg_field_params: BackgroundFieldParams,
        perturb: &mut P,
    ) {
        let tri_verts: Vec<_> = make_test_triangle(0.0, perturb)
            .into_iter()
            .map(|x| x.into_data())
            .collect();
        let tri = TriMesh::new(tri_verts, vec![[0, 1, 2]]);
        let qs = vec![
            Vector3::new([0.0, 0.2, 0.0]),
            Vector3::new([0.0, 0.0001, 0.0]),
            Vector3::new([0.0, -0.4, 0.0]),
        ];

        for q in qs.into_iter() {
            contact_hessian_at_tester(q, &tri, radius, 0.0, bg_field_params);
            contact_hessian_at_tester(q + perturb(), &tri, radius, 0.0, bg_field_params);
            contact_hessian_at_tester(q + perturb(), &tri, radius, 1.0, bg_field_params);
        }
    }

    /// Test the `contact_hessian_at` function. This verifies that it indeed produces the derivative
    /// of the `contact_jacobian_at` function.
    fn contact_hessian_at_tester(
        q: Vector3<f64>,
        mesh: &TriMesh<f64>,
        radius: f64,
        max_step: f64,
        bg_field_params: BackgroundFieldParams,
    ) {
        let q = q.mapd(|x| F1::cst(x)); // convert to autodiff
        let mut tri_verts: Vec<_> = mesh
            .vertex_position_iter()
            .map(|&x| Vector3::new(x).mapd(|x| F1::cst(x)).into())
            .collect();
        let tri_faces = mesh.indices.as_slice();
        let num_verts = tri_verts.len();

        let samples = new_test_samples(SampleType::Face, &tri_faces, &tri_verts);

        let neighbors: Vec<_> = samples
            .iter()
            .filter(|s| (q - s.pos).norm() < F1::cst(radius + max_step))
            .map(|sample| sample.index)
            .collect();

        if neighbors.is_empty() {
            return;
        }

        // Radius is such that samples are captured by the query point.
        let kernel = kernel::LocalApproximate::new(radius, 0.00001);

        let view = SamplesView::new(neighbors.as_ref(), &samples);

        let mut success = true;

        for k in 0..1 {
            let mut multiplier: Vector3<F1> = Vector3::zero();
            multiplier[k] = F1::cst(1.0);

            // Compute the complete Hessian.
            let hess: Vec<(usize, usize, [[F1; 3]; 3])> = contact_hessian_at(
                q,
                view,
                kernel,
                &tri_verts,
                tri_faces,
                bg_field_params,
                multiplier,
            )
            .collect();
            // dbg!(&tri_faces);

            let mut hess_full = vec![vec![0.0; 3 * num_verts]; 3 * num_verts];

            for &(r, c, h) in hess.iter() {
                for i in 0..3 {
                    for j in 0..3 {
                        hess_full[3 * c + i][3 * r + j] += h[i][j].value();
                    }
                }
            }

            let mut ad_hess_full = vec![vec![0.0; 3 * num_verts]; 3 * num_verts];

            // Test the accuracy of each component of the hessian against an autodiff version of the
            // second derivative.
            for vtx in 0..num_verts {
                for i in 0..3 {
                    // eprintln!("vtx: {vtx}; i: {i}");
                    // Set a variable to take the derivative with respect to, using autodiff.
                    tri_verts[vtx][i] = F1::var(tri_verts[vtx][i]);

                    // We need to update samples to make sure the normals and centroids are recomputed
                    // using the correct wrt autodiff variable.
                    let samples = new_test_samples(SampleType::Face, &tri_faces, &tri_verts);
                    let view = SamplesView::new(neighbors.as_ref(), &samples);

                    // Debug code
                    // End of debug code

                    // Compute the Jacobian. After calling this function, calling
                    // `.deriv()` on the output will give us the second derivative.
                    let sample_jac: Vec<_> =
                        jacobian::contact_jacobian_at(q, view, kernel, bg_field_params)
                            .0
                            .collect();

                    let vert_jac =
                        consolidate_contact_jacobian(view, &sample_jac, &tri_faces, num_verts);

                    // Consolidate the hessian to this particular vertex and component.
                    let mut hess_vtx_i = vec![Vector3::zero(); num_verts];
                    for &(r, c, h) in hess.iter() {
                        if c == vtx {
                            hess_vtx_i[r] += Matrix3::from(h)[i];
                        }
                    }

                    for (vtx_idx, (jac, hes)) in vert_jac.iter().zip(hess_vtx_i).enumerate() {
                        for j in 0..3 {
                            // Check the derivative of the autodiff with our previously computed Jacobian.
                            if !relative_eq!(
                                hes[j].value(),
                                jac[k][j].deriv(),
                                max_relative = 1e-6,
                                epsilon = 1e-12
                            ) {
                                println!(
                                    "col_vtx = {}; j = {}; {:14.10} vs {:14.10}",
                                    vtx_idx,
                                    j,
                                    hes[j].value(),
                                    jac[k][j].deriv()
                                );
                                success = false;
                            }
                            ad_hess_full[3 * vtx + i][3 * vtx_idx + j] += jac[k][j].deriv();
                        }
                    }

                    // Reset the variable back to being a constant.
                    tri_verts[vtx][i] = F1::cst(tri_verts[vtx][i]);
                }
            }

            if !success {
                print_full_hessian(&hess_full, 3 * num_verts, "Block Lower Triangular Hessian");
                print_full_hessian(&ad_hess_full, 3 * num_verts, "Full Autodiff Hessian");
            }
        }

        assert!(success, "Hessian does not match its AutoDiff counterpart");
    }

    #[test]
    fn one_tet_contact_hessian_test() {
        let mut no_perturb = || Vector3::zero();
        let mut perturb = make_perturb_fn();
        for i in 1..25 {
            let radius = 0.3 * (i as f64);
            let mut run_test = |field_type, weighted| {
                let bg_params = BackgroundFieldParams {
                    field_type,
                    weighted,
                };
                // Unperturbed tet
                one_tet_contact_hessian(radius, bg_params, &mut no_perturb);
                // Perturbed tet
                one_tet_contact_hessian(radius, bg_params, &mut perturb);
            };

            run_test(BackgroundFieldType::Zero, false);
            run_test(BackgroundFieldType::Zero, true);
        }
    }

    #[test]
    fn one_triangle_contact_hessian_test() {
        let mut no_perturb = || Vector3::zero();
        let mut perturb = make_perturb_fn();
        for i in 1..25 {
            let radius = 0.2 * (i as f64);

            let mut run_test = |field_type, weighted| {
                let bg_params = BackgroundFieldParams {
                    field_type,
                    weighted,
                };
                // Unperturbed triangle. This test is easier to debug.
                one_triangle_contact_hessian(radius, bg_params, &mut no_perturb);
                // Test with a perturbed triangle. This test verifies that the derivative is robust
                // against perturbation.
                one_triangle_contact_hessian(radius, bg_params, &mut perturb);
            };

            run_test(BackgroundFieldType::Zero, false);
            run_test(BackgroundFieldType::Zero, true);
        }
    }

    /// Sample Hessian tester is parametrized by the query point location `q`, triangle mesh `mesh`,
    /// kernel radius `radius`, sparsity extension (value added to the radius) `max_step`, and the
    /// background field params `bg_field_params`.
    fn sample_hessian_tester(
        q: Vector3<f64>,
        mesh: &TriMesh<f64>,
        radius: f64,
        max_step: f64,
        bg_field_params: BackgroundFieldParams,
    ) {
        let q = q.mapd(|x| F1::cst(x)); // convert to autodiff
        let tri_verts: Vec<_> = mesh
            .vertex_position_iter()
            .map(|&x| Vector3::new(x).mapd(|x| F1::cst(x)))
            .collect();
        let tri_faces = mesh.indices.as_slice();

        let mut samples = new_test_samples(SampleType::Face, &tri_faces, &tri_verts);

        let neighbors: Vec<_> = samples
            .iter()
            .filter(|s| (q - s.pos).norm() < F1::cst(radius + max_step))
            .map(|sample| sample.index)
            .collect();

        if neighbors.is_empty() {
            return;
        }

        let num_samples = samples.len();

        // Radius is such that samples are captured by the query point.
        let kernel = kernel::LocalApproximate::new(radius, 0.00001);

        // Compute the complete hessian.
        let hess: Vec<(usize, usize, Matrix3<F1>)> = {
            let view = SamplesView::new(neighbors.as_ref(), &samples);

            let bg = BackgroundField::local(q, view, kernel, bg_field_params, None).unwrap();

            sample_hessian_at(q, view, kernel, bg.clone()).collect()
        };

        let mut success = true;

        let mut hess_full2 = vec![vec![0.0; 3 * num_samples]; 3 * num_samples];
        let mut ad_hess_full = vec![vec![0.0; 3 * num_samples]; 3 * num_samples];

        // Test the accuracy of each component of the hessian against an autodiff version of the
        // second derivative.
        for sample_idx in 0..num_samples {
            for i in 0..3 {
                // Set a variable to take the derivative with respect to, using autodiff.
                samples.positions[sample_idx][i] = F1::var(samples.positions[sample_idx][i]);
                //println!("row = {}; i = {}", sample_idx, i);

                let view = SamplesView::new(neighbors.as_ref(), &samples);

                let bg = BackgroundField::local(q, view, kernel, bg_field_params, None).unwrap();

                // Compute the Jacobian. After calling this function, calling
                // `.deriv()` on the output will give us the second derivative.
                let mut jac = vec![Vector3::zero(); num_samples];
                for (jac_val, &idx) in
                    jacobian::sample_jacobian_at(q, view, kernel, bg.clone()).zip(neighbors.iter())
                {
                    jac[idx] = jac_val;
                }

                // Compute the potential and test the jacobian for good measure.
                let mut p = F1::cst(0.0);
                compute_potential_at(q, view, kernel, bg_field_params, &mut p);

                // Test the surface Jacobian against autodiff on the potential computation.
                if !p.deriv().is_nan() {
                    assert_relative_eq!(
                        jac[sample_idx][i].value(),
                        p.deriv(),
                        max_relative = 1e-5,
                        epsilon = 1e-10
                    );
                }
                //println!(
                //    "jac {:9.5} vs {:9.5}",
                //    jac[sample_idx][i].value(),
                //    p.deriv()
                //);

                // Consolidate the hessian to this particular sample and component.
                let mut hess_sample = vec![Vector3::<F1>::zero(); num_samples];
                for &(r, c, h) in hess.iter() {
                    // TODO: make hessian only lower triangular and check it here
                    //assert!(r >= c, "Hessian is not block lower triangular.");
                    if r == c {
                        for x in 0..3 {
                            for y in 0..3 {
                                assert_relative_eq!(
                                    h[y][x].value(),
                                    h[x][y].value(),
                                    max_relative = 1e-6,
                                    epsilon = 1e-10
                                );
                            }
                        }
                    }

                    if r == sample_idx {
                        hess_sample[c] += h[i];
                    }

                    // Add upper triangular part to hess_sample.
                    if r > c && c == sample_idx {
                        hess_sample[r] += h.transpose()[i];
                    }
                }

                for (sample_j, (jac, hes)) in jac.iter().zip(hess_sample).enumerate() {
                    for j in 0..3 {
                        // Check the derivative of the autodiff with our previously computed Jacobian.
                        if !relative_eq!(
                            hes[j].value(),
                            jac[j].deriv(),
                            max_relative = 1e-6,
                            epsilon = 1e-7
                        ) {
                            println!(
                                "col = {}; j = {}; {:11.7} vs {:11.7}",
                                sample_j,
                                j,
                                hes[j].value(),
                                jac[j].deriv()
                            );
                            success = false;
                        }
                        ad_hess_full[3 * sample_j + j][3 * sample_idx + i] += jac[j].deriv();
                        hess_full2[3 * sample_j + j][3 * sample_idx + i] += hes[j].value();
                    }
                }

                // Reset the variable back to being a constant.
                samples.positions[sample_idx][i] = F1::cst(samples.positions[sample_idx][i]);
            }
        }

        if !success {
            print_full_hessian(&hess_full2, 3 * num_samples, "Full Hessian");
            print_full_hessian(&ad_hess_full, 3 * num_samples, "Full Autodiff Hessian");
        }
        assert!(success, "Hessian does not match its AutoDiff counterpart");
    }

    /// Test the part of the Hessian that excludes any normal derivatives. (i.e. assume the normals
    /// are constant )
    #[test]
    fn sample_hessian_test() {
        let mut no_perturb = || Vector3::zero();
        let mut perturb = make_perturb_fn();

        let qs = vec![
            Vector3::new([0.0, 0.2, 0.0]),
            Vector3::new([0.0, 0.0001, 0.0]),
            Vector3::new([0.0, -0.4, 0.0]),
        ];

        let run_tester = |tri, field_type, weighted| {
            for i in 1..50 {
                let radius = 0.1 * (i as f64);
                let bg = BackgroundFieldParams {
                    field_type,
                    weighted,
                };
                for &q in qs.iter() {
                    sample_hessian_tester(q, &tri, radius, 0.0, bg);
                }
            }
        };

        let run_tester_on_mesh = |tri: TriMesh<f64>| {
            run_tester(tri.clone(), BackgroundFieldType::Zero, false);
            run_tester(tri.clone(), BackgroundFieldType::Zero, true);
            run_tester(tri.clone(), BackgroundFieldType::FromInput, false);
            run_tester(tri.clone(), BackgroundFieldType::FromInput, true);
            run_tester(tri.clone(), BackgroundFieldType::DistanceBased, false);
            run_tester(tri.clone(), BackgroundFieldType::DistanceBased, true);
        };

        let build_mesh_and_run_test = |(tri_verts, tri_indices)| {
            let tri = TriMesh::new(tri_verts, tri_indices);
            run_tester_on_mesh(tri);
        };

        // One flat triangle test
        build_mesh_and_run_test((make_test_triangle(0.0, &mut no_perturb), vec![[0, 1, 2]]));

        // One perturbed triangle test
        build_mesh_and_run_test((make_test_triangle(0.0, &mut perturb), vec![[0, 1, 2]]));

        // Two flat triangles test
        build_mesh_and_run_test(make_two_test_triangles(0.0, &mut no_perturb));

        // Two perturbed triangles test
        build_mesh_and_run_test(make_two_test_triangles(0.0, &mut perturb));

        // Three perturbed triangles test
        build_mesh_and_run_test(make_three_test_triangles(0.0, &mut perturb));

        // Regular tetrahedron test
        run_tester_on_mesh(TriMesh::from(
            PlatonicSolidBuilder::new().build_tetrahedron(),
        ));
    }

    /// Test the second order derivatives of our normal computation method for face normals.
    #[test]
    fn face_normal_hessian_test() {
        // Simple test with two triangles
        let (tri_verts, tri_faces) = make_two_test_triangles(0.0, &mut || Vector3::zero());
        face_normal_hessian_tester(&tri_verts, &tri_faces);

        // More complex test with a whole tet
        let (tet_verts, tet_faces) = make_tet();
        face_normal_hessian_tester(&tet_verts, &tet_faces);
    }

    fn face_normal_hessian_tester(verts: &[[f64; 3]], faces: &[[usize; 3]]) {
        let samples = Samples::new_triangle_samples(faces, verts, vec![0.0; faces.len()]);

        let neighbors: Vec<usize> = (0..faces.len()).collect(); // look at all the faces

        // Set a random product vector.
        let multipliers = utils::random_vectors(faces.len());
        let ad_multipliers: Vec<_> = multipliers
            .iter()
            .map(|&v| Vector3::new(v).mapd(|x| F1::cst(x)))
            .collect();

        let multiplier = move |Sample { index, .. }| Vector3::new(multipliers[index]);

        let ad_multiplier = move |Sample { index, .. }| ad_multipliers[index];

        // Compute the normal hessian product.
        let view = SamplesView::new(neighbors.as_ref(), &samples);
        let hess_iter =
            compute_face_unit_normals_hessian_products(view, verts, faces, multiplier.clone());

        let mut num_hess_entries = 0;
        let mut hess = [[0.0; 12]; 12]; // Dense matrix
        for (r, c, m) in hess_iter {
            // map to tet vertices instead of surface vertices
            for j in 0..3 {
                for i in 0..3 {
                    hess[3 * c + j][3 * r + i] += m[i][j];
                    if i >= j {
                        // Only record lower triangular non-zeros
                        num_hess_entries += 1;
                    }
                }
            }
        }

        let mut ad_hess = [[0.0; 12]; 12]; // ad Dense matrix

        assert_eq!(
            ImplicitSurface::<f64>::num_face_unit_normals_hessian_entries(neighbors.len()),
            num_hess_entries
        );

        // Convert tet vertices into varibales because we are taking the derivative with respect to
        // vertices.
        let mut ad_verts: Vec<[F1; 3]> = verts
            .iter()
            .cloned()
            .map(|v| Vector3::new(v).mapd(|x| F1::cst(x)).into())
            .collect();

        let mut success = true;
        for r in 0..4 {
            for i in 0..3 {
                ad_verts[r][i] = F1::var(ad_verts[r][i]);

                let ad_samples =
                    Samples::new_triangle_samples(faces, &ad_verts, vec![F1::cst(0.0); 4]);
                let ad_view = SamplesView::new(neighbors.as_ref(), &ad_samples);

                // Convert the samples to use autodiff constants.
                let grad = super::jacobian::compute_face_unit_normal_derivative(
                    &ad_verts,
                    faces,
                    ad_view,
                    ad_multiplier.clone(),
                );

                for c in 0..4 {
                    for j in 0..3 {
                        // Only check lower triangular part.
                        if 3 * c + j <= 3 * r + i {
                            if !relative_eq!(
                                hess[3 * c + j][3 * r + i],
                                grad[c][j].deriv(),
                                max_relative = 1e-5,
                                epsilon = 1e-10
                            ) {
                                println!(
                                    "col = {}; j = {}; {:11.7} vs {:11.7}",
                                    c,
                                    j,
                                    hess[3 * c + j][3 * r + i],
                                    grad[c][j].deriv(),
                                );
                                success = false;
                            }
                        }
                        ad_hess[3 * c + j][3 * r + i] += grad[c][j].deriv();
                    }
                }

                ad_verts[r][i] = F1::cst(ad_verts[r][i]);
            }
        }
        if !success {
            fn print_full_hessian_arr(hess: [[f64; 12]; 12], name: &str) {
                println!("{} = ", name);
                for r in 0..12 {
                    for c in 0..12 {
                        if relative_eq!(hess[c][r], 0.0, max_relative = 1e-6, epsilon = 1e-12) {
                            print!("     .    ",);
                        } else {
                            print!("{:9.5} ", hess[c][r]);
                        }
                    }
                    println!();
                }
                println!();
            }

            print_full_hessian_arr(hess, "Full Hessian");
            print_full_hessian_arr(ad_hess, "Full AutoDiff Hessian");
        }
        assert!(success, "Hessian does not match its AutoDiff counterpart");
    }

    /// Test the second order derivatives of query computation.
    #[test]
    fn query_hessian_at_test() {
        // Simple test with two triangles
        let (tri_verts, tri_faces) = make_two_test_triangles(0.0, &mut || Vector3::zero());
        query_hessian_at_tester(&tri_verts, &tri_faces);

        // More complex test with a whole tet
        let (tet_verts, tet_faces) = make_tet();
        query_hessian_at_tester(&tet_verts, &tet_faces);
    }

    fn query_hessian_at_tester(verts: &[[f64; 3]], faces: &[[usize; 3]]) {
        let samples = Samples::new_triangle_samples(faces, verts, vec![0.0; faces.len()]);

        let neighbors: Vec<usize> = (0..faces.len()).collect(); // look at all the faces

        let bg_params = BackgroundFieldParams {
            field_type: BackgroundFieldType::Zero,
            weighted: false,
        };
        let kernel = LocalCubic::new(1.5);

        let q = Vector3::new([0.0, 0.01, 0.0]);
        // Compute the normal hessian product.
        let view = SamplesView::new(neighbors.as_ref(), &samples);
        let hess = query_hessian_at(q, view, kernel, bg_params);

        let mut ad_hess = Matrix3::<F1>::zero();

        // Convert tet vertices into varibales because we are taking the derivative with respect to
        // vertices.
        let ad_verts: Vec<[F1; 3]> = verts
            .iter()
            .cloned()
            .map(|v| Vector3::new(v).mapd(|x| F1::cst(x)).into())
            .collect();

        let mut q_ad = q.cast::<F1>();

        let mut success = true;
        for i in 0..3 {
            q_ad[i] = F1::var(q_ad[i]);

            let ad_samples = Samples::new_triangle_samples(faces, &ad_verts, vec![F1::cst(0.0); 4]);
            let ad_view = SamplesView::new(neighbors.as_ref(), &ad_samples);

            // Convert the samples to use autodiff constants.
            let grad = super::jacobian::query_jacobian_at(q_ad, ad_view, None, kernel, bg_params);

            for j in 0..3 {
                // Only check lower triangular part.
                if !relative_eq!(
                    hess[j][i],
                    grad[j].deriv(),
                    max_relative = 1e-5,
                    epsilon = 1e-10
                ) {
                    println!(
                        "({}, {}); {:11.7} vs {:11.7}",
                        j,
                        i,
                        hess[j][i],
                        grad[j].deriv(),
                    );
                    success = false;
                }
                ad_hess[j][i] += grad[j].deriv();
            }

            q_ad[i] = F1::cst(q_ad[i]);
        }
        if !success {
            dbg!(hess);
            dbg!(ad_hess);
        }
        assert!(success, "Hessian does not match its AutoDiff counterpart");
    }
}
