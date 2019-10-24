use super::*;
use crate::Error;
use rayon::iter::Either;

impl<T: Real + Send + Sync> ImplicitSurface<T> {
    /*
     * Query jacobian
     */

    pub fn num_query_jacobian_entries(&self) -> Result<usize, Error> {
        match self {
            ImplicitSurface::MLS(mls) => mls.num_query_jacobian_entries(),
            _ => Err(Error::UnsupportedKernel),
        }
    }

    pub fn query_jacobian_indices_iter(
        &self,
    ) -> Result<impl Iterator<Item = (usize, usize)> + Clone, Error> {
        match self {
            ImplicitSurface::MLS(mls) => mls.query_jacobian_indices_iter(),
            _ => Err(Error::UnsupportedKernel),
        }
    }

    /// Compute the Jacobian of this implicit surface function with respect to query points.
    /// This is a more convenient version of the `query_jacobian_values` function where the values
    /// are already expected to be packed into triplets.
    pub fn query_jacobian(
        &self,
        query_points: &[[T; 3]],
        values: &mut [[T; 3]],
    ) -> Result<(), Error> {
        match self {
            ImplicitSurface::MLS(mls) => mls.query_jacobian(query_points, values),
            _ => Err(Error::UnsupportedKernel),
        }
    }

    /// Compute the Jacobian of this implicit surface function with respect to query points.
    pub fn query_jacobian_values(
        &self,
        query_points: &[[T; 3]],
        values: &mut [T],
    ) -> Result<(), Error> {
        self.query_jacobian(query_points, reinterpret::reinterpret_mut_slice(values))
    }

    /// Compute the Jacobian of this implicit surface function with respect to query points.
    /// This version of the query Jacobian returns all diagonal values of the Jacobian, including
    /// values for points with empty neighbourhoods. This is especially valuable for projection
    /// where the background potential can help. The other Jacobian functions ignore these values
    /// altogether. This also means we don't need to worry about the size of `values` since it will
    /// always be the same as the size of `query_points`.
    pub fn query_jacobian_full(
        &self,
        query_points: &[[T; 3]],
        values: &mut [[T; 3]],
    ) -> Result<(), Error> {
        match self {
            ImplicitSurface::MLS(mls) => mls.query_jacobian_full(query_points, values),
            _ => Err(Error::UnsupportedKernel),
        }
    }

    /*
     * Surface Jacobian
     */
    /// Compute the number of indices (non-zeros) needed for the implicit surface potential
    /// Jacobian with respect to surface points. If neighbourhoods haven't been precomputed or an
    /// error occured, this function will return `None`.
    pub fn num_surface_jacobian_entries(&self) -> Option<usize> {
        match self {
            ImplicitSurface::MLS(mls) => {
                let num_neigh_points = match mls.base().sample_type {
                    SampleType::Vertex => mls
                        .extended_neighbourhood_seq()
                        .ok()?
                        .map(|x| x.len())
                        .sum::<usize>(),
                    SampleType::Face => mls
                        .trivial_neighbourhood_seq()
                        .ok()?
                        .map(|x| x.len())
                        .sum::<usize>(),
                };
                let num_pts_per_sample = match mls.base().sample_type {
                    SampleType::Vertex => 1,
                    SampleType::Face => 3,
                };
                Some(num_neigh_points * 3 * num_pts_per_sample)
            }
            _ => None,
        }
    }

    /// Compute the indices for the implicit surface potential Jacobian with respect to surface
    /// points.
    pub fn surface_jacobian_indices_par_iter<'a>(
        &'a self,
    ) -> Result<impl ParallelIterator<Item = (usize, usize)> + 'a, Error> {
        match self {
            ImplicitSurface::MLS(mls) => mls.surface_jacobian_indices_par_iter(),
            _ => Err(Error::UnsupportedKernel),
        }
    }

    /// Compute the indices for the implicit surface potential Jacobian with respect to surface
    /// points.
    pub fn surface_jacobian_indices_iter<'a>(
        &'a self,
    ) -> Result<impl Iterator<Item = (usize, usize)> + 'a, Error> {
        match self {
            ImplicitSurface::MLS(mls) => mls.surface_jacobian_indices_iter(),
            _ => Err(Error::UnsupportedKernel),
        }
    }

    /// Compute the indices for the implicit surface potential Jacobian with respect to surface
    /// points.
    pub fn surface_jacobian_indices(
        &self,
        rows: &mut [usize],
        cols: &mut [usize],
    ) -> Result<(), Error> {
        match self {
            ImplicitSurface::MLS(mls) => mls.surface_jacobian_indices(rows, cols),
            _ => Err(Error::UnsupportedKernel),
        }
    }

    /// Compute the Jacobian of this implicit surface function with respect to surface
    /// points.
    pub fn surface_jacobian_values(
        &self,
        query_points: &[[T; 3]],
        values: &mut [T],
    ) -> Result<(), Error> {
        match self {
            ImplicitSurface::MLS(mls) => mls.surface_jacobian_values(query_points, values),
            _ => Err(Error::UnsupportedKernel),
        }
    }

    /*
     * Contact Jacobian
     */

    /// Compute the contact Jacobian of this implicit surface function with respect to surface
    /// points.
    pub fn contact_jacobian_product_values(
        &self,
        query_points: &[[T; 3]],
        multiplier: &[[T; 3]],
        values: &mut [[T; 3]],
    ) -> Result<(), Error> {
        match self {
            ImplicitSurface::MLS(mls) => {
                mls.contact_jacobian_product_values(query_points, multiplier, values)
            }
            _ => Err(Error::UnsupportedKernel),
        }
    }

    pub fn num_contact_jacobian_entries(&self) -> Result<usize, Error> {
        match self {
            ImplicitSurface::MLS(mls) => mls.num_contact_jacobian_entries(),
            _ => Err(Error::UnsupportedKernel),
        }
    }

    /// Compute the contact Jacobian of this implicit surface function with respect to surface
    /// points.
    pub fn contact_jacobian_values(
        &self,
        query_points: &[[T; 3]],
        values: &mut [T],
    ) -> Result<(), Error> {
        match self {
            ImplicitSurface::MLS(mls) => mls.contact_jacobian_values(query_points, values),
            _ => Err(Error::UnsupportedKernel),
        }
    }

    pub fn contact_jacobian_indices_iter(
        &self,
    ) -> Result<impl Iterator<Item = (usize, usize)> + Clone, Error> {
        match self {
            ImplicitSurface::MLS(mls) => mls.contact_jacobian_indices_iter(),
            _ => Err(Error::UnsupportedKernel),
        }
    }

    pub fn num_contact_jacobian_matrices(&self) -> Result<usize, Error> {
        match self {
            ImplicitSurface::MLS(mls) => mls.num_contact_jacobian_matrices(),
            _ => Err(Error::UnsupportedKernel),
        }
    }

    /// Compute the contact Jacobian of this implicit surface function with respect to surface
    /// points. The returned 2D arrays are column major 3x3 matrices.
    pub fn contact_jacobian_matrices(
        &self,
        query_points: &[[T; 3]],
        matrices: &mut [[[T; 3]; 3]],
    ) -> Result<(), Error> {
        match self {
            ImplicitSurface::MLS(mls) => mls.contact_jacobian_matrices(query_points, matrices),
            _ => Err(Error::UnsupportedKernel),
        }
    }

    pub fn contact_jacobian_matrix_indices_iter(
        &self,
    ) -> Result<impl Iterator<Item = (usize, usize)> + Clone, Error> {
        match self {
            ImplicitSurface::MLS(mls) => mls.contact_jacobian_matrix_indices_iter(),
            _ => Err(Error::UnsupportedKernel),
        }
    }
}

impl<T: Real + Send + Sync> MLS<T> {
    /*
     * Query Jacobian
     */
    pub fn num_query_jacobian_entries(&self) -> Result<usize, Error> {
        self.num_neighbourhoods().map(|n| 3 * n)
    }

    /// Compute the Jacobian of this implicit surface function with respect to query points.
    /// This is a more convenient version of the `query_jacobian_values` function where the values
    /// are already expected to be packed into triplets.
    pub fn query_jacobian(
        &self,
        query_points: &[[T; 3]],
        values: &mut [[T; 3]],
    ) -> Result<(), Error> {
        apply_kernel_fn!(self, |kernel| self.query_jacobian_values_impl(
            query_points,
            kernel,
            values,
            false
        ))
    }

    /// Compute the Jacobian of this implicit surface function with respect to query points.
    pub fn query_jacobian_values(
        &self,
        query_points: &[[T; 3]],
        values: &mut [T],
    ) -> Result<(), Error> {
        self.query_jacobian(query_points, reinterpret::reinterpret_mut_slice(values))
    }

    /// Compute the Jacobian of this implicit surface function with respect to query points.
    /// This version of the query Jacobian returns all diagonal values of the Jacobian, including
    /// values for points with empty neighbourhoods. This is especially valuable for projection
    /// where the background potential can help. The other Jacobian functions ignore these values
    /// altogether. This also means we don't need to worry about the size of `values` since it will
    /// always be the same as the size of `query_points`.
    pub fn query_jacobian_full(
        &self,
        query_points: &[[T; 3]],
        values: &mut [[T; 3]],
    ) -> Result<(), Error> {
        apply_kernel_fn!(self, |kernel| self.query_jacobian_values_impl(
            query_points,
            kernel,
            values,
            true
        ))
    }

    pub fn query_jacobian_indices_iter(
        &self,
    ) -> Result<impl Iterator<Item = (usize, usize)> + Clone, Error> {
        let indices: Result<Vec<_>, Error> = self.trivial_neighbourhood_seq().map(move |s| {
            s.enumerate()
                .filter(move |(_, nbrs)| nbrs.len() != 0)
                .flat_map(move |(i, _)| (0..3).map(move |j| (i, 3 * i + j)))
                .collect()
        });
        indices.map(std::iter::IntoIterator::into_iter)
    }

    pub(crate) fn query_jacobian_values_impl<K>(
        &self,
        query_points: &[[T; 3]],
        kernel: K,
        value_vecs: &mut [[T; 3]],
        full: bool,
    ) -> Result<(), Error>
    where
        K: SphericalKernel<T> + std::fmt::Debug + Copy + Sync + Send,
    {
        let neigh_points = self.trivial_neighbourhood_seq()?;
        let closest_points = self.closest_samples_seq()?;

        let ImplicitSurfaceBase {
            ref samples,
            bg_field_params,
            ..
        } = *self.base();

        // For each row (query point)
        zip!(query_points.iter(), neigh_points, closest_points)
            .filter(|(_, nbrs, _)| full || nbrs.len() != 0)
            .zip(value_vecs.iter_mut())
            .for_each(move |((q, nbr_points, closest), vec)| {
                let view = SamplesView::new(nbr_points, samples);
                *vec = Self::query_jacobian_at(
                    Vector3(*q),
                    view,
                    Some(closest),
                    kernel,
                    bg_field_params,
                )
                .into();
            });
        Ok(())
    }

    /// Compute the Jacobian of the potential field with respect to the given query point.
    pub(crate) fn query_jacobian_at<'a, K: 'a>(
        q: Vector3<T>,
        view: SamplesView<'a, 'a, T>,
        closest: Option<usize>,
        kernel: K,
        bg_field_params: BackgroundFieldParams,
    ) -> Vector3<T>
    where
        K: SphericalKernel<T> + std::fmt::Debug + Copy + Sync + Send,
    {
        let bg = BackgroundField::new(q, view, closest, kernel, bg_field_params, None).unwrap();

        // Background potential Jacobian.
        let bg_jac = bg.compute_query_jacobian();

        let closest_d = bg.closest_sample_dist();
        let weight_sum_inv = bg.weight_sum_inv();

        // For each surface vertex contribution
        let dw_neigh = Self::normalized_neighbour_weight_gradient(q, view, kernel, bg);

        let main_jac: Vector3<T> = view
            .into_iter()
            .map(
                move |Sample {
                          pos, nml, value, ..
                      }| {
                    let unit_nml = nml * (T::one() / nml.norm());
                    Self::sample_query_jacobian_at(
                        q,
                        pos,
                        value,
                        kernel,
                        unit_nml,
                        dw_neigh,
                        weight_sum_inv,
                        closest_d,
                    )
                },
            )
            .sum();

        main_jac + bg_jac
    }

    /// Compute the normalized sum of all sample weight gradients.
    pub(crate) fn normalized_neighbour_weight_gradient<'a, K, V>(
        q: Vector3<T>,
        samples: SamplesView<'a, 'a, T>,
        kernel: K,
        bg: BackgroundField<'a, T, V, K>,
    ) -> Vector3<T>
    where
        K: SphericalKernel<T> + std::fmt::Debug + Copy + Sync + Send + 'a,
        V: Copy + Clone + std::fmt::Debug + PartialEq + num_traits::Zero,
    {
        let closest_d = bg.closest_sample_dist();

        // Background potential adds to the total weight sum, so we should get the updated weight
        // sum from there.
        let weight_sum_inv = bg.weight_sum_inv();

        let mut dw_neigh: Vector3<T> = samples
            .iter()
            .map(|s| kernel.with_closest_dist(closest_d).grad(q, s.pos))
            .sum();

        // Contribution from the background potential
        dw_neigh += bg.background_weight_gradient(None);

        dw_neigh * weight_sum_inv // normalize the neighbourhood derivative
    }

    /// Compute the Jacobian for the implicit surface potential for the given sample with the
    /// specified kernel. This Jacobian is with respect to the query
    /// point `q`, but this is not the complete Jacobian, this implementation returns a mapping
    /// from sample space to physical space in a form of a `1`-by-`3` Jacobian vector where.
    /// The returned iterator returns only non-zero elements.  When
    /// using the unit normal as the multiplier and summing over all samples, this function
    /// produces the true Jacobian of the potential with respect to the query point.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn sample_query_jacobian_at<'a, K: 'a>(
        q: Vector3<T>,
        sample_pos: Vector3<T>,
        sample_value: T,
        kernel: K,
        multiplier: Vector3<T>,
        dw_neigh_normalized: Vector3<T>,
        weight_sum_inv: T,
        closest_d: T,
    ) -> Vector3<T>
    where
        K: SphericalKernel<T> + std::fmt::Debug + Copy,
    {
        let w = kernel.with_closest_dist(closest_d).eval(q, sample_pos);
        let dw = kernel.with_closest_dist(closest_d).grad(q, sample_pos);
        let psi = T::from(sample_value).unwrap() + multiplier.dot(q - sample_pos);
        ((dw - dw_neigh_normalized * w) * psi + (multiplier * w)) * weight_sum_inv
    }

    /*
     * Surface Jacobian
     */

    /// Compute the number of indices (non-zeros) needed for the implicit surface potential
    /// Jacobian with respect to surface points. If neighbourhoods haven't been precomputed or an
    /// error occured, this function will return `None`.
    pub fn num_surface_jacobian_entries(&self) -> Option<usize> {
        let num_neigh_points = match self.base().sample_type {
            SampleType::Vertex => self
                .extended_neighbourhood_seq()
                .ok()?
                .map(|x| x.len())
                .sum::<usize>(),
            SampleType::Face => self
                .trivial_neighbourhood_seq()
                .ok()?
                .map(|x| x.len())
                .sum::<usize>(),
        };
        let num_pts_per_sample = match self.base().sample_type {
            SampleType::Vertex => 1,
            SampleType::Face => 3,
        };
        Some(num_neigh_points * 3 * num_pts_per_sample)
    }

    /// Compute the Jacobian of this implicit surface function with respect to surface
    /// points.
    pub fn surface_jacobian_values(
        &self,
        query_points: &[[T; 3]],
        values: &mut [T],
    ) -> Result<(), Error> {
        apply_kernel_fn!(self, |kernel| self.surface_jacobian_values_impl(
            query_points,
            kernel,
            values
        ))
    }

    /// Return row and column indices for each non-zero entry in the jacobian. This is determined
    /// by the precomputed `neighbour_cache` map.
    pub fn surface_jacobian_indices_par_iter<'a>(
        &'a self,
    ) -> Result<impl ParallelIterator<Item = (usize, usize)> + 'a, Error> {
        Ok(match self.base().sample_type {
            SampleType::Vertex => Either::Left(
                self.extended_neighbourhood_par()?
                    .enumerate()
                    .filter(|(_, nbr_points)| nbr_points.len() != 0)
                    .flat_map(move |(row, nbr_points)| {
                        nbr_points.par_iter().flat_map(move |col| {
                            (0usize..3).into_par_iter().map(move |i| (row, 3 * col + i))
                        })
                    }),
            ),
            SampleType::Face => Either::Right(
                self.trivial_neighbourhood_par()?
                    .enumerate()
                    .filter(|(_, nbr_points)| nbr_points.len() != 0)
                    .flat_map(move |(row, nbr_points)| {
                        nbr_points.par_iter().flat_map(move |&pidx| {
                            self.base().surface_topo[pidx]
                                .par_iter()
                                .flat_map(move |col| {
                                    (0usize..3).into_par_iter().map(move |i| (row, 3 * col + i))
                                })
                        })
                    }),
            ),
        })
    }

    /// Return row and column indices for each non-zero entry in the jacobian. This is determined
    /// by the precomputed `neighbour_cache` map.
    pub fn surface_jacobian_indices_iter<'a>(
        &'a self,
    ) -> Result<impl Iterator<Item = (usize, usize)> + 'a, Error> {
        Ok(match self.base().sample_type {
            SampleType::Vertex => Either::Left(
                self.extended_neighbourhood_seq()?
                    .enumerate()
                    .filter(|(_, nbr_points)| nbr_points.len() != 0)
                    .flat_map(move |(row, nbr_points)| {
                        nbr_points
                            .into_iter()
                            .flat_map(move |col| (0..3).map(move |i| (row, 3 * col + i)))
                    }),
            ),
            SampleType::Face => Either::Right(
                self.trivial_neighbourhood_seq()?
                    .enumerate()
                    .filter(|(_, nbr_points)| nbr_points.len() != 0)
                    .flat_map(move |(row, nbr_points)| {
                        nbr_points.iter().flat_map(move |&pidx| {
                            self.base().surface_topo[pidx]
                                .iter()
                                .flat_map(move |col| (0..3).map(move |i| (row, 3 * col + i)))
                        })
                    }),
            ),
        })
    }

    /// Return row and column indices for each non-zero entry in the jacobian. This is determined
    /// by the precomputed `neighbour_cache` map.
    pub fn surface_jacobian_indices(
        &self,
        rows: &mut [usize],
        cols: &mut [usize],
    ) -> Result<(), Error> {
        // For each row
        match self.base().sample_type {
            SampleType::Vertex => {
                let neigh_points = self.extended_neighbourhood_seq()?;
                let row_col_iter = neigh_points
                    .enumerate()
                    .filter(|(_, nbr_points)| !nbr_points.is_empty())
                    .flat_map(move |(row, nbr_points)| {
                        nbr_points
                            .iter()
                            .flat_map(move |&col| (0..3).map(move |i| (row, 3 * col + i)))
                    });
                for ((row, col), out_row, out_col) in
                    zip!(row_col_iter, rows.iter_mut(), cols.iter_mut())
                {
                    *out_row = row;
                    *out_col = col;
                }
            }
            SampleType::Face => {
                let neigh_points = self.trivial_neighbourhood_seq()?;
                let row_col_iter = neigh_points
                    .enumerate()
                    .filter(|(_, nbr_points)| !nbr_points.is_empty())
                    .flat_map(move |(row, nbr_points)| {
                        nbr_points.iter().flat_map(move |&pidx| {
                            self.base().surface_topo[pidx]
                                .iter()
                                .flat_map(move |&col| (0..3).map(move |i| (row, 3 * col + i)))
                        })
                    });
                for ((row, col), out_row, out_col) in
                    zip!(row_col_iter, rows.iter_mut(), cols.iter_mut())
                {
                    *out_row = row;
                    *out_col = col;
                }
            }
        };
        Ok(())
    }

    pub(crate) fn surface_jacobian_values_impl<K>(
        &self,
        query_points: &[[T; 3]],
        kernel: K,
        values: &mut [T],
    ) -> Result<(), Error>
    where
        K: SphericalKernel<T> + std::fmt::Debug + Copy + Sync + Send,
    {
        let value_vecs: &mut [[T; 3]] = reinterpret::reinterpret_mut_slice(values);

        let ImplicitSurfaceBase {
            ref samples,
            ref surface_topo,
            ref dual_topo,
            ref surface_vertex_positions,
            bg_field_params,
            sample_type,
            ..
        } = *self.base();

        match sample_type {
            SampleType::Vertex => {
                let neigh_points = self.extended_neighbourhood_seq()?;
                // For each row (query point)
                let vtx_jac = zip!(query_points.iter(), neigh_points)
                    .filter(|(_, nbrs)| nbrs.len() != 0)
                    .flat_map(move |(q, nbr_points)| {
                        let view = SamplesView::new(nbr_points, samples);
                        Self::vertex_jacobian_at(
                            Vector3(*q),
                            view,
                            kernel,
                            surface_topo,
                            dual_topo,
                            bg_field_params,
                        )
                    });

                value_vecs
                    .iter_mut()
                    .zip(vtx_jac)
                    .for_each(|(vec, new_vec)| {
                        *vec = new_vec.into();
                    });
            }
            SampleType::Face => {
                let neigh_points = self.trivial_neighbourhood_seq()?;
                let face_jac = zip!(query_points.iter(), neigh_points)
                    .filter(|(_, nbrs)| nbrs.len() != 0)
                    .flat_map(move |(q, nbr_points)| {
                        let view = SamplesView::new(nbr_points, samples);
                        Self::face_jacobian_at(
                            Vector3(*q),
                            view,
                            kernel,
                            surface_topo,
                            surface_vertex_positions,
                            bg_field_params,
                        )
                    });

                value_vecs
                    .iter_mut()
                    .zip(face_jac)
                    .for_each(|(vec, new_vec)| {
                        *vec = new_vec.into();
                    });
            }
        }
        Ok(())
    }

    pub(crate) fn vertex_jacobian_at<'a, K: 'a>(
        q: Vector3<T>,
        view: SamplesView<'a, 'a, T>,
        kernel: K,
        surface_topo: &'a [[usize; 3]],
        dual_topo: &'a [Vec<usize>],
        bg_field_params: BackgroundFieldParams,
    ) -> impl Iterator<Item = Vector3<T>> + 'a
    where
        K: SphericalKernel<T> + std::fmt::Debug + Copy + Sync + Send,
    {
        let bg = BackgroundField::local(q, view, kernel, bg_field_params, None).unwrap();

        let closest_d = bg.closest_sample_dist();
        let weight_sum_inv = bg.weight_sum_inv();

        // For each surface vertex contribution
        let main_jac = Self::sample_jacobian_at(q, view, kernel, bg);

        // Add in the normal gradient multiplied by a vector of given Vector3 values.
        let nml_jac = MLS::compute_vertex_unit_normals_gradient_products(
            view,
            &surface_topo,
            &dual_topo,
            move |Sample { pos, .. }| {
                let w = kernel.with_closest_dist(closest_d).eval(q, pos);
                (q - pos) * (w * weight_sum_inv)
            },
        );

        zip!(main_jac, nml_jac).map(|(m, n)| m + n)
    }

    /// Jacobian of the face based local potential with respect to surface vertex positions.
    pub(crate) fn face_jacobian_at<'a, K: 'a>(
        q: Vector3<T>,
        view: SamplesView<'a, 'a, T>,
        kernel: K,
        surface_topo: &'a [[usize; 3]],
        surface_vertex_positions: &'a [Vector3<T>],
        bg_field_params: BackgroundFieldParams,
    ) -> impl Iterator<Item = Vector3<T>> + 'a
    where
        K: SphericalKernel<T> + std::fmt::Debug + Copy + Sync + Send,
    {
        let bg = BackgroundField::local(q, view, kernel, bg_field_params, None).unwrap();
        let third = T::one() / T::from(3.0).unwrap();

        let closest_d = bg.closest_sample_dist();
        let weight_sum_inv = bg.weight_sum_inv();

        // For each surface vertex contribution
        let main_jac = Self::sample_jacobian_at(q, view, kernel, bg);

        // Add in the normal gradient multiplied by a vector of given Vector3 values.
        let nml_jac = Self::compute_face_unit_normals_gradient_products(
            view,
            surface_vertex_positions,
            &surface_topo,
            move |Sample { pos, .. }| {
                let wk = kernel.with_closest_dist(closest_d).eval(q, pos);
                (q - pos) * (wk * weight_sum_inv)
            },
        );

        // There are 3 contributions from each sample to each vertex.
        main_jac
            .flat_map(move |m| std::iter::repeat(m).take(3))
            .zip(nml_jac)
            .map(move |(m, n)| m * third + n)
    }

    /// Compute the Jacobian for the implicit surface potential given by the samples with the
    /// specified kernel assuming constant normals. This Jacobian is with respect to sample points.
    pub(crate) fn sample_jacobian_at<'a, K: 'a>(
        q: Vector3<T>,
        samples: SamplesView<'a, 'a, T>,
        kernel: K,
        bg: BackgroundField<'a, T, T, K>,
    ) -> impl Iterator<Item = Vector3<T>> + 'a
    where
        K: SphericalKernel<T> + std::fmt::Debug + Copy + Sync + Send,
    {
        // Background potential Jacobian.
        let bg_jac = bg.compute_jacobian();

        let closest_d = bg.closest_sample_dist();

        // Background potential adds to the total weight sum, so we should get the updated weight
        // sum from there.
        let weight_sum_inv = bg.weight_sum_inv();

        let local_pot =
            Self::compute_local_potential_at(q, samples, kernel, weight_sum_inv, closest_d);

        let main_jac = samples.into_iter().map(
            move |Sample {
                      index,
                      pos,
                      nml,
                      value,
                      ..
                  }| {
                let diff = q - pos;

                let norm_inv = T::one() / nml.norm();
                let unit_nml = nml * norm_inv;

                let dw = kernel.with_closest_dist(closest_d).grad(q, pos);
                // Contribution from the background potential
                let dwb = bg.background_weight_gradient(Some(index));
                let mut dwdp = (dw - dwb) * (local_pot * weight_sum_inv);

                dwdp -= dw * (weight_sum_inv * (T::from(value).unwrap() + unit_nml.dot(diff)));

                // Compute the normal component of the derivative
                let w = kernel.with_closest_dist(closest_d).eval(q, pos);
                let nml_deriv = unit_nml * (w * weight_sum_inv);
                dwdp - nml_deriv
            },
        );
        zip!(bg_jac, main_jac).map(|(b, m)| b + m)
    }

    /*
     * Contact Jacobian
     */

    /// Compute the contact Jacobian of this implicit surface function with respect to surface
    /// points.
    pub fn contact_jacobian_product_values(
        &self,
        query_points: &[[T; 3]],
        multiplier: &[[T; 3]],
        values: &mut [[T; 3]],
    ) -> Result<(), Error> {
        apply_kernel_fn!(self, |kernel| self.contact_jacobian_product_values_impl(
            query_points,
            multiplier,
            kernel,
            values
        ))
    }

    pub fn num_contact_jacobian_entries(&self) -> Result<usize, Error> {
        self.num_contact_jacobian_matrices().map(|n| n * 9)
    }

    /// Compute the contact Jacobian of this implicit surface function with respect to surface
    /// points.
    pub fn contact_jacobian_values(
        &self,
        query_points: &[[T; 3]],
        values: &mut [T],
    ) -> Result<(), Error> {
        let matrices: &mut [[[T; 3]; 3]] = reinterpret::reinterpret_mut_slice(values);
        self.contact_jacobian_matrices(query_points, matrices)
    }

    pub fn contact_jacobian_indices_iter(
        &self,
    ) -> Result<impl Iterator<Item = (usize, usize)> + Clone, Error> {
        self.contact_jacobian_matrix_indices_iter().map(|iter| {
            iter.flat_map(move |(row_mtx, col_mtx)| {
                (0..3).flat_map(move |j| (0..3).map(move |i| (3 * row_mtx + i, 3 * col_mtx + j)))
            })
        })
    }

    /// Compute the contact Jacobian of this implicit surface function with respect to surface
    /// points. The returned 2D arrays are column major 3x3 matrices.
    pub fn contact_jacobian_matrices(
        &self,
        query_points: &[[T; 3]],
        matrices: &mut [[[T; 3]; 3]],
    ) -> Result<(), Error> {
        apply_kernel_fn!(self, |kernel| self.contact_jacobian_matrices_impl(
            query_points,
            kernel,
            matrices
        ))
    }

    pub fn num_contact_jacobian_matrices(&self) -> Result<usize, Error> {
        let neigh_points = self.trivial_neighbourhood_seq()?;
        let num_pts_per_sample = match self.base().sample_type {
            SampleType::Vertex => 1,
            SampleType::Face => 3,
        };
        Ok(neigh_points.map(|x| x.len()).sum::<usize>() * num_pts_per_sample)
    }

    pub fn contact_jacobian_matrix_indices_iter(
        &self,
    ) -> Result<impl Iterator<Item = (usize, usize)> + Clone, Error> {
        let neigh_points = self.trivial_neighbourhood_seq()?;

        let ImplicitSurfaceBase {
            sample_type,
            ref surface_topo,
            ..
        } = *self.base();

        let indices = neigh_points
            .enumerate()
            .filter(move |(_, nbrs)| nbrs.len() != 0)
            .flat_map(move |(row, nbr_points)| nbr_points.iter().map(move |&col| (row, col)));

        Ok(match sample_type {
            SampleType::Vertex => Either::Left(indices.collect::<Vec<_>>().into_iter()),
            SampleType::Face => Either::Right(
                indices
                    .flat_map(move |(row, j)| surface_topo[j].iter().map(move |&col| (row, col)))
                    .collect::<Vec<_>>()
                    .into_iter(),
            ),
        })
    }

    /// Multiplier is a stacked velocity stored at samples.
    pub(crate) fn contact_jacobian_matrices_impl<K>(
        &self,
        query_points: &[[T; 3]],
        kernel: K,
        value_mtx: &mut [[[T; 3]; 3]],
    ) -> Result<(), Error>
    where
        K: SphericalKernel<T> + std::fmt::Debug + Copy + Sync + Send,
    {
        let neigh_points = self.trivial_neighbourhood_seq()?;

        let ImplicitSurfaceBase {
            ref samples,
            bg_field_params,
            sample_type,
            ..
        } = *self.base();

        let third = T::one() / T::from(3.0).unwrap();

        assert_eq!(query_points.len(), neigh_points.len());

        // For each row (query point),
        let jac = zip!(query_points.iter(), neigh_points)
            .filter(|(_, nbrs)| nbrs.len() != 0)
            .flat_map(move |(q, nbr_points)| {
                let view = SamplesView::new(nbr_points, samples);
                Self::contact_jacobian_at(Vector3(*q), view, kernel, bg_field_params).0
            });

        match sample_type {
            SampleType::Vertex => {
                value_mtx.iter_mut().zip(jac).for_each(|(mtx, new_mtx)| {
                    *mtx = new_mtx.into();
                });
            }
            SampleType::Face => {
                value_mtx
                    .iter_mut()
                    .zip(jac.flat_map(move |j| std::iter::repeat(j * third).take(3)))
                    .for_each(|(mtx, new_mtx)| {
                        *mtx = new_mtx.into();
                    });
            }
        }
        Ok(())
    }

    /// Multiplier is a stacked velocity stored at samples.
    pub(crate) fn contact_jacobian_product_values_impl<K>(
        &self,
        query_points: &[[T; 3]],
        multiplier: &[[T; 3]],
        kernel: K,
        value_vecs: &mut [[T; 3]],
    ) -> Result<(), Error>
    where
        K: SphericalKernel<T> + std::fmt::Debug + Copy + Sync + Send,
    {
        let neigh_points = self.trivial_neighbourhood_seq()?;

        let ImplicitSurfaceBase {
            ref samples,
            bg_field_params,
            sample_type,
            ref surface_topo,
            ..
        } = *self.base();

        match sample_type {
            SampleType::Vertex => {
                // For each row (query point)
                let vtx_jac = zip!(query_points.iter(), neigh_points)
                    .filter(|(_, nbrs)| nbrs.len() != 0)
                    .map(move |(q, nbr_points)| {
                        let view = SamplesView::new(nbr_points, samples);
                        Self::vertex_contact_jacobian_product_at(
                            Vector3(*q),
                            view,
                            multiplier,
                            kernel,
                            bg_field_params,
                        )
                    });

                value_vecs
                    .iter_mut()
                    .zip(vtx_jac)
                    .for_each(|(vec, new_vec)| {
                        *vec = new_vec.into();
                    });
            }
            SampleType::Face => {
                let face_jac = zip!(query_points.iter(), neigh_points)
                    .filter(|(_, nbrs)| nbrs.len() != 0)
                    .map(move |(q, nbr_points)| {
                        let view = SamplesView::new(nbr_points, samples);
                        Self::face_contact_jacobian_product_at(
                            Vector3(*q),
                            view,
                            multiplier,
                            kernel,
                            bg_field_params,
                            surface_topo,
                        )
                    });

                value_vecs
                    .iter_mut()
                    .zip(face_jac)
                    .for_each(|(vec, new_vec)| {
                        *vec = new_vec.into();
                    });
            }
        }
        Ok(())
    }

    /// Compute the contact Jacobian for the implicit surface potential for the given sample with
    /// the specified kernel.  This is the Jacobian of the query point `q` with respect to the
    /// sample position `sample_pos`.  When multiplied by the unit normal, this coincidentally
    /// produces the query Jacobian (Jacobian of the potential with respect to the query position).
    pub(crate) fn sample_contact_jacobian_at<'a, K: 'a>(
        q: Vector3<T>,
        sample_pos: Vector3<T>,
        sample_nml: Vector3<T>,
        kernel: K,
        mut grad_phi: Vector3<T>,
        weight_sum_inv: T,
        closest_d: T,
    ) -> Matrix3<T>
    where
        K: SphericalKernel<T> + std::fmt::Debug + Copy,
    {
        let w_normalized = kernel.with_closest_dist(closest_d).eval(q, sample_pos) * weight_sum_inv;
        let grad_phi_norm = grad_phi.norm();
        if grad_phi_norm != T::zero() {
            grad_phi /= grad_phi_norm;
        }; // normalize grad_phi
        let nml_dot_grad = sample_nml.dot(grad_phi);
        let rot = if nml_dot_grad != -T::one() {
            let u = sample_nml.cross(grad_phi);
            let ux = u.skew();
            Matrix3::identity() + ux + (ux * ux) / (T::one() + nml_dot_grad)
        } else {
            // TODO: take a convenient unit vector u and compute the rotation
            // as
            //let ux = u.skew();
            //Matrix3::identity() + (ux*ux) * 2
            Matrix3::identity()
        };
        rot * w_normalized

        //let w = kernel.with_closest_dist(closest_d).eval(q, sample_pos);
        //let dw = kernel.with_closest_dist(closest_d).grad(q, sample_pos);
        //((dw - dw_neigh_normalized * w) * (q - sample_pos).transpose() + Matrix3::identity() * w)
        //    * weight_sum_inv
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn sample_contact_jacobian_product_at<'a, K: 'a>(
        q: Vector3<T>,
        sample_pos: Vector3<T>,
        sample_nml: Vector3<T>,
        kernel: K,
        grad_phi: Vector3<T>,
        weight_sum_inv: T,
        closest_d: T,
        multiplier: Vector3<T>,
    ) -> Vector3<T>
    where
        K: SphericalKernel<T> + std::fmt::Debug + Copy,
    {
        let jac = Self::sample_contact_jacobian_at(
            q,
            sample_pos,
            sample_nml,
            kernel,
            grad_phi,
            weight_sum_inv,
            closest_d,
        );
        jac * multiplier

        //let w = kernel.with_closest_dist(closest_d).eval(q, sample_pos);
        //let dw = kernel.with_closest_dist(closest_d).grad(q, sample_pos);
        //let psi = T::from(sample_value).unwrap() + multiplier.dot(q - sample_pos);
        //((dw - dw_neigh_normalized * w) * psi + (multiplier * w)) * weight_sum_inv
    }

    /// Compute the Jacobian of a vector on the surface in physical space with respect to the
    /// mesh vertex positions. Note that this is not a strict Jacobian when the background
    /// field is non-zero. Instead this function becomes an affine map and the background portion
    /// is reported in the second output value.
    pub(crate) fn contact_jacobian_at<'a, K: 'a>(
        q: Vector3<T>,
        samples: SamplesView<'a, 'a, T>,
        kernel: K,
        bg_field_params: BackgroundFieldParams,
    ) -> (impl Iterator<Item = Matrix3<T>> + 'a, Vector3<T>)
    where
        K: SphericalKernel<T> + std::fmt::Debug + Copy + Sync + Send,
    {
        let bg = BackgroundField::local(q, samples, kernel, bg_field_params, None).unwrap();

        let weight_sum_inv = bg.weight_sum_inv();
        let closest_d = bg.closest_sample_dist();

        let bg_jac = bg.compute_query_jacobian();

        let grad_phi = Self::query_jacobian_at(q, samples, None, kernel, bg_field_params);

        let jac_iter = samples.into_iter().map(move |sample| {
            Self::sample_contact_jacobian_at(
                q,
                sample.pos,
                sample.nml,
                kernel,
                grad_phi,
                weight_sum_inv,
                closest_d,
            )
        });

        (jac_iter, bg_jac)
    }

    /// Compute the Jacobian of a vector on the surface in physical space with respect to the
    /// mesh vertex positions. Note that this is not a strict Jacobian product when the background
    /// field is non-zero. Instead this function becomes an affine map. This function assumes that
    /// the samples live on the vertices, which is coincident with the multipliers.
    pub(crate) fn vertex_contact_jacobian_product_at<'a, K: 'a>(
        q: Vector3<T>,
        samples: SamplesView<'a, 'a, T>,
        sample_multipliers: &'a [[T; 3]],
        kernel: K,
        bg_field_params: BackgroundFieldParams,
    ) -> Vector3<T>
    where
        K: SphericalKernel<T> + std::fmt::Debug + Copy + Sync + Send,
    {
        let bg = BackgroundField::local(q, samples, kernel, bg_field_params, None).unwrap();

        let weight_sum_inv = bg.weight_sum_inv();
        let closest_d = bg.closest_sample_dist();

        let bg_jac = bg.compute_query_jacobian();

        let grad_phi = Self::query_jacobian_at(q, samples, None, kernel, bg_field_params);

        let jac = samples
            .into_iter()
            .map(
                move |Sample {
                          index, pos, nml, ..
                      }| {
                    let mult = sample_multipliers[index].into();
                    Self::sample_contact_jacobian_product_at(
                        q,
                        pos,
                        nml,
                        kernel,
                        grad_phi,
                        weight_sum_inv,
                        closest_d,
                        mult,
                    )
                },
            )
            .sum::<Vector3<T>>();
        jac + bg_jac
    }

    /// Compute the Jacobian of a vector on the surface in physical space with respect to the
    /// mesh vertex positions. Note that this is not a strict Jacobian product when the background
    /// field is non-zero. Instead this function becomes an affine map. This function assumes that
    /// the samples live on the faces, which is at odds with the multipliers, which live on
    /// vertices.
    pub(crate) fn face_contact_jacobian_product_at<'a, K: 'a>(
        q: Vector3<T>,
        samples: SamplesView<'a, 'a, T>,
        vertex_multipliers: &'a [[T; 3]],
        kernel: K,
        bg_field_params: BackgroundFieldParams,
        triangles: &'a [[usize; 3]],
    ) -> Vector3<T>
    where
        K: SphericalKernel<T> + std::fmt::Debug + Copy + Sync + Send,
    {
        let bg = BackgroundField::local(q, samples, kernel, bg_field_params, None).unwrap();

        let weight_sum_inv = bg.weight_sum_inv();
        let closest_d = bg.closest_sample_dist();

        let bg_jac = bg.compute_query_jacobian();

        let grad_phi = Self::query_jacobian_at(q, samples, None, kernel, bg_field_params);

        let jac = samples
            .into_iter()
            .map(
                move |Sample {
                          index, pos, nml, ..
                      }| {
                    let mult = (0..3).fold(Vector3::zeros(), |acc, i| {
                        acc + vertex_multipliers[triangles[index][i]].into()
                    }) / T::from(3.0).unwrap();
                    Self::sample_contact_jacobian_product_at(
                        q,
                        pos,
                        nml,
                        kernel,
                        grad_phi,
                        weight_sum_inv,
                        closest_d,
                        mult,
                    )
                },
            )
            .sum::<Vector3<T>>();
        jac + bg_jac
    }
}

/// Compute the face normal derivative with respect to tet vertices.
#[cfg(test)]
pub(crate) fn compute_face_unit_normal_derivative<T: Real + Send + Sync>(
    tet_verts: &[Vector3<T>],
    tet_faces: &[[usize; 3]],
    view: SamplesView<'_, '_, T>,
    multiplier: impl FnMut(Sample<T>) -> Vector3<T>,
) -> Vec<Vector3<T>> {
    // Compute the normal gradient product.
    let grad_iter =
        MLS::compute_face_unit_normals_gradient_products(view, tet_verts, tet_faces, multiplier);

    // Convert to grad wrt tet vertex indices instead of surface triangle vertex indices.
    let tet_indices: &[usize] = reinterpret::reinterpret_slice(&tet_faces);
    let mut vert_grad = vec![Vector3::zeros(); tet_verts.len()];
    for (g, &vtx_idx) in grad_iter.zip(tet_indices) {
        vert_grad[vtx_idx] += g;
    }

    vert_grad
}

/// Make a query triangle in the x-z plane at the given height, perturbed by a 3D perturbation function.
#[cfg(test)]
pub(crate) fn make_test_triangle(
    h: f64,
    perturb: &mut impl FnMut() -> Vector3<f64>,
) -> Vec<Vector3<f64>> {
    vec![
        Vector3([0.5, h, 0.0]) + perturb(),
        Vector3([-0.25, h, 0.433013]) + perturb(),
        Vector3([-0.25, h, -0.433013]) + perturb(),
    ]
}

/// Make two query triangles in the x-z plane at the given height, perturbed by a 3D perturbation function.
#[cfg(test)]
pub(crate) fn make_two_test_triangles(
    h: f64,
    perturb: &mut impl FnMut() -> Vector3<f64>,
) -> (Vec<Vector3<f64>>, Vec<[usize; 3]>) {
    (
        vec![
            Vector3([0.0, h, 0.0]) + perturb(),
            Vector3([0.0, h, 1.0]) + perturb(),
            Vector3([1.0, h, 0.0]) + perturb(),
            Vector3([1.0, h, 1.0]) + perturb(),
        ],
        vec![[0, 1, 2], [1, 3, 2]],
    )
}

/// Make htree query triangles in the x-z plane at the given height, perturbed by a 3D perturbation function.
#[cfg(test)]
pub(crate) fn make_three_test_triangles(
    h: f64,
    perturb: &mut impl FnMut() -> Vector3<f64>,
) -> (Vec<Vector3<f64>>, Vec<[usize; 3]>) {
    (
        vec![
            Vector3([0.0, h, 0.0]) + perturb(),
            Vector3([0.0, h, 1.0]) + perturb(),
            Vector3([1.0, h, 0.0]) + perturb(),
            Vector3([1.0, h + 0.5, 1.0]) + perturb(),
            Vector3([2.0, h, 0.0]) + perturb(),
        ],
        vec![[0, 1, 2], [1, 3, 2], [2, 3, 4]],
    )
}

#[cfg(test)]
pub(crate) fn make_perturb_fn() -> impl FnMut() -> Vector3<f64> {
    use rand::{distributions::Uniform, Rng, SeedableRng, StdRng};
    let mut rng: StdRng = SeedableRng::from_seed([3; 32]);
    let range = Uniform::new(-0.1, 0.1);
    move || Vector3([rng.sample(range), rng.sample(range), rng.sample(range)])
}

/// Reduce the given Jacobian from face vertices to vertices.
#[cfg(test)]
pub(crate) fn consolidate_face_jacobian<T: Real>(
    jac: &[Vector3<T>],
    neighbours: &[usize],
    faces: &[[usize; 3]],
    num_verts: usize,
) -> Vec<Vector3<T>> {
    let tet_indices_iter = neighbours
        .iter()
        .flat_map(|&neigh| faces[neigh].iter().cloned());

    let mut vert_jac = vec![Vector3::zeros(); num_verts];

    for (&jac, vtx_idx) in jac.iter().zip(tet_indices_iter) {
        vert_jac[vtx_idx] += jac;
    }

    vert_jac
}

/// A utility function to generate a new set of samples. This is useful when changing which
/// variable to take the auto-derivative with respect to.
#[cfg(test)]
pub(crate) fn new_test_samples<T, V3>(
    sample_type: SampleType,
    triangles: &[[usize; 3]],
    verts: &[V3],
) -> Samples<T>
where
    T: Real + Send + Sync,
    V3: Into<Vector3<T>> + Clone,
{
    match sample_type {
        SampleType::Face => {
            Samples::new_triangle_samples(&triangles, &verts, vec![T::zero(); triangles.len()])
        }
        SampleType::Vertex => {
            Samples::new_vertex_samples(&triangles, &verts, None, vec![T::zero(); verts.len()])
        }
    }
}

///// Linear search for the closest sample to the given query point `q`.
//#[cfg(test)]
//pub(crate) fn find_closest_sample_index<T: Real + Send + Sync>(q: Vector3<T>, samples: &Samples<T>) -> usize {
//    samples.iter()
//        .min_by(|s,t| (q - s.pos).norm().partial_cmp(&(q - t.pos).norm()).unwrap())
//        .expect("Failed to find closest sample.").index
//}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel;
    use autodiff::F;

    /// Tester for the Jacobian at a single position with respect to a surface defined by a single point.
    fn one_point_potential_derivative_tester(radius: f64, bg_field_params: BackgroundFieldParams) {
        // The set of samples is just one point. These are initialized using a forward
        // differentiator.
        let mut samples = Samples {
            points: vec![Vector3([0.2, 0.1, 0.0]).map(|x| F::cst(x))],
            normals: vec![Vector3([0.3, 1.0, 0.1]).map(|x| F::cst(x))],
            velocities: vec![Vector3([2.3, 3.0, 0.2]).map(|x| F::cst(x))],
            values: vec![F::cst(0.0)],
        };

        // The set of neighbours is the one sample given.
        let neighbours = vec![0];

        // Radius is such that samples are captured by the query point.
        let kernel = kernel::LocalApproximate::new(radius, 0.00001);

        // Initialize the query point.
        let q = Vector3([0.5, 0.3, 0.0]).map(|x| F::cst(x));

        // There is no surface for the set of samples. As a result, the normal derivative should be
        // skipped in this test.
        let surf_topo = vec![];
        let dual_topo = vec![vec![]];

        // Create a view of the samples for the Jacobian computation.
        let view = SamplesView::new(neighbours.as_ref(), &samples);

        // Compute the complete jacobian.
        let vert_jac: Vec<_> =
            MLS::vertex_jacobian_at(q, view, kernel, &surf_topo, &dual_topo, bg_field_params)
                .collect();

        // Test the accuracy of each component of the jacobian against an autodiff version of the
        // derivative.
        for i in 0..3 {
            // Set a variable to take the derivative with respect to, using autodiff.
            samples.points[0][i] = F::var(samples.points[0][i]);

            // Create a view of the samples for the potential function.
            let view = SamplesView::new(neighbours.as_ref(), &samples);

            // Initialize background potential to zero.
            let mut p = F::cst(0.0);

            // Compute the local potential function. After calling this function, calling
            // `.deriv()` on the potential output will give us the derivative with resepct to the
            // preset variable.
            MLS::compute_potential_at(q, view, kernel, bg_field_params, &mut p);

            // Check the derivative of the autodiff with our previously computed Jacobian.
            assert_relative_eq!(
                vert_jac[0][i].value(),
                p.deriv(),
                max_relative = 1e-6,
                epsilon = 1e-12
            );

            // Reset the variable back to being a constant.
            samples.points[0][i] = F::cst(samples.points[0][i]);
        }
    }

    /// Test the Jacobian at a single position with respect to a surface defined by a single point.
    #[test]
    fn one_point_potential_derivative_test() {
        for i in 1..50 {
            let radius = 0.1 * (i as f64);
            let run_test = |field_type, weighted| {
                one_point_potential_derivative_tester(
                    radius,
                    BackgroundFieldParams {
                        field_type,
                        weighted,
                    },
                );
            };
            run_test(BackgroundFieldType::Zero, false);
            run_test(BackgroundFieldType::Zero, true);
            run_test(BackgroundFieldType::FromInput, false);
            run_test(BackgroundFieldType::FromInput, true);
            run_test(BackgroundFieldType::DistanceBased, false);
            run_test(BackgroundFieldType::DistanceBased, true);
        }
    }

    /// A more complex test parametrized by the background potential choice, sample type, radius
    /// and a perturbation function that is expected to generate a random perturbation at every
    /// consequent call.
    fn hard_potential_derivative<P: FnMut() -> Vector3<f64>>(
        bg_field_params: BackgroundFieldParams,
        sample_type: SampleType,
        radius: f64,
        perturb: &mut P,
    ) {
        // This is a similar test to the one above, but has a non-trivial surface topology for the
        // surface.
        let tri_verts = make_test_triangle(1.18032, perturb);
        let (tet_verts, tet_faces) = make_tet();

        let dual_topo = ImplicitSurfaceBuilder::compute_dual_topo(tet_verts.len(), &tet_faces);

        let samples = new_test_samples(sample_type, &tet_faces, &tet_verts);

        let neighbours = vec![0, 1, 2, 3]; // All tet faces or verts (depending on sample_type)

        let kernel = kernel::LocalApproximate::new(radius, 1e-5);

        // Convert tet vertices into varibales because we are taking the derivative with respect to
        // vertices.
        let mut ad_tet_verts: Vec<Vector3<F>> =
            tet_verts.iter().map(|&v| v.map(|x| F::cst(x))).collect();

        for &q in tri_verts.iter() {
            // Compute the Jacobian.
            let view = SamplesView::new(neighbours.as_ref(), &samples);

            let vert_jac = match sample_type {
                SampleType::Face => {
                    let jac: Vec<_> = MLS::face_jacobian_at(
                        q,
                        view,
                        kernel,
                        &tet_faces,
                        &tet_verts,
                        bg_field_params,
                    )
                    .collect();

                    assert_eq!(jac.len(), 3 * neighbours.len());

                    consolidate_face_jacobian(&jac, &neighbours, &tet_faces, tet_verts.len())
                }
                SampleType::Vertex => {
                    let jac: Vec<_> = MLS::vertex_jacobian_at(
                        q,
                        view,
                        kernel,
                        &tet_faces,
                        &dual_topo,
                        bg_field_params,
                    )
                    .collect();

                    assert_eq!(jac.len(), neighbours.len());

                    jac
                }
            };

            let q = q.map(|x| F::cst(x));

            for (vtx, jac) in vert_jac.iter().enumerate() {
                for i in 0..3 {
                    ad_tet_verts[vtx][i] = F::var(ad_tet_verts[vtx][i]);

                    let ad_samples = new_test_samples(sample_type, &tet_faces, &ad_tet_verts);

                    let view = SamplesView::new(neighbours.as_ref(), &ad_samples);
                    let mut p = F::cst(0.0);
                    MLS::compute_potential_at(q, view, kernel, bg_field_params, &mut p);

                    assert_relative_eq!(jac[i], p.deriv(), max_relative = 1e-5, epsilon = 1e-10);

                    ad_tet_verts[vtx][i] = F::cst(ad_tet_verts[vtx][i]);
                }
            }
        }
    }

    #[test]
    fn hard_potential_derivative_test() {
        let mut perturb = make_perturb_fn();

        // Run for some number of perturbations
        for i in 1..50 {
            let radius = 0.1 * (i as f64);
            let mut run_test = |field_type, weighted, sample_type| {
                hard_potential_derivative(
                    BackgroundFieldParams {
                        field_type,
                        weighted,
                    },
                    sample_type,
                    radius,
                    &mut perturb,
                );
            };

            run_test(BackgroundFieldType::Zero, false, SampleType::Vertex);
            run_test(BackgroundFieldType::Zero, true, SampleType::Vertex);
            run_test(BackgroundFieldType::FromInput, false, SampleType::Vertex);
            run_test(BackgroundFieldType::FromInput, true, SampleType::Vertex);
            run_test(
                BackgroundFieldType::DistanceBased,
                false,
                SampleType::Vertex,
            );
            run_test(BackgroundFieldType::DistanceBased, true, SampleType::Vertex);

            run_test(BackgroundFieldType::Zero, false, SampleType::Face);
            run_test(BackgroundFieldType::Zero, true, SampleType::Face);
            run_test(BackgroundFieldType::FromInput, false, SampleType::Face);
            run_test(BackgroundFieldType::FromInput, true, SampleType::Face);
            run_test(BackgroundFieldType::DistanceBased, false, SampleType::Face);
            run_test(BackgroundFieldType::DistanceBased, true, SampleType::Face);
        }
    }

    /// Test the derivatives of our normal computation method.
    fn normal_derivative_test(sample_type: SampleType) {
        let (tet_verts, tet_faces) = make_tet();

        // Vertex to triangle map
        let dual_topo = ImplicitSurfaceBuilder::compute_dual_topo(tet_verts.len(), &tet_faces);

        let samples = new_test_samples(sample_type, &tet_faces, &tet_verts);

        let indices = vec![0, 1, 2, 3]; // look at all the vertices

        // Set a random product vector.
        let dxs = utils::random_vectors(tet_verts.len());
        let dx = move |Sample { index, .. }| Vector3(dxs[index]);

        // Compute the normal gradient product.
        let view = SamplesView::new(indices.as_ref(), &samples);

        let grad: Vec<_> = match sample_type {
            SampleType::Vertex => MLS::compute_vertex_unit_normals_gradient_products(
                view,
                &tet_faces,
                &dual_topo,
                dx.clone(),
            )
            .collect(),
            SampleType::Face => {
                compute_face_unit_normal_derivative(&tet_verts, &tet_faces, view, dx.clone())
            }
        };

        // Convert tet vertices into varibales because we are taking the derivative with respect to
        // vertices.
        let mut ad_tet_verts: Vec<Vector3<F>> =
            tet_verts.iter().map(|&v| v.map(|x| F::cst(x))).collect();

        for (vtx, g) in grad.iter().enumerate() {
            for i in 0..3 {
                ad_tet_verts[vtx][i] = F::var(ad_tet_verts[vtx][i]);

                // Recompute normals by computing new autodiff samples.
                let mut ad_samples = new_test_samples(sample_type, &tet_faces, &ad_tet_verts);

                // Normalize normals
                for nml in ad_samples.normals.iter_mut() {
                    *nml = *nml / nml.norm();
                }

                let mut exp = F::cst(0.0);
                for sample in view.clone().iter() {
                    exp += ad_samples.normals[sample.index].dot(dx(sample).map(|x| F::cst(x)));
                }

                assert_relative_eq!(g[i], exp.deriv(), max_relative = 1e-5, epsilon = 1e-10);

                ad_tet_verts[vtx][i] = F::cst(ad_tet_verts[vtx][i]);
            }
        }
    }

    /// Test the first order derivatives of our normal computation method for face normals.
    #[test]
    fn face_normal_derivative_test() {
        normal_derivative_test(SampleType::Face);
    }
    /// Test the derivatives of our normal computation method for vertex normals.
    #[test]
    fn vertex_normal_derivative_test() {
        normal_derivative_test(SampleType::Vertex);
    }

    #[test]
    fn dynamic_background_potential_derivative_test() {
        // Prepare data
        let q = Vector3([0.1, 0.3, 0.2]);
        let points = vec![
            Vector3([0.3, 0.2, 0.1]),
            Vector3([0.4, 0.2, 0.1]),
            Vector3([0.2, 0.1, 0.3]),
        ];

        let samples = Samples::new_point_samples(points.clone());

        let indices: Vec<usize> = (0..points.len()).collect();

        let radius = 2.0;

        // Initialize kernel.
        let kernel = kernel::LocalApproximate::new(radius, 1e-5);

        // Create a view to the data to be iterated.
        let view = SamplesView::new(indices.as_slice(), &samples);

        // Initialize a background potential. This function takes care of a lot of the setup.
        let bg = BackgroundField::local(
            q,
            view,
            kernel,
            BackgroundFieldParams {
                field_type: BackgroundFieldType::DistanceBased,
                weighted: true,
            },
            None,
        )
        .unwrap();

        // Compute manual Jacobian. This is the function being tested for correctness.
        let jac: Vec<_> = bg.compute_jacobian().collect();

        // Prepare autodiff variables.
        let mut ad_samples =
            Samples::new_point_samples(points.iter().map(|&pos| pos.map(|x| F::cst(x))).collect());

        let q = q.map(|x| F::cst(x));

        // Perform the derivative test on each of the variables.
        for i in 0..points.len() {
            for j in 0..3 {
                ad_samples.points[i][j] = F::var(ad_samples.points[i][j]);

                // Initialize an autodiff version of the potential.
                // This should be done outside the inner loop over samples, but here we make an
                // exception for simplicity.
                let view = SamplesView::new(indices.as_slice(), &ad_samples);
                let ad_bg = BackgroundField::local(
                    q,
                    view,
                    kernel,
                    BackgroundFieldParams {
                        field_type: BackgroundFieldType::DistanceBased,
                        weighted: true,
                    },
                    Some(F::cst(0.0)),
                )
                .unwrap();

                let p = ad_bg.compute_unnormalized_weighted_scalar_field() * ad_bg.weight_sum_inv();

                assert_relative_eq!(jac[i][j], p.deriv());
                ad_samples.points[i][j] = F::cst(ad_samples.points[i][j]);
            }
        }
    }

    /// A test parametrized by the background potential choice, sample type, radius and a
    /// perturbation function that is expected to generate a random perturbation at every
    /// consequent call.  This function tests the surface Jacobian of the implicit function.
    pub fn surface_jacobian<P: FnMut() -> Vector3<f64>>(
        bg_field_params: BackgroundFieldParams,
        sample_type: SampleType,
        radius_multiplier: f64,
        perturb: &mut P,
    ) {
        let tri_vert_vecs = make_test_triangle(1.18032, perturb);

        let tri_verts: Vec<[f64; 3]> = tri_vert_vecs.iter().map(|&v| v.into()).collect();

        let params = crate::Params {
            kernel: KernelType::Approximate {
                tolerance: 0.00001,
                radius_multiplier,
            },
            background_field: bg_field_params,
            sample_type,
            ..Default::default()
        };

        let tet = geo::mesh::TriMesh::from(utils::make_regular_tet());
        let mut surf = crate::surface_from_trimesh(&tet, params)
            .expect("Failed to create a surface for a tet.");

        let (tet_verts, _) = make_tet(); // expect this to be the same as in make_regular_tet.

        // Convert tet vertices into varibales because we are taking the derivative with respect to
        // vertices.
        let mut ad_tet_verts: Vec<[F; 3]> = tet_verts
            .iter()
            .cloned()
            .map(|v| v.map(|x| F::cst(x)).into())
            .collect();

        let mut ad_surf = crate::surface_from_trimesh::<F>(&tet, params)
            .expect("Failed to create a surface for a autodiff tet.");
        let ad_tri_verts: Vec<[F; 3]> = tri_vert_vecs
            .iter()
            .map(|&v| v.map(|x| F::cst(x)).into())
            .collect();

        surf.compute_neighbours(&tri_verts);
        ad_surf.compute_neighbours(&ad_tri_verts);
        let nnz = surf.num_surface_jacobian_entries().unwrap();
        let mut jac_vals = vec![0.0; nnz];
        let mut jac_rows = vec![0; nnz];
        let mut jac_cols = vec![0; nnz];
        surf.surface_jacobian_indices(&mut jac_rows, &mut jac_cols)
            .expect("Failed to compute surface jacobian indices");
        surf.surface_jacobian_values(&tri_verts, &mut jac_vals)
            .expect("Failed to compute surface jacobian");

        let mut jac = [[0.0; 3]; 12];

        // Make sure the indices are the same as when using iter.
        for ((i, idx), &val) in surf
            .surface_jacobian_indices_iter()
            .unwrap()
            .enumerate()
            .zip(jac_vals.iter())
        {
            assert_eq!(idx.0, jac_rows[i]);
            assert_eq!(idx.1, jac_cols[i]);
            jac[idx.1][idx.0] += val;
        }

        for pidx in 0..ad_tet_verts.len() {
            for i in 0..3 {
                ad_tet_verts[pidx][i] = F::var(ad_tet_verts[pidx][i]);
                ad_surf.update(ad_tet_verts.iter().cloned());

                let mut potential = vec![F::cst(0.0); ad_tri_verts.len()];
                ad_surf
                    .potential(&ad_tri_verts, &mut potential)
                    .expect("Failed to compute autodiff potential");

                let col = 3 * pidx + i;
                for row in 0..3 {
                    //if !relative_eq!(
                    //    jac[col][row],
                    //    potential[row].deriv(),
                    //    max_relative = 1e-5,
                    //    epsilon = 1e-10
                    //) {
                    //    println!("({:?}, {:?}) => {:?} vs {:?}", row, col, jac[col][row], potential[row].deriv());
                    //}
                    assert_relative_eq!(
                        jac[col][row],
                        potential[row].deriv(),
                        max_relative = 1e-5,
                        epsilon = 1e-10
                    );
                }

                ad_tet_verts[pidx][i] = F::cst(ad_tet_verts[pidx][i]);
            }
        }
    }

    #[test]
    fn surface_jacobian_test() {
        let mut perturb = make_perturb_fn();

        // Run for some number of perturbations
        for i in 0..50 {
            let radius_multiplier = 1.0 + 0.1 * (i as f64);

            let mut run_test = |field_type, weighted, sample_type| {
                surface_jacobian(
                    BackgroundFieldParams {
                        field_type,
                        weighted,
                    },
                    sample_type,
                    radius_multiplier,
                    &mut perturb,
                );
            };

            run_test(BackgroundFieldType::Zero, false, SampleType::Vertex);
            run_test(BackgroundFieldType::Zero, true, SampleType::Vertex);
            run_test(BackgroundFieldType::FromInput, false, SampleType::Vertex);
            run_test(BackgroundFieldType::FromInput, true, SampleType::Vertex);
            run_test(
                BackgroundFieldType::DistanceBased,
                false,
                SampleType::Vertex,
            );
            run_test(BackgroundFieldType::DistanceBased, true, SampleType::Vertex);

            run_test(BackgroundFieldType::Zero, false, SampleType::Face);
            run_test(BackgroundFieldType::Zero, true, SampleType::Face);
            run_test(BackgroundFieldType::FromInput, false, SampleType::Face);
            run_test(BackgroundFieldType::FromInput, true, SampleType::Face);
            run_test(BackgroundFieldType::DistanceBased, false, SampleType::Face);
            run_test(BackgroundFieldType::DistanceBased, true, SampleType::Face);
        }
    }

    /// A test parametrized by the background potential choice, radius and a perturbation
    /// function that is expected to generate a random perturbation at every consequent call.
    /// This function tests the query Jacobian of the implicit function.
    pub fn query_jacobian<P: FnMut() -> Vector3<f64>>(
        bg_field_params: BackgroundFieldParams,
        radius: f64,
        perturb: &mut P,
    ) {
        let tri_verts = make_test_triangle(1.18032, perturb);

        let (tet_verts, tet_faces) = make_tet();

        let samples = Samples::new_triangle_samples(&tet_faces, &tet_verts, vec![0.0; 4]);

        let neighbours = vec![0, 1, 2, 3]; // All tet faces

        let kernel = kernel::LocalApproximate::new(radius, 1e-5);

        // Convert tet vertices into varibales because we are taking the derivative with respect to
        // vertices.
        let ad_tet_verts: Vec<Vector3<F>> = tet_verts
            .iter()
            .cloned()
            .map(|v| v.map(|x| F::cst(x)))
            .collect();

        for &q in tri_verts.iter() {
            // Compute the Jacobian.
            let view = SamplesView::new(neighbours.as_ref(), &samples);

            let jac = MLS::query_jacobian_at(q, view, None, kernel, bg_field_params);

            let mut q = q.map(|x| F::cst(x));

            for i in 0..3 {
                q[i] = F::var(q[i]);

                let ad_samples =
                    Samples::new_triangle_samples(&tet_faces, &ad_tet_verts, vec![F::cst(0.0); 4]);

                let view = SamplesView::new(neighbours.as_ref(), &ad_samples);

                let mut p = F::cst(0.0);
                MLS::compute_potential_at(q, view, kernel, bg_field_params, &mut p);

                assert_relative_eq!(jac[i], p.deriv(), max_relative = 1e-5, epsilon = 1e-10);

                q[i] = F::cst(q[i]);
            }
        }
    }

    #[test]
    fn query_jacobian_test() {
        let mut perturb = make_perturb_fn();

        // Run for some number of perturbations
        for i in 0..50 {
            let radius = 0.1 * (i as f64);
            query_jacobian(
                BackgroundFieldParams {
                    field_type: BackgroundFieldType::Zero,
                    weighted: false,
                },
                radius,
                &mut perturb,
            );
            query_jacobian(
                BackgroundFieldParams {
                    field_type: BackgroundFieldType::Zero,
                    weighted: true,
                },
                radius,
                &mut perturb,
            );
            query_jacobian(
                BackgroundFieldParams {
                    field_type: BackgroundFieldType::FromInput,
                    weighted: false,
                },
                radius,
                &mut perturb,
            );
            query_jacobian(
                BackgroundFieldParams {
                    field_type: BackgroundFieldType::FromInput,
                    weighted: true,
                },
                radius,
                &mut perturb,
            );
            query_jacobian(
                BackgroundFieldParams {
                    field_type: BackgroundFieldType::DistanceBased,
                    weighted: false,
                },
                radius,
                &mut perturb,
            );
            query_jacobian(
                BackgroundFieldParams {
                    field_type: BackgroundFieldType::DistanceBased,
                    weighted: true,
                },
                radius,
                &mut perturb,
            );
        }
    }

    /// Verify that the contact jacobian can interpolate an accurate normal.
    /// Given a triangle with identical normals at each vertex, we compute the
    /// corresponding normal at the triangle centroid and verify that it is the
    /// same.
    #[test]
    fn contact_jacobian_normal_test() -> Result<(), Error> {
        use crate::*;
        use geo::NumVertices;

        let tri_vert_pos = make_test_triangle(0.0, &mut || Vector3::zeros());
        let tri_verts: Vec<[f64; 3]> = reinterpret::reinterpret_vec(tri_vert_pos);
        let area = 0.32475975;
        let centroid = [0.0; 3];
        let query_points = vec![centroid];

        let surf_params = Params {
            kernel: kernel::KernelType::Approximate {
                radius_multiplier: 2.0,
                tolerance: 1e-5,
            },
            background_field: BackgroundFieldParams {
                field_type: BackgroundFieldType::DistanceBased,
                weighted: false,
            },
            sample_type: SampleType::Vertex,
            max_step: 0.0,
        };

        let trimesh = geo::mesh::TriMesh::new(tri_verts, vec![0, 2, 1]);
        let mut surf = surface_from_trimesh(&trimesh, surf_params).unwrap();
        surf.compute_neighbours(&query_points);

        let mut jac = vec![[[0.0; 3]; 3]; trimesh.num_vertices()];

        surf.contact_jacobian_matrices(&query_points, &mut jac)?;
        let num_jac_entries = surf.num_contact_jacobian_matrices()?;
        assert_eq!(num_jac_entries, 3);
        let weighted_normal = Vector3([0.0, area, 0.0]);

        let mut result = Vector3::zeros();
        for &jac_mtx in jac.iter() {
            result += Matrix3(jac_mtx) * weighted_normal;
        }

        let expected = weighted_normal;
        for i in 0..3 {
            assert_relative_eq!(result[i], expected[i], max_relative = 1e-5, epsilon = 1e-10);
        }
        Ok(())
    }

    /// Verify that the contact jacobian can interpolate an accurate tangent vector.
    /// Given a triangle with identical tangent vectors at each vertex, we compute the
    /// corresponding vector at the triangle centroid and verify that it is the
    /// same.
    #[test]
    fn contact_jacobian_identity_test() -> Result<(), Error> {
        use crate::*;
        use geo::NumVertices;

        let tri_vert_pos = make_test_triangle(0.0, &mut || Vector3::zeros());
        let tri_verts: Vec<[f64; 3]> = reinterpret::reinterpret_vec(tri_vert_pos);
        let centroid = [0.0; 3];
        let query_points = vec![centroid];

        let kernel = kernel::KernelType::Approximate {
            radius_multiplier: 2.0,
            tolerance: 1e-5,
        };
        let surf_params = Params {
            kernel,
            background_field: BackgroundFieldParams {
                field_type: BackgroundFieldType::DistanceBased,
                weighted: false,
            },
            sample_type: SampleType::Vertex,
            max_step: 0.0,
        };

        let mut trimesh = geo::mesh::TriMesh::new(tri_verts, vec![0, 2, 1]);
        let test_vector = Vector3([1.5, 0.3, 0.5]);
        trimesh.add_attrib_data::<[f32; 3], VertexIndex>("V", vec![test_vector.into(); 3])?;
        trimesh.add_attrib_data::<[f32; 3], VertexIndex>("N", vec![[0.0, 1.0, 0.0]; 3])?;

        let mut surf = surface_from_trimesh(&trimesh, surf_params).unwrap();
        surf.compute_neighbours(&query_points);

        let mut jac = vec![[[0.0; 3]; 3]; trimesh.num_vertices()];

        surf.contact_jacobian_matrices(&query_points, &mut jac)?;
        let num_jac_entries = surf.num_contact_jacobian_matrices()?;
        assert_eq!(num_jac_entries, 3);

        let mut result = Vector3::zeros();
        for &jac_mtx in jac.iter() {
            result += Matrix3(jac_mtx) * test_vector;
        }

        // Verify that the contact jacobian produces the same result as when computing the
        // quantities on an input mesh, which is often used for debugging and prototyping.
        let mut ptcld = geo::mesh::PointCloud::new(query_points.clone());
        surf.compute_potential_on_mesh(&mut ptcld, || false)?;
        let result_attrib = ptcld.remove_attrib::<VertexIndex>("tangents")?;
        let tangents_vec = result_attrib.clone_into_vec()?;
        let result2: [f32; 3] = tangents_vec[0];

        for i in 0..3 {
            assert_relative_eq!(result[i], result2[i], max_relative = 1e-5, epsilon = 1e-10);
        }

        // Finally verify that the produced vector is indeed the same as the input test_vector.
        // That is interpolating the same vector better produce that same vector.
        let expected = test_vector;
        for i in 0..3 {
            assert_relative_eq!(result[i], expected[i], max_relative = 1e-5, epsilon = 1e-10);
        }
        Ok(())
    }

    /// Tester for the contact jacobian. This tester is parameterized by background field type,
    /// radius and a perturb function.
    fn contact_jacobian<P: FnMut() -> Vector3<f64>>(
        bg_field_params: BackgroundFieldParams,
        radius_multiplier: f64,
        perturb: &mut P,
    ) -> Result<(), Error> {
        use crate::*;
        use geo::NumVertices;
        use utils::*;

        let tri_vert_pos = make_test_triangle(1.18032, perturb);

        let tri_verts: Vec<[f64; 3]> = reinterpret::reinterpret_vec(tri_vert_pos);

        let mut tet = make_regular_tet();

        let multiplier_vecs = utils::random_vectors(tet.num_vertices());
        let multipliers_f32: Vec<_> = multiplier_vecs
            .iter()
            .cloned()
            .map(|v| Vector3(v).map(|x| x as f32).into())
            .collect();
        tet.set_attrib_data::<[f32; 3], VertexIndex>("V", &multipliers_f32)
            .unwrap();

        let surf_params = Params {
            kernel: kernel::KernelType::Approximate {
                radius_multiplier,
                tolerance: 1e-5,
            },
            background_field: bg_field_params,
            sample_type: SampleType::Face,
            max_step: 100.0 * radius_multiplier, // essentially unlimited
        };

        let trimesh = geo::mesh::TriMesh::from(tet);

        let multipliers: Vec<_> = trimesh
            .attrib_as_slice::<[f32; 3], VertexIndex>("V")
            .unwrap()
            .iter()
            .map(|&x| Vector3(x).map(|x| f64::from(x)).into_inner())
            .collect();
        let mut surf = surface_from_trimesh(&trimesh, surf_params).unwrap();
        surf.compute_neighbours(&tri_verts);

        let mut jac_prod = vec![[0.0; 3]; tri_verts.len()];

        // Compute and test the contact Jacobian product.
        surf.contact_jacobian_product_values(&tri_verts, &multipliers, &mut jac_prod)?;

        let mut expected = vec![[0.0; 3]; tri_verts.len()];
        surf.vector_field(&tri_verts, &mut expected).unwrap();
        for (jac, &exp) in jac_prod.into_iter().zip(expected.iter()) {
            for i in 0..3 {
                assert_relative_eq!(jac[i], exp[i], max_relative = 1e-5, epsilon = 1e-10);
            }
        }

        // Compute and test the contact Jacobian matrix.
        let num_jac_entries = surf.num_contact_jacobian_entries()?;
        let indices_iter = surf.contact_jacobian_indices_iter()?;
        let mut jac = vec![0.0; num_jac_entries];
        surf.contact_jacobian_values(&tri_verts, &mut jac)?;
        let multiplier_values: &[f64] = reinterpret::reinterpret_slice(&multipliers);
        let mut alt_jac_prod_vals = vec![0.0; tri_verts.len() * 3];
        for ((row, col), jac) in indices_iter.zip(jac.into_iter()) {
            alt_jac_prod_vals[row] += jac * multiplier_values[col];
        }

        let alt_jac_prod_vecs: &[[f64; 3]] = reinterpret::reinterpret_slice(&alt_jac_prod_vals);

        for (jac, &exp) in alt_jac_prod_vecs.into_iter().zip(expected.iter()) {
            for i in 0..3 {
                if !relative_eq!(jac[i], exp[i], max_relative = 1e-5, epsilon = 1e-10) {
                    println!("{:?} vs {:?}", jac[i], exp[i]);
                }
                //assert_relative_eq!(jac[i], exp[i], max_relative = 1e-5, epsilon = 1e-10);
            }
        }
        Ok(())
    }

    #[test]
    fn contact_jacobian_test() -> Result<(), Error> {
        let mut perturb = make_perturb_fn();

        // Run for some number of perturbations
        for i in 0..50 {
            let radius_multiplier = 1.0 + 0.1 * (i as f64);
            contact_jacobian(
                BackgroundFieldParams {
                    field_type: BackgroundFieldType::Zero,
                    weighted: false,
                },
                radius_multiplier,
                &mut perturb,
            )?;
            contact_jacobian(
                BackgroundFieldParams {
                    field_type: BackgroundFieldType::Zero,
                    weighted: true,
                },
                radius_multiplier,
                &mut perturb,
            )?;
            contact_jacobian(
                BackgroundFieldParams {
                    field_type: BackgroundFieldType::FromInput,
                    weighted: false,
                },
                radius_multiplier,
                &mut perturb,
            )?;
            contact_jacobian(
                BackgroundFieldParams {
                    field_type: BackgroundFieldType::FromInput,
                    weighted: true,
                },
                radius_multiplier,
                &mut perturb,
            )?;
        }
        Ok(())
    }
}
