use super::*;
use crate::Error;

impl<T: Real + Send + Sync> ImplicitSurface<T> {
    /// Compute the number of indices (non-zeros) needed for the implicit surface potential
    /// Jacobian with respect to surface points.
    pub fn num_surface_jacobian_entries(&self) -> usize {
        let cache = self.neighbour_cache.borrow();
        let num_pts_per_sample = match self.sample_type {
            SampleType::Vertex => 1,
            SampleType::Face => 3,
        };
        cache
            .cached_neighbour_points()
            .iter()
            .map(|pts| pts.len())
            .sum::<usize>()
            * 3
            * num_pts_per_sample
    }

    /// Compute the indices for the implicit surface potential Jacobian with respect to surface
    /// points.
    pub fn surface_jacobian_indices_iter(
        &self,
    ) -> Result<Box<dyn Iterator<Item = (usize, usize)>>, Error> {
        match self.kernel {
            KernelType::Approximate { .. } => Ok(self.mls_surface_jacobian_indices_iter()),
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
        match self.kernel {
            KernelType::Approximate { .. } => {
                self.mls_surface_jacobian_indices(rows, cols);
                Ok(())
            }
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
        let ImplicitSurface {
            ref kernel,
            ref spatial_tree,
            max_step,
            ..
        } = *self;

        match *kernel {
            KernelType::Approximate { tolerance, radius } => {
                let kernel = kernel::LocalApproximate::new(radius, tolerance);
                let radius_ext = radius + cast::<_, f64>(max_step).unwrap();
                let radius2 = radius_ext * radius_ext;
                let neigh = |q| {
                    let q_pos = Vector3(q).cast::<f64>().unwrap().into();
                    spatial_tree
                        .lookup_in_circle(&q_pos, &radius2)
                        .into_iter()
                        .cloned()
                };
                self.mls_surface_jacobian_values(query_points, kernel, neigh, values);
                Ok(())
            }
            _ => Err(Error::UnsupportedKernel),
        }
    }

    /// Return row and column indices for each non-zero entry in the jacobian. This is determined
    /// by the precomputed `neighbour_cache` map.
    pub(crate) fn mls_surface_jacobian_indices_iter(
        &self,
    ) -> Box<dyn Iterator<Item = (usize, usize)>> {
        match self.sample_type {
            SampleType::Vertex => {
                let cached_pts = {
                    let cache = self.neighbour_cache.borrow();
                    cache.cached_neighbour_points().to_vec()
                };
                Box::new(
                    cached_pts
                        .into_iter()
                        .filter(|c| !c.is_empty())
                        .enumerate()
                        .flat_map(move |(row, nbr_points)| {
                            nbr_points
                                .into_iter()
                                .flat_map(move |col| (0..3).map(move |i| (row, 3 * col + i)))
                        }),
                )
            }
            SampleType::Face => {
                let cached: Vec<_> = {
                    let cache = self.neighbour_cache.borrow();
                    cache
                        .cached_neighbour_points()
                        .iter()
                        .filter(|c| !c.is_empty())
                        .enumerate()
                        .flat_map(|(row, nbr_points)| {
                            nbr_points.iter().flat_map(move |&pidx| {
                                self.surface_topo[pidx]
                                    .iter()
                                    .flat_map(move |col| (0..3).map(move |i| (row, 3 * col + i)))
                            })
                        })
                        .collect()
                };

                Box::new(cached.into_iter())
            }
        }
    }

    /// Return row and column indices for each non-zero entry in the jacobian. This is determined
    /// by the precomputed `neighbour_cache` map.
    fn mls_surface_jacobian_indices(&self, rows: &mut [usize], cols: &mut [usize]) {
        // For each row
        let cache = self.neighbour_cache.borrow();
        match self.sample_type {
            SampleType::Vertex => {
                let row_col_iter = cache
                    .cached_neighbour_points()
                    .iter()
                    .filter(|c| !c.is_empty())
                    .enumerate()
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
                let row_col_iter = cache
                    .cached_neighbour_points()
                    .iter()
                    .filter(|c| !c.is_empty())
                    .enumerate()
                    .flat_map(move |(row, nbr_points)| {
                        nbr_points.iter().flat_map(move |&pidx| {
                            self.surface_topo[pidx]
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
    }

    pub(crate) fn mls_surface_jacobian_values<'a, I, K, N>(
        &self,
        query_points: &[[T; 3]],
        kernel: K,
        neigh: N,
        values: &mut [T],
    ) where
        I: Iterator<Item = Sample<T>> + 'a,
        K: SphericalKernel<T> + LocalKernel<T> + std::fmt::Debug + Copy + Sync + Send,
        N: Fn([T; 3]) -> I + Sync + Send,
    {
        let value_vecs: &mut [[T; 3]] = reinterpret::reinterpret_mut_slice(values);

        let neigh_points = self.cached_neighbours_borrow(query_points, neigh);

        let ImplicitSurface {
            ref samples,
            ref surface_topo,
            ref dual_topo,
            ref surface_vertex_positions,
            bg_field_type,
            sample_type,
            ..
        } = *self;

        match sample_type {
            SampleType::Vertex => {
                // For each row (query point)
                let vtx_jac = zip!(query_points.iter(), neigh_points.iter())
                    .filter(|(_, nbrs)| !nbrs.is_empty())
                    .flat_map(move |(q, nbr_points)| {
                        let view = SamplesView::new(nbr_points, samples);
                        Self::vertex_jacobian_at(
                            Vector3(*q),
                            view,
                            kernel,
                            surface_topo,
                            dual_topo,
                            bg_field_type,
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
                let face_jac = zip!(query_points.iter(), neigh_points.iter())
                    .filter(|(_, nbrs)| !nbrs.is_empty())
                    .flat_map(move |(q, nbr_points)| {
                        let view = SamplesView::new(nbr_points, samples);

                        Self::face_jacobian_at(
                            Vector3(*q),
                            view,
                            kernel,
                            surface_topo,
                            surface_vertex_positions,
                            bg_field_type,
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
    }

    pub(crate) fn vertex_jacobian_at<'a, K: 'a>(
        q: Vector3<T>,
        view: SamplesView<'a, 'a, T>,
        kernel: K,
        surface_topo: &'a [[usize; 3]],
        dual_topo: &'a [Vec<usize>],
        bg_field_type: BackgroundFieldType,
    ) -> impl Iterator<Item = Vector3<T>> + 'a
    where
        K: SphericalKernel<T> + LocalKernel<T> + std::fmt::Debug + Copy + Sync + Send,
    {
        let bg = Self::compute_background_potential(q, view, kernel, bg_field_type);

        // Background potential Jacobian.
        let bg_jac = bg.compute_jacobian();

        let closest_d = bg.closest_sample_dist();
        let weight_sum_inv = bg.weight_sum_inv();

        // For each surface vertex contribution
        let main_jac = Self::sample_jacobian_at(q, view, kernel, bg);

        // Add in the normal gradient multiplied by a vector of given Vector3 values.
        let nml_jac = ImplicitSurface::compute_vertex_unit_normals_gradient_products(
            view,
            &surface_topo,
            &dual_topo,
            move |Sample { pos, .. }| {
                let wk = kernel.with_closest_dist(closest_d).eval(q, pos);
                (q - pos) * (wk * weight_sum_inv)
            },
        );

        zip!(bg_jac, main_jac, nml_jac).map(|(b, m, n)| b + m + n)
    }

    pub(crate) fn face_jacobian_at<'a, K: 'a>(
        q: Vector3<T>,
        view: SamplesView<'a, 'a, T>,
        kernel: K,
        surface_topo: &'a [[usize; 3]],
        surface_vertex_positions: &'a [Vector3<T>],
        bg_field_type: BackgroundFieldType,
    ) -> impl Iterator<Item = Vector3<T>> + 'a
    where
        K: SphericalKernel<T> + LocalKernel<T> + std::fmt::Debug + Copy + Sync + Send,
    {
        let bg = Self::compute_background_potential(q, view, kernel, bg_field_type);

        // Background potential Jacobian.
        let bg_jac = bg.compute_jacobian();

        let closest_d = bg.closest_sample_dist();
        let weight_sum_inv = bg.weight_sum_inv();

        // For each surface vertex contribution
        let main_jac = Self::sample_jacobian_at(q, view, kernel, bg);

        let third = T::from(1.0 / 3.0).unwrap();

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
        zip!(bg_jac, main_jac)
            .flat_map(move |(b, m)| std::iter::repeat(b + m).take(3))
            .zip(nml_jac)
            .map(move |(m, n)| m * third + n)
    }

    /// Compute the Jacobian of the potential field with respect to the given query point.
    pub(crate) fn query_jacobian_at<'a, K: 'a>(
        q: Vector3<T>,
        view: SamplesView<'a, 'a, T>,
        kernel: K,
        bg_field_type: BackgroundFieldType,
    ) -> Vector3<T>
    where
        K: SphericalKernel<T> + LocalKernel<T> + std::fmt::Debug + Copy + Sync + Send,
    {
        let bg = Self::compute_background_potential(q, view, kernel, bg_field_type);

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
                    Self::sample_contact_jacobian_at(
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
        let closest_d = bg.closest_sample_dist();

        // Background potential adds to the total weight sum, so we should get the updated weight
        // sum from there.
        let weight_sum_inv = bg.weight_sum_inv();
        let weight_sum_inv2 = weight_sum_inv * weight_sum_inv;

        samples.into_iter().map(
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

                let mut dw_neigh = T::zero();

                for Sample {
                    pos: posk,
                    nml: nmlk,
                    value: offk,
                    ..
                } in samples.iter()
                {
                    let wk = kernel.with_closest_dist(closest_d).eval(q, posk);
                    let diffk = q - posk;
                    let pk = T::from(offk).unwrap() + (nmlk.dot(diffk) / nmlk.norm());
                    dw_neigh -= wk * pk;
                }

                let dw = -kernel.with_closest_dist(closest_d).grad(q, pos);
                let mut dw_p = dw * (dw_neigh * weight_sum_inv2);

                // Contribution from the background potential
                let dwb = bg.background_weight_gradient(Some(index));
                dw_p += dwb * (dw_neigh * weight_sum_inv2);

                dw_p += dw * (weight_sum_inv * (T::from(value).unwrap() + unit_nml.dot(diff)));

                // Compute the normal component of the derivative
                let w = kernel.with_closest_dist(closest_d).eval(q, pos);
                let nml_deriv = unit_nml * (w * weight_sum_inv);
                dw_p - nml_deriv
            },
        )
    }

    /// Compute the Jacobian for the implicit surface potential for the given sample with the
    /// specified kernel. This Jacobian is with respect to the query
    /// point `q`, but this is not the complete Jacobian, this implementation returns a mapping
    /// from sample space to physical space in a form of a `1`-by-`3` Jacobian vector where.
    /// The returned iterator returns only non-zero elements.  When
    /// using the unit normal as the multiplier and summing over all samples, this function
    /// produces the true Jacobian of the potential with respect to the query point.
    pub(crate) fn sample_contact_jacobian_at<'a, K: 'a>(
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
        let mut dw_p = dw * weight_sum_inv;
        dw_p -= dw_neigh_normalized * (w * weight_sum_inv);
        dw_p *= T::from(sample_value).unwrap() + multiplier.dot(q - sample_pos);

        // Compute the normal component of the derivative
        dw_p + multiplier * (w * weight_sum_inv)
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
        V: Copy + Clone + std::fmt::Debug + PartialEq,
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

    /// Compute the contact Jacobian of this implicit surface function with respect to surface
    /// points.
    pub fn contact_jacobian_product(
        &self,
        query_points: &[[T; 3]],
        multipliers: &[[T; 3]],
        values: &mut [T],
    ) -> Result<(), Error> {
        let ImplicitSurface {
            ref kernel,
            ref spatial_tree,
            max_step,
            ..
        } = *self;

        match *kernel {
            KernelType::Approximate { tolerance, radius } => {
                let kernel = kernel::LocalApproximate::new(radius, tolerance);
                let radius_ext = radius + cast::<_, f64>(max_step).unwrap();
                let radius2 = radius_ext * radius_ext;
                let neigh = |q| {
                    let q_pos = Vector3(q).cast::<f64>().unwrap().into();
                    spatial_tree
                        .lookup_in_circle(&q_pos, &radius2)
                        .into_iter()
                        .cloned()
                };
                self.mls_contact_jacobian_product(query_points, multipliers, kernel, neigh, values);
                Ok(())
            }
            _ => Err(Error::UnsupportedKernel),
        }
    }

    /// Multiplier is a stacked velocity stored at samples.
    pub(crate) fn mls_contact_jacobian_product<'a, I, K, N>(
        &self,
        query_points: &[[T; 3]],
        multiplier: &[[T; 3]],
        kernel: K,
        neigh: N,
        values: &mut [T],
    ) where
        I: Iterator<Item = Sample<T>> + 'a,
        K: SphericalKernel<T> + LocalKernel<T> + std::fmt::Debug + Copy + Sync + Send,
        N: Fn([T; 3]) -> I + Sync + Send,
    {
        let value_vecs: &mut [[T; 3]] = reinterpret::reinterpret_mut_slice(values);

        let neigh_points = self.cached_neighbours_borrow(query_points, neigh);

        let ImplicitSurface {
            ref samples,
            bg_field_type,
            sample_type,
            ref surface_topo,
            ..
        } = *self;

        match sample_type {
            SampleType::Vertex => {
                // For each row (query point)
                let vtx_jac = zip!(query_points.iter(), neigh_points.iter())
                    .filter(|(_, nbrs)| !nbrs.is_empty())
                    .map(move |(q, nbr_points)| {
                        let view = SamplesView::new(nbr_points, samples);
                        Self::vertex_contact_jacobian_product_at(
                            Vector3(*q),
                            view,
                            multiplier,
                            kernel,
                            bg_field_type,
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
                let face_jac = zip!(query_points.iter(), neigh_points.iter())
                    .filter(|(_, nbrs)| !nbrs.is_empty())
                    .map(move |(q, nbr_points)| {
                        let view = SamplesView::new(nbr_points, samples);
                        Self::face_contact_jacobian_product_at(
                            Vector3(*q),
                            view,
                            multiplier,
                            kernel,
                            bg_field_type,
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
        bg_field_type: BackgroundFieldType,
    ) -> Vector3<T>
    where
        K: SphericalKernel<T> + LocalKernel<T> + std::fmt::Debug + Copy + Sync + Send,
    {
        let bg = Self::compute_background_potential(q, samples, kernel, bg_field_type);

        let weight_sum_inv = bg.weight_sum_inv();
        let closest_d = bg.closest_sample_dist();

        let bg_jac = bg.compute_query_jacobian();

        let dw_neigh = Self::normalized_neighbour_weight_gradient(q, samples, kernel, bg);

        let jac = samples
            .into_iter()
            .map(
                move |Sample {
                          index, pos, value, ..
                      }| {
                    let mult = sample_multipliers[index].into();
                    Self::sample_contact_jacobian_at(
                        q,
                        pos,
                        value,
                        kernel,
                        mult,
                        dw_neigh,
                        weight_sum_inv,
                        closest_d,
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
        bg_field_type: BackgroundFieldType,
        triangles: &'a [[usize; 3]],
    ) -> Vector3<T>
    where
        K: SphericalKernel<T> + LocalKernel<T> + std::fmt::Debug + Copy + Sync + Send,
    {
        let bg = Self::compute_background_potential(q, samples, kernel, bg_field_type);

        let weight_sum_inv = bg.weight_sum_inv();
        let closest_d = bg.closest_sample_dist();

        let bg_jac = bg.compute_query_jacobian();

        let dw_neigh = Self::normalized_neighbour_weight_gradient(q, samples, kernel, bg);

        let jac = samples
            .into_iter()
            .map(
                move |Sample {
                          index, pos, value, ..
                      }| {
                    let mult = (0..3).fold(Vector3::zeros(), |acc, i| {
                        acc + vertex_multipliers[triangles[index][i]].into()
                    }) / T::from(3.0).unwrap();
                    Self::sample_contact_jacobian_at(
                        q,
                        pos,
                        value,
                        kernel,
                        mult,
                        dw_neigh,
                        weight_sum_inv,
                        closest_d,
                    )
                },
            )
            .sum::<Vector3<T>>();
        jac + bg_jac
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use autodiff::F;

    fn easy_potential_derivative(radius: f64, bg_field_type: BackgroundFieldType) {
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
        let jac: Vec<Vector3<F>> = ImplicitSurface::vertex_jacobian_at(
            q,
            view,
            kernel,
            &surf_topo,
            &dual_topo,
            bg_field_type,
        )
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
            ImplicitSurface::compute_local_potential_at(
                q,
                view,
                F::cst(radius),
                kernel,
                bg_field_type,
                &mut p,
            );

            // Check the derivative of the autodiff with our previously computed Jacobian.
            assert_relative_eq!(
                jac[0][i].value(),
                p.deriv(),
                max_relative = 1e-6,
                epsilon = 1e-12
            );

            // Reset the variable back to being a constant.
            samples.points[0][i] = F::cst(samples.points[0][i]);
        }
    }

    #[test]
    fn easy_potential_derivative_test() {
        for i in 1..50 {
            let radius = 0.1 * (i as f64);
            easy_potential_derivative(radius, BackgroundFieldType::None);
            easy_potential_derivative(radius, BackgroundFieldType::Zero);
            easy_potential_derivative(radius, BackgroundFieldType::FromInput);
            easy_potential_derivative(radius, BackgroundFieldType::DistanceBased);
            easy_potential_derivative(radius, BackgroundFieldType::NormalBased);
        }
    }

    fn easy_potential_hessian(radius: f64, bg_field_type: BackgroundFieldType) {
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
        let jac: Vec<Vector3<F>> = ImplicitSurface::vertex_jacobian_at(
            q,
            view,
            kernel,
            &surf_topo,
            &dual_topo,
            bg_field_type,
        )
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
            ImplicitSurface::compute_local_potential_at(
                q,
                view,
                F::cst(radius),
                kernel,
                bg_field_type,
                &mut p,
            );

            // Check the derivative of the autodiff with our previously computed Jacobian.
            assert_relative_eq!(
                jac[0][i].value(),
                p.deriv(),
                max_relative = 1e-6,
                epsilon = 1e-12
            );

            // Reset the variable back to being a constant.
            samples.points[0][i] = F::cst(samples.points[0][i]);
        }
    }

    #[test]
    fn easy_potential_hessian_test() {
        for i in 1..50 {
            let radius = 0.1 * (i as f64);
            easy_potential_hessian(radius, BackgroundFieldType::None);
            easy_potential_hessian(radius, BackgroundFieldType::Zero);
            easy_potential_hessian(radius, BackgroundFieldType::FromInput);
            easy_potential_hessian(radius, BackgroundFieldType::DistanceBased);
            easy_potential_hessian(radius, BackgroundFieldType::NormalBased);
        }
    }

    /// Convert samples to autodiff number type constants.
    fn samples_to_autodiff(samples: Samples<f64>) -> Samples<F> {
        let Samples {
            points,
            normals,
            velocities,
            values,
        } = samples;

        Samples {
            points: points
                .into_iter()
                .map(|vec| vec.map(|x| F::cst(x)))
                .collect(),
            normals: normals
                .into_iter()
                .map(|vec| vec.map(|x| F::cst(x)))
                .collect(),
            velocities: velocities
                .into_iter()
                .map(|vec| vec.map(|x| F::cst(x)))
                .collect(),
            values: values.into_iter().map(|x| F::cst(x)).collect(),
        }
    }

    /// A more complex test parametrized by the background potential choice, radius and a perturbation
    /// function that is expected to generate a random perturbation at every consequent call.
    /// This function tests vertex based implicit surfaaces.
    fn hard_vertex_potential_derivative<P: FnMut() -> Vector3<f64>>(
        bg_field_type: BackgroundFieldType,
        radius: f64,
        perturb: &mut P,
    ) {
        // This is a similar test to the one above, but has a non-trivial surface topology for the
        // surface.

        let h = 1.18032;
        let tri_verts = vec![
            Vector3([0.5, h, 0.0]) + perturb(),
            Vector3([-0.25, h, 0.433013]) + perturb(),
            Vector3([-0.25, h, -0.433013]) + perturb(),
        ];

        let (tet_verts, tet_faces) = make_tet();

        let dual_topo = ImplicitSurfaceBuilder::compute_dual_topo(tet_verts.len(), &tet_faces);

        // Initialize the samples with regular f64 for now to keep debug output clean.
        // Compute normals. Make sure this is done the same way as everywhere else.
        let mut normals = vec![Vector3::zeros(); tet_verts.len()];
        ImplicitSurface::compute_vertex_area_normals(&tet_faces, &tet_verts, &mut normals);

        let samples = Samples {
            points: tet_verts.clone(),
            normals: normals.clone(),
            velocities: vec![Vector3::zeros(); 4],
            values: vec![0.0; 4],
        };

        let neighbours = vec![0, 1, 2, 3];

        let kernel = kernel::LocalApproximate::new(radius, 1e-5);

        // Convert the samples to use autodiff constants.
        let mut ad_samples = samples_to_autodiff(samples.clone());

        for &q in tri_verts.iter() {
            // Compute the Jacobian.
            let view = SamplesView::new(neighbours.as_ref(), &samples);
            let jac: Vec<Vector3<f64>> = ImplicitSurface::vertex_jacobian_at(
                q,
                view,
                kernel,
                &tet_faces,
                &dual_topo,
                bg_field_type,
            )
            .collect();

            assert_eq!(jac.len(), neighbours.len());

            let q = q.map(|x| F::cst(x));

            for &vtx in neighbours.iter() {
                for i in 0..3 {
                    ad_samples.points[vtx][i] = F::var(ad_samples.points[vtx][i]);

                    // Compute normals. This is necessary to capture the normal derivatives.
                    ImplicitSurface::compute_vertex_area_normals(
                        &tet_faces,
                        &ad_samples.points,
                        &mut ad_samples.normals,
                    );

                    let view = SamplesView::new(neighbours.as_ref(), &ad_samples);
                    let mut p = F::cst(0.0);
                    ImplicitSurface::compute_local_potential_at(
                        q,
                        view,
                        F::cst(radius),
                        kernel,
                        bg_field_type,
                        &mut p,
                    );

                    assert_relative_eq!(
                        jac[vtx][i],
                        p.deriv(),
                        max_relative = 1e-5,
                        epsilon = 1e-10
                    );

                    ad_samples.points[vtx][i] = F::cst(ad_samples.points[vtx][i]);
                }
            }
        }
    }

    /// A more complex test parametrized by the background potential choice, radius and a perturbation
    /// function that is expected to generate a random perturbation at every consequent call.
    /// This function tests face centric implicit surfaces.
    fn hard_face_potential_derivative<P: FnMut() -> Vector3<f64>>(
        bg_field_type: BackgroundFieldType,
        radius: f64,
        perturb: &mut P,
    ) {
        // This is a similar test to the one above, but has a non-trivial surface topology for the
        // surface.

        let h = 1.18032;
        let tri_verts = vec![
            Vector3([0.5, h, 0.0]) + perturb(),
            Vector3([-0.25, h, 0.433013]) + perturb(),
            Vector3([-0.25, h, -0.433013]) + perturb(),
        ];

        let (tet_verts, tet_faces) = make_tet();

        let samples = Samples::new_triangle_samples(&tet_faces, &tet_verts, vec![0.0; 4]);

        let neighbours = vec![0, 1, 2, 3]; // All tet faces

        let kernel = kernel::LocalApproximate::new(radius, 1e-5);

        // Convert tet vertices into varibales because we are taking the derivative with respect to
        // vertices.
        let mut ad_tet_verts: Vec<Vector3<F>> = tet_verts
            .iter()
            .cloned()
            .map(|v| v.map(|x| F::cst(x)))
            .collect();

        for &q in tri_verts.iter() {
            // Compute the Jacobian.
            let view = SamplesView::new(neighbours.as_ref(), &samples);
            let jac: Vec<Vector3<f64>> = ImplicitSurface::face_jacobian_at(
                q,
                view,
                kernel,
                &tet_faces,
                &tet_verts,
                bg_field_type,
            )
            .collect();

            assert_eq!(jac.len(), 3 * neighbours.len());

            // Reduce the Jacobian from face vertices to vertices.
            let tet_indices: &[usize] = reinterpret::reinterpret_slice(&tet_faces);
            let mut vert_jac = vec![Vector3::zeros(); tet_verts.len()];
            for (&jac, &vtx_idx) in jac.iter().zip(tet_indices) {
                vert_jac[vtx_idx] += jac;
            }

            assert_eq!(vert_jac.len(), tet_verts.len());

            let q = q.map(|x| F::cst(x));

            for &vtx in neighbours.iter() {
                for i in 0..3 {
                    ad_tet_verts[vtx][i] = F::var(ad_tet_verts[vtx][i]);

                    let ad_samples = Samples::new_triangle_samples(
                        &tet_faces,
                        &ad_tet_verts,
                        vec![F::cst(0.0); 4],
                    );

                    let view = SamplesView::new(neighbours.as_ref(), &ad_samples);
                    let mut p = F::cst(0.0);
                    ImplicitSurface::compute_local_potential_at(
                        q,
                        view,
                        F::cst(radius),
                        kernel,
                        bg_field_type,
                        &mut p,
                    );

                    assert_relative_eq!(
                        vert_jac[vtx][i],
                        p.deriv(),
                        max_relative = 1e-5,
                        epsilon = 1e-10
                    );

                    ad_tet_verts[vtx][i] = F::cst(ad_tet_verts[vtx][i]);
                }
            }
        }
    }

    #[test]
    fn hard_potential_derivative_test() {
        use rand::{distributions::Uniform, Rng, SeedableRng, StdRng};

        let mut rng: StdRng = SeedableRng::from_seed([3; 32]);
        let range = Uniform::new(-0.1, 0.1);

        let mut perturb = || Vector3([rng.sample(range), rng.sample(range), rng.sample(range)]);

        // Run for some number of perturbations
        for i in 1..50 {
            let radius = 0.1 * (i as f64);
            hard_vertex_potential_derivative(BackgroundFieldType::None, radius, &mut perturb);
            hard_vertex_potential_derivative(BackgroundFieldType::Zero, radius, &mut perturb);
            hard_vertex_potential_derivative(BackgroundFieldType::FromInput, radius, &mut perturb);
            hard_vertex_potential_derivative(
                BackgroundFieldType::DistanceBased,
                radius,
                &mut perturb,
            );
            hard_vertex_potential_derivative(
                BackgroundFieldType::NormalBased,
                radius,
                &mut perturb,
            );

            hard_face_potential_derivative(BackgroundFieldType::None, radius, &mut perturb);
            hard_face_potential_derivative(BackgroundFieldType::Zero, radius, &mut perturb);
            hard_face_potential_derivative(BackgroundFieldType::FromInput, radius, &mut perturb);
            hard_face_potential_derivative(
                BackgroundFieldType::DistanceBased,
                radius,
                &mut perturb,
            );
            hard_face_potential_derivative(BackgroundFieldType::NormalBased, radius, &mut perturb);
        }
    }

    /// Compute normalized area weighted vertex normals given a triangle topology.
    /// This is a helper function for the `vertex_normal_derivative_test`.
    /// Note that it is strictly more useful to precompute unnormalized vertex normals because they
    /// cary more information like area.
    pub(crate) fn compute_vertex_unit_normals<T: Real + Send + Sync>(
        surf_topo: &[[usize; 3]],
        points: &[Vector3<T>],
        normals: &mut [Vector3<T>],
    ) {
        // Compute area normals.
        ImplicitSurface::compute_vertex_area_normals(surf_topo, points, normals);

        // Normalize.
        for nml in normals.iter_mut() {
            *nml = *nml / nml.norm();
        }
    }

    /// Generate a tetrahedron with vertex positions and indices for the triangle faces.
    fn make_tet() -> (Vec<Vector3<f64>>, Vec<[usize; 3]>) {
        use geo::mesh::TriMesh;
        use utils::*;
        let tet = make_regular_tet();
        let TriMesh {
            vertex_positions,
            indices,
            ..
        } = TriMesh::from(tet);
        let tet_verts = vertex_positions.into_iter().map(|x| Vector3(x)).collect();
        let tet_faces = reinterpret::reinterpret_vec(indices.into_vec());

        (tet_verts, tet_faces)
    }

    /// Compute the face normal derivative with respect to tet vertices.
    fn compute_face_unit_normal_derivative<T: Real + Send + Sync>(
        tet_verts: &[Vector3<T>],
        tet_faces: &[[usize; 3]],
        view: SamplesView<'_, '_, T>,
        multiplier: impl FnMut(Sample<T>) -> Vector3<T>,
    ) -> Vec<Vector3<T>> {
        // Compute the normal gradient product.
        let grad_iter = ImplicitSurface::compute_face_unit_normals_gradient_products(
            view, tet_verts, tet_faces, multiplier,
        );

        // Convert to grad wrt tet vertex indices instead of surface triangle vertex indices.
        let tet_indices: &[usize] = reinterpret::reinterpret_slice(&tet_faces);
        let mut vert_grad = vec![Vector3::zeros(); tet_verts.len()];
        for (g, &vtx_idx) in grad_iter.zip(tet_indices) {
            vert_grad[vtx_idx] += g;
        }

        vert_grad
    }

    /// Generate a random vector of `Vector3` multipliers.
    fn random_vectors(n: usize) -> Vec<Vector3<f64>> {
        use rand::{distributions::Uniform, Rng, SeedableRng, StdRng};
        let mut rng: StdRng = SeedableRng::from_seed([3; 32]);
        let range = Uniform::new(-1.0, 1.0);
        (0..n)
            .map(move |_| Vector3([rng.sample(range), rng.sample(range), rng.sample(range)]))
            .collect()
    }

    /// Test the first order derivatives of our normal computation method for face normals.
    #[test]
    fn face_normal_derivative_test() {
        let (tet_verts, tet_faces) = make_tet();

        // Initialize the samples with regular f64 for now to keep debug output clean.
        let samples = Samples::new_triangle_samples(&tet_faces, &tet_verts, vec![0.0; 4]);

        let indices = vec![0, 1, 2, 3]; // look at all the faces

        // Set a random product vector.
        let multipliers = random_vectors(tet_faces.len());
        let multiplier = move |Sample { index, .. }| multipliers[index];

        let view = SamplesView::new(indices.as_ref(), &samples);
        let vert_grad =
            compute_face_unit_normal_derivative(&tet_verts, &tet_faces, view, multiplier.clone());

        // Convert tet vertices into varibales because we are taking the derivative with respect to
        // vertices.
        let mut ad_tet_verts: Vec<Vector3<F>> = tet_verts
            .iter()
            .cloned()
            .map(|v| v.map(|x| F::cst(x)))
            .collect();

        for (vtx, g) in vert_grad.iter().enumerate() {
            for i in 0..3 {
                ad_tet_verts[vtx][i] = F::var(ad_tet_verts[vtx][i]);

                // Compute autodiff samples.
                let mut ad_samples =
                    Samples::new_triangle_samples(&tet_faces, &ad_tet_verts, vec![F::cst(0.0); 4]);

                // Normalize face normals
                for nml in ad_samples.normals.iter_mut() {
                    *nml = *nml / nml.norm();
                }

                let mut exp = F::cst(0.0);
                for sample in view.clone().iter() {
                    exp +=
                        ad_samples.normals[sample.index].dot(multiplier(sample).map(|x| F::cst(x)));
                }

                assert_relative_eq!(g[i], exp.deriv(), max_relative = 1e-5, epsilon = 1e-10);

                ad_tet_verts[vtx][i] = F::cst(ad_tet_verts[vtx][i]);
            }
        }
    }

    /// Test the second order derivatives of our normal computation method for face normals.
    #[test]
    fn face_normal_hessian_test() {
        use geo::math::Matrix12;

        let (tet_verts, tet_faces) = make_tet();

        // Initialize the samples with regular f64 for now to keep debug output clean.
        let samples = Samples::new_triangle_samples(&tet_faces, &tet_verts, vec![0.0; 4]);

        let indices = vec![0, 1, 2, 3]; // look at all the faces

        // Set a random product vector.
        let multipliers = random_vectors(tet_faces.len());
        let ad_multipliers: Vec<_> = multipliers.iter().map(|&v| v.map(|x| F::cst(x))).collect();

        let multiplier = move |Sample { index, .. }| multipliers[index];

        let ad_multiplier = move |Sample { index, .. }| ad_multipliers[index];

        // Compute the normal hessian product.
        let view = SamplesView::new(indices.as_ref(), &samples);
        let hess_iter = ImplicitSurface::compute_face_unit_normals_hessian_products(
            view,
            &tet_verts,
            &tet_faces,
            multiplier.clone(),
        );

        let mut num_hess_entries = 0;
        let mut hess: Matrix12<f64> = Matrix12::zeros(); // Dense matrix
        for (r, c, m) in hess_iter {
            // map to tet vertices instead of surface vertices
            for j in 0..3 {
                for i in 0..3 {
                    hess[3 * c + j][3 * r + i] += m[j][i];
                    if i >= j {
                        // Only record lower triangular non-zeros
                        num_hess_entries += 1;
                    }
                }
            }
        }

        assert_eq!(
            ImplicitSurface::<f64>::num_face_unit_normals_hessian_entries(samples.len()),
            num_hess_entries
        );

        // Convert tet vertices into varibales because we are taking the derivative with respect to
        // vertices.
        let mut ad_tet_verts: Vec<Vector3<F>> = tet_verts
            .iter()
            .cloned()
            .map(|v| v.map(|x| F::cst(x)))
            .collect();

        for r in 0..4 {
            for i in 0..3 {
                ad_tet_verts[r][i] = F::var(ad_tet_verts[r][i]);

                let ad_samples =
                    Samples::new_triangle_samples(&tet_faces, &ad_tet_verts, vec![F::cst(0.0); 4]);
                let ad_view = SamplesView::new(indices.as_ref(), &ad_samples);

                // Convert the samples to use autodiff constants.
                let grad = compute_face_unit_normal_derivative(
                    &ad_tet_verts,
                    &tet_faces,
                    ad_view,
                    ad_multiplier.clone(),
                );

                for c in 0..4 {
                    for j in 0..3 {
                        // Only check lower triangular part.
                        if 3 * c + j <= 3 * r + i {
                            assert_relative_eq!(
                                hess[3 * c + j][3 * r + i],
                                grad[c][j].deriv(),
                                max_relative = 1e-5,
                                epsilon = 1e-10
                            );
                            //println!("({:?}, {:?}) => {:?} vs {:?}", 3*r + i, 3*c + j, hess[3*c + j][3*r + i], grad[c][j].deriv());
                        }
                    }
                }

                ad_tet_verts[r][i] = F::cst(ad_tet_verts[r][i]);
            }
        }
    }

    /// Test the derivatives of our normal computation method for vertex normals.
    #[test]
    fn vertex_normal_derivative_test() {
        let (tet_verts, tet_faces) = make_tet();

        let mut normals = vec![Vector3::zeros(); tet_verts.len()];
        ImplicitSurface::compute_vertex_area_normals(
            tet_faces.as_slice(),
            tet_verts.as_slice(),
            &mut normals,
        );

        // Vertex to triangle map
        let dual_topo = ImplicitSurfaceBuilder::compute_dual_topo(tet_verts.len(), &tet_faces);

        // Initialize the samples with regular f64 for now to keep debug output clean.
        let samples = Samples {
            points: tet_verts.clone(),
            normals: normals.clone(),
            velocities: vec![Vector3::zeros(); tet_verts.len()],
            values: vec![0.0; 4], // This is not actually used in this test.
        };

        let indices = vec![0, 1, 2, 3]; // look at all the vertices

        // Convert the samples to use autodiff constants.
        let mut ad_samples = samples_to_autodiff(samples.clone());

        // Set a random product vector.
        let dxs = random_vectors(tet_verts.len());
        let dx = move |Sample { index, .. }| dxs[index];

        // Compute the normal gradient product.
        let view = SamplesView::new(indices.as_ref(), &samples);
        let grad_iter = ImplicitSurface::compute_vertex_unit_normals_gradient_products(
            view,
            &tet_faces,
            &dual_topo,
            dx.clone(),
        );

        for (&vtx, g) in indices.iter().zip(grad_iter) {
            for i in 0..3 {
                ad_samples.points[vtx][i] = F::var(ad_samples.points[vtx][i]);

                // Compute normalized normals. This is necessary to capture the normal derivatives.
                compute_vertex_unit_normals(
                    &tet_faces,
                    &ad_samples.points,
                    &mut ad_samples.normals,
                );

                let mut exp = F::cst(0.0);
                for sample in view.clone().iter() {
                    exp += ad_samples.normals[sample.index].dot(dx(sample).map(|x| F::cst(x)));
                }

                assert_relative_eq!(g[i], exp.deriv(), max_relative = 1e-5, epsilon = 1e-10);

                ad_samples.points[vtx][i] = F::cst(ad_samples.points[vtx][i]);
            }
        }
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

        let samples = Samples {
            points: points.clone(),
            normals: vec![Vector3::zeros(); points.len()], // Not used
            velocities: vec![Vector3::zeros(); points.len()], // Not used
            values: vec![0.0; points.len()],               // Not used
        };

        let indices: Vec<usize> = (0..points.len()).collect();

        let radius = 2.0;

        // Initialize kernel.
        let kernel = kernel::LocalApproximate::new(radius, 1e-5);

        // Create a view to the data to be iterated.
        let view = SamplesView::new(indices.as_slice(), &samples);

        // Initialize a background potential. This function takes care of a lot of the setup.
        let bg = BackgroundField::new(
            q,
            view,
            radius,
            kernel,
            BackgroundFieldValue::jac(BackgroundFieldType::DistanceBased),
        );

        // Compute manual Jacobian. This is the function being tested for correctness.
        let jac: Vec<_> = bg.compute_jacobian().collect();

        // Prepare autodiff variables.
        let mut ad_samples = Samples {
            points: points.iter().map(|&pos| pos.map(|x| F::cst(x))).collect(),
            normals: vec![Vector3::zeros(); points.len()], // Not used
            velocities: vec![Vector3::zeros(); points.len()], // Not used
            values: vec![F::cst(0.0); points.len()],       // Not used
        };

        let q = q.map(|x| F::cst(x));

        // Perform the derivative test on each of the variables.
        for i in 0..points.len() {
            for j in 0..3 {
                ad_samples.points[i][j] = F::var(ad_samples.points[i][j]);

                // Initialize an autodiff version of the potential.
                // This should be done outside the inner loop over samples, but here we make an
                // exception for simplicity.
                let view = SamplesView::new(indices.as_slice(), &ad_samples);
                let ad_bg = BackgroundField::new(
                    q,
                    view,
                    F::cst(radius),
                    kernel,
                    BackgroundFieldValue::val(BackgroundFieldType::DistanceBased, F::cst(0.0)),
                );

                let p = ad_bg.compute_unnormalized_weighted_scalar_field() * ad_bg.weight_sum_inv();

                assert_relative_eq!(jac[i][j], p.deriv());
                ad_samples.points[i][j] = F::cst(ad_samples.points[i][j]);
            }
        }
    }

    /// A test parametrized by the background potential choice, radius and a perturbation
    /// function that is expected to generate a random perturbation at every consequent call.
    /// This function tests the query Jacobian of the implicit function.
    pub fn query_jacobian<P: FnMut() -> Vector3<f64>>(
        bg_field_type: BackgroundFieldType,
        radius: f64,
        perturb: &mut P,
    ) {
        let h = 1.18032;
        let tri_verts = vec![
            Vector3([0.5, h, 0.0]) + perturb(),
            Vector3([-0.25, h, 0.433013]) + perturb(),
            Vector3([-0.25, h, -0.433013]) + perturb(),
        ];

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
            let jac = ImplicitSurface::query_jacobian_at(q, view, kernel, bg_field_type);

            let mut q = q.map(|x| F::cst(x));

            for i in 0..3 {
                q[i] = F::var(q[i]);

                let ad_samples =
                    Samples::new_triangle_samples(&tet_faces, &ad_tet_verts, vec![F::cst(0.0); 4]);

                let view = SamplesView::new(neighbours.as_ref(), &ad_samples);

                let mut p = F::cst(0.0);
                ImplicitSurface::compute_local_potential_at(
                    q,
                    view,
                    F::cst(radius),
                    kernel,
                    bg_field_type,
                    &mut p,
                );

                assert_relative_eq!(jac[i], p.deriv(), max_relative = 1e-5, epsilon = 1e-10);

                q[i] = F::cst(q[i]);
            }
        }
    }

    #[test]
    fn query_jacobian_test() {
        use rand::{distributions::Uniform, Rng, SeedableRng, StdRng};

        let mut rng: StdRng = SeedableRng::from_seed([3; 32]);
        let range = Uniform::new(-0.1, 0.1);

        let mut perturb = || Vector3([rng.sample(range), rng.sample(range), rng.sample(range)]);

        // Run for some number of perturbations
        for i in 1..50 {
            let radius = 0.1 * (i as f64);
            query_jacobian(BackgroundFieldType::None, radius, &mut perturb);
            query_jacobian(BackgroundFieldType::Zero, radius, &mut perturb);
            query_jacobian(BackgroundFieldType::FromInput, radius, &mut perturb);
            query_jacobian(BackgroundFieldType::DistanceBased, radius, &mut perturb);
            query_jacobian(BackgroundFieldType::NormalBased, radius, &mut perturb);
        }
    }

    /// Tester for the contact jacobian. This tester is parameterized by background field type,
    /// radius and a perturb function.
    fn contact_jacobian<P: FnMut() -> Vector3<f64>>(
        bg_field_type: BackgroundFieldType,
        radius: f64,
        perturb: &mut P,
    ) {
        use crate::*;
        use geo::NumVertices;
        use utils::*;

        let h = 1.18032;
        let tri_vert_pos = vec![
            Vector3([0.5, h, 0.0]) + perturb(),
            Vector3([-0.25, h, 0.433013]) + perturb(),
            Vector3([-0.25, h, -0.433013]) + perturb(),
        ];

        let tri_verts: Vec<[f64; 3]> = reinterpret::reinterpret_vec(tri_vert_pos);

        let mut tet = make_regular_tet();

        let multiplier_vecs = random_vectors(tet.num_vertices());
        let multipliers_f32: Vec<_> = multiplier_vecs
            .iter()
            .cloned()
            .map(|v| v.map(|x| x as f32).into())
            .collect();
        tet.set_attrib_data::<[f32; 3], VertexIndex>("V", &multipliers_f32)
            .unwrap();

        let surf_params = Params {
            kernel: kernel::KernelType::Approximate {
                radius,
                tolerance: 1e-5,
            },
            background_field: bg_field_type,
            sample_type: SampleType::Face,
            max_step: 100.0 * radius, // essentially unlimited
        };

        let trimesh = geo::mesh::TriMesh::from(tet);

        let multipliers: Vec<_> = trimesh
            .attrib_as_slice::<[f32; 3], VertexIndex>("V")
            .unwrap()
            .iter()
            .map(|&x| Vector3(x).map(|x| f64::from(x)).into_inner())
            .collect();
        let surf = surface_from_trimesh(&trimesh, surf_params).unwrap();

        let mut jac_prod_values = vec![0.0; 3 * tri_verts.len()];

        // Compute the Jacobian.
        assert!(surf
            .contact_jacobian_product(&tri_verts, &multipliers, &mut jac_prod_values,)
            .is_ok());

        let jac_prod: Vec<[f64; 3]> = reinterpret::reinterpret_vec(jac_prod_values);

        let mut expected = vec![[0.0; 3]; tri_verts.len()];
        surf.vector_field(&tri_verts, &mut expected).unwrap();
        for (jac, exp) in jac_prod.into_iter().zip(expected.into_iter()) {
            for i in 0..3 {
                assert_relative_eq!(jac[i], exp[i], max_relative = 1e-5, epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn contact_jacobian_test() {
        use rand::{distributions::Uniform, Rng, SeedableRng, StdRng};

        let mut rng: StdRng = SeedableRng::from_seed([3; 32]);
        let range = Uniform::new(-0.1, 0.1);

        let mut perturb = || Vector3([rng.sample(range), rng.sample(range), rng.sample(range)]);

        // Run for some number of perturbations
        for i in 1..50 {
            let radius = 0.1 * (i as f64);
            contact_jacobian(BackgroundFieldType::None, radius, &mut perturb);
            //contact_jacobian(BackgroundFieldType::Zero, radius, &mut perturb);
            //contact_jacobian(
            //    BackgroundFieldType::FromInput,
            //    radius,
            //    &mut perturb,
            //);
            //contact_jacobian(
            //    BackgroundFieldType::DistanceBased,
            //    radius,
            //    &mut perturb,
            //);
            //contact_jacobian(
            //    BackgroundFieldType::NormalBased,
            //    radius,
            //    &mut perturb,
            //);
        }
    }
}
