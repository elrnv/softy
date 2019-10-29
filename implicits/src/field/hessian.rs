use super::*;
use crate::Error;

/// Symmetric outer product of two vectors: a*b' + b*a'
pub(crate) fn sym_outer<T: Real>(a: Vector3<T>, b: Vector3<T>) -> Matrix3<T> {
    a * b.transpose() + b * a.transpose()
}

impl<T: Real> ImplicitSurface<T> {
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

impl<T: Real + Send + Sync> QueryTopo<T> {
    /*
     * Query Hessian
     */

    pub fn num_query_hessian_product_entries(&self) -> usize {
        self.num_neighbourhoods() * 6
    }

    pub fn query_hessian_product_indices_iter<'a>(
        &'a self,
    ) -> impl Iterator<Item = (usize, usize)> + 'a {
        self.trivial_neighbourhood_seq()
            .enumerate()
            .filter(move |(_, nbrs)| nbrs.len() != 0)
            .flat_map(move |(i, _)| {
                (0..3).flat_map(move |c| (c..3).map(move |r| (3 * i + r, 3 * i + c)))
            })
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
        K: SphericalKernel<T> + std::fmt::Debug + Copy + Sync + Send,
    {
        let neigh_points = self.trivial_neighbourhood_seq();

        let ImplicitSurfaceBase {
            ref samples,
            bg_field_params,
            ..
        } = *self.base();

        // For each row (query point)
        let hess_iter = zip!(query_points.iter(), neigh_points)
            .filter(|(_, nbrs)| !nbrs.is_empty())
            .zip(multipliers.iter())
            .map(move |((q, nbr_points), &mult)| {
                let view = SamplesView::new(nbr_points, samples);
                query_hessian_at(Vector3(*q), view, kernel, bg_field_params) * mult
            });

        let value_mtxs: &mut [[T; 6]] = reinterpret::reinterpret_mut_slice(values);

        let mut count = 0;
        value_mtxs
            .iter_mut()
            .zip(hess_iter)
            .for_each(|(mtx, new_mtx)| {
                let mut i = 0;
                for (c, new_col) in new_mtx.into_inner().iter().enumerate() {
                    for &new_val in new_col.iter().skip(c) {
                        mtx[i] = new_val * scale;
                        i += 1;
                    }
                }
                count += 1;
            });

        // Ensure that all values are filled
        debug_assert_eq!(value_mtxs.len(), count);
    }

    /*
     * Surface Hessian
     */

    /// Get the total number of entries for the sparse Hessian non-zeros. The Hessian is taken with
    /// respect to sample points. This estimate is based on the current neighbour data, which
    /// gives the number of query points, if the neighbourhood was not precomputed this function
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
    pub fn surface_hessian_product_indices_iter<'a>(
        &'a self,
    ) -> Result<impl Iterator<Item = (usize, usize)> + 'a, Error> {
        let neigh_points = self.trivial_neighbourhood_seq();

        let ImplicitSurfaceBase {
            ref surface_topo,
            sample_type,
            bg_field_params,
            ..
        } = *self.base();

        match sample_type {
            SampleType::Vertex => Err(Error::UnsupportedSampleType),
            SampleType::Face => Ok(neigh_points
                .flat_map(move |nbr_points| {
                    face_hessian_indices_iter(
                        nbr_points.iter().cloned(),
                        surface_topo,
                        bg_field_params.weighted,
                    )
                })
                .flat_map(move |(row, col)| {
                    (0..3).flat_map(move |r| (0..3).map(move |c| (3 * row + r, 3 * col + c)))
                })
                .filter(move |(row, col)| row >= col)),
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
        K: SphericalKernel<T> + std::fmt::Debug + Copy + Sync + Send,
    {
        let neigh_points = self.trivial_neighbourhood_seq();

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
                            Vector3(*q),
                            view,
                            kernel,
                            surface_topo,
                            surface_vertex_positions,
                            bg_field_params,
                            *lambda,
                        )
                    });

                values
                    .iter_mut()
                    .zip(face_hess.flat_map(move |(row, col, mtx)| {
                        (0..3).flat_map(move |r| {
                            (0..3)
                                .filter(move |c| 3 * row + r >= 3 * col + c)
                                .map(move |c| mtx[c][r])
                        })
                    }))
                    .for_each(|(val, new_val)| {
                        *val = new_val * scale;
                    });
                Ok(())
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
/// Compute the Jacobian of the potential field with respect to the given query point.
pub(crate) fn query_hessian_at<'a, T, K: 'a>(
    q: Vector3<T>,
    view: SamplesView<'a, 'a, T>,
    kernel: K,
    bg_field_params: BackgroundFieldParams,
) -> Matrix3<T>
where
    T: Real + Send + Sync,
    K: SphericalKernel<T> + std::fmt::Debug + Copy + Sync + Send,
{
    let bg: BackgroundField<T, T, K> =
        BackgroundField::local(q, view, kernel, bg_field_params, None).unwrap();

    // Background potential Jacobian.
    let closest_d = bg.closest_sample_dist();
    let weight_sum_inv = bg.weight_sum_inv();

    let bg_hess = bg.compute_query_hessian();

    // For each surface vertex contribution
    let dw_neigh = jacobian::normalized_neighbour_weight_gradient(q, view, kernel, bg.clone());
    let ddw_neigh = normalized_neighbour_weight_hessian(q, view, kernel, bg);

    // For vectors a and b, this computes a*b' + b*a'.
    let sym_outer = |a: Vector3<T>, b: Vector3<T>| a * b.transpose() + b * a.transpose();

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
pub(crate) fn normalized_neighbour_weight_hessian<'a, T, K, V>(
    q: Vector3<T>,
    samples: SamplesView<'a, 'a, T>,
    kernel: K,
    bg: BackgroundField<'a, T, V, K>,
) -> Matrix3<T>
where
    T: Real + Send + Sync,
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

    ddw_neigh * weight_sum_inv // normalize the neighbourhood derivative
}

/*
 * Surface hessian components
 */

pub(crate) fn face_hessian_at<'a, T, K: 'a>(
    q: Vector3<T>,
    view: SamplesView<'a, 'a, T>,
    kernel: K,
    surface_topo: &'a [[usize; 3]],
    surface_vertex_positions: &'a [Vector3<T>],
    bg_field_params: BackgroundFieldParams,
    multiplier: T,
) -> impl Iterator<Item = (usize, usize, Matrix3<T>)> + 'a
where
    T: Real + Send + Sync,
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
    let nml_hess = normal_hessian_at(q, view, kernel, &surface_topo, surface_vertex_positions, bg);

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
    T: Real + Send + Sync,
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
    surface_vertex_positions: &'a [Vector3<T>],
    bg: BackgroundField<'a, T, T, K>,
) -> impl Iterator<Item = (usize, usize, Matrix3<T>)> + 'a
where
    T: Real + Send + Sync,
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
        &surface_vertex_positions,
        &surface_topo,
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

        face_unit_normal_gradient_iter(sample_l, surface_vertex_positions, surface_topo)
            .enumerate()
            .flat_map(move |(i, dn_li)| {
                let row_vtx = surface_topo[index_l][i];
                (0..3).map(move |j| {
                    let col_vtx = surface_topo[index_l][j];

                    let mtx = (dn_li * (q - pos_l) * dw_l.transpose()) * (ws_inv * third);

                    if row_vtx > col_vtx {
                        (row_vtx, col_vtx, mtx)
                    } else if row_vtx < col_vtx {
                        (col_vtx, row_vtx, mtx.transpose())
                    } else {
                        (row_vtx, col_vtx, mtx + mtx.transpose())
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
                face_unit_normal_gradient_iter(sample_k, surface_vertex_positions, surface_topo)
                    .enumerate()
                    .flat_map(move |(i, dn_ki)| {
                        let row_vtx = surface_topo[index_k][i];
                        (0..3).map(move |j| {
                            let col_vtx = surface_topo[index_l][j];

                            let mtx = -dn_ki
                                * (q - pos_k)
                                * dws_l.transpose()
                                * (wk * ws_inv * ws_inv * third);

                            if row_vtx > col_vtx {
                                (row_vtx, col_vtx, mtx)
                            } else if row_vtx < col_vtx {
                                (col_vtx, row_vtx, mtx.transpose())
                            } else {
                                (row_vtx, col_vtx, mtx + mtx.transpose())
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
    surface_vertices: &'a [Vector3<T>],
    surface_topo: &'a [[usize; 3]],
    mut multiplier: F,
) -> impl Iterator<Item = (usize, usize, Matrix3<T>)> + 'a
where
    T: Real + Send + Sync,
    F: FnMut(Sample<T>) -> T + 'a,
{
    let third = T::one() / T::from(3.0).unwrap();
    samples.into_iter().flat_map(move |sample| {
        let tri_indices = &surface_topo[sample.index];
        let nml_proj = scaled_tangent_projection(sample); // symmetric
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
                    let mtx = tri.area_normal_gradient(k) * nml_proj
                        - nml_proj * tri.area_normal_gradient(l);
                    (vtx_row, vtx_col, mtx * (lambda * third))
                })
        })
    })
}

/// Block lower triangular part of the unit normal Hessian multiplied by the given multiplier.
pub(crate) fn compute_face_unit_normals_hessian_products<'a, T, F>(
    samples: SamplesView<'a, 'a, T>,
    surface_vertices: &'a [Vector3<T>],
    surface_topo: &'a [[usize; 3]],
    mut multiplier: F,
) -> impl Iterator<Item = (usize, usize, Matrix3<T>)> + 'a
where
    T: Real + Send + Sync,
    F: FnMut(Sample<T>) -> Vector3<T> + 'a,
{
    // For each triangle contribution (one element in a sum)
    samples.into_iter().flat_map(move |sample| {
        let tri_indices = &surface_topo[sample.index];
        let norm_inv = T::one() / sample.nml.norm();
        let nml = sample.nml * norm_inv;
        let nml_proj = scaled_tangent_projection(sample);
        let mult = multiplier(sample);
        let tri = Triangle::from_indexed_slice(tri_indices, surface_vertices);
        let grad = [
            tri.area_normal_gradient(0),
            tri.area_normal_gradient(1),
            tri.area_normal_gradient(2),
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
                    let m = Triangle::area_normal_hessian_product(j, i, proj_mult)
                        + (grad[j] * nml_mult_prod * grad[i]) * norm_inv;
                    (vtx_row, vtx_col, m)
                })
        })
    })
}

/// Helper function to print the dense hessian given by a vector of vectors.
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
    use crate::kernel;
    use autodiff::F;
    use geo::mesh::{TriMesh, VertexPositions};
    use jacobian::{
        consolidate_face_jacobian, make_perturb_fn, make_test_triangle, make_three_test_triangles,
        make_two_test_triangles, new_test_samples,
    };

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
        };

        let surf = crate::mls_from_trimesh::<F>(&surf_mesh, params)
            .expect("Failed to construct an implicit surface.");

        let mut ad_tri_verts: Vec<_> = surf_mesh
            .vertex_position_iter()
            .map(|&x| Vector3(x).map(|x| F::cst(x)).into_inner())
            .collect();
        let num_verts = ad_tri_verts.len();
        let ad_query_points: Vec<_> = query_points
            .iter()
            .map(|&a| Vector3(a).map(|x| F::cst(x)).into_inner())
            .collect();
        let num_query_points = query_points.len();

        let mut query_surf = surf.query_topo(&ad_query_points);
        let num_hess_entries = query_surf
            .num_surface_hessian_product_entries()
            .expect("Uncached query points.");
        let num_jac_entries = query_surf.num_surface_jacobian_entries();

        // Compute the complete hessian.
        let mut hess_rows = vec![0; num_hess_entries];
        let mut hess_cols = vec![0; num_hess_entries];
        let mut hess_values = vec![F::cst(0.0); num_hess_entries];
        let mut jac_rows = vec![0; num_jac_entries];
        let mut jac_cols = vec![0; num_jac_entries];
        let mut jac_values = vec![F::cst(0.0); num_jac_entries];

        let num_neighs = query_surf.num_neighbourhoods();
        let mut multipliers = vec![F::cst(0.0); num_neighs];
        query_surf
            .surface_hessian_product_indices(&mut hess_rows, &mut hess_cols)
            .expect("Failed to compute hessian indices");

        let mut hess_full = vec![vec![0.0; 3 * num_verts]; 3 * num_verts];
        let mut ad_hess_full = vec![vec![0.0; 3 * num_verts]; 3 * num_verts];

        let query_neighbourhood_sizes = query_surf.neighbourhood_sizes();
        dbg!(&query_neighbourhood_sizes);

        // We use the multipliers to isolate the hessian for each query point.
        for (mult_idx, q_idx) in (0..num_query_points)
            .filter(|&q_idx| query_neighbourhood_sizes[q_idx] != 0)
            .enumerate()
        {
            multipliers[mult_idx] = F::cst(1.0);

            query_surf
                .surface_hessian_product_values(&ad_query_points, &multipliers, &mut hess_values)
                .expect("Failed to compute hessian product");

            let mut success = true;

            // Test the accuracy of each component of the hessian against an autodiff version of the
            // second derivative.
            for vtx in 0..num_verts {
                for i in 0..3 {
                    // Set a variable to take the derivative with respect to, using autodiff.
                    ad_tri_verts[vtx][i] = F::var(ad_tri_verts[vtx][i]);
                    query_surf.update_surface(ad_tri_verts.iter().cloned());

                    query_surf.surface_jacobian_values(&ad_query_points, &mut jac_values);
                    query_surf.surface_jacobian_indices(&mut jac_rows, &mut jac_cols);

                    // Get the jacobian for the specific query point we are interested in.
                    let mut jac_q = vec![F::cst(0.0); num_verts * 3];
                    for (&r, &c, &jac) in zip!(jac_rows.iter(), jac_cols.iter(), jac_values.iter())
                    {
                        if r == q_idx {
                            jac_q[c] += jac;
                        }
                    }

                    // Consolidate the Hessian to the particular vertex and component we are
                    // interested in.
                    let mut hess_vtx = vec![F::cst(0.0); num_verts * 3];
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
                    ad_tri_verts[vtx][i] = F::cst(ad_tri_verts[vtx][i]);
                }

                if !success {
                    print_full_hessian(&hess_full, 3 * num_verts, "Full Hessian");
                    print_full_hessian(&ad_hess_full, 3 * num_verts, "Full Autodiff Hessian");
                }
                assert!(success, "Hessian does not match its AutoDiff counterpart");
            }
            multipliers[mult_idx] = F::cst(0.0); // reset multiplier
        }
        Ok(())
    }

    /// Test the highest level surface Hessian functions with a tetrahedron.
    #[test]
    fn one_tet_hessian_test() -> Result<(), Error> {
        let qs: Vec<_> = (0..4)
            .map(|i| Vector3([0.0, -0.5 + 0.25 * i as f64, 0.0]).into_inner())
            .collect();

        let trimesh = TriMesh::from(utils::make_regular_tet());

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
            .map(|x| x.into_inner())
            .collect();
        let tri = geo::mesh::TriMesh::new(tri_verts, vec![0, 1, 2]);

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

    fn one_tet_face_hessian<P: FnMut() -> Vector3<f64>>(
        radius: f64,
        bg_field_params: BackgroundFieldParams,
        perturb: &mut P,
    ) {
        let tet = TriMesh::from(utils::make_regular_tet());
        let qs: Vec<_> = (0..4)
            .map(|i| Vector3([0.0, -0.5 + 0.25 * i as f64, 0.0]))
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
        let tri_verts: Vec<[f64; 3]> = reinterpret::reinterpret_vec(tri_verts);
        let tri_indices: Vec<usize> = reinterpret::reinterpret_vec(tri_indices);
        let tri = TriMesh::new(tri_verts, tri_indices);
        let qs = vec![
            Vector3([0.0, 0.2, 0.0]),
            Vector3([0.0, 0.0001, 0.0]),
            Vector3([0.0, -0.4, 0.0]),
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
            .map(|x| x.into_inner())
            .collect();
        let tri_indices = vec![0usize, 1, 2];
        let tri = TriMesh::new(tri_verts, tri_indices);
        let qs = vec![
            Vector3([0.0, 0.2, 0.0]),
            Vector3([0.0, 0.0001, 0.0]),
            Vector3([0.0, -0.4, 0.0]),
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
        let q = q.map(|x| F::cst(x)); // convert to autodiff
        let mut tri_verts: Vec<_> = mesh
            .vertex_position_iter()
            .map(|&x| Vector3(x).map(|x| F::cst(x)))
            .collect();
        let tri_faces = reinterpret::reinterpret_slice(mesh.indices.as_slice());
        let num_verts = tri_verts.len();

        let samples = new_test_samples(SampleType::Face, &tri_faces, &tri_verts);

        let neighbours: Vec<_> = samples
            .iter()
            .filter(|s| (q - s.pos).norm() < F::cst(radius + max_step))
            .map(|sample| sample.index)
            .collect();

        if neighbours.is_empty() {
            return;
        }

        // Radius is such that samples are captured by the query point.
        let kernel = kernel::LocalApproximate::new(radius, 0.00001);

        let view = SamplesView::new(neighbours.as_ref(), &samples);

        // Compute the complete hessian.
        let hess: Vec<(usize, usize, Matrix3<F>)> = face_hessian_at(
            q,
            view,
            kernel,
            tri_faces,
            &tri_verts,
            bg_field_params,
            F::cst(1.0),
        )
        .collect();

        let mut hess_full = vec![vec![0.0; 3 * num_verts]; 3 * num_verts];
        let mut hess_full2 = vec![vec![0.0; 3 * num_verts]; 3 * num_verts];

        for &(r, c, h) in hess.iter() {
            for j in 0..3 {
                for i in 0..3 {
                    hess_full[3 * c + j][3 * r + i] += h[j][i].value();
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
                tri_verts[vtx][i] = F::var(tri_verts[vtx][i]);
                //println!("row_vtx = {}; i = {}", vtx, i);

                // We need to update samples to make sure the normals and centroids are recomputed
                // using the correct wrt autodiff variable.
                let samples = new_test_samples(SampleType::Face, &tri_faces, &tri_verts);
                //for (p, tri_indices) in samples.points.iter_mut().zip(tri_faces.iter()) {
                //    use geo::ops::Centroid;
                //    let tri = Triangle::from_indexed_slice(tri_indices, &tri_verts);
                //    *p = tri.centroid();
                //}
                let view = SamplesView::new(neighbours.as_ref(), &samples);

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

                let vert_jac = consolidate_face_jacobian(&jac, &neighbours, tri_faces, num_verts);

                // Compute the potential and test the jacobian for good measure.
                let mut p = F::cst(0.0);
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
                let mut hess_vtx = vec![Vector3::zeros(); num_verts];
                for &(r, c, h) in hess.iter() {
                    assert!(r >= c, "Hessian is not block lower triangular.");
                    if r == vtx {
                        hess_vtx[c] += h.transpose()[i];
                    } else if c == vtx {
                        hess_vtx[r] += h[i];
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
                tri_verts[vtx][i] = F::cst(tri_verts[vtx][i]);
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
        let mut no_perturb = || Vector3::zeros();
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
        let mut no_perturb = || Vector3::zeros();
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
        let mut no_perturb = || Vector3::zeros();
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
        let q = q.map(|x| F::cst(x)); // convert to autodiff
        let tri_verts: Vec<_> = mesh
            .vertex_position_iter()
            .map(|&x| Vector3(x).map(|x| F::cst(x)))
            .collect();
        let tri_faces = reinterpret::reinterpret_slice(mesh.indices.as_slice());

        let mut samples = new_test_samples(SampleType::Face, &tri_faces, &tri_verts);

        let neighbours: Vec<_> = samples
            .iter()
            .filter(|s| (q - s.pos).norm() < F::cst(radius + max_step))
            .map(|sample| sample.index)
            .collect();

        if neighbours.is_empty() {
            return;
        }

        let num_samples = samples.len();

        // Radius is such that samples are captured by the query point.
        let kernel = kernel::LocalApproximate::new(radius, 0.00001);

        // Compute the complete hessian.
        let hess: Vec<(usize, usize, Matrix3<F>)> = {
            let view = SamplesView::new(neighbours.as_ref(), &samples);

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
                samples.points[sample_idx][i] = F::var(samples.points[sample_idx][i]);
                //println!("row = {}; i = {}", sample_idx, i);

                let view = SamplesView::new(neighbours.as_ref(), &samples);

                let bg = BackgroundField::local(q, view, kernel, bg_field_params, None).unwrap();

                // Compute the Jacobian. After calling this function, calling
                // `.deriv()` on the output will give us the second derivative.
                let mut jac = vec![Vector3::zeros(); num_samples];
                for (jac_val, &idx) in
                    jacobian::sample_jacobian_at(q, view, kernel, bg.clone()).zip(neighbours.iter())
                {
                    jac[idx] = jac_val;
                }

                // Compute the potential and test the jacobian for good measure.
                let mut p = F::cst(0.0);
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
                let mut hess_sample = vec![Vector3::<F>::zeros(); num_samples];
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
                        hess_sample[c] += h.transpose()[i];
                    }

                    // Add upper triangular part to hess_sample.
                    if r > c && c == sample_idx {
                        hess_sample[r] += h[i];
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
                samples.points[sample_idx][i] = F::cst(samples.points[sample_idx][i]);
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
        use reinterpret::reinterpret_vec;
        let mut no_perturb = || Vector3::zeros();
        let mut perturb = make_perturb_fn();

        let qs = vec![
            Vector3([0.0, 0.2, 0.0]),
            Vector3([0.0, 0.0001, 0.0]),
            Vector3([0.0, -0.4, 0.0]),
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
            let tri = TriMesh::new(reinterpret_vec(tri_verts), reinterpret_vec(tri_indices));
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
        run_tester_on_mesh(TriMesh::from(utils::make_regular_tet()));
    }

    /// Test the second order derivatives of our normal computation method for face normals.
    #[test]
    fn face_normal_hessian_test() {
        // Simple test with two triangles
        let (tri_verts, tri_faces) = make_two_test_triangles(0.0, &mut || Vector3::zeros());
        face_normal_hessian_tester(&tri_verts, &tri_faces);

        // More complex test with a whole tet
        let (tet_verts, tet_faces) = make_tet();
        face_normal_hessian_tester(&tet_verts, &tet_faces);
    }

    fn face_normal_hessian_tester(verts: &[Vector3<f64>], faces: &[[usize; 3]]) {
        let samples = Samples::new_triangle_samples(faces, verts, vec![0.0; faces.len()]);

        let neighbours: Vec<usize> = (0..faces.len()).collect(); // look at all the faces

        // Set a random product vector.
        let multipliers = utils::random_vectors(faces.len());
        let ad_multipliers: Vec<_> = multipliers
            .iter()
            .map(|&v| Vector3(v).map(|x| F::cst(x)))
            .collect();

        let multiplier = move |Sample { index, .. }| Vector3(multipliers[index]);

        let ad_multiplier = move |Sample { index, .. }| ad_multipliers[index];

        // Compute the normal hessian product.
        let view = SamplesView::new(neighbours.as_ref(), &samples);
        let hess_iter =
            compute_face_unit_normals_hessian_products(view, verts, faces, multiplier.clone());

        let mut num_hess_entries = 0;
        let mut hess = [[0.0; 12]; 12]; // Dense matrix
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
            ImplicitSurface::<f64>::num_face_unit_normals_hessian_entries(neighbours.len()),
            num_hess_entries
        );

        // Convert tet vertices into varibales because we are taking the derivative with respect to
        // vertices.
        let mut ad_verts: Vec<Vector3<F>> = verts
            .iter()
            .cloned()
            .map(|v| v.map(|x| F::cst(x)))
            .collect();

        for r in 0..4 {
            for i in 0..3 {
                ad_verts[r][i] = F::var(ad_verts[r][i]);

                let ad_samples =
                    Samples::new_triangle_samples(faces, &ad_verts, vec![F::cst(0.0); 4]);
                let ad_view = SamplesView::new(neighbours.as_ref(), &ad_samples);

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
                            assert_relative_eq!(
                                hess[3 * c + j][3 * r + i],
                                grad[c][j].deriv(),
                                max_relative = 1e-5,
                                epsilon = 1e-10
                            );
                        }
                    }
                }

                ad_verts[r][i] = F::cst(ad_verts[r][i]);
            }
        }
    }
}
