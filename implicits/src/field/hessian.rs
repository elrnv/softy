use super::*;
use crate::Error;

impl<T: Real> ImplicitSurface<T> {
    /// Get the number of Hessian non-zeros for the face unit normal Hessian.
    /// This is essentially the number of items returned by
    /// `compute_face_unit_normals_hessian_products`.
    pub(crate) fn num_face_unit_normals_hessian_entries(num_samples: usize) -> usize {
        num_samples * 6 * 6
    }
}

impl<T: Real + Send + Sync> ImplicitSurface<T> {
    /// Block lower triangular part of the unit normal Hessian.
    pub(crate) fn compute_face_unit_normals_hessian_products<'a, F>(
        samples: SamplesView<'a, 'a, T>,
        surface_vertices: &'a [Vector3<T>],
        surface_topo: &'a [[usize; 3]],
        mut multiplier: F,
    ) -> impl Iterator<Item = (usize, usize, Matrix3<T>)> + 'a
    where
        F: FnMut(Sample<T>) -> Vector3<T> + 'a,
    {
        // For each triangle contribution (one element in a sum)
        samples
            .into_iter()
            .zip(surface_topo.iter())
            .flat_map(move |(sample, tri_indices)| {
                let norm_inv = T::one() / sample.nml.norm();
                let nml = sample.nml * norm_inv;
                let nml_proj = Matrix3::identity() - nml * nml.transpose();
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
                            let nml_dot_mult_div_norm = nml.dot(mult) * norm_inv;
                            let proj_mult = nml_proj * (mult * norm_inv); // projected multiplier
                            let nml_mult_prod = nml_proj * nml_dot_mult_div_norm
                                + proj_mult * nml.transpose()
                                + nml * proj_mult.transpose();
                            let m = Triangle::area_normal_hessian_product(j, i, proj_mult)
                                + (grad[j] * nml_mult_prod * grad[i]) * norm_inv;
                            (vtx_row, vtx_col, m)
                        })
                })
            })
    }

    /// Get the total number of entries for the sparse Hessian non-zeros. The Hessian is taken with
    /// respect to sample points. This estimate is based on the current neighbour cache, which
    /// gives the number of query points, if the neighbourhood was not precomputed this function
    /// returns `None`.
    pub fn num_surface_hessian_product_entries(&self) -> Option<usize> {
        let neigh_points = self.extended_neighbourhood_borrow().ok()?;
        let num_pts_per_sample = match self.sample_type {
            SampleType::Vertex => unimplemented!(),
            SampleType::Face => 3,
        };
        Some(neigh_points.iter()
            .map(|pts| pts.len())
            .sum::<usize>()
            * 3
            * num_pts_per_sample)
    }

    /// Compute the indices for the implicit surface potential Hessian with respect to surface
    /// points.
    pub fn surface_hessian_product_indices_iter(
        &self,
    ) -> Result<Box<dyn Iterator<Item = (usize, usize)>>, Error> {
        match self.kernel {
            KernelType::Approximate { .. } => self.mls_surface_jacobian_indices_iter(),
            _ => Err(Error::UnsupportedKernel),
        }
    }

    /// Compute the Hessian of this implicit surface function with respect to surface
    /// points multiplied by a vector of multipliers (one for each query point).
    pub fn surface_hessian_product_values(
        &self,
        query_points: &[[T; 3]],
        multipliers: &[T],
        values: &mut [T],
    ) -> Result<(), Error> {
        match self.kernel {
            KernelType::Approximate { tolerance, radius } => {
                let kernel = kernel::LocalApproximate::new(radius, tolerance);
                self.mls_surface_hessian_product_values(query_points, multipliers, kernel, values)
            }
            _ => Err(Error::UnsupportedKernel),
        }
    }

    pub(crate) fn mls_surface_hessian_product_values<'a, K>(
        &self,
        query_points: &[[T; 3]],
        multipliers: &[T],
        kernel: K,
        values: &mut [T],
    ) -> Result<(), Error>
        where
        K: SphericalKernel<T> + LocalKernel<T> + std::fmt::Debug + Copy + Sync + Send,
    {
        let value_vecs: &mut [[T; 3]] = reinterpret::reinterpret_mut_slice(values);

        self.cache_neighbours(query_points);
        let neigh_points = self.extended_neighbourhood_borrow()?;

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
                        Self::vertex_hessian_at(
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
                        Self::face_hessian_at(
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
        Ok(())
    }

    pub(crate) fn vertex_hessian_at<'a, K: 'a>(
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

    pub(crate) fn face_hessian_at<'a, K: 'a>(
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use autodiff::F;

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

        // Compute the complete hessian.
        let jac: Vec<Vector3<F>> = ImplicitSurface::vertex_hessian_at(
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

    /// Test the second order derivatives of our normal computation method for face normals.
    #[test]
    fn face_normal_hessian_test() {
        use geo::math::Matrix12;

        let (tet_verts, tet_faces) = make_tet();

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
                let grad = super::jacobian::compute_face_unit_normal_derivative(
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
                        }
                    }
                }

                ad_tet_verts[r][i] = F::cst(ad_tet_verts[r][i]);
            }
        }
    }

}
