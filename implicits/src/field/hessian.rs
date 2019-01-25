use super::*;
use crate::Error;

impl<T: Real> ImplicitSurface<T> {
    /// Get the number of Hessian non-zeros for the face unit normal Hessian.
    /// This is essentially the number of items returned by
    /// `compute_face_unit_normals_hessian_products`.
    pub fn num_face_unit_normals_hessian_entries(num_samples: usize) -> usize {
        num_samples * 6 * 6
    }
}

impl<T: Real + Send + Sync> ImplicitSurface<T> {
    /// Block lower triangular part of the unit normal Hessian multiplied by the given multiplier.
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
        let neigh_points = self.trivial_neighbourhood_borrow().ok()?;
        let num_pts_per_sample = match self.sample_type {
            SampleType::Vertex => unimplemented!(),
            SampleType::Face => 3,
        };
        Some(neigh_points.iter().map(|pts| pts.len()).sum::<usize>() * 3 * num_pts_per_sample)
    }

    /// Compute the indices for the implicit surface potential Hessian with respect to surface
    /// points.
    pub fn surface_hessian_product_indices(
        &self,
        query_points: &[[T; 3]],
        multipliers: &[T],
        rows: &mut [usize],
        cols: &mut [usize],
    ) -> Result<(), Error> {
        match self.kernel {
            KernelType::Approximate { tolerance, radius } => {
                let kernel = kernel::LocalApproximate::new(radius, tolerance);
                self.mls_surface_hessian_product_indices(query_points, multipliers, kernel, rows, cols)
            }
            _ => Err(Error::UnsupportedKernel),
        }
    }

    pub(crate) fn mls_surface_hessian_product_indices<K>(
        &self,
        query_points: &[[T; 3]],
        multipliers: &[T],
        kernel: K,
        rows: &mut [usize],
        cols: &mut [usize],
    ) -> Result<(), Error>
    where
        K: SphericalKernel<T> + std::fmt::Debug + Copy + Sync + Send,
    {
        self.cache_neighbours(query_points);
        let neigh_points = self.trivial_neighbourhood_borrow()?;

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
                unimplemented!();
            }
            SampleType::Face => {
                let face_hess = zip!(query_points.iter(), multipliers.iter(), neigh_points.iter())
                    .filter(|(_, _, nbrs)| !nbrs.is_empty())
                    .flat_map(move |(q, lambda, nbr_points)| {
                        let view = SamplesView::new(nbr_points, samples);
                        Self::face_hessian_at(
                            Vector3(*q),
                            view,
                            kernel,
                            surface_topo,
                            surface_vertex_positions,
                            dual_topo,
                            bg_field_type,
                            *lambda,
                        )
                    });

                for (i, (r,c)) in face_hess.flat_map(move |(row,col,_)| {
                    (0..3).flat_map(move |r| {
                        (0..3).filter(move |c| 3*row + r >= 3*col + c)
                            .map(move |c| (3*row + r, 3*col + c))
                    })
                }).enumerate() {
                    rows[i] = r;
                    cols[i] = c;
                }

            }
        }
        Ok(())
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

    pub(crate) fn mls_surface_hessian_product_values<K>(
        &self,
        query_points: &[[T; 3]],
        multipliers: &[T],
        kernel: K,
        values: &mut [T],
    ) -> Result<(), Error>
    where
        K: SphericalKernel<T> + std::fmt::Debug + Copy + Sync + Send,
    {
        self.cache_neighbours(query_points);
        let neigh_points = self.trivial_neighbourhood_borrow()?;

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
                unimplemented!();
            }
            SampleType::Face => {
                let face_jac = zip!(query_points.iter(), multipliers.iter(), neigh_points.iter())
                    .filter(|(_, _, nbrs)| !nbrs.is_empty())
                    .flat_map(move |(q, lambda, nbr_points)| {
                        let view = SamplesView::new(nbr_points, samples);
                        Self::face_hessian_at(
                            Vector3(*q),
                            view,
                            kernel,
                            surface_topo,
                            surface_vertex_positions,
                            dual_topo,
                            bg_field_type,
                            *lambda,
                        )
                    });

                values
                    .iter_mut()
                    .zip(face_jac.flat_map(move |(row,col,mtx)| {
                        (0..3).flat_map(move |r| {
                            (0..3).filter(move |c| 3*row + r >= 3*col + c)
                                .map(move |c| mtx[c][r])
                        })
                    }))
                    .for_each(|(val, new_val)| {
                        *val = new_val;
                    });
            }
        }
        Ok(())
    }

    pub(crate) fn face_hessian_at<'a, K: 'a>(
        q: Vector3<T>,
        view: SamplesView<'a, 'a, T>,
        kernel: K,
        surface_topo: &'a [[usize; 3]],
        surface_vertex_positions: &'a [Vector3<T>],
        dual_topo: &'a [Vec<usize>],
        bg_field_type: BackgroundFieldType,
        multiplier: T,
    ) -> impl Iterator<Item = (usize, usize, Matrix3<T>)> + 'a
    where
        K: SphericalKernel<T> + std::fmt::Debug + Copy + Sync + Send,
    {
        let bg = Self::compute_background_potential(q, view, kernel, bg_field_type);

        let ninth = T::one() / T::from(9.0).unwrap();

        // For each surface vertex contribution
        let main_jac = Self::sample_hessian_at(q, view, kernel, bg.clone())
            .flat_map(move |jac| {
                (0..3).flat_map(move |i| {
                    (0..3).map(move |j| (surface_topo[jac.0][i], surface_topo[jac.1][j], jac.2 * ninth))
                })
            });

        // Add in the normal gradient multiplied by a vector of given Vector3 values.
        let nml_jac = Self::normal_hessian_at(
            q,
            view,
            kernel,
            &surface_topo,
            surface_vertex_positions,
            dual_topo,
            bg,
        );

        // There are 3 contributions from each sample to each vertex.
        nml_jac.chain(main_jac).map(move |jac| (jac.0, jac.1, jac.2 * multiplier))
    }

    /// Hessian part with respect to samples.
    pub(crate) fn sample_hessian_at<'a, K: 'a>(
        q: Vector3<T>,
        samples: SamplesView<'a, 'a, T>,
        kernel: K,
        bg: BackgroundField<'a, T, T, K>,
    ) -> impl Iterator<Item = (usize, usize, Matrix3<T>)> + 'a
    where
        K: SphericalKernel<T> + std::fmt::Debug + Copy + Sync + Send,
    {
        let csd = bg.closest_sample_dist();
        let ws_inv = bg.weight_sum_inv();

        let local_pot = Self::compute_local_potential_at(q, samples, kernel, ws_inv, csd);

        samples.into_iter().flat_map(
            move |Sample {
                index: i,
                pos: pos_i,
                nml: nml_i,
                value: phi_i,
                ..
            }| {
                let bg = bg.clone();
                let psi_i = phi_i + (q - pos_i).dot(nml_i) * (T::one() / nml_i.norm());
                let dw_i = kernel.with_closest_dist(csd).grad(q, pos_i);
                samples.into_iter().map(
                    move |Sample {
                        index: j,
                        pos: pos_j,
                        nml: nml_j,
                        value: phi_j,
                        ..
                    }| {
                        let psi_j = phi_j + (q - pos_j).dot(nml_j) * (T::one() / nml_j.norm());

                        let mut h = Matrix3::zeros();

                        if i == j {
                            // Diagonal entries
                            let ddw = kernel.with_closest_dist(csd).hess(q, pos_i);
                            h += ddw * ws_inv * psi_i;

                            h -= ddw * ws_inv * local_pot;
                        }

                        let dw_j = kernel.with_closest_dist(csd).grad(q, pos_j);
                        let dwb_j = bg.background_weight_gradient(Some(j));
                        let dws_j = dw_j - dwb_j;

                        h -= dw_i * (dws_j.transpose() * (psi_i * ws_inv));

                        h -= dw_i * (dw_j.transpose() * (psi_j * ws_inv * ws_inv) - dws_j.transpose() * (T::from(2.0).unwrap() * local_pot * ws_inv * ws_inv));

                        (j, i, h)
                    },
                )
            },
        )
    }

    /// A helper function to compute the normal part of the hessian for this field.
    fn normal_hessian_at<'a, K: 'a>(
        q: Vector3<T>,
        samples: SamplesView<'a, 'a, T>,
        kernel: K,
        surface_topo: &'a [[usize; 3]],
        surface_vertex_positions: &'a [Vector3<T>],
        dual_topo: &'a [Vec<usize>],
        bg: BackgroundField<'a, T, T, K>,
    ) -> impl Iterator<Item = (usize, usize, Matrix3<T>)> + 'a
    where
        K: SphericalKernel<T> + std::fmt::Debug + Copy + Sync + Send,
    {
        let csd = bg.closest_sample_dist();
        let ws_inv = bg.weight_sum_inv();

        let sym_mult = move |Sample { pos, .. }| {
            kernel.with_closest_dist(csd).eval(q, pos) * ws_inv
        };

        let sym =
            Self::face_unit_normals_symmetric_jacobian(samples, surface_vertex_positions, surface_topo, dual_topo, sym_mult);

        let nml_hess_multiplier = move |Sample { pos, .. }| {
            let w = kernel.with_closest_dist(csd).eval(q, pos);
            (q - pos) * (w * ws_inv)
        };

        // Compute the unit normal hessian product.
        let nml_hess_iter = ImplicitSurface::compute_face_unit_normals_hessian_products(
            samples,
            &surface_vertex_positions,
            &surface_topo,
            nml_hess_multiplier,
        );

        let third = T::one() / T::from(3.0).unwrap();

        // Remaining hessian terms
        let coupling_nml_hess_iter = samples.into_iter().flat_map(move |sample| {
            let Sample { index, pos, nml, .. } = sample;

            let dwbdp = bg.background_weight_gradient(Some(index));
            let dwdp = -kernel.with_closest_dist(csd).grad(q, pos);
            let dwsdp = dwdp + dwbdp;
            
            (0..3).flat_map(move |i| {
                let grad = Self::face_unit_normal_gradient_iter(sample, surface_vertex_positions, surface_topo);
                grad.enumerate().map(move |(j, g)| {
                    let row_vtx = surface_topo[index][j];
                    let col_vtx = surface_topo[index][i];

                    let num_neighs = T::from(dual_topo[row_vtx].len()).unwrap();

                    let mtx = ((g * (q - pos) * num_neighs - nml * third) * dwdp.transpose()) * (ws_inv * third);

                    if row_vtx > col_vtx {
                        (row_vtx, col_vtx, mtx) 
                    } else if row_vtx < col_vtx {
                        (col_vtx, row_vtx, mtx.transpose())
                    } else { // == 
                        (row_vtx, col_vtx, mtx + mtx.transpose())
                    }
                })
            }).chain(
                (0..3).flat_map(move |i| {
                    samples.clone().into_iter().flat_map(move |sample_j| {
                        let wj = kernel.with_closest_dist(csd).eval(q, sample_j.pos) * ws_inv;
                        (0..3).map(move |j| {
                            let row_vtx = surface_topo[sample_j.index][j];
                            let col_vtx = surface_topo[index][i];

                            let neigh_d = samples.from_view(&dual_topo[row_vtx]).into_iter().map(|sample| {
                                let wk = kernel.with_closest_dist(csd).eval(q, sample.pos);
                                let j_vtx = surface_topo[sample.index].iter().position(|&vtx| vtx == row_vtx).expect("Corrupt topo");
                                let grad = Self::face_unit_normal_gradient(sample, j_vtx, surface_vertex_positions, surface_topo);
                                grad * (q - sample.pos) * wk
                            }).sum::<Vector3<T>>() * ws_inv;

                            let mtx = (sample_j.nml * (wj * third) - neigh_d) * dwsdp.transpose() * (ws_inv * third);
                            if row_vtx > col_vtx {
                                (row_vtx, col_vtx, mtx) 
                            } else if row_vtx < col_vtx {
                                (col_vtx, row_vtx, mtx.transpose())
                            } else { // == 
                                (row_vtx, col_vtx, mtx + mtx.transpose())
                            }
                        })
                    })
                })
            )
        });

        zip!(sym, nml_hess_iter).map(move |(s, n)| {
            assert_eq!(s.0, n.0);
            assert_eq!(s.1, n.1);
            (s.0, s.1, n.2 - s.2)
        }).chain(coupling_nml_hess_iter)
    }
}

#[cfg(test)]
mod tests {
    use geo::mesh::{TriMesh, VertexPositions};
    use super::*;
    use jacobian::{new_test_samples, make_test_tri, make_perturb_fn, consolidate_face_jacobian};
    use autodiff::F;

    /// High level test of the Hessian as the derivative of the Jacobian.
    fn surface_hessian_tester(
        query_points: &[[f64;3]],
        surf_mesh: &TriMesh<f64>,
        radius: f64,
        max_step: f64,
        bg_field_type: BackgroundFieldType,
    ) {
        let params = crate::Params {
            kernel: KernelType::Approximate {
                tolerance: 0.00001,
                radius,
            },
            background_field: bg_field_type,
            sample_type: SampleType::Face,
            max_step,
        };

        let mut surf = crate::surface_from_trimesh::<F>(&surf_mesh, params).expect("Failed to construct an implicit surface.");

        let mut ad_tri_verts: Vec<_> = surf_mesh.vertex_position_iter().map(|&x| Vector3(x).map(|x| F::cst(x)).into_inner()).collect();
        let num_verts = ad_tri_verts.len();
        let ad_query_points: Vec<_> = query_points.iter().map(|&a| Vector3(a).map(|x| F::cst(x)).into_inner()).collect();
        let num_query_points = query_points.len();

        surf.cache_neighbours(&ad_query_points);
        let num_hess_entries = surf.num_surface_hessian_product_entries().expect("Uncached query points.");
        let num_jac_entries = surf.num_surface_jacobian_entries().expect("Uncached query points.");

        // Compute the complete hessian.
        let mut hess_rows = vec![0; num_hess_entries];
        let mut hess_cols = vec![0; num_hess_entries];
        let mut hess_values = vec![F::cst(0.0); num_hess_entries];
        let mut jac_rows = vec![0; num_jac_entries];
        let mut jac_cols = vec![0; num_jac_entries];
        let mut jac_values = vec![F::cst(0.0); num_jac_entries];

        let mut multipliers = vec![F::cst(0.0); num_query_points];
        surf.surface_hessian_product_indices(
            &ad_query_points,
            &multipliers,
            &mut hess_rows,
            &mut hess_cols,
        ).expect("Failed to compute hessian indices");

        // We use the multipliers to isolate the hessian for each query point.
        for q_idx in 0..num_query_points {
            multipliers[q_idx] = F::cst(1.0);

            surf.surface_hessian_product_values(
                &ad_query_points,
                &multipliers,
                &mut hess_values
            ).expect("Failed to compute hessian product");

            // Test the accuracy of each component of the hessian against an autodiff version of the
            // second derivative.
            for vtx in 0..num_verts {
                for i in 0..3 {
                    // Set a variable to take the derivative with respect to, using autodiff.
                    ad_tri_verts[vtx][i] = F::var(ad_tri_verts[vtx][i]);
                    surf.update(ad_tri_verts.iter().cloned());

                    surf.surface_jacobian_values(&ad_query_points, &mut jac_values).expect("Failed to compute Jacobian values.");
                    surf.surface_jacobian_indices(&mut jac_rows, &mut jac_cols).expect("Failed to compute Jacobian indices.");

                    // Get the jacobian for the specific query point we are interested in.
                    let mut jac_q = vec![F::cst(0.0); num_verts*3];
                    for (&r, &c, &jac) in zip!(jac_rows.iter(), jac_cols.iter(), jac_values.iter()) {
                        if r == q_idx {
                            jac_q[c] += jac;
                        }
                    }

                    // Consolidate the Hessian to the particular vertex and component we are
                    // interested in.
                    let mut hess_vtx = vec![F::cst(0.0); num_verts*3];
                    for (&r, &c, &h) in zip!(hess_rows.iter(), hess_cols.iter(), hess_values.iter()) {
                        if r == 3*vtx + i {
                            hess_vtx[c] += h;
                        }
                    }

                    for (jac, hes) in jac_q.iter().zip(hess_vtx) {
                        // Check the derivative of the autodiff with our previously computed Jacobian.
                        println!("{:.5} vs {:.5}", hes.value(), jac.deriv());
                        //assert_relative_eq!(
                        //    hes[j].value(),
                        //    jac[j].deriv(),
                        //    max_relative = 1e-6,
                        //    epsilon = 1e-12
                        //);
                    }

                    // Reset the variable back to being a constant.
                    ad_tri_verts[vtx][i] = F::cst(ad_tri_verts[vtx][i]);
                }
            }
            multipliers[q_idx] = F::cst(0.0); // reset multiplier
        }
    }

    /// Test the highest level surface Hessian functions.
    #[test]
    fn one_triangle_hessian_test() {
        let qs = vec![[0.0, 0.4, 0.0], [0.0, 0.0, 0.0], [0.0, -0.4, 0.0]];
        let mut perturb = make_perturb_fn();
        let tri_verts = make_test_tri(0.0, &mut perturb).into_iter().map(|x| x.into_inner()).collect();
        let tri = geo::mesh::TriMesh::new(tri_verts, vec![0, 1, 2]);

        for i in 1..50 {
            let radius = 0.1 * (i as f64);
            surface_hessian_tester(&qs, &tri, radius, 0.0, BackgroundFieldType::None);
        }
    }

    fn one_tet_face_hessian<P: FnMut() -> Vector3<f64>>(
        radius: f64,
        bg_field_type: BackgroundFieldType,
        perturb: &mut P,
    ) {
        let tet = TriMesh::from(utils::make_regular_tet());
        let qs: Vec<_> = (0..10).map(|i| Vector3([0.0, -0.5 + 0.1*i as f64, 0.0])).collect();

        for q in qs.into_iter() {
            face_hessian_tester(q, &tet, radius, 0.0, bg_field_type);
            face_hessian_tester(q + perturb(), &tet, radius, 0.0, bg_field_type);
            face_hessian_tester(q + perturb(), &tet, radius, 1.0, bg_field_type);
        }
    }

    fn one_triangle_face_hessian<P: FnMut() -> Vector3<f64>>(
        radius: f64,
        bg_field_type: BackgroundFieldType,
        perturb: &mut P,
    ) {
        let tri_verts: Vec<_> = make_test_tri(0.0, perturb).into_iter().map(|x| x.into_inner()).collect();
        let tri_indices = vec![0usize, 1, 2];
        let tri = TriMesh::new(tri_verts, tri_indices);
        let qs = vec![Vector3([0.0, 0.2, 0.0])];//, Vector3([0.0, 0.0, 0.0]), Vector3([0.0, -0.4, 0.0])];

        for q in qs.into_iter() {
            face_hessian_tester(q, &tri, radius, 0.0, bg_field_type);
            //face_hessian_tester(q + perturb(), &tri, radius, 0.0, bg_field_type);
            //face_hessian_tester(q + perturb(), &tri, radius, 1.0, bg_field_type);
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
        bg_field_type: BackgroundFieldType,
    ) {
        dbg!(q);

        let q = q.map(|x| F::cst(x)); // convert to autodiff
        dbg!(mesh.vertex_positions());
        let mut tri_verts: Vec<_> = mesh.vertex_position_iter().map(|&x| Vector3(x).map(|x| F::cst(x))).collect();
        let tri_faces = reinterpret::reinterpret_slice(mesh.indices.as_slice());
        dbg!(&tri_faces);
        dbg!(radius);
        let num_verts = tri_verts.len();

        let samples = new_test_samples(SampleType::Face, &tri_faces, &tri_verts);

        let neighbours: Vec<_> = samples.iter()
            .filter(|s| (q - s.pos).norm() < F::cst(radius + max_step))
            .map(|sample| sample.index).collect();

        if neighbours.is_empty() {
            return;
        }

        // Radius is such that samples are captured by the query point.
        let kernel = kernel::LocalApproximate::new(radius, 0.00001);

        let view = SamplesView::new(neighbours.as_ref(), &samples);

        // Compute the complete hessian.
        let hess: Vec<(usize, usize, Matrix3<F>)> = ImplicitSurface::face_hessian_at(
            q,
            view,
            kernel,
            tri_faces,
            &tri_verts,
            &vec![vec![0];3],
            bg_field_type,
            F::cst(1.0),
        ).collect();

        let mut hess_full = vec![vec![0.0; 3*num_verts]; 3*num_verts];
        let mut hess_full2 = vec![vec![0.0; 3*num_verts]; 3*num_verts];

        for &(r,c,h) in hess.iter() {
            for j in 0..3 {
                for i in 0..3 {
                    hess_full[3*c + j][3*r + i] += h[j][i].value();
                }
            }
        }

        let print_full_hess = |hess: Vec<Vec<f64>>, name| {
            println!("{} = ", name);
            for r in 0..3*num_verts {
                for c in 0..3*num_verts {
                    print!("{:9.5} ", hess[c][r]);
                }
                println!("");
            }
            println!("");
        };

        let mut ad_hess_full = vec![vec![0.0; 3*num_verts]; 3*num_verts];

        // Test the accuracy of each component of the hessian against an autodiff version of the
        // second derivative.
        for vtx in 0..num_verts {
            for i in 0..3 {
                // Set a variable to take the derivative with respect to, using autodiff.
                tri_verts[vtx][i] = F::var(tri_verts[vtx][i]);
                println!("vtx = {}; i = {}", vtx, i);

                // We need to update samples to make sure the normals and centroids are recomputed
                // using the correct wrt autodiff variable.
                let samples = new_test_samples(SampleType::Face, &tri_faces, &tri_verts);
                let view = SamplesView::new(neighbours.as_ref(), &samples);

                // Compute the Jacobian. After calling this function, calling
                // `.deriv()` on the output will give us the second derivative.
                let jac: Vec<_> = ImplicitSurface::face_jacobian_at(
                    q,
                    view,
                    kernel,
                    tri_faces,
                    &tri_verts,
                    bg_field_type,
                ).collect();

                let vert_jac = consolidate_face_jacobian(&jac, &neighbours, tri_faces, num_verts);

                // Compute the potential and test the jacobian for good measure.
                let mut p = F::cst(0.0);
                ImplicitSurface::compute_potential_at(
                    q,
                    view,
                    kernel,
                    bg_field_type,
                    &mut p
                );

                // Test the surface Jacobian against autodiff on the potential computation.
                assert_relative_eq!(
                    vert_jac[vtx][i].value(),
                    p.deriv(),
                    max_relative = 1e-5,
                    epsilon = 1e-10
                );
                println!("jac {:9.5} vs {:9.5}", vert_jac[vtx][i].value(), p.deriv());

                // Consolidate the hessian to this particular vertex and component.
                let mut hess_vtx = vec![Vector3::zeros(); num_verts];
                for &(r, c, h) in hess.iter() {
                    assert!(r >= c, "Hessian is not block lower triangular.");
                    if r == c {
                        for x in 0..3 {
                            for y in 0..3 {
                                assert_relative_eq!(h[y][x].value(), h[x][y].value(), max_relative = 1e-6, epsilon = 1e-12);
                            }
                        }
                    }
                    if r == vtx {
                        hess_vtx[c] += h.transpose()[i];
                    } else if c == vtx {
                        hess_vtx[r] += h[i];
                    }
                }

                for (vtx_idx, (jac, hes)) in vert_jac.iter().zip(hess_vtx).enumerate() {
                    for j in 0..3 {
                        // Check the derivative of the autodiff with our previously computed Jacobian.
                        println!("{:9.5} vs {:9.5}", hes[j].value(), jac[j].deriv());
                        ad_hess_full[3*vtx_idx + j][3*vtx + i] += jac[j].deriv();
                        hess_full2[3*vtx_idx + j][3*vtx + i] += hes[j].value();
                        //assert_relative_eq!(
                        //    hes[j].value(),
                        //    jac[j].deriv(),
                        //    max_relative = 1e-6,
                        //    epsilon = 1e-12
                        //);
                    }
                }
                println!("");

                // Reset the variable back to being a constant.
                tri_verts[vtx][i] = F::cst(tri_verts[vtx][i]);
            }
        }

        print_full_hess(hess_full, "Block Lower Triangular Hessian");
        print_full_hess(hess_full2, "Full Hessian");
        print_full_hess(ad_hess_full, "Full Autodiff Hessian");
    }

    #[test]
    fn one_triangle_face_hessian_test() {
        let mut perturb = || Vector3::zeros();//make_perturb_fn();
        for i in 3..4 {
            let radius = 0.1 * (i as f64);
            one_triangle_face_hessian(radius, BackgroundFieldType::None, &mut perturb);
            //one_triangle_face_hessian(radius, BackgroundFieldType::Zero, &mut perturb);
            //one_triangle_face_hessian(radius, BackgroundFieldType::FromInput, &mut perturb);
            //one_triangle_face_hessian(radius, BackgroundFieldType::DistanceBased, &mut perturb);
            //one_triangle_face_hessian(radius, BackgroundFieldType::NormalBased, &mut perturb);
        }
    }

    #[test]
    fn one_tet_face_hessian_test() {
        let mut perturb = make_perturb_fn();
        for i in 1..50 {
            let radius = 0.1 * (i as f64);
            one_tet_face_hessian(radius, BackgroundFieldType::None, &mut perturb);
            //one_tet_face_hessian(radius, BackgroundFieldType::Zero, &mut perturb);
            //one_tet_face_hessian(radius, BackgroundFieldType::FromInput, &mut perturb);
            //one_tet_face_hessian(radius, BackgroundFieldType::DistanceBased, &mut perturb);
            //one_tet_face_hessian(radius, BackgroundFieldType::NormalBased, &mut perturb);
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
