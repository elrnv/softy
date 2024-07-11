use super::samples::*;
use crate::Real;
use num_traits::Zero;
use tensr::{zip, Matrix3, Vector3};

use rayon::prelude::*;

/*
 * Potential function compoenents
 *
 * The following routines compute parts of various potential functions defining implicit surfaces.
 */

/// Compute the potential field (excluding background field) at a given query point.
pub(crate) fn alt_compute_local_potential_at<T, K>(
    q: Vector3<T>,
    samples: SamplesView<T>,
    kernel: K,
    weight_sum_inv: T,
    closest_d: T,
) -> T
where
    K: SphericalKernel<T> + Copy + std::fmt::Debug + Sync + Send,
    T: na::RealField + Real,
{
    use na::{DMatrix, DVector};
    let basis = DMatrix::from_iterator(
        4,
        samples.len(),
        samples
            .iter()
            .flat_map(|s| vec![T::one(), s.pos[0], s.pos[1], s.pos[2]].into_iter()),
    );

    let diag_weights: Vec<T> = samples
        .iter()
        .map(|s| kernel.with_closest_dist(closest_d).eval(q, s.pos) * weight_sum_inv)
        .collect();

    let weights = DMatrix::from_diagonal(&DVector::from_vec(diag_weights));

    let basis_view = &basis;
    let h = basis_view * &weights * basis_view.transpose();

    let sample_data: Vec<T> = samples
        .iter()
        .map(|s| s.value + s.nml.dot(q - s.pos) / s.nml.norm())
        .collect();

    let rhs = basis * weights * DVector::from_vec(sample_data);

    h.svd(true, true)
        .solve(&rhs, T::from(1e-9).unwrap())
        .map(|c| c[0] + q[0] * c[1] + q[1] * c[2] + q[2] * c[3])
        .unwrap_or_else(|_| T::from(std::f64::NAN).unwrap())
}

impl<T: Real> MLS<T> {
    /*
     * Compute MLS potential on mesh
     */

    /// Implementation of the Moving Least Squares algorithm for computing an implicit surface.
    pub(crate) fn compute_on_mesh<K, F, M>(
        self,
        mesh: &mut M,
        kernel: K,
        interrupt: F,
    ) -> Result<(), Error>
    where
        K: SphericalKernel<T> + std::fmt::Debug + Copy + Sync + Send,
        F: Fn() -> bool + Sync + Send,
        M: VertexMesh<T>,
        T: na::RealField,
    {
        let query_surf = QueryTopo::new(mesh.vertex_positions(), self);

        let ImplicitSurfaceBase {
            ref samples,
            bg_field_params,
            ..
        } = *query_surf.base();

        // Move the potential attrib out of the mesh. We will reinsert it after we are done.
        let potential_attrib = mesh
            .remove_attrib::<VertexIndex>("potential")
            .ok() // convert to option (None when it doesn't exist)
            .unwrap_or_else(|| Attribute::direct_from_vec(vec![0.0f32; mesh.num_vertices()]));

        let mut potential = potential_attrib
            .into_data()
            .cast_into_vec::<f32>()
            .unwrap_or_else(|| {
                // Couldn't cast, which means potential is of some non-numeric type.
                // We overwrite it because we need that attribute spot.
                vec![0.0f32; mesh.num_vertices()]
            });

        // Alternative potential for prototyping
        let alt_potential_attrib = mesh
            .remove_attrib::<VertexIndex>("alt_potential")
            .ok() // convert to option (None when it doesn't exist)
            .unwrap_or_else(|| Attribute::direct_from_vec(vec![0.0f32; mesh.num_vertices()]));

        let mut alt_potential = alt_potential_attrib
            .into_data()
            .cast_into_vec::<f32>()
            .unwrap_or_else(|| {
                // Couldn't cast, which means potential is of some non-numeric type.
                // We overwrite it because we need that attribute spot.
                vec![0.0f32; mesh.num_vertices()]
            });

        // Overwrite these attributes.
        mesh.remove_attrib::<VertexIndex>("normals").ok();
        let mut normals = vec![[0.0f32; 3]; mesh.num_vertices()];
        mesh.remove_attrib::<VertexIndex>("tangents").ok();
        let mut tangents = vec![[0.0f32; 3]; mesh.num_vertices()];

        let query_points = mesh.vertex_positions();
        let neigh_points = query_surf.trivial_neighborhood_par();
        let closest_points = query_surf.closest_samples_par();

        // Initialize extra debug info.
        let mut num_neighs_attrib_data = vec![0i32; mesh.num_vertices()];
        let mut neighs_attrib_data = vec![[-1i32; 11]; mesh.num_vertices()];
        let mut bg_weight_attrib_data = vec![0f32; mesh.num_vertices()];
        let mut weight_sum_attrib_data = vec![0f32; mesh.num_vertices()];

        let result = zip!(
            query_points.par_iter(),
            neigh_points,
            closest_points,
            num_neighs_attrib_data.par_iter_mut(),
            neighs_attrib_data.par_iter_mut(),
            bg_weight_attrib_data.par_iter_mut(),
            weight_sum_attrib_data.par_iter_mut(),
            potential.par_iter_mut(),
            alt_potential.par_iter_mut(),
            normals.par_iter_mut(),
            tangents.par_iter_mut(),
        )
        .map(
            |(
                q,
                neighs,
                closest,
                num_neighs,
                out_neighs,
                bg_weight,
                weight_sum,
                potential,
                alt_potential,
                normal,
                tangent,
            )| {
                if interrupt() {
                    return Err(Error::Interrupted);
                }

                let q = Vector3::new(*q);

                let view = SamplesView::new(neighs, samples);

                // Record number of neighbors in total.
                *num_neighs = view.len() as i32;

                // Record up to 11 neighbors
                for (k, neigh) in view.iter().take(11).enumerate() {
                    out_neighs[k] = neigh.index as i32;
                }

                let bg = BackgroundField::global(
                    q,
                    view,
                    closest,
                    kernel,
                    bg_field_params,
                    Some(T::from(*potential).unwrap()),
                );

                let closest_d = bg.closest_sample_dist();
                *bg_weight = bg.background_weight().to_f32().unwrap();
                *weight_sum = bg.weight_sum.to_f32().unwrap();
                let weight_sum_inv = bg.weight_sum_inv();

                *potential = (weight_sum_inv * bg.compute_unnormalized_weighted_scalar_field())
                    .to_f32()
                    .unwrap();

                *alt_potential = (weight_sum_inv * bg.compute_unnormalized_weighted_scalar_field())
                    .to_f32()
                    .unwrap();

                if !view.is_empty() {
                    let mut grad_w_sum_normalized = Vector3::zero();
                    for grad in samples
                        .iter()
                        .map(|Sample { pos, .. }| kernel.with_closest_dist(closest_d).grad(q, pos))
                    {
                        grad_w_sum_normalized += grad;
                    }
                    grad_w_sum_normalized *= weight_sum_inv;

                    let mut out_normal = Vector3::zero();
                    let mut out_tangent = Vector3::zero();

                    let p = compute_local_potential_at(q, view, kernel, weight_sum_inv, closest_d);

                    let alt_p =
                        alt_compute_local_potential_at(q, view, kernel, weight_sum_inv, closest_d);

                    for Sample { pos, nml, vel, .. } in view.iter() {
                        let w = kernel.with_closest_dist(closest_d).eval(q, pos);
                        let grad_w = kernel.with_closest_dist(closest_d).grad(q, pos);
                        let w_normalized = w * weight_sum_inv;
                        let grad_w_normalized =
                            grad_w * weight_sum_inv - grad_w_sum_normalized * w_normalized;

                        out_normal += grad_w_normalized * (q - pos).dot(nml) + nml * w_normalized;

                        // Compute vector interpolation
                        let grad_phi = jacobian::query_jacobian_at(
                            q,
                            view,
                            Some(closest),
                            kernel,
                            bg_field_params,
                        );

                        let nml_dot_grad = nml.dot(grad_phi);
                        // Handle degenerate case when nml and grad are exactly opposing. In
                        // this case the solution is not unique, so we pick one.
                        let rot = if nml_dot_grad != -T::one() {
                            let u = nml.cross(grad_phi);
                            let ux = u.skew();
                            Matrix3::identity() + ux + (ux * ux) / (T::one() + nml_dot_grad)
                        } else {
                            // TODO: take a convenient unit vector u that is
                            // orthogonal to nml and compute the rotation as
                            //let ux = u.skew();
                            //Matrix3::identity() + (ux*ux) * 2
                            Matrix3::identity()
                        };

                        out_tangent += (rot * vel) * w_normalized;
                    }

                    *potential += p.to_f32().unwrap();
                    *alt_potential += alt_p.to_f32().unwrap();
                    *normal = out_normal.map(|x| x.to_f32().unwrap()).into();
                    *tangent = out_tangent.map(|x| x.to_f32().unwrap()).into();
                }
                Ok(())
            },
        )
        .reduce(|| Ok(()), |acc, result| acc.and(result));

        {
            mesh.set_attrib_data::<_, VertexIndex>("num_neighbors", num_neighs_attrib_data)?;
            mesh.set_attrib_data::<_, VertexIndex>("neighbors", neighs_attrib_data)?;
            mesh.set_attrib_data::<_, VertexIndex>("bg_weight", bg_weight_attrib_data)?;
            mesh.set_attrib_data::<_, VertexIndex>("weight_sum", weight_sum_attrib_data)?;
            mesh.set_attrib_data::<_, VertexIndex>("potential", potential)?;
            mesh.set_attrib_data::<_, VertexIndex>("alt_potential", alt_potential)?;
            mesh.set_attrib_data::<_, VertexIndex>("normals", normals)?;
            mesh.set_attrib_data::<_, VertexIndex>("tangents", tangents)?;
        }

        result
    }
}
