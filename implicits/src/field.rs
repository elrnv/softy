//!
//! Implicit surface module. This module defines an interface for computing an implicit surface
//! potential and its derivatives.
//!

use crate::kernel::*;
use crate::Error;
use crate::Real;
use geo::attrib::*;
use geo::mesh::{topology::VertexIndex, VertexMesh};
use geo::prim::Triangle;
use num_traits::ToPrimitive;
use num_traits::Zero;
use rayon::prelude::*;
use rstar::RTree;
use tensr::{zip, IntoData, Matrix, Matrix3, Scalar, Vector3};

macro_rules! apply_kernel_query_fn_impl_iter {
    ($surf:expr, $f:expr, ?) => {{
        match *$surf {
            QueryTopo::Local {
                surf:
                    LocalMLS {
                        kernel,
                        base_radius,
                        ..
                    },
                ..
            } => Either::Left(apply_as_spherical_impl_iter!(kernel, base_radius, $f, ?)),
            QueryTopo::Global {
                surf: GlobalMLS { kernel, .. },
                ..
            } => Either::Right(apply_as_spherical_impl_iter!(kernel, $f, ?)),
        }
    }};
    ($surf:expr, $f:expr) => {{
        match *$surf {
            QueryTopo::Local {
                surf:
                    LocalMLS {
                        kernel,
                        base_radius,
                        ..
                    },
                ..
            } => Either::Left(apply_as_spherical_impl_iter!(kernel, base_radius, $f)),
            QueryTopo::Global {
                surf: GlobalMLS { kernel, .. },
                ..
            } => Either::Right(apply_as_spherical_impl_iter!(kernel, $f)),
        }
    }};
}

macro_rules! apply_kernel_query_fn {
    ($surf:expr, $f:expr) => {
        match *$surf {
            QueryTopo::Local {
                surf:
                    LocalMLS {
                        kernel,
                        base_radius,
                        ..
                    },
                ..
            } => apply_as_spherical!(kernel, base_radius, $f),
            QueryTopo::Global {
                surf: GlobalMLS { kernel, .. },
                ..
            } => apply_as_spherical!(kernel, $f),
        }
    };
}

macro_rules! apply_kernel_fn {
    ($surf:expr, $f:expr) => {
        match $surf {
            MLS::Local(LocalMLS {
                kernel,
                base_radius,
                ..
            }) => apply_as_spherical!(kernel, base_radius, $f),
            MLS::Global(GlobalMLS { kernel, .. }) => apply_as_spherical!(kernel, $f),
        }
    };
}

pub mod background_field;
pub mod builder;
pub mod hessian;
pub mod jacobian;
pub mod neighbor_cache;
pub mod query;
pub mod samples;
pub mod spatial_tree;

pub use self::builder::*;
pub use self::query::*;
pub use self::samples::*;
pub use self::spatial_tree::*;

pub(crate) use self::background_field::*;
pub use self::background_field::{BackgroundFieldParams, BackgroundFieldType};
pub(crate) use self::neighbor_cache::Neighborhood;

#[derive(Copy, Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum SampleType {
    Vertex,
    Face,
}

/// Side of the implicit field. This is used to indicate a side of the implicit field with respect
/// to some iso-value, where `Above` refers to the potential above the iso-value and `Below` refers
/// to the potential below a certain iso-value.
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum Side {
    Above,
    Below,
}

/// Implicit surface type.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde_all", derive(serde::Serialize, serde::Deserialize))]
pub struct ImplicitSurfaceBase<T = f64>
where
    T: Scalar,
{
    /// Enum for choosing how to compute a background potential field that may be mixed in with
    /// the local potentials.
    pub bg_field_params: BackgroundFieldParams,

    /// Surface triangles representing the surface discretization to be approximated.
    /// This topology also defines the normals to the surface.
    pub surface_topo: Vec<[usize; 3]>,

    /// Save the vertex positions of the mesh because the samples may not coincide (e.g. face
    /// centered samples).
    pub surface_vertex_positions: Vec<[T; 3]>,

    /// Sample points defining the entire implicit surface.
    pub samples: Samples<T>,

    /// Vertex neighborhood topology. For each vertex, this vector stores all the indices to
    /// adjacent triangles.
    pub dual_topo: Vec<Vec<usize>>,

    /// The type of implicit surface. For example should the samples be centered at vertices or
    /// face centroids.
    pub sample_type: SampleType,

    /// Local search tree for fast proximity queries.
    pub spatial_tree: RTree<Sample<T>>,
}

#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde_all", derive(serde::Serialize, serde::Deserialize))]
pub enum MLS<T = f64>
where
    T: Scalar,
{
    Local(LocalMLS<T>),
    Global(GlobalMLS<T>),
}

#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde_all", derive(serde::Serialize, serde::Deserialize))]
pub struct LocalMLS<T = f64>
where
    T: Scalar,
{
    pub kernel: LocalKernel,

    pub base_radius: f64,

    /// The `max_step` parameter sets the maximum position change allowed between calls to retrieve
    /// the derivative sparsity pattern.
    ///
    /// If this is set too large, the derivative may be denser than then needed, which typically
    /// results in slower performance.  If it is set too low, there may be errors in the
    /// derivative. It is the callers responsibility to set this step accurately using
    /// `update_max_step`. If the implicit surface is not changing, leave this at
    /// 0.0.
    pub max_step: T,

    pub surf_base: Box<ImplicitSurfaceBase<T>>,
}

impl<T: Scalar> LocalMLS<T> {
    /// Return the absolute radius of this kernel.
    #[inline]
    pub fn radius(&self) -> f64 {
        self.base_radius * self.kernel.radius_multiplier()
    }
}
impl<T: Real> LocalMLS<T> {
    /// Creates a clone of this `LocalMLS` with all reals cast to the given type.
    pub fn clone_cast<S: Real>(&self) -> LocalMLS<S> {
        LocalMLS {
            kernel: self.kernel,
            base_radius: self.base_radius,
            max_step: S::from(self.max_step).unwrap(),
            surf_base: Box::new(self.surf_base.clone_cast::<S>()),
        }
    }
}

impl<T: Real> GlobalMLS<T> {
    /// Creates a clone of this `GlobalMLS` with all reals cast to the given type.
    pub fn clone_cast<S: Real>(&self) -> GlobalMLS<S> {
        GlobalMLS {
            kernel: self.kernel,
            surf_base: Box::new(self.surf_base.clone_cast::<S>()),
        }
    }
}

#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde_all", derive(serde::Serialize, serde::Deserialize))]
pub struct GlobalMLS<T = f64>
where
    T: Scalar,
{
    pub kernel: GlobalKernel,
    pub surf_base: Box<ImplicitSurfaceBase<T>>,
}

#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct HrbfSurface<T = f64>
where
    T: Scalar,
{
    pub surf_base: Box<ImplicitSurfaceBase<T>>,
}

#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde_all", derive(serde::Serialize, serde::Deserialize))]
pub enum ImplicitSurface<T = f64>
where
    T: Scalar,
{
    Hrbf(HrbfSurface<T>),
    MLS(MLS<T>),
}

impl<T: Real> ImplicitSurfaceBase<T> {
    /// Creates a clone of this `ImplicitSurfaceBase` with all reals cast to the given type.
    pub fn clone_cast<S: Real>(&self) -> ImplicitSurfaceBase<S> {
        use tensr::AsTensor;
        let samples = self.samples.clone_cast::<S>();
        let spatial_tree = build_rtree_from_samples(&samples);
        ImplicitSurfaceBase {
            bg_field_params: self.bg_field_params,
            surface_topo: self.surface_topo.clone(),
            surface_vertex_positions: self
                .surface_vertex_positions
                .iter()
                .map(|x| x.as_tensor().cast::<S>().into())
                .collect(),
            samples,
            dual_topo: self.dual_topo.clone(),
            sample_type: self.sample_type,
            spatial_tree,
        }
    }
    /// Reverse the direction of the normals.
    ///
    /// This effectively swaps the sign of the implicit potential.
    pub fn reverse(&mut self) -> &mut Self {
        self.samples
            .normals
            .iter_mut()
            .for_each(|[ref mut x, ref mut y, ref mut z]| {
                *x = -*x;
                *y = -*y;
                *z = -*z
            });
        self
    }

    /// Reverse the direction of the normals in parallel.
    ///
    /// This can be beneficial for very large meshes.
    pub fn reverse_par(&mut self) -> &mut Self {
        self.samples
            .normals
            .par_iter_mut()
            .for_each(|[ref mut x, ref mut y, ref mut z]| {
                *x = -*x;
                *y = -*y;
                *z = -*z
            });
        self
    }

    /// Update the stored samples. This assumes that vertex positions have been updated.
    fn update_samples(&mut self) {
        let ImplicitSurfaceBase {
            ref mut samples,
            ref surface_topo,
            ref surface_vertex_positions,
            sample_type,
            ..
        } = self;

        match sample_type {
            SampleType::Vertex => {
                let Samples {
                    ref mut positions,
                    ref mut normals,
                    // ref mut tangents
                    ..
                } = samples;

                surface_vertex_positions.iter().zip(positions.iter_mut()).for_each(|(vertex_pos, sample_pos)| {
                    *sample_pos = *vertex_pos;
                });

                // Compute unnormalized area weighted vertex normals given a triangle topology.
                geo::algo::compute_vertex_area_weighted_normals(positions, surface_topo, normals);
            }
            SampleType::Face => {
                samples.update_triangle_samples(surface_topo, surface_vertex_positions);
            }
        }
    }

    /// Update the stored samples. This assumes that vertex positions have been updated.
    fn update_samples_par(&mut self) {
        let ImplicitSurfaceBase {
            ref mut samples,
            ref surface_topo,
            ref surface_vertex_positions,
            sample_type,
            ..
        } = self;

        match sample_type {
            SampleType::Vertex => {
                let Samples {
                    ref mut positions,
                    ref mut normals,
                    // ref mut tangents
                    ..
                } = samples;

                surface_vertex_positions.par_iter().zip(positions.par_iter_mut()).for_each(|(vertex_pos, sample_pos)| {
                    *sample_pos = *vertex_pos;
                });

                // Compute unnormalized area weighted vertex normals given a triangle topology.
                // (Not parallel)
                geo::algo::compute_vertex_area_weighted_normals(positions, surface_topo, normals);
            }
            SampleType::Face => {
                samples.update_triangle_samples_par(surface_topo, surface_vertex_positions);
            }
        }
    }

    /// Update vertex positions and samples using an iterator over mesh vertices.
    ///
    /// This is a very
    /// permissive `update` function, which will update as many positions as possible and recompute
    /// the implicit surface data (like samples and spatial tree if needed) whether or not enough
    /// positions were specified to cover all surface vertices. This function will return the
    /// number of vertices that were indeed updated.
    pub fn update<I>(&mut self, vertex_iter: I) -> usize
    where
        I: Iterator<Item = [T; 3]>,
    {
        // First we update the surface vertex positions.
        let mut num_updated = 0;
        for (p, new_p) in self.surface_vertex_positions.iter_mut().zip(vertex_iter) {
            *p = new_p;
            num_updated += 1;
        }

        // Then update the samples that determine the shape of the implicit surface.
        self.update_samples();

        // Finally update the rtree responsible for neighbor search.
        self.spatial_tree = build_rtree_from_samples(&self.samples);

        num_updated
    }

    /// Update vertex positions and samples using an iterator over mesh vertices.
    ///
    /// Parallel version of `update`.
    pub fn update_par<I>(&mut self, vertex_iter: I) -> usize
        where
            I: IndexedParallelIterator<Item = [T; 3]>,
    {
        // First we update the surface vertex positions.
        let num_updated = self.surface_vertex_positions.par_iter_mut().zip(vertex_iter).map(|(p, new_p)| {
            *p = new_p;
        }).count();

        // Then update the samples that determine the shape of the implicit surface.
        self.update_samples_par();

        // Finally update the rtree responsible for neighbor search.
        self.spatial_tree = build_rtree_from_samples(&self.samples);

        num_updated
    }
}

impl<T: Real> MLS<T> {
    fn base(&self) -> &ImplicitSurfaceBase<T> {
        match self {
            MLS::Local(LocalMLS { surf_base, .. }) | MLS::Global(GlobalMLS { surf_base, .. }) => {
                surf_base
            }
        }
    }

    fn base_mut(&mut self) -> &mut ImplicitSurfaceBase<T> {
        match self {
            MLS::Local(LocalMLS { surf_base, .. }) | MLS::Global(GlobalMLS { surf_base, .. }) => {
                surf_base
            }
        }
    }

    /// Radius of influence ( kernel radius ) for this implicit surface.
    pub fn radius(&self) -> f64 {
        match self {
            MLS::Local(local) => local.radius(),
            MLS::Global(_) => std::f64::INFINITY,
        }
    }

    /// Return the surface vertex positions used by this implicit surface.
    pub fn surface_vertex_positions(&self) -> &[[T; 3]] {
        &self.base().surface_vertex_positions
    }

    /// Return the surface topology used by this implicit surface.
    pub fn surface_topology(&self) -> &[[usize; 3]] {
        &self.base().surface_topo
    }

    /// Return the number of samples used by this implicit surface.
    pub fn samples(&self) -> &Samples<T> {
        &self.base().samples
    }

    /// Return the number of samples used by this implicit surface.
    pub fn num_samples(&self) -> usize {
        self.base().samples.len()
    }

    /// Build a query topology type to be able to ask questions about the surface potential and
    /// derivatives at a set of query points.
    ///
    /// This type contains the necessary neighborhood information to make queries fast.
    pub fn query_topo<Q>(self, query_points: Q) -> QueryTopo<T>
    where
        Q: AsRef<[[T; 3]]>,
    {
        QueryTopo::new(query_points.as_ref(), self)
    }

    /// Reverse the direction of the normals.
    ///
    /// This effectively swaps the sign of the implicit potential.
    pub fn reverse(&mut self) -> &mut Self {
        self.base_mut().reverse();
        self
    }

    /// Reverse the direction of the normals in parallel.
    ///
    /// This can be beneficial for very large meshes.
    pub fn reverse_par(&mut self) -> &mut Self {
        self.base_mut().reverse_par();
        self
    }

    /// Update vertex positions and samples using an iterator over mesh vertices.
    ///
    /// This is a very permissive `update` function, which will update as many positions as
    /// possible and recompute the implicit surface data (like samples and spatial tree if needed)
    /// whether or not enough positions were specified to cover all surface vertices. This function
    /// will return the number of vertices that were indeed updated.
    pub fn update<I>(&mut self, vertex_iter: I) -> usize
    where
        I: Iterator<Item = [T; 3]>,
    {
        self.base_mut().update(vertex_iter)
    }

    /// The `max_step` parameter sets the maximum position change allowed between calls to
    /// retrieve the derivative sparsity pattern (this function). If this is set too large, the
    /// derivative will be denser than then needed, which typically results in slower performance.
    /// If it is set too low, there will be errors in the derivative. It is the callers
    /// responsibility to set this step accurately.
    pub fn update_max_step(&mut self, max_step: T) {
        if let MLS::Local(local) = self {
            local.max_step = max_step;
        }
    }

    pub fn update_radius_multiplier(&mut self, new_radius_multiplier: f64) {
        if let MLS::Local(LocalMLS { kernel, .. }) = self {
            *kernel = kernel.with_radius_multiplier(new_radius_multiplier);
        }
    }

    /*
     * Compute MLS potential on mesh
     */

    /// Implementation of the Moving Least Squares algorithm for computing an implicit surface.
    fn compute_on_mesh<K, F, M>(self, mesh: &mut M, kernel: K, interrupt: F) -> Result<(), Error>
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

impl<T: Real> ImplicitSurface<T> {
    fn base(&self) -> &ImplicitSurfaceBase<T> {
        match self {
            ImplicitSurface::MLS(mls) => mls.base(),
            ImplicitSurface::Hrbf(HrbfSurface { surf_base }) => surf_base,
        }
    }

    fn base_mut(&mut self) -> &mut ImplicitSurfaceBase<T> {
        match self {
            ImplicitSurface::MLS(mls) => mls.base_mut(),
            ImplicitSurface::Hrbf(HrbfSurface { surf_base }) => surf_base,
        }
    }

    /// Radius of influence ( kernel radius ) for this implicit surface.
    pub fn radius(&self) -> f64 {
        match self {
            ImplicitSurface::MLS(mls) => mls.radius(),
            ImplicitSurface::Hrbf(_) => std::f64::INFINITY,
        }
    }

    /// Return the surface vertex positions used by this implicit surface.
    pub fn surface_vertex_positions(&self) -> &[[T; 3]] {
        &self.base().surface_vertex_positions
    }

    /// Return the surface topology used by this implicit surface.
    pub fn surface_topology(&self) -> &[[usize; 3]] {
        &self.base().surface_topo
    }

    /// Return the number of samples used by this implicit surface.
    pub fn samples(&self) -> &Samples<T> {
        &self.base().samples
    }

    /// Return the number of samples used by this implicit surface.
    pub fn num_samples(&self) -> usize {
        self.base().samples.len()
    }

    /// Reverse the direction of the normals.
    ///
    /// This effectively swaps the sign of the implicit potential.
    pub fn reverse(&mut self) -> &mut Self {
        self.base_mut().reverse();
        self
    }

    /// Reverse the direction of the normals in parallel.
    ///
    /// This can be beneficial for very large meshes.
    pub fn reverse_par(&mut self) -> &mut Self {
        self.base_mut().reverse_par();
        self
    }

    /// Update vertex positions and samples using an iterator over mesh vertices. This is a very
    /// permissive `update` function, which will update as many positions as possible and recompute
    /// the implicit surface data (like samples and spatial tree if needed) whether or not enough
    /// positions were specified to cover all surface vertices. This function will return the
    /// number of vertices that were indeed updated.
    pub fn update<I>(&mut self, vertex_iter: I) -> usize
    where
        I: Iterator<Item = [T; 3]>,
    {
        self.base_mut().update(vertex_iter)
    }

    /// The `max_step` parameter sets the maximum position change allowed between calls to
    /// retrieve the derivative sparsity pattern (this function). If this is set too large, the
    /// derivative will be denser than then needed, which typically results in slower performance.
    /// If it is set too low, there will be errors in the derivative. It is the callers
    /// responsibility to set this step accurately.
    pub fn update_max_step(&mut self, max_step: T) {
        if let ImplicitSurface::MLS(mls) = self {
            mls.update_max_step(max_step);
        }
    }

    pub fn update_radius_multiplier(&mut self, new_radius_multiplier: f64) {
        if let ImplicitSurface::MLS(mls) = self {
            mls.update_radius_multiplier(new_radius_multiplier);
        }
    }

    /*
     * The methods below are designed for debugging and visualization.
     */

    /// Compute the implicit surface potential on the given polygon mesh.
    pub fn compute_potential_on_mesh<F, M>(self, mesh: &mut M, interrupt: F) -> Result<(), Error>
    where
        F: Fn() -> bool + Sync + Send,
        M: VertexMesh<T>,
    {
        match self {
            ImplicitSurface::MLS(mls) => {
                apply_kernel_fn!(mls, |kernel| mls.compute_on_mesh(mesh, kernel, interrupt))
            }
            ImplicitSurface::Hrbf(HrbfSurface { surf_base }) => {
                Self::compute_hrbf_on_mesh(mesh, &surf_base.samples, interrupt)
            }
        }
    }

    pub fn compute_hrbf(
        query_points: &[[T; 3]],
        samples: &Samples<T>,
        out_potential: &mut [T],
    ) -> Result<(), Error> {
        debug_assert!(
            query_points.iter().all(|&q| q.iter().all(|&x| !x.is_nan())),
            "Detected NaNs in query points. Please report this bug."
        );

        let Samples {
            ref positions,
            ref normals,
            ref values,
            ..
        } = samples;

        let pts: Vec<na::Point3<f64>> = positions
            .iter()
            .map(|&p| {
                let pos: [f64; 3] = Vector3::new(p).cast::<f64>().into();
                na::Point3::from(pos)
            })
            .collect();
        let nmls: Vec<na::Vector3<f64>> = normals
            .iter()
            .map(|&n| {
                let nml: [f64; 3] = Vector3::new(n).cast::<f64>().into();
                na::Vector3::from(nml).normalize()
            })
            .collect();

        let hrbf_values: Vec<f64> = values.iter().map(|&x| x.to_f64().unwrap()).collect();

        let hrbf = hrbf::Pow3HrbfBuilder::<f64>::new(pts)
            .offsets(hrbf_values)
            .normals(nmls)
            .build()
            .unwrap();

        query_points
            .par_iter()
            .zip(out_potential.par_iter_mut())
            .for_each(|(q, potential)| {
                let pos: [f64; 3] = Vector3::new(*q).cast::<f64>().into();
                *potential = T::from(hrbf.eval(na::Point3::from(pos))).unwrap();
            });

        Ok(())
    }

    pub fn compute_hrbf_on_mesh<F, M>(
        mesh: &mut M,
        samples: &Samples<T>,
        interrupt: F,
    ) -> Result<(), Error>
    where
        T: geo::Real,
        F: Fn() -> bool + Sync + Send,
        M: VertexMesh<T>,
    {
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

        let Samples {
            ref positions,
            ref normals,
            ref values,
            ..
        } = samples;
        let sample_pos = mesh.vertex_positions().to_vec();

        let pts: Vec<na::Point3<f64>> = positions
            .iter()
            .map(|&p| {
                let pos: [f64; 3] = Vector3::new(p).cast::<f64>().into();
                na::Point3::from(pos)
            })
            .collect();
        let nmls: Vec<na::Vector3<f64>> = normals
            .iter()
            .map(|&n| {
                let nml: [f64; 3] = Vector3::new(n).cast::<f64>().into();
                na::Vector3::from(nml).normalize()
            })
            .collect();
        let hrbf_values: Vec<f64> = values.iter().map(|&x| x.to_f64().unwrap()).collect();
        let hrbf = hrbf::Pow3HrbfBuilder::<f64>::new(pts)
            .offsets(hrbf_values)
            .normals(nmls)
            .build()
            .unwrap();

        let result = sample_pos
            .par_iter()
            .zip(potential.par_iter_mut())
            .map(|(q, potential)| {
                if interrupt() {
                    return Err(Error::Interrupted);
                }

                let pos: [f64; 3] = Vector3::new(*q).cast::<f64>().into();
                *potential = hrbf.eval(na::Point3::from(pos)) as f32;

                Ok(())
            })
            .reduce(|| Ok(()), |acc, result| acc.and(result));

        mesh.set_attrib_data::<_, VertexIndex>("potential", potential)?;

        result
    }
}

/*
 * Potential function compoenents
 *
 * The following routines compute parts of various potential functions defining implicit surfaces.
 */

/// Compute the potential at a given query point. If the potential is invalid or nieghbourhood
/// is empty, `potential` is not modified, otherwise it's updated.
/// Note: passing the output parameter potential as a mut reference allows us to optionally mix
/// a pre-initialized custom global potential field with the local potential.
pub(crate) fn compute_potential_at<T, K>(
    q: Vector3<T>,
    samples: SamplesView<T>,
    kernel: K,
    bg_potential: BackgroundFieldParams,
    potential: &mut T,
) where
    T: Real,
    K: SphericalKernel<T> + Copy + std::fmt::Debug + Sync + Send,
{
    if samples.is_empty() {
        return;
    }

    let bg = BackgroundField::local(q, samples, kernel, bg_potential, Some(*potential)).unwrap();

    let weight_sum_inv = bg.weight_sum_inv();

    // Generate a background potential field for every query point. This will be mixed
    // in with the computed potentials for local methods.
    *potential = bg.compute_unnormalized_weighted_scalar_field() * weight_sum_inv;

    let local_field =
        compute_local_potential_at(q, samples, kernel, weight_sum_inv, bg.closest_sample_dist());

    *potential += local_field;
}

/// Compute the potential field (excluding background field) at a given query point.
pub(crate) fn compute_local_potential_at<T, K>(
    q: Vector3<T>,
    samples: SamplesView<T>,
    kernel: K,
    weight_sum_inv: T,
    closest_d: T,
) -> T
where
    T: Real,
    K: SphericalKernel<T> + Copy + std::fmt::Debug + Sync + Send,
{
    samples
        .iter()
        .map(
            |Sample {
                 pos, nml, value, ..
             }| {
                let w = kernel.with_closest_dist(closest_d).eval(q, pos);
                let p = value + nml.dot(q - pos) / nml.norm();
                w * p
            },
        )
        .sum::<T>()
        * weight_sum_inv
}

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

pub(crate) fn compute_local_vector_at<T, K>(
    q: Vector3<T>,
    samples: SamplesView<T>,
    kernel: K,
    bg_potential: BackgroundFieldParams,
    vector: &mut [T; 3],
) where
    T: Real,
    K: SphericalKernel<T> + Copy + std::fmt::Debug + Sync + Send,
{
    if samples.is_empty() {
        return;
    }

    let bg = BackgroundField::local(
        q,
        samples,
        kernel,
        bg_potential,
        Some(Vector3::new(*vector)),
    )
    .unwrap();

    // Generate a background potential field for every query point. This will be mixed
    // in with the computed potentials for local methods.
    let mut out_field = bg.compute_unnormalized_weighted_vector_field();

    let closest_dist = bg.closest_sample_dist();

    let weight_sum_inv = bg.weight_sum_inv();

    let grad_phi = jacobian::query_jacobian_at(q, samples, None, kernel, bg_potential);

    for Sample { pos, vel, nml, .. } in samples.iter() {
        out_field += jacobian::sample_contact_jacobian_product_at(
            q,
            pos,
            nml,
            kernel,
            grad_phi,
            weight_sum_inv,
            closest_dist,
            vel,
        );
    }

    *vector = out_field.into();
}

/*
 * Derivative and normal computation utils
 */

/// Compute the gradient vector product of the face normals with respect to
/// surface vertices.
///
/// This function returns an iterator with the same size as `surface_vertices.len()`.
///
/// Note that the product vector is given by a closure `multiplier` which must give a valid
/// vector value for any vertex index, however not all indices will be used since only the
/// neighborhood of vertex at `index` will have non-zero gradients.
pub(crate) fn compute_face_unit_normals_gradient_products<'a, T, F>(
    samples: SamplesView<'a, 'a, T>,
    surface_vertices: &'a [[T; 3]],
    surface_topo: &'a [[usize; 3]],
    mut multiplier: F,
) -> impl Iterator<Item = Vector3<T>> + 'a
where
    T: Real,
    F: FnMut(Sample<T>) -> Vector3<T> + 'a,
{
    samples.into_iter().flat_map(move |sample| {
        let mult = multiplier(sample);
        let grad =
            face_unit_normal_gradient_iter_from_sample(sample, surface_vertices, surface_topo);
        grad.map(move |g| g * mult)
    })
}

/// Parallel version of `compute_face_unit_normals_gradient_products`
#[allow(dead_code)]
pub(crate) fn compute_face_unit_normals_gradient_products_par<'a, T, F>(
    samples: SamplesView<'a, 'a, T>,
    surface_vertices: &'a [[T; 3]],
    surface_topo: &'a [[usize; 3]],
    multiplier: F,
) -> impl IndexedParallelIterator<Item = Matrix3<T>> + 'a
where
    T: Real,
    F: Fn(Sample<T>) -> Vector3<T> + Send + Sync + 'a,
{
    samples.into_par_iter().map(move |sample| {
        let mult = multiplier(sample);
        face_unit_normal_gradient_product_from_sample(sample, surface_vertices, surface_topo, mult)
    })
}

pub(crate) fn face_unit_normal_gradient_iter_from_sample<T>(
    sample: Sample<T>,
    surface_vertices: &[[T; 3]],
    surface_topo: &[[usize; 3]],
) -> impl Iterator<Item = Matrix3<T>>
where
    T: Real,
{
    let mut sample_unit_nml = sample.nml;
    let sample_area = sample_unit_nml.normalize();
    face_unit_normal_gradient_iter(
        sample.index,
        sample_unit_nml,
        T::one() / sample_area,
        surface_vertices,
        surface_topo,
    )
}

/// Compute the gradient of the face normal at the given sample with respect to
/// its vertices. The returned triple of `Matrix3`s corresponds to the block column vector of
/// three matrices corresponding to each triangle vertex, which together construct the actual
/// `9x3` component-wise gradient.
pub(crate) fn face_unit_normal_gradient_iter<T>(
    sample_index: usize,
    sample_unit_nml: Vector3<T>,
    sample_area_inv: T,
    surface_vertices: &[[T; 3]],
    surface_topo: &[[usize; 3]],
) -> impl Iterator<Item = Matrix3<T>>
where
    T: Real,
{
    let nml_proj = scaled_tangent_projection(sample_unit_nml, sample_area_inv);
    let tri_indices = &surface_topo[sample_index];
    let tri = Triangle::from_indexed_slice(tri_indices, surface_vertices);
    // Note: Implicit transpose when interpreting column-major matrix as a row-major matrix.
    (0..3).map(move |i| -Matrix3::new(tri.area_normal_gradient(i)) * nml_proj)
}

#[allow(dead_code)]
pub(crate) fn face_unit_normal_gradient_product_from_sample<T>(
    sample: Sample<T>,
    surface_vertices: &[[T; 3]],
    surface_topo: &[[usize; 3]],
    mult: Vector3<T>,
) -> Matrix3<T>
where
    T: Real,
{
    let mut sample_unit_nml = sample.nml;
    let sample_area = sample_unit_nml.normalize();
    face_unit_normal_gradient_product(
        sample.index,
        sample_unit_nml,
        T::one() / sample_area,
        surface_vertices,
        surface_topo,
        mult,
    )
}

#[allow(dead_code)]
pub(crate) fn face_unit_normal_gradient_product<T>(
    sample_index: usize,
    sample_unit_nml: Vector3<T>,
    sample_area_inv: T,
    surface_vertices: &[[T; 3]],
    surface_topo: &[[usize; 3]],
    mult: Vector3<T>,
) -> Matrix3<T>
where
    T: Real,
{
    let nml_proj_mult = scaled_tangent_projection(sample_unit_nml, sample_area_inv) * mult;
    let tri_indices = &surface_topo[sample_index];
    let tri = Triangle::from_indexed_slice(tri_indices, surface_vertices);
    // Note: Implicit transpose when interpreting column-major matrix as a row-major matrix.
    Matrix3::new([
        (-Matrix3::new(tri.area_normal_gradient(0)) * nml_proj_mult).into(),
        (-Matrix3::new(tri.area_normal_gradient(1)) * nml_proj_mult).into(),
        (-Matrix3::new(tri.area_normal_gradient(2)) * nml_proj_mult).into(),
    ])
}

/// Compute the matrix for projecting on the tangent plane of the given sample inversely scaled
/// by the local area (normal norm reciprocal).
pub(crate) fn scaled_tangent_projection<T>(
    sample_unit_nml: Vector3<T>,
    scale_factor: T,
) -> Matrix3<T>
where
    T: Real,
{
    Matrix3::from_diag_iter(std::iter::repeat(scale_factor))
        - (sample_unit_nml * scale_factor) * sample_unit_nml.transpose()
}

pub(crate) fn scaled_tangent_projection_from_sample<T>(sample: Sample<T>) -> Matrix3<T>
where
    T: Real,
{
    let mut sample_unit_nml = sample.nml;
    let sample_area = sample_unit_nml.normalize();
    scaled_tangent_projection(sample_unit_nml, T::one() / sample_area)
}

/// Compute the gradient vector product of the `compute_vertex_unit_normals` function with respect to
/// vertices given in the sample view.
///
/// This function returns an iterator with the same size as `samples.len()`.
///
/// Note that the product vector is given by a closure `dx` which must give a valid vector
/// value for any vertex index, however not all indices will be used since only the
/// neighborhood of vertex at `index` will have non-zero gradients.
pub(crate) fn compute_vertex_unit_normals_gradient_products<'a, T, F>(
    samples: SamplesView<'a, 'a, T>,
    surface_topo: &'a [[usize; 3]],
    dual_topo: &'a [Vec<usize>],
    dx: F,
) -> impl Iterator<Item = Vector3<T>> + 'a
where
    T: Real,
    F: Fn(Sample<T>) -> Vector3<T> + Copy + 'a,
{
    samples.into_iter().map(move |sample| {
        compute_vertex_unit_normal_gradient_product(sample, samples, surface_topo, dual_topo, dx)
    })
}
pub(crate) fn compute_vertex_unit_normals_gradient_products_par<'a, T, F>(
    samples: SamplesView<'a, 'a, T>,
    surface_topo: &'a [[usize; 3]],
    dual_topo: &'a [Vec<usize>],
    dx: F,
) -> impl IndexedParallelIterator<Item = Vector3<T>> + 'a
where
    T: Real,
    F: Fn(Sample<T>) -> Vector3<T> + Copy + Send + Sync + 'a,
{
    samples.into_par_iter().map(move |sample| {
        compute_vertex_unit_normal_gradient_product(sample, samples, surface_topo, dual_topo, dx)
    })
}

pub(crate) fn compute_vertex_unit_normal_gradient_product<'a, T, F>(
    sample: Sample<T>,
    samples: SamplesView<'a, 'a, T>,
    surface_topo: &'a [[usize; 3]],
    dual_topo: &'a [Vec<usize>],
    dx: F,
) -> Vector3<T>
where
    T: Real,
    F: Fn(Sample<T>) -> Vector3<T> + 'a,
{
    let Sample { index, nml, .. } = sample;
    let norm_inv = T::one() / nml.norm();
    // Compute the normal component of the derivative
    let nml_proj = Matrix3::identity() - nml * (nml.transpose() * (norm_inv * norm_inv));
    let mut nml_deriv = Vector3::zero();
    // Look at the ring of triangles around the vertex with respect to which we are
    // taking the derivative.
    for &tri_idx in dual_topo[index].iter() {
        let tri_indices = &surface_topo[tri_idx];
        // Pull contributions from all neighbors on the surface, not just ones part of the
        // neighborhood,
        let tri = Triangle::from_indexed_slice(tri_indices, samples.all_points());
        // Note that area_normal_gradient returns column-major matrix, but we use row-major,
        // since area_normal_gradient is skew symmetric, we interpret this as a negation.
        let nml_grad = Matrix3::new(
            tri.area_normal_gradient(
                tri_indices
                    .iter()
                    .position(|&j| j == index)
                    .expect("Triangle mesh topology corruption."),
            ),
        );
        let mut tri_grad = nml_proj * (dx(sample) * norm_inv);
        for sample in SamplesView::from_view(tri_indices, samples).into_iter() {
            if sample.index != index {
                let normk_inv = T::one() / sample.nml.norm();
                let nmlk_proj = Matrix3::identity()
                    - sample.nml * (sample.nml.transpose() * (normk_inv * normk_inv));
                tri_grad += nmlk_proj * (dx(sample) * normk_inv);
            }
        }
        nml_deriv -= nml_grad * tri_grad;
    }
    nml_deriv
}

/// Generate a tetrahedron with vertex positions and indices for the triangle faces.
#[cfg(test)]
pub(crate) fn make_tet() -> (Vec<[f64; 3]>, Vec<[usize; 3]>) {
    use geo::mesh::{builder, TriMesh};
    let tet = builder::PlatonicSolidBuilder::build_tetrahedron();
    let TriMesh {
        vertex_positions,
        indices,
        ..
    } = TriMesh::from(tet);
    let tet_verts = vertex_positions.into_vec();
    let tet_faces = indices.into_vec();

    (tet_verts, tet_faces)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::*;
    use geo::mesh::{builder::PlatonicSolidBuilder, *};
    use geo::ops::transform::*;

    #[test]
    fn size_test() -> Result<(), Error> {
        use std::mem::size_of;
        eprintln!("Vec<usize>: {}", size_of::<Vec<usize>>());
        eprintln!("MLS: {}", size_of::<MLS>());
        eprintln!("LocalMLS: {}", size_of::<LocalMLS>());
        eprintln!("GlobalMLS: {}", size_of::<GlobalMLS>());
        eprintln!("HrbfSurface: {}", size_of::<HrbfSurface>());
        eprintln!("ImplicitSurface: {}", size_of::<ImplicitSurface>());
        eprintln!("ImplicitSurfaceBase: {}", size_of::<ImplicitSurfaceBase>());
        eprintln!("QueryTopo: {}", size_of::<QueryTopo>());
        eprintln!("Neighborhood: {}", size_of::<Neighborhood>());
        Ok(())
    }

    // Helper function for testing. This is an implicit surface and grid mesh pair where each
    // vertex of the grid mesh has a non-empty local neighbpourhood of the implicit surface.
    // The `reverse` option reverses each triangle in the sphere to create an inverted implicit
    // surface.
    fn make_octahedron_and_grid_local(reverse: bool) -> Result<(MLS, PolyMesh<f64>), Error> {
        // Create a surface sample mesh.
        let octahedron_trimesh = PlatonicSolidBuilder::build_octahedron();
        let mut sphere = PolyMesh::from(octahedron_trimesh);
        if reverse {
            sphere.reverse();
        }

        // Translate the mesh slightly in z.
        sphere.translate([0.0, 0.0, 0.2]);

        // Construct the implicit surface.
        let surface = mls_from_polymesh(
            &sphere,
            Params {
                kernel: KernelType::Approximate {
                    tolerance: 0.00001,
                    radius_multiplier: 2.45,
                },
                background_field: BackgroundFieldParams {
                    field_type: BackgroundFieldType::DistanceBased,
                    // Distance based background is discontinuous, this is bad for projection, so
                    // we always opt for an unweighted background to make sure that local
                    // potentials are always high quality
                    weighted: false,
                },
                sample_type: SampleType::Vertex,
                ..Default::default()
            },
        )?;

        // Make a mesh to be projected.
        let grid = make_grid(23, 23);

        Ok((surface, grid))
    }

    // Helper function for testing. This is an implicit surface and grid mesh pair where each
    // vertex of the grid mesh has a non-empty local neighbpourhood of the implicit surface.
    // The `reverse` option reverses each triangle in the sphere to create an inverted implicit
    // surface.
    fn make_octahedron_and_grid(
        reverse: bool,
        radius_multiplier: f64,
    ) -> Result<(MLS, PolyMesh<f64>), Error> {
        // Create a surface sample mesh.
        let octahedron_trimesh = PlatonicSolidBuilder::build_octahedron();
        let mut sphere = PolyMesh::from(octahedron_trimesh);
        if reverse {
            sphere.reverse();
        }

        // Translate the mesh slightly in z.
        sphere.translate([0.0, 0.0, 0.2]);

        // Construct the implicit surface.
        let surface = mls_from_polymesh(
            &sphere,
            Params {
                kernel: KernelType::Approximate {
                    tolerance: 0.00001,
                    radius_multiplier,
                },
                background_field: BackgroundFieldParams {
                    field_type: BackgroundFieldType::DistanceBased,
                    weighted: false,
                },
                sample_type: SampleType::Vertex,
                ..Default::default()
            },
        )?;

        // Make a mesh to be projected.
        let mut grid = make_grid(22, 22);

        grid.uniform_scale(2.0);

        Ok((surface, grid))
    }

    fn projection_tester(
        query_surf: &QueryTopo,
        mut grid: PolyMesh<f64>,
        side: Side,
    ) -> Result<(), Error> {
        let epsilon = 1e-4;

        // Make a copy of the grid for the parallel projection test.
        let mut grid_par = grid.clone();

        let init_potential = {
            // Get grid node positions to be projected.
            let pos = grid.vertex_positions_mut();
            let pos_par = grid_par.vertex_positions_mut();

            // Compute potential before projection.
            let mut init_potential = vec![0.0; pos.len()];
            query_surf.potential(pos, &mut init_potential);

            // Project grid outside the implicit surface.
            assert!(query_surf.project(side, 0.0, epsilon, pos));
            assert!(query_surf.project_par(side, 0.0, epsilon, pos_par));
            init_potential
        };

        // Compute potential after projection.
        let mut final_potential = vec![0.0; init_potential.len()];
        query_surf.potential(grid.vertex_positions(), &mut final_potential);

        let mut final_potential_par = vec![0.0; init_potential.len()];
        query_surf.potential(grid_par.vertex_positions(), &mut final_potential_par);
        for (par, seq) in final_potential_par.iter().zip(final_potential.iter()) {
            assert_eq!(par, seq);
        }

        //use geo::mesh::topology::VertexIndex;
        //grid.set_attrib_data::<_, VertexIndex>("init_potential", &init_potential);
        //grid.set_attrib_data::<_, VertexIndex>("final_potential", &final_potential);
        //geo::io::save_polymesh(&grid, &std::path::PathBuf::from("out/mesh.vtk"))?;

        for (&old, &new) in init_potential.iter().zip(final_potential.iter()) {
            // Check that all vertices are outside the implicit solid.
            match side {
                Side::Above => {
                    assert!(new >= 0.0, "new = {}, old = {}", new, old);
                    if old < 0.0 {
                        // Check that the projected vertices are now within the narrow band of valid
                        // projections (between 0 and epsilon).
                        assert!(new <= epsilon, "new = {}", new);
                    }
                }
                Side::Below => {
                    assert!(new <= 0.0, "new = {}, old = {}", new, old);
                    if old > 0.0 {
                        // Check that the projected vertices are now within the narrow band of valid
                        // projections (between 0 and epsilon).
                        assert!(new >= -epsilon, "new = {}", new);
                    }
                }
            }
        }

        Ok(())
    }

    /// Test projection where each projected vertex has a non-empty local neighborhood of the
    /// implicit surface.
    #[test]
    fn local_projection_test() -> Result<(), Error> {
        let (surface, grid) = make_octahedron_and_grid_local(false)?;
        let query_surf = QueryTopo::new(grid.vertex_positions(), surface);
        projection_tester(&query_surf, grid, Side::Above)?;

        let (surface, grid) = make_octahedron_and_grid_local(true)?;
        let query_surf = QueryTopo::new(grid.vertex_positions(), surface);
        projection_tester(&query_surf, grid, Side::Below)
    }

    /// Test projection where some projected vertices may not have a local neighborhood at all.
    /// This is a more complex test than the local_projection_test
    #[test]
    fn global_projection_test() -> Result<(), Error> {
        let (surface, grid) = make_octahedron_and_grid(false, 2.45)?;
        let query_surf = QueryTopo::new(grid.vertex_positions(), surface);
        projection_tester(&query_surf, grid, Side::Above)?;

        let (surface, grid) = make_octahedron_and_grid(true, 2.45)?;
        let query_surf = QueryTopo::new(grid.vertex_positions(), surface);
        projection_tester(&query_surf, grid, Side::Below)
    }

    /// Test with a radius multiplier less than 1.0. Although not strictly useful, this should not
    /// crash.
    #[test]
    fn narrow_projection_test() -> Result<(), Error> {
        // Make a mesh to be projected.
        use geo::attrib::*;
        use geo::mesh::builder::*;
        use geo::mesh::topology::*;

        let mut grid = GridBuilder {
            rows: 18,
            cols: 19,
            orientation: AxisPlaneOrientation::ZX,
        }
        .build();
        grid.insert_attrib_with_default::<_, VertexIndex>("potential", 0.0f32)
            .unwrap();

        grid.reverse();

        grid.translate([0.0, 0.12639757990837097, 0.0]);

        let torus = geo::io::load_polymesh("assets/projection_torus.vtk")?;

        // Construct the implicit surface.
        let surface = mls_from_polymesh(
            &torus,
            Params {
                kernel: KernelType::Approximate {
                    tolerance: 0.000009999999747378752,
                    radius_multiplier: 0.7599999904632568,
                },
                background_field: BackgroundFieldParams {
                    field_type: BackgroundFieldType::DistanceBased,
                    weighted: false,
                },
                sample_type: SampleType::Face,
                ..Default::default()
            },
        )?;

        let query_surf = QueryTopo::new(grid.vertex_positions(), surface);
        projection_tester(&query_surf, grid, Side::Above)?;

        Ok(())
    }

    #[cfg(feature = "serde")]
    mod test_structs {
        use serde::{Deserialize, Serialize};
        #[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
        pub struct NeighborCache<T> {
            pub points: Vec<T>,
            pub valid: bool,
        }

        #[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
        pub struct Neighborhood {
            closest_set: NeighborCache<usize>,
            trivial_set: NeighborCache<Vec<usize>>,
            extended_set: NeighborCache<Vec<usize>>,
        }

        /// This struct helps deserialize testing assets without having to store an rtree.
        #[derive(Clone, Debug, Serialize, Deserialize)]
        pub struct ImplicitSurfaceNoTree {
            pub kernel: super::KernelType,
            pub base_radius: f64,
            pub bg_field_params: super::BackgroundFieldParams,
            pub surface_topo: Vec<[usize; 3]>,
            pub surface_vertex_positions: Vec<super::Vector3<f64>>,
            pub samples: super::Samples<f64>,
            pub max_step: f64,
            pub query_neighborhood: std::cell::RefCell<Neighborhood>,
            pub dual_topo: Vec<Vec<usize>>,
            pub sample_type: super::SampleType,
        }
    }

    /// Test a specific case where the projection direction can be zero, which could result in
    /// NaNs. This case must not crash.
    #[cfg(feature = "serde")]
    #[test]
    fn zero_step_projection_test() -> Result<(), Error> {
        use std::io::Read;
        let iso_value = 0.0;
        let epsilon = 0.0001;
        let mut query_points: Vec<[f64; 3]> = {
            let mut file = std::fs::File::open("assets/grid_points.json")
                .expect("Failed to open query points file");
            let mut contents = String::new();
            file.read_to_string(&mut contents)
                .expect("Failed to read grid points json.");
            serde_json::from_str(&contents).expect("Failed to deserialize grid points.")
        };

        let surface: MLS<f64> = {
            let mut file = std::fs::File::open("assets/torus_surf_no_tree.json")
                .expect("Failed to open torus surface file");
            let mut contents = String::new();
            file.read_to_string(&mut contents)
                .expect("Failed to read torus surface json.");
            let test_structs::ImplicitSurfaceNoTree {
                kernel,
                base_radius,
                bg_field_params,
                surface_topo,
                surface_vertex_positions,
                samples,
                max_step,
                dual_topo,
                sample_type,
                ..
            } = serde_json::from_str(&contents).expect("Failed to deserialize torus surface.");
            let spatial_tree = build_rtree_from_samples(&samples);
            MLS::Local(LocalMLS {
                kernel: kernel.into(),
                base_radius,
                max_step,
                surf_base: Box::new(ImplicitSurfaceBase {
                    bg_field_params,
                    surface_topo,
                    surface_vertex_positions,
                    samples,
                    dual_topo,
                    sample_type,
                    spatial_tree,
                }),
            })
        };

        let query_surf = QueryTopo::new(&query_points, surface);

        let init_potential = {
            // Compute potential before projection.
            let mut init_potential = vec![0.0; query_points.len()];
            query_surf.potential(&query_points, &mut init_potential);
            init_potential
        };

        // Project grid outside the implicit surface.
        assert!(query_surf.project_to_above(iso_value, epsilon, &mut query_points));

        // Compute potential after projection.
        let mut final_potential = vec![0.0; init_potential.len()];
        query_surf.potential(&query_points, &mut final_potential);

        for (i, (&old, &new)) in init_potential
            .iter()
            .zip(final_potential.iter())
            .enumerate()
        {
            // Check that all vertices are outside the implicit solid.
            assert!(new >= 0.0, "new = {}, old = {}, i = {}", new, old, i);
            if old < 0.0 {
                // Check that the projected vertices are now within the narrow band of valid
                // projections (between 0 and epsilon).
                assert!(new <= epsilon, "new = {}", new);
            }
        }

        Ok(())
    }

    #[test]
    fn neighborhoods() -> Result<(), Error> {
        // Local test
        let (surface, grid) = make_octahedron_and_grid_local(false)?;
        let query_surf = QueryTopo::new(grid.vertex_positions(), surface);
        assert_eq!(
            query_surf.num_neighborhoods(),
            query_surf.nonempty_neighborhood_indices().len()
        );

        // Non-local test
        let (surface, grid) = make_octahedron_and_grid(false, 2.45)?;
        let query_surf = QueryTopo::new(grid.vertex_positions(), surface);
        assert_eq!(
            query_surf.num_neighborhoods(),
            query_surf.nonempty_neighborhood_indices().len()
        );
        Ok(())
    }
}
