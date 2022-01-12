use super::point_contact::PointContactConstraint;
use crate::constraints::{ContactConstraint, ContactSurface};
use crate::matrix::MatrixElementIndex;
use crate::{
    ContactBasis, ContactJacobian, ContactJacobianView, Error, FrictionParams, FrictionWorkspace,
    Index, Real, TriMesh, ORIGINAL_VERTEX_INDEX_ATTRIB,
};
use autodiff as ad;
use flatk::{
    Chunked, Chunked1, Chunked3, CloneWithStorage, Offsets, Select, SelectView, Set, Sparse,
    Storage, StorageMut, Subset, SubsetView, UniChunked, View, U1, U3,
};
use geo::attrib::Attrib;
use geo::mesh::VertexMesh;
use geo::{NumVertices, VertexIndex};
use implicits::{KernelType, QueryTopo};
use lazycell::LazyCell;
use num_traits::Zero;
use rayon::iter::Either;
use rayon::prelude::*;
use tensr::{
    AsMutTensor, AsTensor, BlockDiagonalMatrix3, CwiseBinExprImpl, Expr, ExprMut, IndexedExpr,
    IntoData, IntoExpr, IntoTensor, Matrix, Matrix2, Matrix3, MulExpr, Multiplication, Scalar,
    Tensor, Vector2, Vector3,
};

pub type DistanceGradient<T = f64> = Tensor![T; S S 3 1];

#[derive(Clone, Debug)]
pub struct MappedDistanceGradient<T: Scalar> {
    /// Compressed sparse sparse row sparse column gradient matrix.
    pub matrix: DistanceGradient<T>,
    /// A Mapping from original triplets to the final compressed sparse matrix.
    pub mapping: Vec<Index>,
}

impl<T: Real> MappedDistanceGradient<T> {
    fn clone_cast<S: Real>(&self) -> MappedDistanceGradient<S> {
        let MappedDistanceGradient { matrix, mapping } = self;

        let storage: Vec<_> = matrix
            .storage()
            .iter()
            .map(|&x| S::from(x).unwrap())
            .collect();
        let new_matrix = matrix.clone().clone_with_storage(storage);

        MappedDistanceGradient {
            matrix: new_matrix,
            mapping: mapping.clone(),
        }
    }
}

fn clone_cast_ssblock_mtx<T: Real, S: Real>(jac: &Tensor![T; S S 3 3]) -> Tensor![S; S S 3 3] {
    let storage: Vec<_> = jac.storage().iter().map(|&x| S::from(x).unwrap()).collect();
    jac.clone().clone_with_storage(storage)
}

/*
 * Functions defining the friction presliding profile
 */

/// This function is the sliding profile divided by `x`.
///
/// The sliding profile defines the relationship between velocity magnitude (the input)
/// and the friction force magnitude (the output).
///
/// The division is done for numerical stability to avoid division by zero.
/// `s(x) = 1 / (x + 0.1 * eps)`
fn stabilized_sliding_profile<T: Real>(x: T, epsilon: T) -> T {
    // Quadratic smoothing function with compact support.
    // s(x) = 2/eps - x/eps^2
    //let s = |x| {
    //    if x < epsilon {
    //        T::from(2.0).unwrap()  / epsilon - x / (epsilon * epsilon)
    //    } else {
    //        T::one()
    //    }
    //};
    // Note that denominator is always >= 0.1eps since x > 0.
    T::one() / (x + T::from(0.1).unwrap() * epsilon)
}

/// Derivative of the sliding profile.
fn stabilized_sliding_profile_derivative<T: Real>(x: T, epsilon: T) -> T {
    let denom = x + T::from(0.1).unwrap() * epsilon;
    -T::one() / (denom * denom)
}

/// The full sliding profile including 1D direction.
///
/// This is the function eta in the paper.
pub fn eta<T: Real>(v: Vector2<T>, factor: T, epsilon: T) -> Vector2<T> {
    // This is similar to function s but with the norm of v multiplied through to avoid
    // degeneracies.
    let s = |x| stabilized_sliding_profile(x, epsilon);
    let norm_v = v.norm();
    if norm_v > T::zero() {
        v * (factor * s(norm_v))
    } else {
        Vector2::zero()
    }
}

/// Jacobian of the full directional 1D sliding profile
pub fn eta_jac<T: Real>(v: Vector2<T>, factor: T, epsilon: T) -> Matrix2<T> {
    let s = |x| stabilized_sliding_profile(x, epsilon);
    let ds = |x| stabilized_sliding_profile_derivative(x, epsilon);
    let norm_v = v.norm();
    if norm_v > T::zero() {
        v * (ds(norm_v) / norm_v) * (v.transpose() * factor)
            + Matrix2::identity() * (s(norm_v) * factor)
    } else {
        Matrix2::zero()
    }
}

// Helper function to compute the relative tangent velocity at the point of contact from the given Jacobians and surface velocities `v0`.
pub fn contact_tangent_velocity<T: Real>(
    jac: ContactJacobianView<T>,
    collider_v: SubsetView<SubsetView<Chunked3<&[T]>>>,
    v0: SubsetView<Chunked3<&[T]>>,
) -> Chunked3<Vec<T>> {
    // Compute relative velocity at the point of contact: `vc = J(x)v`
    let mut vc = (jac.into_tensor() * v0.into_tensor()).into_data();
    *&mut vc.expr_mut() -= collider_v.expr();
    vc
}

type SSBlock3<T> = Tensor![T; S S 3 3];

#[derive(Clone, Debug)]
struct FrictionJacobianWorkspace<T> {
    bc: Chunked3<Vec<T>>,
    contact_gradient_basis_eta_jac_basis_contact_jac: SSBlock3<T>,
    contact_gradient_basis_eta_jac_basis_jac_contact_jac: SSBlock3<T>,
    contact_gradient_basis_eta_jac_basis_jac: SSBlock3<T>,
    contact_gradient_jac_basis: SSBlock3<T>,
}

impl<T: Real> Default for FrictionJacobianWorkspace<T> {
    fn default() -> Self {
        let mtx = Sparse::from_dim(vec![], 0, Chunked::from_offsets(vec![0], Sparse::from_dim(vec![], 0, Chunked3::from_flat(Chunked3::default()))));
        FrictionJacobianWorkspace {
            bc: Default::default(),
            contact_gradient_basis_eta_jac_basis_contact_jac: mtx.clone(),
            contact_gradient_basis_eta_jac_basis_jac_contact_jac:mtx.clone(),
            contact_gradient_basis_eta_jac_basis_jac: mtx.clone(),
            contact_gradient_jac_basis: mtx,
        }
    }
}

/// A penalty based point contact constraint.
///
/// This is similar to `PointContactConstraint` but this constraint applies a penalty instead of
/// inequality to enforce the contact constraint. This makes it an *equality* constraint.
///
/// This constraint also keeps track of where each vertex maps within the global array unlike
/// `PointContactConstraint` which expects the caller to manage this information.
#[derive(Clone, Debug)]
pub struct PenaltyPointContactConstraint<T = f64>
where
    T: Scalar,
{
    pub point_constraint: PointContactConstraint<T>,
    /// Indices of original vertices for the implicit surface.
    pub implicit_surface_vertex_indices: Vec<usize>,
    /// Indices of original vertices for the collider.
    pub collider_vertex_indices: Vec<usize>,

    pub(crate) lambda: Vec<T>,
    pub distance_potential: Vec<T>,

    contact_jacobian: LazyCell<ContactJacobian<T>>,
    distance_gradient: LazyCell<MappedDistanceGradient<T>>,

    friction_jacobian_workspace: FrictionJacobianWorkspace<T>,
}

impl<T: Real> PenaltyPointContactConstraint<T> {
    pub fn clone_cast<S: Real>(&self) -> PenaltyPointContactConstraint<S> {
        let mut distance_gradient = LazyCell::new();
        if self.distance_gradient.filled() {
            distance_gradient.replace(self.distance_gradient.borrow().unwrap().clone_cast::<S>());
        }
        let mut contact_jacobian = LazyCell::new();
        if self.contact_jacobian.filled() {
            contact_jacobian.replace(clone_cast_ssblock_mtx::<T, S>(
                self.contact_jacobian.borrow().unwrap(),
            ));
        }
        PenaltyPointContactConstraint {
            point_constraint: self.point_constraint.clone_cast(),
            implicit_surface_vertex_indices: self.implicit_surface_vertex_indices.clone(),
            collider_vertex_indices: self.collider_vertex_indices.clone(),
            distance_gradient,
            lambda: self.lambda.iter().map(|&x| S::from(x).unwrap()).collect(),
            distance_potential: self
                .distance_potential
                .iter()
                .map(|&x| S::from(x).unwrap())
                .collect(),
            contact_jacobian,
            friction_jacobian_workspace: FrictionJacobianWorkspace::default(),
        }
    }

    pub fn new<VP: VertexMesh<f64>>(
        // Main object experiencing contact against its implicit surface representation.
        object: ContactSurface<&TriMesh, f64>,
        // Collision object consisting of points pushing against the solid object.
        collider: ContactSurface<&VP, f64>,
        kernel: KernelType,
        friction_params: Option<FrictionParams>,
        contact_offset: f64,
        num_vertices: usize,
    ) -> Result<Self, Error> {
        let implicit_surface_vertex_indices = object
            .mesh
            .attrib_clone_into_vec::<usize, VertexIndex>(ORIGINAL_VERTEX_INDEX_ATTRIB)
            .unwrap_or_else(|_| (0..object.mesh.num_vertices()).collect::<Vec<_>>());
        let collider_vertex_indices = collider
            .mesh
            .attrib_clone_into_vec::<usize, VertexIndex>(ORIGINAL_VERTEX_INDEX_ATTRIB)
            .unwrap_or_else(|_| (0..collider.mesh.num_vertices()).collect::<Vec<_>>());

        let constraint = PointContactConstraint::new(
            object,
            collider,
            kernel,
            friction_params,
            contact_offset,
            false, // Linearized penalty constraints are not supported
        )?;

        let mut penalty_constraint = PenaltyPointContactConstraint {
            point_constraint: constraint,
            implicit_surface_vertex_indices,
            collider_vertex_indices,
            distance_gradient: LazyCell::new(),
            contact_jacobian: LazyCell::new(),
            lambda: Vec::new(),
            distance_potential: Vec::new(),
            friction_jacobian_workspace: FrictionJacobianWorkspace::default(),
        };

        penalty_constraint.reset_distance_gradient(num_vertices);

        Ok(penalty_constraint)
    }

    /// Constructs a clone of this constraint with autodiff variables.
    pub fn clone_as_autodiff<S: Real>(&self) -> PenaltyPointContactConstraint<ad::FT<S>> {
        self.clone_cast::<ad::FT<S>>()
    }

    // Same as `update_collider_vertex_positions` but without knowledge about original vertex indices.
    pub fn update_surface_vertex_positions(&mut self, x: Chunked3<&[T]>) -> usize {
        let x = SubsetView::from_unique_ordered_indices(&self.implicit_surface_vertex_indices, x);
        self.point_constraint.update_surface_with_mesh_pos(x)
    }

    // Same as `update_collider_vertex_positions` but without knowledge about original vertex indices.
    pub fn update_collider_vertex_positions(&mut self, x: Chunked3<&[T]>) {
        let x = SubsetView::from_unique_ordered_indices(&self.collider_vertex_indices, x);
        self.point_constraint.update_collider_vertex_positions(x);
    }

    pub fn cached_distance_potential(&self, num_vertices: usize) -> Vec<T> {
        let mut output = vec![T::zero(); num_vertices];
        let distance_potential = self.point_constraint.cached_constraint_value();
        for (&idx, &dist) in self
            .collider_vertex_indices
            .iter()
            .zip(distance_potential.as_slice())
        {
            output[idx] = dist;
        }
        output
    }

    pub fn build_distance_gradient<S: Real>(
        indices: &[MatrixElementIndex],
        blocks: Chunked3<&[S]>,
        num_rows: usize,
        num_cols: usize,
    ) -> MappedDistanceGradient<S> {
        let nnz = indices.len();
        assert_eq!(nnz, blocks.len());
        // Construct a mapping from original triplets to final compressed matrix.
        let mut entries = (0..nnz).collect::<Vec<_>>();

        // Sort indices into row major order
        entries.sort_by(|&a, &b| {
            indices[a]
                .row
                .cmp(&indices[b].row)
                .then_with(|| indices[a].col.cmp(&indices[b].col))
        });

        let mut mapping = vec![Index::INVALID; entries.len()];
        let entries = entries
            .into_iter()
            .filter(|&i| indices[i].row < num_rows && indices[i].col < num_cols)
            .collect::<Vec<_>>();

        // We use tensr to build the CSR matrix since it allows us to track
        // where each element goes after compression.
        let triplet_iter = entries
            .iter()
            .map(|&i| (indices[i].row, indices[i].col, blocks[i]));

        let uncompressed = tensr::SSBlockMatrix3x1::from_block_triplets_iter_uncompressed(
            triplet_iter,
            num_rows,
            num_cols,
        );

        // Compress the CSR matrix.
        let matrix = uncompressed.pruned(
            |_, _, _| true,
            |src, dst| {
                mapping[entries[src]] = Index::new(dst);
            },
        );

        // DEBUG CODE
        //        use tensr::{BlockMatrix, Get};
        //        let mut j_dense =
        //            flatk::ChunkedN::from_flat_with_stride(matrix.num_total_cols(), vec![S::zero(); matrix.num_total_rows() * matrix.num_total_cols()]);
        //
        //        dbg!(matrix.num_total_cols());
        //        dbg!(matrix.num_total_rows());
        //
        //        // Clear j_dense
        //        for jd in j_dense.storage_mut().iter_mut() {
        //            *jd = S::zero();
        //        }
        //
        //        // Copy j_vals to j_dense
        //        for (row_idx, row, _) in matrix.as_data().iter() {
        //            for (col_idx, block, _) in row.iter() {
        //                for i in 0..3 {
        //                    let val = block.at(i)[0];
        //                    j_dense[3*row_idx + i][col_idx] += val;
        //                }
        //            }
        //        }
        //
        //        eprintln!("G = [");
        //        for jp in j_dense.iter() {
        //            for j in jp.iter() {
        //                eprint!("{:?} ", j);
        //            }
        //            eprintln!(";");
        //        }
        //        eprintln!("]");

        // END OF DEBUG CODE
        MappedDistanceGradient {
            matrix: matrix.into_data(),
            mapping,
        }
    }

    /// Initializes the constraint gradient sparsity pattern.
    pub fn reset_distance_gradient<'a>(&mut self, num_vertices: usize) {
        let (indices, blocks): (Vec<_>, Chunked3<Vec<_>>) = self
            .distance_jacobian_blocks_iter()
            .map(|(row, col, block)| (MatrixElementIndex { row: col, col: row }, block))
            .unzip();
        let num_constraints = self.constraint_size();
        self.distance_gradient
            .replace(Self::build_distance_gradient(
                indices.as_slice(),
                blocks.view(),
                num_vertices,
                num_constraints,
            ));
    }

    pub fn update_neighbors<'a>(&mut self, x: Chunked3<&'a [T]>) -> bool {
        self.update_state(x);

        let updated = self
            .point_constraint
            .implicit_surface
            .reset(self.point_constraint.collider_vertex_positions.as_arrays());

        // Updating neighbours invalidates the constraint gradient so we must recompute
        // the sparsity pattern here.

        if updated {
            self.reset_distance_gradient(x.len());
        }
        updated
    }

    /// Update the current state using the given position vector.
    pub fn update_state(&mut self, x: Chunked3<&[T]>) {
        let num_vertices_updated = self.update_surface_vertex_positions(x);
        assert_eq!(
            num_vertices_updated,
            self.point_constraint
                .implicit_surface
                .surface_vertex_positions()
                .len()
        );
        self.update_collider_vertex_positions(x);
        self.update_distance_potential();
    }

    /// Update the cached constraint gradient for efficient future derivative computations.
    ///
    /// This function assumes that the `constraint_gradient` field sparsity has already been
    /// initialized.
    pub fn update_constraint_gradient(&mut self) {
        use flatk::Isolate;

        let MappedDistanceGradient { matrix, mapping } = self
            .distance_gradient
            .borrow_mut()
            .expect("Uninitialized constraint gradient.");

        // Clear matrix.
        matrix
            .storage_mut()
            .par_iter_mut()
            .for_each(|x| *x = T::zero());
        let mut matrix_blocks = Chunked3::from_flat(matrix.storage_mut().as_mut_slice());

        // Fill Gradient matrix with values from triplets according to our precomputed mapping.
        let triplets = Self::distance_jacobian_blocks_iter_fn(
            &self.point_constraint,
            &self.implicit_surface_vertex_indices,
            &self.collider_vertex_indices,
        );
        for (&pos, (_, _, block)) in mapping.iter().zip(triplets) {
            if let Some(pos) = pos.into_option() {
                *matrix_blocks.view_mut().isolate(pos).as_mut_tensor() += block.as_tensor();
            }
        }
    }

    /// Prune contacts with zero contact_impulse and contacts without neighboring samples.
    /// This function outputs the indices of contacts as well as a pruned vector of impulses.
    pub fn in_contact_indices(
        &self,
        contact_impulse: &[T],
        potential: &[T],
    ) -> (Vec<usize>, Vec<usize>, Vec<T>) {
        self.point_constraint
            .in_contact_indices(contact_impulse, potential)
    }

    pub fn num_collider_vertices(&self) -> usize {
        self.point_constraint.collider_vertex_positions.len()
    }

    pub fn update_distance_potential(&mut self) {
        // Take a slice of lambda for this particular contact constraint.
        let num_constraints = self.constraint_size();

        self.distance_potential.clear();
        self.distance_potential.resize(num_constraints, T::zero());
        self.point_constraint
            .compute_nonlinear_constraint(self.distance_potential.as_mut_slice());
    }

    /// Computes the derivative of a cubic penalty function for contacts multiplied by `-κ`.
    pub fn update_multipliers(&mut self, delta: f32, kappa: f32) {
        let dist = self.distance_potential.as_slice();
        self.lambda.clear();
        self.lambda.resize(dist.len(), T::zero());
        let kappa = T::from(kappa).unwrap();
        self.lambda.iter_mut().zip(dist.iter()).for_each(|(l, &d)| {
            *l = -kappa * ContactPenalty::new(delta).db(d);
        });
    }

    pub(crate) fn constraint_size(&self) -> usize {
        self.point_constraint.constraint_size()
    }

    // Helper function to construct subsets from x using internal indices.
    //pub(crate) fn input_and_constraint<'a>(
    //    &mut self,
    //    x: Chunked3<&'a [T]>,
    //) -> (
    //    [SubsetView<Chunked3<&'a [T]>>; 2],
    //    &mut PointContactConstraint<T>,
    //) {
    //    let x0 = SubsetView::from_unique_ordered_indices(&self.implicit_surface_vertex_indices, x);
    //    let x1 = SubsetView::from_unique_ordered_indices(&self.collider_vertex_indices, x);
    //    ([x0, x1], &mut self.point_constraint)
    //}

    pub fn subtract_constraint_force(&self, mut f: Chunked3<&mut [T]>) {
        self.distance_jacobian_blocks_iter()
            .for_each(|(row, col, j)| {
                *f[col].as_mut_tensor() -= *j.as_tensor() * self.lambda[row];
            });
    }

    // Compute friction force `f_f(x,v) = -μT(x)H(T(x)'v)λ(x)` and subtract it from `f`.
    pub fn subtract_friction_force(&mut self, mut f: Chunked3<&mut [T]>, v: Chunked3<&[T]>) {
        if let Some((obj_f, col_f)) = self.compute_friction_force(v) {
            for (&i, obj_f) in self
                .implicit_surface_vertex_indices
                .iter()
                .zip(obj_f.iter())
            {
                *f[i].as_mut_tensor() -= obj_f.as_tensor();
            }
            for (i, col_f, _) in col_f.iter() {
                let i = self.collider_vertex_indices[i];
                *f[i].as_mut_tensor() -= col_f.as_tensor();
            }
        }
    }

    // Compute `b(x,v) = H(T(x)'v)λ(x)`, friction force in contact space.
    //
    // This function uses current state. To get an upto date friction impulse call update_state.
    // Computes the jacobian with rows indexed by query points if `full_jacobian` is `true`.
    pub fn compute_contact_friction_force(
        &mut self,
        // Velocity
        v: Chunked3<&[T]>,
        active_contact_indices: &[usize],
        active_constraint_subset: &[usize],
        full_jacobian: bool,
    ) -> Option<Chunked3<Vec<T>>> {
        let v0 = SubsetView::from_unique_ordered_indices(&self.implicit_surface_vertex_indices, v);
        let v1 = SubsetView::from_unique_ordered_indices(&self.collider_vertex_indices, v);

        let lambda = &self.lambda;
        let pc = &mut self.point_constraint;

        if pc.friction_workspace.is_none() {
            return None;
        }

        // Construct contact (or "sliding") basis.
        let normals = pc.contact_normals();
        let normals_subset =
            Subset::from_unique_ordered_indices(active_constraint_subset, normals.as_slice());
        let mut normals = Chunked3::from_array_vec(vec![[T::zero(); 3]; normals_subset.len()]);
        normals_subset.clone_into_other(&mut normals);

        // TODO: memoize the contact jacobian:
        //       Store the jacobian sparsity at the beginning of each step with all of potential contact points.
        //       At each residual or jacobian function call, replace the sparse rows with those
        //       of just the active contact points creating an even more sparse jacobian equivalent
        //       to this one but only a *view*. The values of the contact jacobian can be updated
        //       also but this way we avoid doing any new allocations. This is left as a todo since
        //       we need to profile if this is actually worthwhile.
        // Contact Jacobian is defined for object vertices only. Contact Jacobian for collider vertices is trivial.
        let jac = self
            .contact_jacobian
            .borrow_with(|| pc.compute_contact_jacobian(&active_contact_indices, full_jacobian));
        let collider_v = Subset::from_unique_ordered_indices(active_contact_indices, v1);

        let FrictionWorkspace {
            contact_basis,
            // TODO: use collider_force in place of vc to avoid additional allocations.
            object_force: _,
            collider_force: _, // for active point contacts
            ..
        } = pc.friction_workspace.as_mut().unwrap();

        contact_basis.update_from_normals(normals.into());

        // Compute relative velocity at the point of contact: `vc = J(x)v`
        let mut vc = contact_tangent_velocity(jac.view(), collider_v.view(), v0);

        dbg!(vc.len());
        dbg!(lambda.len());
        assert_eq!(vc.len(), lambda.len());

        // Compute sliding bases velocity product.

        // Define the smoothing function.
        // This is s(x;eps)/x from the paper. We integrate the division by x
        // to avoid generating large values near zero.

        // Compute `vc <- H(B'(x) vc) λ(x)`.
        vc.view_mut()
            .into_iter()
            .zip(lambda)
            .enumerate()
            .for_each(|(i, (vc, lambda))| {
                let [_, v1, v2] = contact_basis.to_contact_coordinates(*vc, i);
                let vc_t = [v1, v2].into_tensor();
                let vc_t_smoothed = eta(vc_t, *lambda, T::from(1e-5).unwrap()).into_data();
                *vc = [T::zero(), vc_t_smoothed[0], vc_t_smoothed[1]];
            });
        Some(vc)
    }

    // Compute `f(x,v) = -μT(x)H(T(x)'v)λ(x)`.
    //
    // This function uses current state. To get an upto date friction impulse call update_state.
    pub fn compute_friction_force(
        &mut self,
        // Velocity
        v: Chunked3<&[T]>,
    ) -> Option<(Chunked3<Vec<T>>, Sparse<Chunked3<Vec<T>>>)> {
        let lambda = self.lambda.as_slice();
        let dist = self.distance_potential.as_slice();
        let (active_constraint_subset, active_contact_indices, _) =
            self.point_constraint.in_contact_indices(lambda, dist);

        let mut vc = self.compute_contact_friction_force(
            v,
            &active_contact_indices,
            &active_constraint_subset,
            false,
        )?;

        let jac = self.contact_jacobian.borrow_with(|| {
            self.point_constraint
                .compute_contact_jacobian(&active_contact_indices, false)
        });

        let FrictionWorkspace {
            contact_basis,
            params,
            ..
        } = self.point_constraint.friction_workspace.as_ref()?;
        let mu = T::from(params.dynamic_friction).unwrap();

        // Compute `vc <- mu B(x) H(B'(x) vc) λ(x)`.
        vc.view_mut().into_iter().enumerate().for_each(|(i, v)| {
            let vc = v.as_tensor();
            *v = contact_basis.from_contact_coordinates(*vc * (-mu), i);
        });

        // Compute object force (compute `f = J'(x)vc`)
        let obj_f = jac.view().into_tensor().transpose() * vc.view().into_tensor();
        let col_f = Sparse::from_dim(
            active_contact_indices.clone(),
            self.point_constraint.collider_vertex_positions.len(),
            vc.into_data(),
        );
        Some((obj_f.into_data(), col_f))
    }

    /// Jacobian of the contact basis matrix with respect to vertex positions.
    ///
    /// If `transpose` is `true` this function computes the jacobian of the contact basis transpose matrix.
    pub fn contact_basis_matrix_jacobian_product<'a>(
        surf: &'a QueryTopo<T>,
        collider_vertex_positions: Chunked3<&'a [T]>,
        contact_basis: &'a ContactBasis<T>,
        multipliers: &'a [[T; 3]],
        transpose: bool,
    ) -> Result<impl Iterator<Item = ((usize, usize), Matrix3<T>)> + 'a, Error> {
        // Contacts occur at the vertex positions of the colliding mesh.
        let normals = &contact_basis.normals;

        // Compute the unnormalized normals (need this for the norms).
        let mut normal_coords = vec![T::zero(); surf.num_query_jacobian_entries()];
        surf.query_jacobian_values(collider_vertex_positions.view().into(), &mut normal_coords);
        let grad_psi = Chunked3::from_flat(normal_coords).into_arrays();

        Ok(surf
            .sample_query_hessian_indexed_blocks_iter(collider_vertex_positions.into())?
            .map(move |(contact_idx, vtx_idx, nml_jac)| {
                let n = normals[contact_idx].into_tensor();

                // Compute the Jacobian of the normalized negative grad psi. (see compute_normals for reference on why).
                let nml_jac = (Matrix3::identity() - n * n.transpose())
                    * nml_jac.into_tensor()
                    * (-T::one() / grad_psi[contact_idx].into_tensor().norm());

                // Find the axis that is most aligned with the normal, then use the next axis for the
                // tangent.
                let mut t = Vector3::zero();
                let tangent_axis = (n.iamax() + 1) % 3;
                t[tangent_axis] = T::one();

                // Project out the normal component.
                t -= n * n[tangent_axis];

                let neg_t_proj = t * t.transpose() - Matrix3::identity();

                let tangent_jac = neg_t_proj
                    * (n * nml_jac[tangent_axis].transpose() + nml_jac * n[tangent_axis])
                    * (T::one() / t.norm());
                let bitangent_jac = [
                    (nml_jac[1] * t[2] - nml_jac[2] * t[1] + tangent_jac[2] * n[1]
                        - tangent_jac[1] * n[2])
                        .into_data(),
                    (nml_jac[2] * t[0] - nml_jac[0] * t[2] + tangent_jac[0] * n[2]
                        - tangent_jac[2] * n[0])
                        .into_data(),
                    (nml_jac[0] * t[1] - nml_jac[1] * t[0] + tangent_jac[1] * n[0]
                        - tangent_jac[0] * n[1])
                        .into_data(),
                ]
                .into_tensor();

                let multiplier = multipliers[contact_idx].into_tensor();
                if transpose {
                    let result = [
                        (multiplier.transpose() * nml_jac)[0].into_data(),
                        (multiplier.transpose() * tangent_jac)[0].into_data(),
                        (multiplier.transpose() * bitangent_jac)[0].into_data(),
                    ]
                    .into_tensor();
                    ((contact_idx, vtx_idx), result)
                } else {
                    (
                        (contact_idx, vtx_idx),
                        nml_jac * multiplier[0]
                            + tangent_jac * multiplier[1]
                            + bitangent_jac * multiplier[2],
                    )
                }
            }))
    }

    /// Builds a contact basis gradient product.
    ///
    /// Computes `d/dq B(q) b` (or `d/dq B^T(q) b` if `transpose` is set to `true`) where `b` is the multiplier.
    pub fn build_contact_basis_gradient_product_from_selection<'a>(
        surf: &implicits::QueryTopo<T>,
        contact_basis: &ContactBasis<T>,
        active_contact_points: SelectView<'a, Chunked3<&'a [T]>>,
        multipliers: &[[T; 3]],
        num_vertices: usize,
        transpose: bool,
    ) -> Result<Tensor![T; S S 3 3], Error> {
        let query_points = active_contact_points.target;
        let (indices, blocks): (Vec<_>, Vec<_>) = Self::contact_basis_matrix_jacobian_product(
            surf,
            query_points,
            contact_basis,
            multipliers,
            transpose,
        )?
        .filter(|((_, col), _)| col < &num_vertices)
        .map(|((row, col), mtx)| (MatrixElementIndex { row: col, col: row }, mtx.transpose()))
        .unzip(); // transpose to produce gradient

        let nnz = indices.len();
        assert_eq!(nnz, blocks.len());

        let mut entries = (0..nnz).collect::<Vec<_>>();

        // Sort indices into row major order
        entries.sort_by(|&a, &b| {
            indices[a]
                .row
                .cmp(&indices[b].row)
                .then_with(|| indices[a].col.cmp(&indices[b].col))
        });

        let triplet_iter = entries
            .iter()
            .map(|&i| (indices[i].row, indices[i].col, blocks[i].into_data()));

        Ok(tensr::SSBlockMatrix3::from_block_triplets_iter(
            triplet_iter,
            num_vertices,
            query_points.len(),
        )
        .into_data())
    }

    /// Builds a contact jacobian gradient product.
    ///
    /// Computes `grad_q J(q) b` where `b` is the multiplier.
    ///
    pub fn build_contact_jacobian_gradient_product<'a>(
        surf: &implicits::QueryTopo<T>,
        query_points: &[[T; 3]],
        multipliers: &[[T; 3]],
        num_vertices: usize,
    ) -> Result<Tensor![T; S S 3 3], Error> {
        let (indices, blocks): (Vec<_>, Vec<_>) = surf
            .contact_jacobian_jacobian_product_indexed_blocks_iter(
                query_points.view(),
                multipliers,
            )?
            .filter(|(_, col, _)| col < &num_vertices)
            .map(|(row, col, mtx)| {
                (
                    MatrixElementIndex { row: col, col: row },
                    mtx.into_tensor().transpose().into_data(),
                )
            })
            .unzip(); // transpose to produce gradient

        let nnz = indices.len();
        assert_eq!(nnz, blocks.len());

        let mut entries = (0..nnz).collect::<Vec<_>>();

        // Sort indices into row major order
        entries.sort_by(|&a, &b| {
            indices[a]
                .row
                .cmp(&indices[b].row)
                .then_with(|| indices[a].col.cmp(&indices[b].col))
        });

        let triplet_iter = entries
            .iter()
            .map(|&i| (indices[i].row, indices[i].col, blocks[i]));

        Ok(tensr::SSBlockMatrix3::from_block_triplets_iter(
            triplet_iter,
            num_vertices,
            query_points.len(),
        )
        .into_data())
    }

    // Compute the product of friction matrix `mu J'B H(B'Jv)` with constraint Jacobian term `-k ddb dd/dq`.
    // Both input matrices are column major.
    // Output blocks are row-major.
    fn contact_constraint_jac_product_map<'a>(
        lambda: &'a [T],
        dist: &'a [T],
        pc: &'a PointContactConstraint<T>,
        implicit_surface_vertex_indices: &'a [usize],
        collider_vertex_indices: &'a [usize],
        // Relative velocity in contact space: `vc = J(x)v`
        mut vc: Chunked3<Vec<T>>,
        contact_jac: ContactJacobianView<'a, T>, // J
        constraint_jac: Tensor![T; &'a S S 1 3], // dd/dq
        delta: f32,
        kappa: f32,
    ) -> impl Iterator<Item = (usize, usize, Matrix3<T>)> + 'a {
        assert_eq!(constraint_jac.len(), contact_jac.into_tensor().num_cols());

        let kappa = T::from(kappa).unwrap();
        let delta = T::from(delta).unwrap();

        dbg!(vc.len());
        dbg!(lambda.len());
        assert_eq!(vc.len(), lambda.len());

        // Compute contact jacobian
        let FrictionWorkspace {
            contact_basis,
            params,
            // TODO: use collider_force in place of vc to avoid additional allocations.
            object_force: _,
            collider_force: _, // for active point contacts
            ..
        } = pc.friction_workspace.as_ref().unwrap();

        let mu = T::from(params.dynamic_friction).unwrap();

        // Compute `vc <- mu B(x) H(B'(x) vc)` this is now a diagonal block matrix stored as the vector vc.
        vc.view_mut().iter_mut().enumerate().for_each(|(i, vc)| {
            let [_, v1, v2] = contact_basis.to_contact_coordinates(*vc, i);
            let vc_t = [v1, v2].into_tensor();
            let vc_t_smoothed = eta(vc_t, -mu, T::from(1e-5).unwrap()).into_data();
            *vc = contact_basis
                .from_contact_coordinates([T::zero(), vc_t_smoothed[0], vc_t_smoothed[1]], i)
        });

        let contact_jac_iter_refcell = std::cell::RefCell::new(contact_jac.into_iter().peekable());
        let contact_jac_iter = &contact_jac_iter_refcell;
        let vc = vc.view();

        // Iterate over each column of the constraint Jacobian.
        let res: Vec<_> = constraint_jac
            .into_iter()
            .flat_map(|(col_idx, col)| {
                col.into_iter()
                    .flat_map(move |(constraint_idx, constraint_jac_block)| {
                        let constraint_jac_block = *constraint_jac_block.into_arrays().as_tensor();
                        while let Some(&(contact_idx, _)) = contact_jac_iter.borrow_mut().peek() {
                            if constraint_idx < contact_idx {
                                return Either::Left(std::iter::empty());
                            } else if contact_idx < constraint_idx {
                                let _ = contact_jac_iter.borrow_mut().next().unwrap();
                            } else {
                                let (contact_idx, contact_col) =
                                    contact_jac_iter.borrow_mut().next().unwrap();
                                let index = contact_idx;
                                let rhs = *vc[index].as_tensor()
                                    * (constraint_jac_block
                                        * (-kappa * ContactPenalty::new(delta).ddb(dist[index])));
                                return Either::Right(
                                    contact_col
                                        .into_iter()
                                        .map(move |(row_idx, contact_block)| {
                                            let block = (*contact_block.into_arrays().as_tensor())
                                                .transpose()
                                                * rhs;
                                            let row_idx =
                                                implicit_surface_vertex_indices[row_idx];
                                            (row_idx, col_idx, block)
                                        })
                                        .chain(std::iter::once((
                                            collider_vertex_indices[index],
                                            col_idx,
                                            rhs,
                                        ))),
                                );
                            }
                        }
                        Either::Left(std::iter::empty())
                    })
            })
            .collect();
        res.into_iter()
    }

    // Assume all the state (including friction workspace) has been updated
    pub(crate) fn friction_jacobian_indexed_value_iter<'a>(
        &'a mut self,
        v: Chunked3<&'a [T]>,
        delta: f32,
        kappa: f32,
        time_step: T,
        max_index: usize,
    ) -> Option<impl Iterator<Item = (usize, usize, T)> + 'a> {
        // If there is anything I regret in this life,
        // writing this function is at the top of that list.

        if self.point_constraint.friction_workspace.is_none() {
            return None;
        }

        // Note that there is a distinction between active *contacts* and active
        // *constraints*. Active *constraints* correspond to to those points
        // that are in the MLS neighborhood of influence to be part of the
        // optimization. Active *contacts* are a subset of those that are
        // considered to be in contact and thus are producing friction.
        let (active_constraint_subset, active_contact_indices, _) = self
            .point_constraint
            .in_contact_indices(self.lambda.as_slice(), self.distance_potential.as_slice());

        let c = self.compute_contact_friction_force(
            v,
            &active_contact_indices,
            &active_constraint_subset,
            true,
        )?;

        // TODO: memoize the contact jacobian:
        //       Store the jacobian sparsity at the beginning of each step with all of potential contact points.
        //       At each residual or jacobian function call, replace the sparse rows with those
        //       of just the active contact points creating an even more sparse jacobian equivalent
        //       to this one but only a *view*. The values of the contact jacobian can be updated
        //       also but this way we avoid doing any new allocations. This is left as a todo since
        //       we need to profile if this is actually worthwhile.
        // Contact Jacobian is defined for object vertices only. Contact Jacobian for collider vertices is trivial.
        let contact_jacobian = self.contact_jacobian.borrow_with(|| {
            self.point_constraint
                .compute_contact_jacobian(&active_contact_indices, true)
        });
        let contact_gradient = self
            .point_constraint
            .compute_contact_gradient(&active_contact_indices, true);

        let mu = T::from(
            self.point_constraint
                .friction_workspace
                .as_ref()
                .unwrap()
                .params
                .dynamic_friction,
        )
        .unwrap();

        let v0 = SubsetView::from_unique_ordered_indices(&self.implicit_surface_vertex_indices, v);
        let v1 = SubsetView::from_unique_ordered_indices(&self.collider_vertex_indices, v);
        let collider_v = Subset::from_unique_ordered_indices(active_contact_indices.as_slice(), v1);

        let MappedDistanceGradient { matrix: g, .. } = self
            .distance_gradient
            .borrow()
            .expect("Uninitialized constraint gradient.");
        let j_view = Self::constraint_gradient_column_major_transpose(g.view());

        let (f_lambda_jac, jac_contact_gradient) = {
            {
                let Self {
                    friction_jacobian_workspace: FrictionJacobianWorkspace {
                        bc,
                        ..
                    },
                    point_constraint,
                    ..
                } = self;

                // Compute `c(x,v) = H(T(x)'v)λ(x)`, friction force in contact space.
                *bc = c
                    .iter()
                    .enumerate()
                    .map(|(i, &v)| {
                        point_constraint
                            .friction_workspace
                            .as_ref()
                            .unwrap()
                            .contact_basis
                            .from_contact_coordinates(v, i)
                    })
                    .collect();
            }

            // Compute relative velocity at the point of contact: `vc = J(x)v`
            let vc = contact_tangent_velocity(contact_jacobian.view(), collider_v.view(), v0);

            // Derivative of lambda term.
            let f_lambda_jac = Self::contact_constraint_jac_product_map(
                self.lambda.as_slice(),
                self.distance_potential.as_slice(),
                &self.point_constraint,
                &self.implicit_surface_vertex_indices,
                &self.collider_vertex_indices,
                vc.clone(),
                contact_jacobian.view(),
                j_view,
                delta,
                kappa,
            );

            let surf = &self.point_constraint.implicit_surface;
            let query_points = &self.point_constraint.collider_vertex_positions;

            // The following should be multiplied by mu.

            // Compute (A)
            let jac_contact_gradient = surf
                .contact_hessian_product_indexed_blocks_iter(
                    query_points.view().into_arrays(),
                    self.friction_jacobian_workspace.bc.as_arrays(),
                )
                .ok()?;

            // Compute (B)

            let active_contact_points = Select::new(active_contact_indices, query_points.view());

            let num_vertices = self.implicit_surface_vertex_indices.len();

            let jac_basis = Self::build_contact_basis_gradient_product_from_selection(
                surf,
                &self
                    .point_constraint
                    .friction_workspace
                    .as_ref()
                    .unwrap()
                    .contact_basis,
                active_contact_points.view(),
                c.view().into_arrays(),
                num_vertices,
                false,
            )
                .ok()?
                .into_tensor()
                .transpose();

            self.friction_jacobian_workspace.contact_gradient_jac_basis = (contact_gradient.view().into_tensor() * jac_basis.view()).into_data(); // (B)

            // Construct full change of basis matrix B
            let b = self
                .point_constraint
                .friction_workspace
                .as_ref()
                .unwrap()
                .contact_basis
                .full_basis_matrix();
            let bt = self
                .point_constraint
                .friction_workspace
                .as_ref()
                .unwrap()
                .contact_basis
                .full_basis_matrix_transpose();

            // Construct the eta matrix H
            let h_blocks: Chunked3<Vec<_>> = (0..query_points.len())
                .flat_map(|contact_idx| {
                    let [_, v1, v2] = self
                        .point_constraint
                        .friction_workspace
                        .as_ref()
                        .unwrap()
                        .contact_basis
                        .to_contact_coordinates(vc[contact_idx], contact_idx);
                    let mtx = eta_jac([v1, v2].into(), T::one(), T::from(1e-5).unwrap())
                        * self.lambda[contact_idx];
                    std::iter::once([T::zero(); 3])
                        .chain(std::iter::once([T::zero(), mtx[0][0], mtx[0][1]]))
                        .chain(std::iter::once([T::zero(), mtx[1][0], mtx[1][1]]))
                })
                .collect();
            let h = BlockDiagonalMatrix3::new(Chunked3::from_flat(h_blocks));

            // Compute (C)

            let mut jac_basis_t = Self::build_contact_basis_gradient_product_from_selection(
                surf,
                &self
                    .point_constraint
                    .friction_workspace
                    .as_ref()
                    .unwrap()
                    .contact_basis,
                active_contact_points.view(),
                v.into_arrays(),
                num_vertices,
                true,
            )
                .ok()?
                .into_tensor()
                .transpose();

            jac_basis_t.premultiply_block_diagonal_mtx(h.view());
            jac_basis_t.premultiply_block_diagonal_mtx(b.view());
            self.friction_jacobian_workspace.contact_gradient_basis_eta_jac_basis_jac =
                (contact_gradient.view().into_tensor() * jac_basis_t.view()).into_data(); // (C)

            // Compute (D)

            let mut jac_contact_jacobian = Self::build_contact_jacobian_gradient_product(
                surf,
                query_points.view().into_arrays(),
                v.view().into_arrays(),
                self.implicit_surface_vertex_indices.len(),
            )
                .ok()?
                .into_tensor()
                .transpose();
            jac_contact_jacobian.premultiply_block_diagonal_mtx(bt.view());
            jac_contact_jacobian.premultiply_block_diagonal_mtx(h.view());
            jac_contact_jacobian.premultiply_block_diagonal_mtx(b.view());
            self.friction_jacobian_workspace.contact_gradient_basis_eta_jac_basis_jac_contact_jac =
                (contact_gradient.view().into_tensor() * jac_contact_jacobian.view()).into_data(); // (D)

            // Compute (E)

            let mut contact_gradient_t = contact_gradient.clone().into_tensor().transpose(); // Make mut
            contact_gradient_t.premultiply_block_diagonal_mtx(bt.view());
            contact_gradient_t.premultiply_block_diagonal_mtx(h.view());
            contact_gradient_t.premultiply_block_diagonal_mtx(b.view());
            self.friction_jacobian_workspace.contact_gradient_basis_eta_jac_basis_contact_jac =
                (contact_gradient.view().into_tensor() * contact_gradient_t.view()).into_data(); // (E)

            (f_lambda_jac, jac_contact_gradient)
        };

        // Combine all matrices.
        let dt_inv = T::one() / T::from(time_step).unwrap();

        Some(
            f_lambda_jac.inspect(|(_,_,m)| log::trace!("f lambda jac: {:?}", m.determinant()))
                .chain(
                    jac_contact_gradient
                        .map(|(row, col, block)| (row, col, Matrix3::new(block).transpose()))
                        .inspect(|(_,_,m)| log::trace!("A: {:?}", m.determinant())) ,
                ) // (A)
                .chain(
                    self.friction_jacobian_workspace.contact_gradient_jac_basis
                        .view()
                        .into_iter()
                        .flat_map(move |(row_idx, row)| {
                            // (B)
                            row.into_iter().map(move |(col_idx, block)| {
                                (row_idx, col_idx, *block.into_arrays().as_tensor() * mu)
                            })
                        })
                        .inspect(|(_,_,m)| log::trace!("B: {:?}", m.determinant()))
                    ,
                )
                .chain(
                    self.friction_jacobian_workspace.contact_gradient_basis_eta_jac_basis_jac
                        .view()
                        .into_iter()
                        .flat_map(move |(row_idx, row)| {
                            // (C)
                            row.into_iter().map(move |(col_idx, block)| {
                                (row_idx, col_idx, *block.into_arrays().as_tensor() * mu)
                            })
                        })
                        .inspect(|(_,_,m)| log::trace!("C: {:?}", m.determinant()))
                    ,
                )
                .chain(
                    self.friction_jacobian_workspace.contact_gradient_basis_eta_jac_basis_jac_contact_jac
                        .view()
                        .into_iter()
                        .flat_map(move |(row_idx, row)| {
                            // (D)
                            row.into_iter().map(move |(col_idx, block)| {
                                (row_idx, col_idx, *block.into_arrays().as_tensor() * mu)
                            })
                        })
                        .inspect(|(_,_,m)| log::trace!("D: {:?}", m.determinant()))
                    ,
                )
                .chain(
                    self.friction_jacobian_workspace.contact_gradient_basis_eta_jac_basis_contact_jac
                        .view()
                        .into_iter()
                        .flat_map(move |(row_idx, row)| {
                            // (E)
                            row.into_iter().map(move |(col_idx, block)| {
                                (
                                    row_idx,
                                    col_idx,
                                    *block.into_arrays().as_tensor() * (mu * dt_inv),
                                )
                            })
                        })
                        .inspect(|(_,_,m)| log::trace!("E: {:?}", m.determinant()))
                    ,
                )
                .filter(move |&(row, col, _)| row < max_index && col < max_index)
                .flat_map(move |(row, col, block)| {
                    (0..3).flat_map(move |r| {
                        (0..3).map(move |c| (3 * row + r, 3 * col + c, block[r][c]))
                    })
                }),
        )
    }

    fn distance_jacobian_blocks_iter<'a>(
        &'a self,
    ) -> impl Iterator<Item = (usize, usize, [T; 3])> + 'a {
        Self::distance_jacobian_blocks_iter_fn(
            &self.point_constraint,
            &self.implicit_surface_vertex_indices,
            &self.collider_vertex_indices,
        )
    }

    fn distance_jacobian_blocks_iter_fn<'a>(
        point_constraint: &'a PointContactConstraint<T>,
        implicit_surface_vertex_indices: &'a [usize],
        collider_vertex_indices: &'a [usize],
    ) -> impl Iterator<Item = (usize, usize, [T; 3])> + 'a {
        point_constraint
            .object_constraint_jacobian_blocks_iter()
            .map(move |(row, col, block)| (row, implicit_surface_vertex_indices[col], block))
            .chain(
                point_constraint
                    .collider_constraint_jacobian_blocks_iter()
                    .map(move |(row, col, block)| (row, collider_vertex_indices[col], block)),
            )
    }

    pub(crate) fn constraint_hessian_size(&self, max_index: usize) -> usize {
        self.constraint_hessian_indices_iter(max_index).count()
    }

    /// Construct a transpose of the constraint gradient (constraint Jacobian).
    ///
    /// The structure is preserved but the inner blocks are transposed.
    pub(crate) fn constraint_gradient_column_major_transpose<'a>(
        matrix: Tensor![T; &'a S S 3 1],
    ) -> Tensor![T; &'a S S 1 3] {
        use flatk::IntoStorage;
        // TODO: update Chunked with from_raw, into_raw functions to avoid exposing unsafe construction.
        let Sparse {
            source:
                Chunked {
                    chunks,
                    data:
                        Sparse {
                            selection: col_selection,
                            source,
                        },
                },
            selection: row_selection,
        } = matrix;

        Sparse {
            source: Chunked::from_offsets(
                chunks.into_inner(),
                Sparse {
                    selection: col_selection,
                    source: Chunked1::from_flat(Chunked3::from_flat(source.into_storage())),
                },
            ),
            selection: row_selection,
        }
    }

    pub(crate) fn num_hessian_diagonal_nnz(&self, max_index: usize) -> usize {
        self.constraint_hessian_indices_iter(max_index)
            .filter(|m| m.col == m.row)
            .count()
    }

    // Assumes surface and contact points are upto date.
    pub(crate) fn object_distance_potential_hessian_indexed_blocks_iter<'a>(
        &'a self,
        lambda: &'a [T],
    ) -> Box<dyn Iterator<Item = (usize, usize, [[T; 3]; 3])> + 'a> {
        Box::new(
            if self.point_constraint.object_is_fixed() {
                None
            } else {
                let surf = &self.point_constraint.implicit_surface;
                surf.surface_hessian_product_indexed_blocks_iter(
                    self.point_constraint
                        .collider_vertex_positions
                        .view()
                        .into(),
                    lambda,
                )
                .ok()
            }
            .into_iter()
            .flatten()
            .map(move |(row, col, mtx)| {
                let row = self.implicit_surface_vertex_indices[row];
                let col = self.implicit_surface_vertex_indices[col];
                if col > row {
                    (col, row, mtx)
                } else {
                    (row, col, mtx)
                }
                .into()
            }),
        )
    }

    // Assumes surface and contact points are upto date.
    pub(crate) fn collider_distance_potential_hessian_indexed_blocks_iter<'a>(
        &'a self,
        lambda: &'a [T],
    ) -> impl Iterator<Item = (usize, [[T; 3]; 3])> + 'a {
        if self.point_constraint.collider_is_fixed() {
            None
        } else {
            let surf = &self.point_constraint.implicit_surface;
            Some(
                surf.query_hessian_product_indexed_blocks_iter(
                    self.point_constraint
                        .collider_vertex_positions
                        .view()
                        .into(),
                    lambda,
                )
                .map(move |(idx, mtx)| (self.collider_vertex_indices[idx], mtx)),
            )
        }
        .into_iter()
        .flatten()
    }

    pub(crate) fn object_distance_potential_hessian_block_indices_iter<'a>(
        &'a self,
    ) -> impl Iterator<Item = (usize, usize)> + 'a {
        if self.point_constraint.object_is_fixed() {
            None
        } else {
            self.point_constraint
                .implicit_surface
                .surface_hessian_product_block_indices_iter()
                .ok()
        }
        .into_iter()
        .flatten()
        .map(move |(row, col)| {
            let row = self.implicit_surface_vertex_indices[row];
            let col = self.implicit_surface_vertex_indices[col];
            if col > row {
                (col, row)
            } else {
                (row, col)
            }
        })
    }

    pub(crate) fn collider_distance_potential_hessian_block_indices_iter<'b>(
        &'b self,
    ) -> impl Iterator<Item = usize> + 'b {
        if self.point_constraint.collider_is_fixed() {
            None
        } else {
            Some(
                self.point_constraint
                    .implicit_surface
                    .query_hessian_product_block_indices_iter()
                    .map(move |idx| self.collider_vertex_indices[idx]),
            )
        }
        .into_iter()
        .flatten()
    }

    pub(crate) fn constraint_hessian_indices_iter<'a>(
        &'a self,
        max_index: usize,
    ) -> impl Iterator<Item = MatrixElementIndex> + 'a {
        let obj_indices_iter = self.object_distance_potential_hessian_block_indices_iter();
        let coll_indices_iter = self
            .collider_distance_potential_hessian_block_indices_iter()
            .map(|idx| (idx, idx));

        let MappedDistanceGradient { matrix: g, .. } = self
            .distance_gradient
            .borrow()
            .expect("Uninitialized constraint gradient.");

        let g_view = g.view();
        let j_view = Self::constraint_gradient_column_major_transpose(g.view());

        let gj_iter = g_view
            .into_iter()
            .filter(move |(row_idx, _)| *row_idx < max_index)
            .flat_map(move |(row_idx, lhs_row)| {
                // Iterate over the columns of transpose g (so rows of g).
                j_view
                    .into_iter()
                    .filter(move |(col_idx, _)| *col_idx <= row_idx)
                    .flat_map(move |(col_idx, rhs_col)| {
                        // Produce an iterator for the row-col block inner product.
                        MulExpr::with_op(lhs_row.into_expr(), rhs_col.into_expr(), Multiplication)
                            .map(move |_| (row_idx, col_idx).into())
                    })
            });

        obj_indices_iter
            .chain(coll_indices_iter)
            .filter(move |(row, col)| *row < max_index && *col < max_index)
            .chain(gj_iter)
            .flat_map(move |(row, col)| {
                if row == col {
                    // Only lower triangular part
                    Either::Left(
                        (0..3).flat_map(move |r| {
                            (0..=r).map(move |c| (3 * row + r, 3 * col + c).into())
                        }),
                    )
                } else {
                    // Entire matrix
                    Either::Right(
                        (0..3).flat_map(move |r| {
                            (0..3).map(move |c| (3 * row + r, 3 * col + c).into())
                        }),
                    )
                }
            })
    }

    pub(crate) fn constraint_hessian_indexed_values_iter<'a>(
        &'a self,
        delta: f32,
        kappa: f32,
        max_index: usize,
    ) -> impl Iterator<Item = (MatrixElementIndex, T)> + 'a {
        let lambda = self.lambda.as_slice();
        let dist = self.distance_potential.as_slice();

        let hessian = self
            .object_distance_potential_hessian_indexed_blocks_iter(lambda)
            .chain(
                self.collider_distance_potential_hessian_indexed_blocks_iter(lambda)
                    .map(|(idx, mtx)| (idx, idx, mtx)),
            );

        let kappa = T::from(kappa).unwrap();

        let MappedDistanceGradient { matrix: g, .. } = self
            .distance_gradient
            .borrow()
            .expect("Uninitialized constraint gradient.");
        let g_view = g.view();
        let j_view = Self::constraint_gradient_column_major_transpose(g.view());

        let gj = g_view
            .into_iter()
            .filter(move |(row_idx, _)| *row_idx < max_index)
            .flat_map(move |(row_idx, lhs_row)| {
                // Iterate over the columns of transpose g (so rows of g).
                j_view
                    .into_iter()
                    .filter(move |(col_idx, _)| *col_idx <= row_idx)
                    .flat_map(move |(col_idx, rhs_col)| {
                        // Produce an iterator for the row-col block inner product.
                        MulExpr::with_op(lhs_row.into_expr(), rhs_col.into_expr(), Multiplication)
                            .map(move |IndexedExpr { index, expr }| {
                                let mtx =
                                    expr * (-kappa * ContactPenalty::new(delta).ddb(dist[index]));
                                (row_idx, col_idx, mtx.into_data())
                            })
                    })
            });

        hessian
            .filter(move |(row, col, _)| *row < max_index && *col < max_index)
            .chain(gj)
            .flat_map(move |(row, col, mtx)| {
                if row == col {
                    Either::Left((0..3).flat_map(move |r| {
                        (0..=r).map(move |c| ((3 * row + r, 3 * col + c).into(), mtx[r][c]))
                    }))
                } else {
                    Either::Right((0..3).flat_map(move |r| {
                        (0..3).map(move |c| ((3 * row + r, 3 * col + c).into(), mtx[r][c]))
                    }))
                }
            })
    }
}

/// The penalty and its derivative alone:
/// ```verbatim
/// b(x;δ) = -((x-δ)^3)/δ if x < δ and 0 otherwise
/// db(x;δ) = -(3/δ)(x-δ)^2 if x < δ and 0 otherwise
/// ddb(x;δ) = -(6/δ)(x-δ) if x < δ and 0 otherwise
/// ```
pub struct ContactPenalty {
    pub delta: f64,
}

impl ContactPenalty {
    pub fn new<T: Real>(delta: T) -> Self {
        ContactPenalty {
            delta: delta.to_f64().unwrap(),
        }
    }

    /// Penalty function.
    ///
    /// This serves as a reference for what the penalty is supposed to represent.
    #[inline]
    #[allow(dead_code)]
    pub fn b<T: Real>(&self, x: T) -> T {
        let delta = T::from(self.delta).unwrap();
        let d = delta - x;
        if d > T::zero() {
            d * d * d / delta
        } else {
            T::zero()
        }
    }

    /// First derivative of the penalty function with respect to `x`.
    #[inline]
    pub fn db<T: Real>(&self, x: T) -> T {
        let delta = T::from(self.delta).unwrap();
        let d = delta - x;
        if d > T::zero() {
            -T::from(3.0).unwrap() * d * d / delta
        } else {
            T::zero()
        }
    }

    /// Second derivative of the penalty function with respect to `x`.
    #[inline]
    pub fn ddb<T: Real>(&self, x: T) -> T {
        let delta = T::from(self.delta).unwrap();
        let d = delta - x;
        if d > T::zero() {
            T::from(6.0).unwrap() * d / delta
        } else {
            T::zero()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ad::F1;

    #[test]
    fn test_contact_penalty_derivative() {
        let delta = 0.001;
        let penalty = ContactPenalty::new(delta);
        let x = F1::var(-0.1);
        let b = penalty.b(x);
        let db_ad = b.deriv();
        let db = penalty.db(x);
        let ddb_ad = db.deriv();
        assert_eq!(db_ad, db.value());
        let ddb = penalty.ddb(x);
        assert_eq!(ddb_ad, ddb.value());
    }
}
