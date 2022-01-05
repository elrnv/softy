use super::point_contact::PointContactConstraint;
use crate::constraints::{ContactConstraint, ContactSurface};
use crate::matrix::MatrixElementIndex;
use crate::{
    Error, FrictionImpulses, FrictionParams, Index, Real, TriMesh, ORIGINAL_VERTEX_INDEX_ATTRIB,
};
use autodiff as ad;
use flatk::{
    Chunked, Chunked1, Chunked3, Offsets, Sparse, StorageMut, Subset, SubsetView, UniChunked, View,
    U1, U3,
};
use geo::attrib::Attrib;
use geo::mesh::VertexMesh;
use geo::{NumVertices, VertexIndex};
use implicits::KernelType;
use lazycell::LazyCell;
use rayon::iter::Either;
use rayon::prelude::*;
use tensr::{
    AsMutTensor, AsTensor, CwiseBinExprImpl, Expr, IndexedExpr, IntoData, IntoExpr, IntoTensor,
    Matrix, MulExpr, Multiplication, Scalar, Tensor, Vector2,
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
        use tensr::{CloneWithStorage, Storage};
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

    pub(crate) distance_gradient: LazyCell<MappedDistanceGradient<T>>,
    pub(crate) lambda: Vec<T>,
    pub distance_potential: Vec<T>,
}

impl<T: Real> PenaltyPointContactConstraint<T> {
    pub fn clone_cast<S: Real>(&self) -> PenaltyPointContactConstraint<S> {
        let mut distance_gradient = LazyCell::new();
        if self.distance_gradient.filled() {
            distance_gradient.replace(self.distance_gradient.borrow().unwrap().clone_cast::<S>());
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
            lambda: Vec::new(),
            distance_potential: Vec::new(),
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
        use flatk::Set;
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
            use flatk::Set;
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
        use flatk::{Isolate, ViewMut};

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
        use flatk::Set;
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

    pub fn subtract_friction_force(&mut self, mut f: Chunked3<&mut [T]>, v: Chunked3<&[T]>) {
        if let Some((obj_f, col_f)) = self.compute_friction_impulse(v) {
            for (&i, obj_f) in self
                .implicit_surface_vertex_indices
                .iter()
                .zip(obj_f.iter())
            {
                *f[i].as_mut_tensor() += obj_f.as_tensor();
            }
            for (i, col_f, _) in col_f.iter() {
                let i = self.collider_vertex_indices[i];
                *f[i].as_mut_tensor() += col_f.as_tensor();
            }
        }
    }

    // Compute `f(x,v) = -μT(x)H(T(x)'v)λ(x)` and subtract it from `fc`.
    //
    // This function uses current state. To get an upto date friction impulse call update_state.
    pub fn compute_friction_impulse(
        &mut self,
        // Contact force magnitude
        v: Chunked3<&[T]>,
    ) -> Option<(Chunked3<Vec<T>>, Sparse<Chunked3<Vec<T>>>)> {
        use flatk::Set;
        use num_traits::Zero;
        use tensr::{AsMutData, ExprMut};

        let v0 = SubsetView::from_unique_ordered_indices(&self.implicit_surface_vertex_indices, v);
        let v1 = SubsetView::from_unique_ordered_indices(&self.collider_vertex_indices, v);

        let lambda = &self.lambda;
        let potential_values = &self.distance_potential;
        let pc = &mut self.point_constraint;

        if pc.friction_impulses.is_none() {
            return None;
        }

        // Note that there is a distinction between active *contacts* and active
        // *constraints*. Active *constraints* correspond to to those points
        // that are in the MLS neighborhood of influence to be part of the
        // optimization. Active *contacts* are a subset of those that are
        // considered to be in contact and thus are producing friction.
        let (active_constraint_subset, active_contact_indices, lambda) =
            pc.in_contact_indices(lambda, potential_values);

        // Construct contact (or "sliding") basis.
        let normals = pc.contact_normals();
        let normals_subset = Subset::from_unique_ordered_indices(
            active_constraint_subset.as_slice(),
            normals.as_slice(),
        );
        let mut normals = Chunked3::from_array_vec(vec![[T::zero(); 3]; normals_subset.len()]);
        normals_subset.clone_into_other(&mut normals);

        pc.friction_impulses
            .as_mut()
            .unwrap()
            .contact_basis
            .update_from_normals(normals.into());

        // Contact Jacobian is defined for object vertices only. Contact Jacobian for collider vertices is trivial.
        let jac = pc.compute_contact_jacobian(&active_contact_indices);
        let collider_v = Subset::from_unique_ordered_indices(active_contact_indices.as_slice(), v1);

        // Compute relative velocity in contact space: `vc = J(x)v`
        let mut vc = jac.view().into_tensor() * v0.into_tensor();
        *&mut vc.expr_mut() -= collider_v.expr();

        dbg!(vc.len());
        dbg!(lambda.len());
        assert_eq!(vc.len(), lambda.len());

        let FrictionImpulses {
            contact_basis,
            params,
            object_impulse: _,
            collider_impulse: _, // for active point contacts
        } = pc.friction_impulses.as_mut().unwrap();

        let mu = T::from(params.dynamic_friction).unwrap();

        // Compute sliding bases velocity product.

        // Define the smoothing function.
        // This is s(x;eps)/x from the paper. We integrate the division by x
        // to avoid generating large values near zero.
        //let smoother = |x, eps| {
        //    if x < eps {
        //        T::from(2.0).unwrap() * x / eps - x * x / (eps * eps)
        //    } else {
        //        T::one()
        //    }
        //};
        let smoother = |x, eps| T::one() / (x + T::from(0.1).unwrap() * eps);

        // Compute `vc <- mu B(x) H(B'(x) vc) λ(x)`.
        vc.as_mut_data()
            .iter_mut()
            .zip(lambda)
            .enumerate()
            .for_each(|(i, (vc, lambda))| {
                let [_, v1, v2] = contact_basis.to_contact_coordinates(*vc, i);
                let vc_t = [v1, v2].into_tensor();
                let norm_vc_t = vc_t.norm();
                let vc_t_smoothed = if norm_vc_t > T::zero() {
                    vc_t * (mu * lambda * smoother(norm_vc_t, T::from(1e-5).unwrap()))
                } else {
                    Vector2::zero()
                }
                .into_data();
                *vc = contact_basis
                    .from_contact_coordinates([T::zero(), vc_t_smoothed[0], vc_t_smoothed[1]], i)
            });

        // Subtract object force (compute `f = J'(x)vc`)
        let obj_f = jac.view().into_tensor().transpose() * vc.view();
        let col_f = Sparse::from_dim(
            active_contact_indices.clone(),
            pc.collider_vertex_positions.len(),
            vc.into_data(),
        );
        Some((obj_f.into_data(), col_f))
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

    /// Construct a transpose of the constraint gradinet (constraint jacobian).
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
