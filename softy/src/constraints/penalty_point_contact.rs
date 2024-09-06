use super::point_contact::PointContactConstraint;
use crate::constraints::{ContactConstraint, ContactSurface};
use crate::matrix::MatrixElementIndex;
use crate::{
    BlockMatrix3Triplets, ContactBasis, ContactGradient, ContactGradientView, ContactJacobian,
    Error, Index, Real, SSBlockMatrix3, TriMesh, TripletContactJacobian,
    ORIGINAL_VERTEX_INDEX_ATTRIB,
};
use autodiff as ad;
use flatk::{
    Chunked, Chunked1, Chunked3, CloneWithStorage, IntoStorage, Isolate, Offsets, Select, Set,
    Sparse, Storage, StorageMut, SubsetView, UniChunked, View, U1, U3,
};
use geo::attrib::Attrib;
use geo::index::CheckedIndex;
use geo::mesh::VertexMesh;
use geo::topology::{NumVertices, VertexIndex};
use implicits::QueryTopo;
use lazycell::LazyCell;
use num_traits::Zero;
use rayon::iter::Either;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::cell::RefCell;
use std::cmp::Ordering;
use std::ops::AddAssign;
use std::time::{Duration, Instant};
use tensr::{
    AsMutTensor, AsTensor, BlockDiagonalMatrix3, CwiseBinExprImpl, Expr, ExprMut, IndexedExpr,
    IntoData, IntoExpr, IntoTensor, Matrix, Matrix2, Matrix3, MulExpr, Multiplication, Scalar,
    Tensor, Vector2, Vector3,
};

// #[derive(Clone, Debug)]
// pub struct MinMaxHeap {
//     min: f64,
//     max_heap: BinaryHeap<NonNan>,
// }
//
// impl MinMaxHeap {
//     pub fn new() -> Self {
//         MinMaxHeap {
//             min: f64::INFINITY,
//             max_heap: BinaryHeap::new(),
//         }
//     }
//     pub fn clear(&mut self) {
//         self.min = f64::INFINITY;
//         self.max_heap.clear();
//     }
//
//     pub fn push(&mut self, item: f64) {
//         self.min = self.min.min(item);
//         self.max_heap.push(item.into());
//     }
//
//     pub fn min(&self) -> f64 {
//         self.min
//     }
//
//     pub fn max(&self) -> Option<f64> {
//         self.max_heap.peek().map(|&x| f64::from(x))
//     }
//
//     pub fn pop_max(&mut self) -> Option<f64> {
//         self.max_heap.pop().map(f64::from)
//     }
// }
//
// // Utility type for BinaryHeap usage.
// #[derive(Copy, Clone, Debug, PartialEq)]
// pub struct NonNan(f64);
//
// impl NonNan {
//     pub fn new(f: f64) -> Self {
//         assert!(!f.is_nan());
//         NonNan(f)
//     }
// }
//
// impl From<f64> for NonNan {
//     fn from(f: f64) -> Self {
//         NonNan::new(f)
//     }
// }
//
// impl From<NonNan> for f64 {
//     fn from(nn: NonNan) -> Self {
//         nn.0
//     }
// }
//
// impl PartialOrd for NonNan {
//     fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
//         self.0.partial_cmp(&other.0)
//     }
// }
//
// impl Eq for NonNan {}
//
// impl std::cmp::Ord for NonNan {
//     fn cmp(&self, other: &NonNan) -> Ordering {
//         self.partial_cmp(other).unwrap()
//     }
// }

pub type DistanceGradient<T = f64> = Tensor![T; S S 3 1];

#[derive(Clone, Debug)]
pub struct MappedDistanceGradient<T: Scalar> {
    /// Compressed sparse row sparse column gradient matrix.
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

    pub fn new(
        indices: &[MatrixElementIndex],
        blocks: Chunked3<&[T]>,
        num_rows: usize,
        num_cols: usize,
    ) -> MappedDistanceGradient<T> {
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
}

#[derive(Clone, Debug)]
pub struct MappedContactJacobian<T> {
    /// Compressed sparse row sparse column contact surface Jacobian matrix.
    pub matrix: ContactJacobian<T>,
    /// A Mapping from original triplets to the final compressed sparse matrix.
    pub mapping: Vec<Index>,
}

impl<T: Real> MappedContactJacobian<T> {
    fn from_triplets(triplet_jac: TripletContactJacobian<T>) -> Self {
        // debug_assert!(triplet_jac
        //     .block_indices
        //     .windows(2)
        //     .all(|w| w[0].0 < w[1].0 || (w[0].0 == w[1].0 && w[0].1 <= w[1].1)));

        let indices = &triplet_jac.block_indices;
        let nnz = indices.len();
        let mut entries = (0..nnz).collect::<Vec<_>>();
        entries.sort_by(|&a, &b| {
            indices[a]
                .0
                .cmp(&indices[b].0)
                .then_with(|| indices[a].1.cmp(&indices[b].1))
        });

        let blocks = triplet_jac.blocks.view();
        // let blocks =
        //     Chunked3::from_flat(triplet_jac.blocks.data.view().into_arrays()).into_arrays();

        let mut mapping = vec![Index::INVALID; blocks.len()];

        let triplet_iter = entries.iter().map(|&i| {
            (
                triplet_jac.block_indices[i].0,
                triplet_jac.block_indices[i].1,
                *blocks.isolate(i).into_arrays(),
            )
        });
        // let triplet_iter = triplet_jac
        //     .block_indices
        //     .iter()
        //     .zip(blocks.iter())
        //     .map(|((row, col), block)| (*row, *col, *block));

        let uncompressed = tensr::SSBlockMatrix3::from_block_triplets_iter_uncompressed(
            triplet_iter,
            triplet_jac.num_rows,
            triplet_jac.num_cols,
        );

        // Compress the CSR matrix.
        let matrix = uncompressed.pruned_with(
            |_, _, _| true,
            |src, dst| {
                mapping[entries[src]] = Index::new(dst);
            },
        );

        MappedContactJacobian {
            matrix: matrix.into_data(),
            mapping,
        }
    }

    fn update_values(&mut self, triplets: &TripletContactJacobian<T>) {
        let MappedContactJacobian { matrix, mapping } = self;

        matrix.storage_mut().fill(T::zero()); // Clear values.

        // Update blocks according to the predetermined mapping.
        for (&pos, block) in mapping.iter().zip(triplets.blocks.iter()) {
            if let Some(pos) = pos.into_option() {
                *matrix
                    .source
                    .data
                    .source
                    .view_mut()
                    .isolate(pos)
                    .into_arrays()
                    .as_mut_tensor() += *block.into_arrays().as_tensor();
            }
        }
    }

    #[allow(dead_code)]
    fn write_mtx(&self, count: u64) {
        use std::io::Write;
        let mut file = std::fs::File::create(&format!("./out/jac_{count:?}.jl")).unwrap();
        let mtx = self.matrix.view();
        // Print the jacobian if it's small enough
        let num_rows = mtx.into_tensor().num_rows() * 3;
        let num_cols = mtx.into_tensor().num_cols() * 3;
        writeln!(file, "Jrows = [").unwrap();
        for (row_idx, row) in mtx.into_iter() {
            for _ in row.into_iter() {
                for i in 0..3 {
                    for _ in 0..3 {
                        write!(file, "{:?}, ", 3 * row_idx + i + 1).unwrap();
                    }
                }
            }
        }
        writeln!(file, "]").unwrap();
        writeln!(file, "Jcols = [").unwrap();
        for (_, row) in mtx.into_iter() {
            for (col_idx, _) in row.into_iter() {
                for _ in 0..3 {
                    for j in 0..3 {
                        write!(file, "{:?}, ", 3 * col_idx + j + 1).unwrap();
                    }
                }
            }
        }
        writeln!(file, "]").unwrap();
        writeln!(file, "Jvals = [").unwrap();
        for (_, row) in mtx.into_iter() {
            for (_, block) in row.into_iter() {
                let arrays = block.into_arrays();
                for i in 0..3 {
                    for j in 0..3 {
                        write!(file, "{:?}, ", arrays[i][j].to_f64().unwrap()).unwrap();
                    }
                }
            }
        }
        writeln!(file, "]").unwrap();
        writeln!(
            file,
            "J = sparse(Jrows, Jcols, Jvals, {:?}, {:?})",
            num_rows, num_cols
        )
        .unwrap();
    }

    pub fn mul(
        &self,
        v: Chunked3<&[T]>,
        constrained_collider_vertices: &[usize],
        implicit_surface_vertex_indices: &[usize],
        collider_vertex_indices: &[usize],
    ) -> Chunked3<Vec<T>> {
        let surf_v = SubsetView::from_unique_ordered_indices(implicit_surface_vertex_indices, v);
        let mut vc = (self.matrix.view().into_tensor() * surf_v.into_tensor()).into_data();
        let col_v = SubsetView::from_unique_ordered_indices(collider_vertex_indices, v);
        *&mut vc.expr_mut() -=
            SubsetView::from_unique_ordered_indices(constrained_collider_vertices, col_v).expr();
        vc
    }

    pub fn transpose_mul(
        &self,
        vc: Chunked3<&[T]>,
        constrained_collider_vertices: &[usize],
        implicit_surface_vertex_indices: &[usize],
        collider_vertex_indices: &[usize],
        num_variables: usize,
    ) -> Chunked3<Vec<T>> {
        let surf_v = (self.matrix.view().into_tensor().transpose() * vc.into_tensor()).into_data();
        let mut v = Chunked3::from_flat(vec![T::zero(); num_variables]);
        for (&i, &[v0, v1, v2]) in implicit_surface_vertex_indices.iter().zip(surf_v.iter()) {
            v[i] = [v0, v1, v2];
        }
        for (&i, &[v0, v1, v2]) in constrained_collider_vertices.iter().zip(vc.iter()) {
            v[collider_vertex_indices[i]] = [-v0, -v1, -v2];
        }
        v
    }

    fn num_rows(&self) -> usize {
        self.matrix.view().into_tensor().num_rows()
    }
}

/// Unlike the contact Jacobian, this matrix is the full contact jacobian including the collider part.
#[derive(Clone, Debug)]
pub struct MappedContactGradient<T> {
    /// Compressed sparse row sparse column contact surface Jacobian matrix.
    pub matrix: ContactGradient<T>,
    /// A Mapping from original triplets to the final compressed sparse matrix.
    pub mapping: Vec<Index>,
}

impl<T: Real> MappedContactGradient<T> {
    // Given the Jacobian triplets we transpose them and construct the Row-major gradient matrix.
    fn from_triplets(triplet_jac: &TripletContactJacobian<T>) -> Self {
        let mut entries = (0..triplet_jac.block_indices.len()).collect::<Vec<_>>();

        // Sort indices into col major order
        entries.sort_by(|&a, &b| {
            triplet_jac.block_indices[a]
                .1
                .cmp(&triplet_jac.block_indices[b].1)
                .then_with(|| {
                    triplet_jac.block_indices[a]
                        .0
                        .cmp(&triplet_jac.block_indices[b].0)
                })
        });

        let blocks = triplet_jac.blocks.view();

        // Transpose triplet iterator.
        let triplet_iter = entries.iter().map(|&i| {
            (
                triplet_jac.block_indices[i].1,
                triplet_jac.block_indices[i].0,
                blocks
                    .isolate(i)
                    .into_arrays()
                    .as_tensor()
                    .transpose()
                    .into_data(),
            )
        });

        let uncompressed = tensr::SSBlockMatrix3::from_block_triplets_iter_uncompressed(
            triplet_iter,
            triplet_jac.num_cols,
            triplet_jac.num_rows,
        );

        let mut mapping = vec![Index::INVALID; entries.len()];

        // Compress the CSR matrix.
        let matrix = uncompressed.pruned_with(
            |_, _, _| true,
            |src, dst| {
                mapping[entries[src]] = Index::new(dst);
            },
        );

        MappedContactGradient {
            matrix: matrix.into_data(),
            mapping,
        }
    }

    fn update_values(&mut self, triplets: &TripletContactJacobian<T>) {
        // *self = MappedContactGradient::from_triplets(triplets);
        let MappedContactGradient { matrix, mapping } = self;

        matrix.storage_mut().fill(T::zero()); // Clear values.

        // Update blocks according to the predetermined mapping.
        for (&pos, block) in mapping.iter().zip(triplets.blocks.iter()) {
            if let Some(pos) = pos.into_option() {
                *matrix
                    .source
                    .data
                    .source
                    .view_mut()
                    .isolate(pos)
                    .into_arrays()
                    .as_mut_tensor() += block.into_arrays().as_tensor().transpose();
            }
        }
    }
}

#[derive(Clone, Debug)]
pub struct MappedSSBlockMatrix3<T> {
    /// Compressed sparse row sparse block matrix.
    pub matrix: SSBlockMatrix3<T>,
    /// A Mapping from original triplets to the final compressed sparse matrix.
    pub mapping: Vec<Index>,
}

impl<T: Real> MappedSSBlockMatrix3<T> {
    fn from_triplets(triplets: &BlockMatrix3Triplets<T>) -> Self {
        let indices = &triplets.block_indices;
        let nnz = indices.len();
        let mut entries = (0..nnz).collect::<Vec<_>>();
        entries.sort_by(|&a, &b| {
            indices[a]
                .0
                .cmp(&indices[b].0)
                .then_with(|| indices[a].1.cmp(&indices[b].1))
        });

        let blocks = triplets.blocks.view();

        let mut mapping = vec![Index::INVALID; blocks.len()];

        let triplet_iter = entries.iter().map(|&i| {
            (
                triplets.block_indices[i].0,
                triplets.block_indices[i].1,
                *blocks.isolate(i).into_arrays(),
            )
        });

        let uncompressed = tensr::SSBlockMatrix3::from_block_triplets_iter_uncompressed(
            triplet_iter,
            triplets.num_rows,
            triplets.num_cols,
        );

        // Compress the CSR matrix.
        let matrix = uncompressed.pruned_with(
            |_, _, _| true,
            |src, dst| {
                mapping[entries[src]] = Index::new(dst);
            },
        );

        MappedSSBlockMatrix3 {
            matrix: matrix.into_data(),
            mapping,
        }
    }

    fn update_values(&mut self, triplets: &BlockMatrix3Triplets<T>) {
        let Self { matrix, mapping } = self;

        matrix.storage_mut().fill(T::zero()); // Clear values.

        // Update blocks according to the predetermined mapping.
        for (&pos, block) in mapping.iter().zip(triplets.blocks.iter()) {
            if let Some(pos) = pos.into_option() {
                *matrix
                    .source
                    .data
                    .source
                    .view_mut()
                    .isolate(pos)
                    .into_arrays()
                    .as_mut_tensor() += *block.into_arrays().as_tensor();
            }
        }
    }
}

fn clone_cast_ssblock_mtx<T: Real, S: Real>(jac: &Tensor![T; S S 3 3]) -> Tensor![S; S S 3 3] {
    let storage: Vec<_> = jac.storage().iter().map(|&x| S::from(x).unwrap()).collect();
    jac.clone().clone_with_storage(storage)
}

/*
 * Functions defining the friction presliding profile.
 */

/// Antiderivative of the stabilized sliding profile multiplied by x.
/// This is used to implement the lagged friction potential.
#[inline]
fn stabilized_sliding_potential<T: Real>(x: T, mut epsilon: T) -> T {
    //   x - εlog(ε + x) + εlog(2ε)
    // = x + εlog(2ε / (ε + x))
    epsilon *= T::from(0.1).unwrap();
    x + epsilon * num_traits::Float::ln(T::from(2.0).unwrap() * epsilon / (epsilon + x))
}

/// Antiderivative of the quadratic sliding profile multiplied by x.
/// This is used to implement the lagged friction potential.
#[inline]
fn quadratic_sliding_potential<T: Real>(x: T, epsilon: T) -> T {
    let three = T::from(3.0).unwrap();
    if x < epsilon {
        (x / epsilon) * (x - (x * x) / (three * epsilon)) + epsilon / three
    } else {
        x
    }
}

/// This function is the C-infty sliding profile divided by `x`.
///
/// The sliding profile defines the relationship between velocity magnitude (the input)
/// and the friction force magnitude (the output).
///
/// The division is done for numerical stability to avoid division by zero.
/// `s(x) = 1 / (x + 0.1 * eps)`
#[inline]
fn stabilized_sliding_profile<T: Real>(x: T, epsilon: T) -> T {
    // Note that denominator is always >= 0.1eps since x > 0.
    T::one() / (x + T::from(0.1).unwrap() * epsilon)
}

/// This function is the quadratic C1 sliding profile divided by `x`, proposed by IPC.
///
/// The sliding profile defines the relationship between velocity magnitude (the input)
/// and the friction force magnitude (the output).
///
/// `s(x) = 2/eps - x/eps^2 if x < eps and 1 otherwise`
#[inline]
fn quadratic_sliding_profile<T: Real>(x: T, epsilon: T) -> T {
    // Quadratic smoothing function with compact support.
    // `s(x) = 2/eps - x/eps^2`
    if x < epsilon {
        (T::from(2.0).unwrap() - x / epsilon) / epsilon
    } else {
        T::one() / x
    }
}

/// Derivative of the sliding profile.
#[inline]
fn stabilized_sliding_profile_derivative<T: Real>(x: T, epsilon: T) -> T {
    let denom = x + T::from(0.1).unwrap() * epsilon;
    -T::one() / (denom * denom)
}

/// Derivative of the quadratic sliding profile.
#[inline]
fn quadratic_sliding_profile_derivative<T: Real>(x: T, epsilon: T) -> T {
    // `s(x) = -1/eps^2`
    if x < epsilon {
        -T::one() / (epsilon * epsilon)
    } else {
        -T::one() / (x * x)
    }
}

/// Helper function to compute the falloff anti-derivative.
///
/// Input is assumed to be non-negative.
///
/// We use the compact cubic falloff, which approximates compact cosine
/// falloff (see https://www.desmos.com/calculator/ra7t8ddwx6).
#[allow(dead_code)]
fn falloff_int<T: Real>(mut x: T, w: T) -> T {
    let half = T::from(0.5).unwrap();
    if x > w {
        half
    } else {
        if w > T::zero() {
            x /= w;
            let x3 = x * x * x;
            let x4 = x3 * x;
            half * x4 - x3 + x
        } else {
            T::zero()
        }
    }
}

/// Helper function to compute the falloff.
///
/// Input is assumed to be non-negative.
///
/// We use the compact cubic falloff, which approximates compact cosine
/// falloff (see https://www.desmos.com/calculator/ra7t8ddwx6).
fn falloff<T: Real>(x: T, w: T) -> T {
    if x > w {
        T::zero()
    } else {
        let _2 = T::from(2.0).unwrap();
        let xmw = x - w;
        let w3 = w * w * w;
        if w3 > T::zero() {
            (_2 * x + w) * xmw * xmw / w3
        } else {
            T::one()
        }
    }
}

/// Derivative of the falloff function.
///
/// Input is assumed to be non-negative.
///
/// See https://www.desmos.com/calculator/mag1naq3vk .
fn dfalloff<T: Real>(x: T, w: T) -> T {
    let _1 = T::one();
    if x > w {
        T::zero()
    } else {
        let _6 = T::from(6.0).unwrap();
        let w2 = w * w;
        if w2 > T::zero() {
            _6 * x * (x - w) / w2
        } else {
            T::zero()
        }
    }
}

/// The sliding potential.
///
/// This is the antiderivative of the function eta in the paper.
#[inline]
pub fn eta_int<T: Real>(
    v: Vector2<T>,
    factor: T,
    friction_params: &FrictionParams,
    is: impl FnOnce(T, T) -> T,
) -> T {
    let &FrictionParams {
        epsilon,
        // dynamic_friction,
        // static_friction,
        // stribeck_velocity,
        // viscous_friction,
        ..
    } = friction_params;

    // v /= T::from(stribeck_velocity).unwrap();
    let v_norm = v.norm();
    // let v_norm2 = v_norm*v_norm;

    let eps = T::from(epsilon).unwrap();
    // let mu_d = T::from(dynamic_friction).unwrap();
    // let mu_s = T::from(static_friction).unwrap();
    // let k = T::from(viscous_friction).unwrap();
    // let h = falloff(v_norm);

    // let half = T::from(0.5).unwrap();

    // TODO: optional: implement this for lagged friction.
    // let static_part = (mu_s - mu_d) * h) * is(v_norm, eps)
    // factor * is(v.norm(), eps) // Simplified model.
    // factor * (mu_d * is(v_norm, eps)) + static_part) + half * k * v_norm2
    factor * is(v_norm, eps) // Simplified model.
}

/// The full sliding profile including 1D direction.
///
/// This is the function eta in the paper.
///
/// See [desmos plot](https://www.desmos.com/calculator/nniv0lnlol) for complete friction model.
#[inline]
pub fn eta<T: Real>(
    v: Vector2<T>,
    factor: T,
    friction_params: &FrictionParams,
    s: impl FnOnce(T, T) -> T,
) -> Vector2<T> {
    // This is similar to function s but with the norm of v multiplied through to avoid
    // degeneracies.
    // let s = |x| stabilized_sliding_profile(x, epsilon);
    let &FrictionParams {
        epsilon,
        dynamic_friction,
        static_friction,
        stribeck_velocity,
        viscous_friction,
        ..
    } = friction_params;

    let vs = T::from(stribeck_velocity).unwrap();
    let v_norm = v.norm();

    let eps = T::from(epsilon).unwrap();
    let mu_d = T::from(dynamic_friction).unwrap();
    let mu_s = T::from(static_friction).unwrap();
    let k = T::from(viscous_friction).unwrap();
    let h = falloff(v_norm, vs);
    //v * (factor * s(v_norm, epsilon)) // Simplified model.
    v * (factor * (mu_d + (mu_s - mu_d) * h) * s(v_norm, eps) + k)
}

/// Jacobian of the full directional 1D sliding profile
#[inline]
pub fn eta_jac<T: Real>(
    v: Vector2<T>,
    factor: T,
    friction_params: &FrictionParams,
    s: impl FnOnce(T, T) -> T,
    ds: impl FnOnce(T, T) -> T,
) -> Matrix2<T> {
    let &FrictionParams {
        epsilon,
        dynamic_friction,
        static_friction,
        stribeck_velocity,
        viscous_friction,
        ..
    } = friction_params;

    let eps = T::from(epsilon).unwrap();
    let mu_d = T::from(dynamic_friction).unwrap();
    let mu_s = T::from(static_friction).unwrap();
    let k = T::from(viscous_friction).unwrap();
    let vs = T::from(stribeck_velocity).unwrap();
    let v_norm = v.norm();
    let s = s(v_norm, eps);
    let ds = ds(v_norm, eps);
    let h = falloff(v_norm, vs);
    let dh = dfalloff(v_norm, vs);
    let mut out = Matrix2::identity() * (k + (mu_d + (mu_s - mu_d) * h) * s * factor);
    if v_norm > T::zero() {
        out +=
            v * (mu_d * ds + (mu_s - mu_d) * (dh * s + h * ds)) * (v.transpose() * factor / v_norm);
    }
    out
}

#[derive(Copy, Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum FrictionProfile {
    Stabilized,
    Quadratic,
}

impl Default for FrictionProfile {
    fn default() -> Self {
        FrictionProfile::Stabilized
    }
}

impl FrictionProfile {
    /// The sliding potential.
    ///
    /// This is the antiderivative of the function eta in the paper.
    #[inline]
    pub fn potential<T: Real>(
        self,
        v: Vector2<T>,
        factor: T,
        friction_params: &FrictionParams,
    ) -> T {
        match self {
            FrictionProfile::Stabilized => eta_int(
                v,
                factor,
                friction_params,
                stabilized_sliding_potential::<T>,
            ),
            FrictionProfile::Quadratic => {
                eta_int(v, factor, friction_params, quadratic_sliding_potential::<T>)
            }
        }
    }

    /// The full sliding profile including 1D direction.
    ///
    /// This is the function eta in the paper.
    ///
    /// The complete friction model is illustrated on [Desmos](https://www.desmos.com/calculator/fpypkf9zsx).
    #[inline]
    pub fn profile<T: Real>(
        self,
        v: Vector2<T>,
        factor: T,
        friction_params: &FrictionParams,
    ) -> Vector2<T> {
        match self {
            FrictionProfile::Stabilized => {
                eta(v, factor, friction_params, stabilized_sliding_profile::<T>)
            }
            FrictionProfile::Quadratic => {
                eta(v, factor, friction_params, quadratic_sliding_profile::<T>)
            }
        }
    }

    /// Jacobian of the full directional 1D sliding profile
    #[inline]
    pub fn jacobian<T: Real>(
        self,
        v: Vector2<T>,
        factor: T,
        friction_params: &FrictionParams,
    ) -> Matrix2<T> {
        match self {
            FrictionProfile::Stabilized => eta_jac(
                v,
                factor,
                friction_params,
                stabilized_sliding_profile::<T>,
                stabilized_sliding_profile_derivative::<T>,
            ),
            FrictionProfile::Quadratic => eta_jac(
                v,
                factor,
                friction_params,
                quadratic_sliding_profile::<T>,
                quadratic_sliding_profile_derivative::<T>,
            ),
        }
    }
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
        let mtx = Sparse::from_dim(
            vec![],
            0,
            Chunked::from_offsets(
                vec![0],
                Sparse::from_dim(vec![], 0, Chunked3::from_flat(Chunked3::default())),
            ),
        );
        FrictionJacobianWorkspace {
            bc: Default::default(),
            contact_gradient_basis_eta_jac_basis_contact_jac: mtx.clone(),
            contact_gradient_basis_eta_jac_basis_jac_contact_jac: mtx.clone(),
            contact_gradient_basis_eta_jac_basis_jac: mtx.clone(),
            contact_gradient_jac_basis: mtx,
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Default)]
pub struct UpdateTimings {
    pub stash: Duration,
    pub contact_jac: Duration,
    pub contact_basis: Duration,
    pub contact_grad: Duration,
    pub collect_triplets: Duration,
    pub redistribute_triplets: Duration,
    pub jac_collect_triplets: Duration,
    pub jac_redistribute_triplets: Duration,
}

impl AddAssign<UpdateTimings> for UpdateTimings {
    fn add_assign(&mut self, rhs: UpdateTimings) {
        self.stash += rhs.stash;
        self.contact_jac += rhs.contact_jac;
        self.contact_basis += rhs.contact_basis;
        self.contact_grad += rhs.contact_grad;
        self.collect_triplets += rhs.collect_triplets;
        self.redistribute_triplets += rhs.redistribute_triplets;
        self.jac_collect_triplets += rhs.jac_collect_triplets;
        self.jac_redistribute_triplets += rhs.jac_redistribute_triplets;
    }
}

impl UpdateTimings {
    pub fn clear(&mut self) {
        self.stash = Duration::new(0, 0);
        self.contact_jac = Duration::new(0, 0);
        self.contact_basis = Duration::new(0, 0);
        self.contact_grad = Duration::new(0, 0);
        self.collect_triplets = Duration::new(0, 0);
        self.redistribute_triplets = Duration::new(0, 0);
        self.jac_collect_triplets = Duration::new(0, 0);
        self.jac_redistribute_triplets = Duration::new(0, 0);
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Default)]
pub struct FrictionTimings {
    pub total: Duration,
    pub jac_basis_mul: Duration,
}

impl AddAssign<FrictionTimings> for FrictionTimings {
    fn add_assign(&mut self, rhs: FrictionTimings) {
        self.total += rhs.total;
        self.jac_basis_mul += rhs.jac_basis_mul;
    }
}

impl FrictionTimings {
    pub fn clear(&mut self) {
        self.total = Duration::new(0, 0);
        self.jac_basis_mul = Duration::new(0, 0);
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Default)]
pub struct FrictionJacobianTimings {
    pub constraint_friction_force: Duration,
    pub contact_jacobian: Duration,
    pub contact_gradient: Duration,
    pub constraint_jacobian: Duration,
    pub constraint_gradient: Duration,
    pub f_lambda_jac: Duration,
    pub a: Duration,
    pub b: Duration,
    pub c: Duration,
    pub d: Duration,
    pub d_half: Duration,
    pub e: Duration,
}

impl AddAssign<FrictionJacobianTimings> for FrictionJacobianTimings {
    fn add_assign(&mut self, rhs: FrictionJacobianTimings) {
        self.constraint_friction_force += rhs.constraint_friction_force;
        self.contact_jacobian += rhs.contact_jacobian;
        self.contact_gradient += rhs.contact_gradient;
        self.constraint_jacobian += rhs.constraint_jacobian;
        self.constraint_gradient += rhs.constraint_gradient;
        self.f_lambda_jac += rhs.f_lambda_jac;
        self.a += rhs.a;
        self.b += rhs.b;
        self.c += rhs.c;
        self.d += rhs.d;
        self.d_half += rhs.d_half;
        self.e += rhs.e;
    }
}

impl FrictionJacobianTimings {
    pub fn clear(&mut self) {
        self.constraint_friction_force = Duration::new(0, 0);
        self.contact_jacobian = Duration::new(0, 0);
        self.contact_gradient = Duration::new(0, 0);
        self.constraint_jacobian = Duration::new(0, 0);
        self.constraint_gradient = Duration::new(0, 0);
        self.f_lambda_jac = Duration::new(0, 0);
        self.a = Duration::new(0, 0);
        self.b = Duration::new(0, 0);
        self.c = Duration::new(0, 0);
        self.d = Duration::new(0, 0);
        self.d_half = Duration::new(0, 0);
        self.e = Duration::new(0, 0);
    }
}

// Stashed data used for linesearch assist.
#[derive(Clone, Debug)]
pub struct LineSearchAssistStash<T = f64>
where
    T: Scalar,
{
    pub contact_basis: ContactBasis<T>,
    pub contact_jacobian: Option<MappedContactJacobian<T>>,
    pub distance_potential: Vec<T>,
    // pub candidate_alphas: MinMaxHeap,
}

impl<T: Real> LineSearchAssistStash<T> {
    pub fn clone_cast<S: Real>(&self) -> LineSearchAssistStash<S> {
        let mut contact_jacobian = None;
        if let Some(self_contact_jacobian) = self.contact_jacobian.as_ref() {
            contact_jacobian.replace(MappedContactJacobian {
                mapping: self_contact_jacobian.mapping.clone(),
                matrix: clone_cast_ssblock_mtx::<T, S>(&self_contact_jacobian.matrix),
            });
        }
        LineSearchAssistStash {
            contact_basis: self.contact_basis.clone_cast(),
            contact_jacobian,
            distance_potential: self
                .distance_potential
                .iter()
                .map(|&x| S::from(x).unwrap())
                .collect(),
            // candidate_alphas: self.candidate_alphas.clone(),
        }
    }

    pub fn new() -> Self {
        LineSearchAssistStash {
            contact_basis: ContactBasis::new(),
            contact_jacobian: None,
            distance_potential: Vec::new(),
            // candidate_alphas: MinMaxHeap::new(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct ContactState<T = f64>
where
    T: Scalar,
{
    pub contact_basis: ContactBasis<T>,
    pub contact_gradient: Option<MappedContactGradient<T>>,
    pub contact_jacobian: Option<MappedContactJacobian<T>>,
    pub point_constraint: PointContactConstraint<T>,
    pub lambda: Vec<T>,
    pub distance_potential: Vec<T>,
    pub distance_gradient: LazyCell<MappedDistanceGradient<T>>,
    /// Collider vertex indices for each active constraint.
    pub constrained_collider_vertices: Vec<usize>,
}

impl<T: Real> ContactState<T> {
    pub fn clone_cast<S: Real>(&self) -> ContactState<S> {
        let mut distance_gradient = LazyCell::new();
        if self.distance_gradient.filled() {
            distance_gradient.replace(self.distance_gradient.borrow().unwrap().clone_cast::<S>());
        }
        let mut contact_jacobian = None;
        if let Some(self_contact_jacobian) = self.contact_jacobian.as_ref() {
            contact_jacobian.replace(MappedContactJacobian {
                mapping: self_contact_jacobian.mapping.clone(),
                matrix: clone_cast_ssblock_mtx::<T, S>(&self_contact_jacobian.matrix),
            });
        }
        let mut contact_gradient = None;
        if let Some(self_contact_gradient) = self.contact_gradient.as_ref() {
            contact_gradient.replace(MappedContactGradient {
                mapping: self_contact_gradient.mapping.clone(),
                matrix: clone_cast_ssblock_mtx::<T, S>(&self_contact_gradient.matrix),
            });
        }
        ContactState {
            contact_basis: self.contact_basis.clone_cast(),
            contact_jacobian,
            contact_gradient,
            point_constraint: self.point_constraint.clone_cast(),
            distance_gradient,
            lambda: self.lambda.iter().map(|&x| S::from(x).unwrap()).collect(),
            distance_potential: self
                .distance_potential
                .iter()
                .map(|&x| S::from(x).unwrap())
                .collect(),
            constrained_collider_vertices: self.constrained_collider_vertices.clone(),
        }
    }
    pub fn new<VP: VertexMesh<f64>>(
        object: ContactSurface<&TriMesh, f64>,
        // Collision object consisting of points pushing against the solid object.
        collider: ContactSurface<&VP, f64>,
        contact_params: FrictionalContactParams,
    ) -> Result<Self, Error> {
        let constraint = PointContactConstraint::new(
            object,
            collider,
            contact_params.kernel,
            contact_params.friction_params.into(),
            contact_params.contact_offset,
            false, // Linearized penalty constraints are not supported
        )?;
        Ok(ContactState {
            contact_basis: ContactBasis::new(),
            contact_gradient: None,
            contact_jacobian: None,
            point_constraint: constraint,
            distance_gradient: LazyCell::new(),
            lambda: Vec::new(),
            distance_potential: Vec::new(),
            constrained_collider_vertices: Vec::new(),
        })
    }
    // Same as `update_collider_vertex_positions` but without knowledge about original vertex indices.
    pub fn update_surface_vertex_positions(
        &mut self,
        x: Chunked3<&[T]>,
        implicit_surface_vertex_indices: &[usize],
        rebuild_tree: bool,
    ) -> usize {
        let x = SubsetView::from_unique_ordered_indices(implicit_surface_vertex_indices, x);
        self.point_constraint
            .update_surface_with_mesh_pos_with_rebuild(x, rebuild_tree)
    }

    // Same as `update_collider_vertex_positions` but without knowledge about original vertex indices.
    pub fn update_collider_vertex_positions(
        &mut self,
        x: Chunked3<&[T]>,
        collider_vertex_indices: &[usize],
    ) {
        let x = SubsetView::from_unique_ordered_indices(collider_vertex_indices, x);
        self.point_constraint.update_collider_vertex_positions(x);
    }

    // Same as `update_collider_vertex_positions` but without knowledge about original vertex indices.
    pub fn update_surface_vertex_positions_cast<S: Real>(
        &mut self,
        x: Chunked3<&[S]>,
        implicit_surface_vertex_indices: &[usize],
    ) -> usize {
        let x = SubsetView::from_unique_ordered_indices(implicit_surface_vertex_indices, x);
        self.point_constraint.update_surface_with_mesh_pos_cast(x)
    }

    // Same as `update_collider_vertex_positions` but without knowledge about original vertex indices.
    pub fn update_collider_vertex_positions_cast<S: Real>(
        &mut self,
        x: Chunked3<&[S]>,
        collider_vertex_indices: &[usize],
    ) {
        let x = SubsetView::from_unique_ordered_indices(collider_vertex_indices, x);
        self.point_constraint
            .update_collider_vertex_positions_cast(x);
    }

    pub fn cached_distance_potential(
        &self,
        collider_vertex_indices: &[usize],
        num_vertices: usize,
    ) -> Vec<T> {
        let mut output = vec![T::zero(); num_vertices];
        let distance_potential = self.point_constraint.cached_constraint_value();
        for (&idx, &dist) in collider_vertex_indices
            .iter()
            .zip(distance_potential.as_slice())
        {
            output[idx] = dist;
        }
        output
    }

    /// Initializes the constraint gradient sparsity pattern.
    pub fn reset_distance_gradient(
        &mut self,
        implicit_surface_vertex_indices: &[usize],
        collider_vertex_indices: &[usize],
        num_vertices: usize,
    ) {
        let (indices, blocks): (Vec<_>, Vec<_>) = distance_jacobian_blocks_par_iter_fn(
            &self.point_constraint,
            implicit_surface_vertex_indices,
            collider_vertex_indices,
        )
        .map(|(row, col, block)| (MatrixElementIndex { row: col, col: row }, block))
        .unzip();
        let num_constraints = self.point_constraint.constraint_size();
        self.distance_gradient.replace(MappedDistanceGradient::new(
            indices.as_slice(),
            Chunked3::from_array_slice(blocks.as_slice()),
            num_vertices,
            num_constraints,
        ));
    }

    pub(crate) fn constraint_size(&self) -> usize {
        self.point_constraint.constraint_size()
    }

    pub fn update_distance_potential(&mut self) {
        // Take a slice of lambda for this particular contact constraint.
        let num_constraints = self.constraint_size();

        self.distance_potential.clear();
        self.distance_potential.resize(num_constraints, T::zero());
        self.point_constraint
            .compute_nonlinear_constraint(self.distance_potential.as_mut_slice());
    }

    /// Update the current state using the given position vector.
    ///
    /// If unsure whether the tree should be rebuilt, rebuild it.
    pub fn update_state_with_rebuild(
        &mut self,
        x: Chunked3<&[T]>,
        implicit_surface_vertex_indices: &[usize],
        collider_vertex_indices: &[usize],
        rebuild_tree: bool,
    ) {
        // eprintln!(
        //     "updating state for {:?}; rebuild: {:?}",
        //     x.storage(),
        //     rebuild_tree
        // );
        let num_vertices_updated =
            self.update_surface_vertex_positions(x, &implicit_surface_vertex_indices, rebuild_tree);
        assert_eq!(
            num_vertices_updated,
            self.point_constraint
                .implicit_surface
                .surface_vertex_positions()
                .len()
        );
        self.update_collider_vertex_positions(x, collider_vertex_indices);
    }

    pub fn update_state_cast<S: Real>(
        &mut self,
        x: Chunked3<&[S]>,
        implicit_surface_vertex_indices: &[usize],
        collider_vertex_indices: &[usize],
    ) {
        let num_vertices_updated =
            self.update_surface_vertex_positions_cast(x, implicit_surface_vertex_indices);
        assert_eq!(
            num_vertices_updated,
            self.point_constraint
                .implicit_surface
                .surface_vertex_positions()
                .len()
        );
        self.update_collider_vertex_positions_cast(x, collider_vertex_indices);
    }

    /// Update the cached constraint gradient for efficient future derivative computations.
    ///
    /// This function assumes that the `constraint_gradient` field sparsity has already been
    /// initialized.
    pub fn update_constraint_gradient(
        &mut self,
        implicit_surface_vertex_indices: &[usize],
        collider_vertex_indices: &[usize],
        timings: &mut UpdateTimings,
    ) {
        let MappedDistanceGradient { matrix, mapping } = self
            .distance_gradient
            .borrow_mut()
            .expect("Uninitialized constraint gradient.");

        // Clear matrix.
        matrix.storage_mut().fill(T::zero());
        {
            let mut matrix_blocks = Chunked3::from_flat(matrix.storage_mut().as_mut_slice());

            let t_begin = Instant::now();
            // Fill Gradient matrix with values from triplets according to our precomputed mapping.
            let triplets: Vec<_> = distance_jacobian_blocks_par_iter_fn(
                &self.point_constraint,
                implicit_surface_vertex_indices,
                collider_vertex_indices,
            )
            .collect();
            let t_collect_triplets = Instant::now();
            // Fill Gradient matrix with values from triplets according to our precomputed mapping.
            for (&pos, (_, _, block)) in mapping.iter().zip(triplets.into_iter()) {
                if let Some(pos) = pos.into_option() {
                    *matrix_blocks.view_mut().isolate(pos).as_mut_tensor() += block.as_tensor();
                }
            }
            let t_redistribute_triplets = Instant::now();
            timings.collect_triplets += t_collect_triplets - t_begin;
            timings.redistribute_triplets += t_redistribute_triplets - t_collect_triplets;
        }
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

    pub fn update_contact_jacobian(&mut self, update_timings: &mut UpdateTimings) {
        update_contact_jacobian(
            self.contact_jacobian.as_mut().unwrap(),
            &self.point_constraint.implicit_surface,
            self.point_constraint.collider_vertex_positions.view(),
            self.constrained_collider_vertices.as_slice(),
            update_timings,
        );
    }

    pub fn update_contact_basis(&mut self) {
        let normals = self.point_constraint.contact_normals();
        assert_eq!(normals.len(), self.lambda.len());
        self.contact_basis.update_from_normals(normals);
    }
}
//
// use std::sync::Mutex;
// use once_cell::sync::Lazy;
//
// static COUNTER: Lazy<Mutex<u64>> = Lazy::new(|| { Mutex::new(0) });

pub(crate) fn update_contact_jacobian<'a, T: Real>(
    contact_jacobian: &'a mut MappedContactJacobian<T>,
    surf: &QueryTopo<T>,
    query_points: Chunked3<&[T]>,
    constrained_collider_vertices: &'a [usize],
    timings: &mut UpdateTimings,
) {
    let constrained_collider_vertex_positions =
        Select::new(constrained_collider_vertices, query_points.view());

    let t_begin = Instant::now();
    let jac_triplets =
        TripletContactJacobian::from_selection(surf, constrained_collider_vertex_positions);
    let t_collect_triplets = Instant::now();

    contact_jacobian.update_values(&jac_triplets);
    // contact_jacobian.write_mtx(*COUNTER.lock().unwrap());
    // *COUNTER.lock().unwrap() += 1;

    let t_redistribute_triplets = Instant::now();
    timings.jac_collect_triplets += t_collect_triplets - t_begin;
    timings.jac_redistribute_triplets += t_redistribute_triplets - t_collect_triplets;
}

fn distance_jacobian_blocks_par_chunks_fn<'a, T, OP, TWS>(
    point_constraint: &'a PointContactConstraint<T>,
    implicit_surface_vertex_indices: &'a [usize],
    collider_vertex_indices: &'a [usize],
    ws: &mut [TWS],
    op: OP,
) where
    T: Real,
    TWS: Send + Sync,
    OP: Fn(&mut TWS, (usize, usize, [T; 3])) + Send + Sync + 'a,
{
    point_constraint.implicit_object_constraint_jacobian_blocks_par_chunks(
        ws,
        // Remap indices.
        |tws, (row, col, block)| op(tws, (row, implicit_surface_vertex_indices[col], block)),
    );
    point_constraint.implicit_collider_constraint_jacobian_blocks_par_chunks(
        ws,
        // Remap indices.
        |tws, (row, col, block)| op(tws, (row, collider_vertex_indices[col], block)),
    );
}

fn distance_jacobian_blocks_par_iter_fn<'a, T: Real>(
    point_constraint: &'a PointContactConstraint<T>,
    implicit_surface_vertex_indices: &'a [usize],
    collider_vertex_indices: &'a [usize],
) -> impl ParallelIterator<Item = (usize, usize, [T; 3])> + 'a {
    point_constraint
        .object_constraint_jacobian_blocks_par_iter()
        .map(move |(row, col, block)| (row, implicit_surface_vertex_indices[col], block))
        .chain(
            point_constraint
                .collider_constraint_jacobian_blocks_par_iter()
                .map(move |(row, col, block)| (row, collider_vertex_indices[col], block)),
        )
}

fn distance_jacobian_blocks_iter_fn<'a, T: Real>(
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

fn default_epsilon() -> f64 {
    0.0001
}

#[derive(Copy, Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct FrictionParams {
    pub dynamic_friction: f64,
    #[serde(default)]
    pub static_friction: f64,
    #[serde(default)]
    pub viscous_friction: f64,
    #[serde(default)]
    pub stribeck_velocity: f64,
    // Friction tolerance
    #[serde(default = "default_epsilon")]
    pub epsilon: f64,
    #[serde(default)]
    pub friction_profile: FrictionProfile,
    /// Use lagged friction.
    #[serde(default)]
    pub lagged: bool,
    /// Use Jacobian approximation
    #[serde(default)]
    pub incomplete_jacobian: bool,
}

impl FrictionParams {
    pub fn is_none(&self) -> bool {
        self.dynamic_friction == 0.0 && self.static_friction == 0.0 && self.viscous_friction == 0.0
    }
    pub fn is_some(&self) -> bool {
        !self.is_none()
    }
    pub fn into_option(self) -> Option<Self> {
        if self.is_none() {
            None
        } else {
            Some(self)
        }
    }
}

fn default_tolerance() -> f32 {
    0.0001
}

#[derive(Copy, Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FrictionalContactParams {
    pub kernel: implicits::KernelType,
    #[serde(default)]
    pub contact_offset: f64,
    /// Stiffness of the contact response.
    ///
    /// If not specified or set to zero, this value is initialized using `tolerance` as `stiffness = 1.0/tolerance`.
    ///
    /// Typically denoted by kappa.
    #[serde(default)]
    pub stiffness: f32,
    /// Distance from the surface along which a non-zero contact penalty is applied.
    ///
    /// This is typically denoted by delta or dhat in literature involving contacts.
    #[serde(default = "default_tolerance")]
    pub tolerance: f32,
    #[serde(skip_serializing_if = "FrictionParams::is_none", default)]
    pub friction_params: FrictionParams,
}

impl Default for FrictionalContactParams {
    fn default() -> Self {
        FrictionalContactParams {
            kernel: implicits::KernelType::Approximate {
                radius_multiplier: 1.0,
                tolerance: 1.0e-5,
            },
            stiffness: 0.0,
            tolerance: 0.0,
            contact_offset: 0.0,
            friction_params: Default::default(), // Frictionless by default
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
    /// Indices of original vertices for the implicit surface.
    pub implicit_surface_vertex_indices: Vec<usize>,
    /// Indices of original vertices for the collider.
    pub collider_vertex_indices: Vec<usize>,

    pub params: FrictionalContactParams,

    pub(crate) contact_state: ContactState<T>,
    pub(crate) contact_state_prev: ContactState<T>,

    // Used by linesearch assist.
    pub(crate) assist_stash: LineSearchAssistStash<T>,

    force_workspace: std::cell::RefCell<Vec<Vec<T>>>,

    jac_contact_jacobian: Option<MappedSSBlockMatrix3<T>>,
    jac_contact_gradient: Option<BlockMatrix3Triplets<T>>,

    friction_jacobian_workspace: FrictionJacobianWorkspace<T>,

    /// Constraint indices for each collider vertex.
    ///
    /// Unconstrained vertices are set to `INVALID`.
    collider_vertex_constraints: Vec<Index>,

    pub update_timings: std::cell::RefCell<UpdateTimings>,
    pub friction_timings: std::cell::RefCell<FrictionTimings>,
    pub jac_timings: std::cell::RefCell<FrictionJacobianTimings>,
}

impl<T: Real> PenaltyPointContactConstraint<T> {
    pub fn advance_state(&mut self) {
        self.contact_state_prev.clone_from(&self.contact_state);
    }

    pub fn clone_cast<S: Real>(&self) -> PenaltyPointContactConstraint<S> {
        let contact_state = self.contact_state.clone_cast();
        let contact_state_prev = self.contact_state_prev.clone_cast();
        let assist_stash = self.assist_stash.clone_cast();

        // let mut contact_jacobian = None;
        // if let Some(self_contact_jacobian) = self.contact_jacobian.as_ref() {
        //     contact_jacobian.replace(MappedContactJacobian {
        //         mapping: self_contact_jacobian.mapping.clone(),
        //         matrix: clone_cast_ssblock_mtx::<T, S>(&self_contact_jacobian.matrix),
        //     });
        // }
        // let mut contact_gradient = LazyCell::new();
        // if self.contact_gradient.filled() {
        //     contact_gradient.replace(clone_cast_ssblock_mtx::<T, S>(
        //         self.contact_gradient.borrow().unwrap(),
        //     ));
        // }
        let mut jac_contact_jacobian = None;
        if let Some(self_jac_contact_jacobian) = self.jac_contact_jacobian.as_ref() {
            jac_contact_jacobian.replace(MappedSSBlockMatrix3 {
                mapping: self_jac_contact_jacobian.mapping.clone(),
                matrix: clone_cast_ssblock_mtx::<T, S>(&self_jac_contact_jacobian.matrix),
            });
        }
        let mut jac_contact_gradient = None;
        if let Some(self_jac_contact_gradient) = self.jac_contact_gradient.as_ref() {
            jac_contact_gradient.replace(BlockMatrix3Triplets {
                block_indices: self_jac_contact_gradient.block_indices.clone(),
                blocks: {
                    let storage: Vec<_> = self_jac_contact_gradient
                        .blocks
                        .storage()
                        .iter()
                        .map(|&x| S::from(x).unwrap())
                        .collect();
                    self_jac_contact_gradient.blocks.clone_with_storage(storage)
                },
                num_rows: self_jac_contact_gradient.num_rows,
                num_cols: self_jac_contact_gradient.num_cols,
            });
        }
        PenaltyPointContactConstraint {
            implicit_surface_vertex_indices: self.implicit_surface_vertex_indices.clone(),
            collider_vertex_indices: self.collider_vertex_indices.clone(),
            params: self.params,
            contact_state,
            contact_state_prev,
            assist_stash,
            jac_contact_jacobian,
            jac_contact_gradient,
            friction_jacobian_workspace: FrictionJacobianWorkspace::default(),
            force_workspace: std::cell::RefCell::new(Vec::new()),
            // constrained_collider_vertices: self.constrained_collider_vertices.clone(),
            collider_vertex_constraints: self.collider_vertex_constraints.clone(),
            friction_timings: RefCell::new(FrictionTimings::default()),
            update_timings: RefCell::new(UpdateTimings::default()),
            jac_timings: RefCell::new(FrictionJacobianTimings::default()),
        }
    }

    pub fn new<VP: VertexMesh<f64>>(
        // Main object experiencing contact against its implicit surface representation.
        object: ContactSurface<&TriMesh, f64>,
        // Collision object consisting of points pushing against the solid object.
        collider: ContactSurface<&VP, f64>,
        params: FrictionalContactParams,
        num_vertices: usize,
        precompute_hessian_matrices: bool,
    ) -> Result<Self, Error> {
        let implicit_surface_vertex_indices = object
            .mesh
            .attrib_clone_into_vec::<usize, VertexIndex>(ORIGINAL_VERTEX_INDEX_ATTRIB)
            .unwrap_or_else(|_| (0..object.mesh.num_vertices()).collect::<Vec<_>>());
        let collider_vertex_indices = collider
            .mesh
            .attrib_clone_into_vec::<usize, VertexIndex>(ORIGINAL_VERTEX_INDEX_ATTRIB)
            .unwrap_or_else(|_| (0..collider.mesh.num_vertices()).collect::<Vec<_>>());

        let contact_state = ContactState::new(object, collider, params)?;
        let mut penalty_constraint = PenaltyPointContactConstraint {
            contact_state_prev: contact_state.clone(),
            contact_state,
            assist_stash: LineSearchAssistStash::new(),
            implicit_surface_vertex_indices,
            collider_vertex_indices,
            jac_contact_jacobian: None,
            jac_contact_gradient: None,
            force_workspace: RefCell::new(Vec::new()),
            friction_jacobian_workspace: FrictionJacobianWorkspace::default(),
            // constrained_collider_vertices: Vec::new(),
            collider_vertex_constraints: Vec::new(),
            params,
            friction_timings: RefCell::new(FrictionTimings::default()),
            update_timings: RefCell::new(UpdateTimings::default()),
            jac_timings: RefCell::new(FrictionJacobianTimings::default()),
        };

        penalty_constraint.precompute_contact_jacobian(num_vertices, precompute_hessian_matrices);
        penalty_constraint.update_distance_potential();
        penalty_constraint.update_multipliers();
        penalty_constraint.reset_distance_gradient(num_vertices);
        penalty_constraint.contact_state.update_contact_basis();
        penalty_constraint
            .contact_state_prev
            .clone_from(&penalty_constraint.contact_state);

        Ok(penalty_constraint)
    }

    /// Constructs a clone of this constraint with autodiff variables.
    #[inline]
    pub fn clone_as_autodiff<S: Real>(&self) -> PenaltyPointContactConstraint<ad::FT<S>> {
        self.clone_cast::<ad::FT<S>>()
    }

    // Same as `update_collider_vertex_positions` but without knowledge about original vertex indices.
    #[inline]
    pub fn update_surface_vertex_positions(
        &mut self,
        x: Chunked3<&[T]>,
        rebuild_tree: bool,
    ) -> usize {
        self.contact_state.update_surface_vertex_positions(
            x,
            &self.implicit_surface_vertex_indices,
            rebuild_tree,
        )
    }

    // Same as `update_collider_vertex_positions` but without knowledge about original vertex indices.
    #[inline]
    pub fn update_collider_vertex_positions(&mut self, x: Chunked3<&[T]>) {
        self.contact_state
            .update_collider_vertex_positions(x, &self.collider_vertex_indices);
    }

    // Same as `update_collider_vertex_positions` but without knowledge about original vertex indices.
    #[inline]
    pub fn update_surface_vertex_positions_cast<S: Real>(&mut self, x: Chunked3<&[S]>) -> usize {
        self.contact_state
            .update_surface_vertex_positions_cast(x, &self.implicit_surface_vertex_indices)
    }

    // Same as `update_collider_vertex_positions` but without knowledge about original vertex indices.
    #[inline]
    pub fn update_collider_vertex_positions_cast<S: Real>(&mut self, x: Chunked3<&[S]>) {
        self.contact_state
            .update_collider_vertex_positions_cast(x, &self.collider_vertex_indices);
    }

    #[inline]
    pub fn update_state(&mut self, x: Chunked3<&[T]>) {
        self.contact_state.update_state_with_rebuild(
            x,
            &self.implicit_surface_vertex_indices,
            &self.collider_vertex_indices,
            true,
        );
    }

    /// Update the current state using the given position vector.
    ///
    /// If unsure whether the tree should be rebuilt, rebuild it.
    #[inline]
    pub fn update_state_with_rebuild(&mut self, x: Chunked3<&[T]>, rebuild_tree: bool) {
        self.contact_state.update_state_with_rebuild(
            x,
            &self.implicit_surface_vertex_indices,
            &self.collider_vertex_indices,
            rebuild_tree,
        );
    }

    #[inline]
    pub fn update_state_cast<S: Real>(&mut self, x: Chunked3<&[S]>) {
        self.contact_state.update_state_cast(
            x,
            &self.implicit_surface_vertex_indices,
            &self.collider_vertex_indices,
        );
    }

    #[inline]
    pub fn point_constraint(&self) -> &PointContactConstraint<T> {
        &self.contact_state.point_constraint
    }

    #[inline]
    pub fn point_constraint_mut(&mut self) -> &mut PointContactConstraint<T> {
        &mut self.contact_state.point_constraint
    }

    #[inline]
    pub fn cached_distance_potential(&self, num_vertices: usize) -> Vec<T> {
        self.contact_state
            .cached_distance_potential(&self.collider_vertex_indices, num_vertices)
    }

    pub fn max_step_violated(&self, vel: Chunked3<&[T]>, dt: f64) -> bool {
        if self.contact_state.contact_jacobian.is_none() {
            // If contact_jacobian hasn't yet been initialized, this is impossible to answer, so assume everything is ok.
            return false;
        }
        let jac = self.contact_state.contact_jacobian.as_ref().unwrap();
        let contact_basis = &self.contact_state.contact_basis;

        let vc = jac.mul(
            vel,
            &self.contact_state.constrained_collider_vertices,
            &self.implicit_surface_vertex_indices,
            &self.collider_vertex_indices,
        );
        let max_vel = vc
            .iter()
            .enumerate()
            .map(|(i, &v)| {
                let [vn, _, _] = contact_basis.to_contact_coordinates(v, i);
                vn
            })
            .max_by(|x, y| x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Less))
            .unwrap_or_else(T::zero)
            .to_f64()
            .unwrap();

        let radius = self.point_constraint().implicit_surface.radius();
        let cur_max_step = self.point_constraint().implicit_surface.max_step() + radius;
        let result = max_vel * dt > cur_max_step;
        if result {
            log::debug!("Max step violation: {}", max_vel * dt - cur_max_step);
        }
        result
    }

    pub fn set_max_step(&mut self, max_step: f64) {
        self.point_constraint_mut()
            .implicit_surface
            .update_max_step(T::from(max_step).unwrap());
    }

    pub fn compute_max_step(&mut self, vel: Chunked3<&[T]>, dt: f64) -> f64 {
        if self.contact_state.contact_jacobian.is_none() {
            // Contact jacobian not yet initialized, just skip this step.
            // No need to increase max_step since we don't even know the velocity.
            return 0.0;
        }
        // Compute maximum relative velocity.
        let jac = self.contact_state.contact_jacobian.as_ref().unwrap();
        let contact_basis = &self.contact_state.contact_basis;

        let vc = jac.mul(
            vel,
            &self.contact_state.constrained_collider_vertices,
            &self.implicit_surface_vertex_indices,
            &self.collider_vertex_indices,
        );
        let max_vel = vc
            .iter()
            .enumerate()
            .map(|(i, &v)| {
                let [vn, _, _] = contact_basis.to_contact_coordinates(v, i);
                vn
            })
            .max_by(|x, y| x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Less))
            .unwrap_or_else(T::zero)
            .to_f64()
            .unwrap();

        let radius = self.point_constraint().implicit_surface.radius();
        let max_step = 0.0_f64.max(1.001 * (max_vel * dt) - radius);
        log::debug!("Updating max_step to: {max_step}");
        max_step
    }

    pub(crate) fn precompute_contact_jacobian(&mut self, num_vertices: usize, and_hessian: bool) {
        let pc = &mut self.contact_state.point_constraint;
        self.contact_state.constrained_collider_vertices = pc.active_constraint_indices();

        if self.params.friction_params.is_none() {
            return;
        }

        let num_constraints = self.contact_state.constrained_collider_vertices.len();

        let constrained_collider_vertices = &self.contact_state.constrained_collider_vertices;
        let constrained_collider_vertex_positions = Select::new(
            constrained_collider_vertices.as_slice(),
            pc.collider_vertex_positions.view(),
        );

        // Contact Jacobian
        let jac_triplets = TripletContactJacobian::from_selection(
            &pc.implicit_surface,
            constrained_collider_vertex_positions.view(),
        );
        self.contact_state.contact_jacobian =
            Some(MappedContactJacobian::from_triplets(jac_triplets));

        self.collider_vertex_constraints =
            vec![Index::invalid(); pc.collider_vertex_positions.len()];
        for (constraint_idx, &query_idx) in self
            .contact_state
            .constrained_collider_vertices
            .iter()
            .enumerate()
        {
            self.collider_vertex_constraints[query_idx] = constraint_idx.into();
        }

        // let jac1 = self.contact_jacobian.clone().unwrap();
        //
        // self.contact_jacobian.as_mut().unwrap().update_values(&jac_triplets);
        // let jac2 = self.contact_jacobian.clone().unwrap();
        // assert_eq!(&jac1.mapping, &jac2.mapping);
        // for ((row_idx1, row1), (row_idx2, row2)) in jac1.matrix.view().into_iter().zip(jac2.matrix.view().into_iter()) {
        //     assert_eq!(row1.len(), row2.len());
        //     assert_eq!(row_idx1, row_idx2);
        //     for ((idx1, entry1), (idx2, entry2)) in row1.into_iter().zip(row2.into_iter()) {
        //         assert_eq!(idx1, idx2);
        //         assert_eq!(entry1, entry2);
        //     }
        // }

        if and_hessian {
            // Contact gradient (used only during contact hessian computation)
            let jac_triplets = TripletContactJacobian::from_selection_reindexed_full(
                &pc.implicit_surface,
                constrained_collider_vertex_positions.view(),
                &self.collider_vertex_constraints,
                &self.implicit_surface_vertex_indices,
                &self.collider_vertex_indices,
                num_vertices,
            );
            self.contact_state.contact_gradient =
                Some(MappedContactGradient::from_triplets(&jac_triplets));

            let multipliers = vec![[T::zero(); 3]; num_vertices];
            let jac_triplets = Self::build_contact_jacobian_gradient_product(
                &pc.implicit_surface,
                pc.collider_vertex_positions.view().into_arrays(),
                &self.collider_vertex_constraints,
                &self.implicit_surface_vertex_indices,
                &self.collider_vertex_indices,
                &multipliers,
                num_vertices,
                num_constraints,
            )
            .unwrap();
            self.jac_contact_jacobian = Some(MappedSSBlockMatrix3::from_triplets(&jac_triplets));

            let multipliers = vec![[T::zero(); 3]; num_constraints];
            let jac_triplets = Self::build_contact_jacobian_jacobian_product(
                &pc.implicit_surface,
                pc.collider_vertex_positions.view().into_arrays(),
                &self.implicit_surface_vertex_indices,
                &self.collider_vertex_indices,
                &multipliers,
                num_vertices,
            )
            .unwrap();
            self.jac_contact_gradient = Some(jac_triplets);
        }
    }

    pub(crate) fn update_contact_gradient<'a>(
        contact_gradient: &'a mut MappedContactGradient<T>,
        surf: &QueryTopo<T>,
        query_points: Chunked3<&[T]>,
        constrained_collider_vertices: &'a [usize],
        collider_vertex_constraints: &'a [Index],
        implicit_surface_vertex_indices: &'a [usize],
        collider_vertex_indices: &'a [usize],
        num_vertices: usize,
    ) {
        let constrained_collider_vertex_positions =
            Select::new(constrained_collider_vertices, query_points.view());

        let triplets = TripletContactJacobian::from_selection_reindexed_full(
            surf,
            constrained_collider_vertex_positions,
            collider_vertex_constraints,
            implicit_surface_vertex_indices,
            collider_vertex_indices,
            num_vertices,
        );

        contact_gradient.update_values(&triplets);
    }

    pub(crate) fn update_jac_contact_jacobian<'a>(
        jac_contact_jacobian: &'a mut MappedSSBlockMatrix3<T>,
        surf: &QueryTopo<T>,
        query_points: Chunked3<&[T]>,
        collider_vertex_constraints: &'a [Index],
        implicit_surface_vertex_indices: &'a [usize],
        collider_vertex_indices: &'a [usize],
        multiplier: &'a [[T; 3]],
        num_vertices: usize,
        num_constraints: usize,
    ) {
        let triplets = Self::build_contact_jacobian_gradient_product(
            surf,
            query_points.view().into_arrays(),
            collider_vertex_constraints,
            implicit_surface_vertex_indices,
            collider_vertex_indices,
            multiplier,
            num_vertices,
            num_constraints,
        )
        .unwrap();

        jac_contact_jacobian.update_values(&triplets);
    }

    pub(crate) fn update_jac_contact_gradient<'a>(
        jac_contact_gradient: &'a mut BlockMatrix3Triplets<T>,
        surf: &QueryTopo<T>,
        query_points: Chunked3<&[T]>,
        implicit_surface_vertex_indices: &'a [usize],
        collider_vertex_indices: &'a [usize],
        multiplier: &'a [[T; 3]],
        num_vertices: usize,
    ) {
        jac_contact_gradient.blocks = Self::build_contact_jacobian_jacobian_product(
            surf,
            query_points.view().into_arrays(),
            implicit_surface_vertex_indices,
            collider_vertex_indices,
            multiplier,
            num_vertices,
        )
        .unwrap()
        .blocks;
    }

    /// Initializes the constraint gradient sparsity pattern.
    #[inline]
    pub fn reset_distance_gradient(&mut self, num_vertices: usize) {
        self.contact_state.reset_distance_gradient(
            &self.implicit_surface_vertex_indices,
            &self.collider_vertex_indices,
            num_vertices,
        );
    }

    pub fn update_neighbors(&mut self, x: Chunked3<&[T]>, and_hessian: bool) -> bool {
        self.update_state(x);

        let updated = self.contact_state.point_constraint.implicit_surface.reset(
            self.contact_state
                .point_constraint
                .collider_vertex_positions
                .as_arrays(),
        );

        // Updating neighbours invalidates the constraint gradient so we must recompute
        // the sparsity pattern here.

        if updated {
            self.precompute_contact_jacobian(x.len(), and_hessian);
            self.update_distance_potential();
            self.update_multipliers();
            self.reset_distance_gradient(x.len());
            self.contact_state.update_contact_basis();
        }
        updated
    }

    /// Update the cached constraint gradient for efficient future derivative computations.
    ///
    /// This function assumes that the `constraint_gradient` field sparsity has already been
    /// initialized.
    #[inline]
    pub fn update_constraint_gradient(&mut self) {
        self.contact_state.update_constraint_gradient(
            &self.implicit_surface_vertex_indices,
            &self.collider_vertex_indices,
            &mut self.update_timings.borrow_mut(),
        );
    }

    #[inline]
    pub fn num_collider_vertices(&self) -> usize {
        self.point_constraint().collider_vertex_positions.len()
    }

    #[inline]
    pub fn update_distance_potential(&mut self) {
        self.contact_state.update_distance_potential();
    }

    /// Computes the derivative of a cubic penalty function for contacts multiplied by `-κ`.
    #[inline]
    pub fn update_multipliers(&mut self) {
        self.contact_state
            .update_multipliers(self.params.tolerance, self.params.stiffness);
    }

    pub(crate) fn assist_line_search_for_contact(
        &mut self,
        mut alpha: T,
        // pos_cur: Chunked3<&[T]>,
        pos_next: Chunked3<&[T]>,
        f1: Chunked3<&[T]>,
        f2: Chunked3<&[T]>,
    ) -> T {
        let delta = T::from(self.params.tolerance).unwrap();
        // self.update_state(pos_cur);
        // self.update_distance_potential();
        // {
        //     let d1 = &mut self.distance_potential_alt;
        //     d1.resize(self.distance_potential.len(), T::zero());
        //     d1.copy_from_slice(&self.distance_potential);
        // }

        self.assist_stash
            .distance_potential
            .clone_from(&self.contact_state.distance_potential);
        self.update_state(pos_next);
        self.update_distance_potential();
        let d1 = &self.assist_stash.distance_potential;
        let d2 = &self.contact_state.distance_potential;
        // eprintln!("d1 = {:?}", d1);
        // eprintln!("d2 = {:?}", d2);

        let constrained_collider_vertices =
            self.contact_state.constrained_collider_vertices.as_slice();

        assert_eq!(d2.len(), constrained_collider_vertices.len());
        assert_eq!(d2.len(), d1.len());

        for (i, (&d1, &d2)) in d1.iter().zip(d2.iter()).enumerate() {
            if d1 > delta && d2 <= T::zero() {
                let vtx_idx = self.collider_vertex_indices[constrained_collider_vertices[i]];
                let f1 = f1[vtx_idx];
                let f2 = f2[vtx_idx];
                // eprintln!("contact {} is sliding at vtx {}", i, vtx_idx);
                // eprintln!("f2 = {f2:?}; f1 = {f1:?}");
                if f2.into_tensor().norm_squared() > f1.into_tensor().norm_squared() {
                    let candidate_alpha = (d1 - T::from(0.5).unwrap() * delta) / (d1 - d2);
                    alpha = num_traits::Float::min(alpha, candidate_alpha);
                }
            }
        }
        alpha
    }

    pub(crate) fn assist_line_search_for_friction(
        &mut self,
        alpha: T,
        p: Chunked3<&[T]>,
        vel: Chunked3<&[T]>,
        f1: Chunked3<&[T]>,
        f2: Chunked3<&[T]>,
        // pos_next: Chunked3<&[T]>,
    ) -> (T, u64) {
        if self.params.friction_params.is_none() {
            return (T::zero(), 0);
        }

        let LineSearchAssistStash {
            contact_jacobian,
            contact_basis,
            ..
        } = &mut self.assist_stash;

        let delta = T::from(self.params.tolerance).unwrap();
        // self.update_state(pos_cur);
        // self.update_distance_potential();
        // {
        //     let d1 = &mut self.distance_potential_alt;
        //     d1.resize(self.distance_potential.len(), T::zero());
        //     d1.copy_from_slice(&self.distance_potential);
        // }

        // Update state to get a current estimate for the potential.

        let alpha_min = T::from(1e-10).unwrap();

        // let mut f1 = Chunked3::from_flat(vec![T::zero(); vel.len() * 3]);
        // let mut f2 = Chunked3::from_flat(vec![T::zero(); vel.len() * 3]);
        // self.subtract_friction_force(f1.view_mut(), vel, epsilon);
        // self.subtract_friction_force(f2.view_mut(), vel_next, epsilon);
        // let epsilon = T::from(epsilon).unwrap();

        let constrained_collider_vertices =
            self.contact_state.constrained_collider_vertices.as_slice();

        // Contact jacobian and contact basis from previous step.
        let jac = contact_jacobian.as_ref().unwrap();
        //eprintln!("contact_basis = {:?}", &contact_basis);

        // Compute relative velocity at the point of contact: `vc = J(x)v`
        // assert_eq!(jac.view().into_tensor().num_cols(), vel.len());
        assert_eq!(p.len(), vel.len());
        //let vc = (jac.view().into_tensor() * vel.into_tensor()).into_data();
        let vc = jac.mul(
            vel,
            constrained_collider_vertices,
            &self.implicit_surface_vertex_indices,
            &self.collider_vertex_indices,
        );
        //let pc = (jac.view().into_tensor() * p.into_tensor()).into_data();
        let mut pc = jac.mul(
            p,
            constrained_collider_vertices,
            &self.implicit_surface_vertex_indices,
            &self.collider_vertex_indices,
        );

        // let d1 = &self.distance_potential_alt;
        let d2 = &self.contact_state.distance_potential;
        // eprintln!("alpha before: {alpha}");
        // eprintln!("vc = {:?}", &vc);
        // eprintln!("pc = {:?}", &pc);
        // eprintln!("d2 = {:?}", d2);

        assert_eq!(d2.len(), pc.len());

        let mut sum_alpha = T::zero();
        let mut num_alphas = 0;

        for (i, (&d2, (p, v))) in d2.iter().zip(pc.iter_mut().zip(vc.iter())).enumerate() {
            if d2 <= delta {
                let vtx_idx = self.collider_vertex_indices[constrained_collider_vertices[i]];
                let f1 = f1[vtx_idx];
                let f2 = f2[vtx_idx];
                // eprintln!("contact {} is sliding at vtx {}", i, vtx_idx);
                // eprintln!("f2 = {f2:?}; f1 = {f1:?}");
                if f2.into_tensor().norm_squared() > f1.into_tensor().norm_squared() {
                    // eprintln!("armijo violated");
                    let [_v0, v1, v2] = contact_basis.to_contact_coordinates(*v, i);
                    let [_p0, p1, p2] = contact_basis.to_contact_coordinates(*p, i);
                    // eprintln!("p = {:?}; v = {:?}", [p1, p2], [v1, v2]);
                    let v2 = Vector2::from([v1, v2]);
                    let p2 = Vector2::from([p1, p2]);
                    let p_dot_v = alpha * p2.dot(v2);
                    let v_dot_v = v2.dot(v2);
                    if p_dot_v <= -v_dot_v {
                        // eprintln!("direction switched");
                        let candidate_alpha = -v_dot_v / p_dot_v;
                        // Only add alphas if they are sufficiently smaller than the minimimum.
                        // This keeps the heaps from getting too large.
                        if candidate_alpha < alpha && candidate_alpha > alpha_min {
                            sum_alpha += candidate_alpha;
                            num_alphas += 1;
                        }
                    }
                }
            }
        }
        (sum_alpha, num_alphas)
    }

    #[inline]
    pub fn subtract_constraint_force(&self, mut f: Chunked3<&mut [T]>) {
        self.distance_jacobian_blocks_iter()
            .for_each(|(row, col, j)| {
                // if row == 0 && col == 0 {
                //     eprintln!(
                //         "contact ({row},{col}): {:?}",
                //         -*j.as_tensor() * self.contact_state.lambda[row]
                //     );
                // }
                *f[col].as_mut_tensor() -= *j.as_tensor() * self.contact_state.lambda[row];
            });
    }

    pub fn contact_constraint(&self) -> T {
        let dist = self.contact_state.distance_potential.as_slice();
        let kappa = T::from(self.params.stiffness).unwrap();
        let delta = self.params.tolerance;
        let constraint = dist
            .iter()
            .map(|&d| kappa * ContactPenalty::new(delta).b(d))
            .sum();
        constraint
    }

    pub fn lagged_friction_potential(&self, v: Chunked3<&[T]>, dqdv: T) -> T {
        // Compute friction potential.
        let state_prev = &self.contact_state_prev;
        let lambda = &state_prev.lambda;
        let constrained_collider_vertices = state_prev.constrained_collider_vertices.as_slice();
        let contact_basis = &state_prev.contact_basis;

        if self.params.friction_params.is_none() {
            return T::zero();
        }
        // Contact Jacobian is defined for object vertices only. Contact Jacobian for collider vertices is negative identity.
        let jac = state_prev.contact_jacobian.as_ref().unwrap();
        assert_eq!(jac.matrix.len(), lambda.len());

        // Compute relative velocity at the point of contact: `vc = J(x)v`
        // assert_eq!(jac.view().into_tensor().num_cols(), v.len());
        // let vc = (jac.view().into_tensor() * v.into_tensor()).into_data();
        let vc = jac.mul(
            v,
            constrained_collider_vertices,
            &self.implicit_surface_vertex_indices,
            &self.collider_vertex_indices,
        );

        assert_eq!(lambda.len(), vc.len());

        // Compute sliding bases velocity product.
        let params = &self.params.friction_params;

        let eta = params.friction_profile;
        let friction_potential: T = vc
            .iter()
            .zip(lambda.iter())
            .enumerate()
            .map(|(i, (vc, lambda))| {
                let [_v1, v1, v2] = contact_basis.to_contact_coordinates(*vc, i);
                let vc_t = [v1, v2].into_tensor();
                eta.potential(vc_t, *lambda, params)
            })
            .sum();

        friction_potential * dqdv
    }

    pub fn subtract_constraint_force_par(&self, mut f: Chunked3<&mut [T]>) {
        let Self {
            contact_state:
                ContactState {
                    point_constraint,
                    lambda,
                    ..
                },
            force_workspace,
            implicit_surface_vertex_indices,
            collider_vertex_indices,
            ..
        } = self;
        let fws = &mut *force_workspace.borrow_mut();
        let ncpus = num_cpus::get();
        fws.resize(ncpus, Vec::new());
        for fws_vec in fws.iter_mut() {
            fws_vec.resize(f.storage().len(), T::zero());
            fws_vec.fill(T::zero());
        }

        distance_jacobian_blocks_par_chunks_fn(
            point_constraint,
            implicit_surface_vertex_indices,
            collider_vertex_indices,
            fws,
            |fws, (row, col, j)| {
                let mut fws = Chunked3::from_flat(fws.as_mut_slice());
                *fws[col].as_mut_tensor() -= *j.as_tensor() * lambda[row];
            },
        );

        // Accumulate results into f.
        for fw in fws.iter() {
            f.storage_mut()
                .par_iter_mut()
                .zip(fw.par_iter())
                .for_each(|(out, f)| {
                    *out += *f;
                })
        }
    }

    // Compute friction force `f_f(x,v) = -μT(x)H(T(x)'v)λ(x)` and subtract it from `f`.
    pub fn subtract_friction_force(
        &mut self,
        mut f: Chunked3<&mut [T]>,
        v: Chunked3<&[T]>,
        lagged: bool,
    ) {
        if let Some(f_f) = self.compute_friction_force(v, lagged) {
            assert_eq!(f.len(), v.len());
            *&mut f.expr_mut() -= f_f.expr();
        }
    }

    // Information about the contact space for debugging.
    // Normals tangents, bitangents and relative velocities.
    #[allow(dead_code)]
    pub fn add_contact_data(
        &mut self,
        v: Chunked3<&[T]>,
        mut n_out: Chunked3<&mut [T]>,
        mut t_out: Chunked3<&mut [T]>,
        mut b_out: Chunked3<&mut [T]>,
        mut v_rel_out: Chunked3<&mut [T]>,
    ) {
        let params = self.params.friction_params;

        if params.is_none() {
            return;
        }

        let state = if params.lagged {
            &self.contact_state_prev
        } else {
            &self.contact_state
        };

        let jac = state.contact_jacobian.as_ref().unwrap();

        // Compute relative velocity at the point of contact: `vc = J(x)v`
        //assert_eq!(jac.num_cols() + pc.collider_vertex_positions.len(), v.len());
        let vc = jac.mul(
            v,
            &state.constrained_collider_vertices,
            &self.implicit_surface_vertex_indices,
            &self.collider_vertex_indices,
        );

        // let mut grad_psi = vec![[T::zero(); 3]; state.constrained_collider_vertices.len()];
        // state.point_constraint.implicit_surface.query_jacobian(state.point_constraint.collider_vertex_positions.view().into(), &mut grad_psi);

        for (i, vc) in vc.iter().enumerate() {
            let mtx = state.contact_basis.contact_basis_matrix(i);
            let vtx_idx = self.collider_vertex_indices[state.constrained_collider_vertices[i]];
            *n_out[vtx_idx].as_mut_tensor() += mtx[0];
            // *n_out[vtx_idx].as_mut_tensor() += grad_psi[i].into_tensor();
            *t_out[vtx_idx].as_mut_tensor() += mtx[1];
            *b_out[vtx_idx].as_mut_tensor() += mtx[2];
            *v_rel_out[vtx_idx].as_mut_tensor() += *vc.as_tensor();
        }
    }

    // Compute `b(x,v) = H(T(x)'v)λ(x)`, friction force in contact space.
    //
    // This function uses current state. To get an upto date friction impulse call update_state.
    // This version of computing friction force works in the space of all active constraints
    // as opposed to all active contacts.
    pub fn compute_constraint_friction_force(
        state: &ContactState<T>,
        constrained_collider_vertices: &[usize],
        implicit_surface_vertex_indices: &[usize],
        collider_vertex_indices: &[usize],
        // Velocity
        v: Chunked3<&[T]>,
        friction_params: &FrictionParams,
    ) -> Chunked3<Vec<T>> {
        let lambda = &state.lambda;

        let jac = state.contact_jacobian.as_ref().unwrap();
        assert_eq!(jac.num_rows(), lambda.len());

        // Compute relative velocity at the point of contact: `vc = J(x)v`
        //assert_eq!(jac.num_cols() + pc.collider_vertex_positions.len(), v.len());
        let mut vc = jac.mul(
            v,
            constrained_collider_vertices,
            implicit_surface_vertex_indices,
            collider_vertex_indices,
        );

        assert_eq!(lambda.len(), vc.len());

        // Compute sliding bases velocity product.

        // Compute `vc <- H(B'(x) vc) λ(x)`.
        let eta = friction_params.friction_profile;
        vc.view_mut()
            .into_iter()
            .zip(lambda.iter())
            .enumerate()
            .for_each(|(i, (vc, lambda))| {
                let [_v0, v1, v2] = state.contact_basis.to_contact_coordinates(*vc, i);
                let vc_t = [v1, v2].into_tensor();
                let fc_smoothed = eta.profile(vc_t, *lambda, friction_params).into_data();
                *vc = [T::zero(), fc_smoothed[0], fc_smoothed[1]];
            });
        vc
    }

    /// Contact gradient is used for computing explicit friction derivatives.
    #[inline]
    pub fn update_sliding_basis(&mut self, and_gradient: bool, stash: bool, num_vertices: usize) {
        if self.params.friction_params.is_none() {
            return;
        }

        let t_begin = Instant::now();
        if stash {
            self.assist_stash
                .contact_jacobian
                .clone_from(&self.contact_state.contact_jacobian);
            self.assist_stash
                .contact_basis
                .clone_from(&self.contact_state.contact_basis);
        }
        let t_stash = Instant::now();
        self.contact_state
            .update_contact_jacobian(&mut *self.update_timings.borrow_mut());
        let t_contact_jac = Instant::now();
        self.contact_state.update_contact_basis();
        let t_contact_basis = Instant::now();

        if and_gradient {
            Self::update_contact_gradient(
                self.contact_state.contact_gradient.as_mut().unwrap(),
                &self.contact_state.point_constraint.implicit_surface,
                self.contact_state
                    .point_constraint
                    .collider_vertex_positions
                    .view(),
                &self.contact_state.constrained_collider_vertices,
                &self.collider_vertex_constraints,
                &self.implicit_surface_vertex_indices,
                &self.collider_vertex_indices,
                num_vertices,
            );
        }
        let t_contact_grad = Instant::now();
        let timings = &mut *self.update_timings.borrow_mut();
        timings.stash += t_stash - t_begin;
        timings.contact_jac += t_contact_jac - t_stash;
        timings.contact_basis += t_contact_basis - t_contact_jac;
        timings.contact_grad += t_contact_grad - t_contact_basis;
    }

    // Compute `f(x,v) = -μT(x)H(T(x)'v)λ(x)`.
    //
    // This is the force of friction *on* the implicit surface.
    //
    // This function uses current state. To get an upto date friction impulse call update_state.
    pub fn compute_friction_force(
        &mut self,
        // Velocity
        v: Chunked3<&[T]>,
        lagged: bool,
    ) -> Option<Chunked3<Vec<T>>> {
        let t_begin = Instant::now();
        let params = self.params.friction_params.into_option()?;
        let state = if lagged || params.lagged {
            &self.contact_state_prev
        } else {
            &self.contact_state
        };

        // Computes `b(x,v) = H(T(x)'v) λ(x)`.
        let mut vc = Self::compute_constraint_friction_force(
            &state,
            &state.constrained_collider_vertices,
            &self.implicit_surface_vertex_indices,
            &self.collider_vertex_indices,
            v,
            &params,
        );

        let t_vc = Instant::now();

        let jac = state.contact_jacobian.as_ref().unwrap();

        // Compute `vc <- -mu B(x) vc`.
        vc.view_mut().into_iter().enumerate().for_each(|(i, v)| {
            let vc = v.as_tensor();
            *v = Vector3::from(
                state
                    .contact_basis
                    .from_contact_coordinates(*vc * (-T::one()), i),
            )
            //.cast::<f64>()
            //.cast::<T>()
            .into_data();
        });

        // Compute object force (compute `f = J'(x)vc`)
        let f = jac.transpose_mul(
            vc.view(),
            &state.constrained_collider_vertices,
            &self.implicit_surface_vertex_indices,
            &self.collider_vertex_indices,
            v.len() * 3,
        );

        let t_vc_jac = Instant::now();
        let timings = &mut *self.friction_timings.borrow_mut();
        timings.total += t_vc_jac - t_begin;
        timings.jac_basis_mul += t_vc_jac - t_vc;
        // eprintln!("f = {:?}", &f.view().storage());
        Some(f.into_data())
    }

    /// Jacobian of the contact basis matrix with respect to vertex positions.
    ///
    /// If `transpose` is `true` this function computes the Jacobian of the contact basis transpose matrix.
    pub fn contact_basis_matrix_jacobian_product<'a>(
        surf: &'a QueryTopo<T>,
        collider_vertex_positions: Chunked3<&'a [T]>,
        query_points_in_contact: &'a [Index],
        implicit_surface_vertex_indices: &'a [usize],
        collider_vertex_indices: &'a [usize],
        contact_basis: &'a ContactBasis<T>,
        multipliers: &'a [[T; 3]],
        transpose: bool,
    ) -> Result<impl Iterator<Item = ((usize, usize), Matrix3<T>)> + 'a, Error> {
        // Contacts occur at the vertex positions of the colliding mesh.
        let normals = &contact_basis.normals;

        // Compute the unnormalized normals (need this for the norms).
        let mut grad_psi = vec![[T::zero(); 3]; query_points_in_contact.len()];
        surf.query_jacobian_full(collider_vertex_positions.view().into(), &mut grad_psi);

        let qhess_mult = vec![T::one(); query_points_in_contact.len()]; // This is more than we need.

        Ok(surf
            .query_hessian_product_indexed_blocks_iter(collider_vertex_positions.into(), qhess_mult)
            .filter_map(move |(vtx_idx, nml_jac)| {
                query_points_in_contact[vtx_idx]
                    .into_option()
                    .map(|contact_idx| {
                        (
                            contact_idx,
                            vtx_idx,
                            collider_vertex_indices[vtx_idx],
                            nml_jac,
                        )
                    })
            })
            .chain(
                surf.sample_query_hessian_indexed_blocks_iter(collider_vertex_positions.into())?
                    .filter_map(move |(query_vtx_idx, vtx_idx, nml_jac)| {
                        query_points_in_contact[query_vtx_idx]
                            .into_option()
                            .map(|contact_idx| {
                                (
                                    contact_idx,
                                    query_vtx_idx,
                                    implicit_surface_vertex_indices[vtx_idx],
                                    nml_jac,
                                )
                            })
                    }),
            )
            .map(move |(contact_idx, query_vtx_idx, vtx_idx, nml_jac)| {
                let n = normals[contact_idx].into_tensor();

                let norm_n_inv = T::one() / grad_psi[query_vtx_idx].into_tensor().norm();

                // Compute the Jacobian of the normalized negative grad psi. (see compute_normals for reference on why).
                let nml_jac = (Matrix3::identity() - n * n.transpose())
                    * nml_jac.into_tensor()
                    * (-norm_n_inv);

                // Find the axis that is most aligned with the normal, then use the next axis for the
                // tangent.
                let mut t = Vector3::zero();
                let tangent_axis = (n.iamax() + 1) % 3;
                t[tangent_axis] = T::one();

                // Project out the normal component.
                t -= n * n[tangent_axis];

                let norm_t_inv = T::one() / t.norm();
                t *= norm_t_inv;

                // Jacobian of unnormalized tangent.
                let jac_t = -(nml_jac * n[tangent_axis] + n * nml_jac[tangent_axis].transpose());
                let t_proj = Matrix3::identity() - t * t.transpose();
                let tangent_jac = t_proj * jac_t * norm_t_inv;

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
        query_points: Chunked3<&'a [T]>,
        collider_vertex_constraints: &[Index],
        multipliers: &[[T; 3]],
        implicit_surface_vertex_indices: &[usize],
        collider_vertex_indices: &[usize],
        num_vertices: usize,
        num_constraints: usize,
        transpose: bool,
    ) -> Result<Tensor![T; S S 3 3], Error> {
        let (indices, blocks): (Vec<_>, Vec<_>) = Self::contact_basis_matrix_jacobian_product(
            surf,
            query_points,
            collider_vertex_constraints,
            implicit_surface_vertex_indices,
            collider_vertex_indices,
            contact_basis,
            multipliers,
            transpose,
        )?
        .filter(|((_, col), _)| col < &num_vertices)
        .map(|((row, col), mtx)| {
            //eprintln!("dBm({},{}): {:?}", col, row, mtx.transpose().into_data());
            (MatrixElementIndex { row: col, col: row }, mtx.transpose())
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
            .map(|&i| (indices[i].row, indices[i].col, blocks[i].into_data()));

        Ok(tensr::SSBlockMatrix3::from_block_triplets_iter(
            triplet_iter,
            num_vertices,
            num_constraints,
        )
        .into_data())
    }

    /// Builds a contact Jacobian gradient product.
    ///
    /// Computes `grad_q (J(q) b)'` where `b` is the multiplier.
    ///
    pub(crate) fn build_contact_jacobian_gradient_product(
        surf: &implicits::QueryTopo<T>,
        query_points: &[[T; 3]],
        collider_vertex_constraints: &[Index],
        implicit_surface_vertex_indices: &[usize],
        collider_vertex_indices: &[usize],
        multipliers: &[[T; 3]],
        num_vertices: usize,
        num_contacts: usize,
    ) -> Result<BlockMatrix3Triplets<T>, Error> {
        let (indices, blocks): (Vec<_>, Vec<_>) = surf
            .contact_jacobian_jacobian_product_indexed_blocks_iter(
                query_points.view(),
                multipliers,
            )?
            .map(|(row, col, mtx)| (row, implicit_surface_vertex_indices[col], mtx))
            .chain(
                surf.contact_query_jacobian_jacobian_product_indexed_blocks_iter(
                    query_points.view(),
                    multipliers,
                )?
                .map(|(row, col, mtx)| (row, collider_vertex_indices[col], mtx)),
            )
            // Remove inactive vertices
            .filter(|(_, col, _)| col < &num_vertices)
            // Reindex rows to be in contact indices
            .filter_map(|(row, col, mtx)| {
                collider_vertex_constraints[row]
                    .into_option()
                    .map(|contact_idx| (contact_idx, col, mtx))
            })
            // Transpose to create gradient
            .map(|(row, col, mtx)| {
                (
                    (col, row),
                    mtx.into_tensor().into_data(), // mtx was column-major
                )
            })
            .unzip();

        Ok(BlockMatrix3Triplets {
            block_indices: indices,
            blocks: Chunked3::from_flat(Chunked3::from_array_vec(
                Chunked3::from_array_vec(blocks).into_storage(),
            )),
            num_rows: num_vertices,
            num_cols: num_contacts,
        })
    }

    /// Builds a contact Jacobian Jacobian product.
    ///
    /// Computes `grad_q J(q) b` where `b` is the multiplier.
    ///
    pub(crate) fn build_contact_jacobian_jacobian_product(
        surf: &implicits::QueryTopo<T>,
        query_points: &[[T; 3]],
        implicit_surface_vertex_indices: &[usize],
        collider_vertex_indices: &[usize],
        multipliers: &[[T; 3]],
        num_vertices: usize,
    ) -> Result<BlockMatrix3Triplets<T>, Error> {
        let (indices, blocks): (Vec<_>, Vec<_>) = surf
            .contact_hessian_product_indexed_blocks_iter(query_points, multipliers)?
            // Reindex
            .filter_map(move |(row, col, block)| {
                let row = implicit_surface_vertex_indices[row];
                let col = implicit_surface_vertex_indices[col];
                if row < num_vertices && col < num_vertices {
                    // eprintln!("Amtx: ({row}, {col}): {block:?}");
                    Some(((row, col), Matrix3::from(block).transpose().into_data()))
                } else {
                    None
                }
            })
            .chain(
                surf.contact_query_hessian_product_indexed_blocks_iter(query_points, multipliers)?
                    // Reindex
                    .filter_map(move |(row, col, block)| {
                        let row = implicit_surface_vertex_indices[row];
                        let col = collider_vertex_indices[col];
                        if row < num_vertices && col < num_vertices {
                            // eprintln!("Amtx: ({row}, {col}): {block:?}");
                            Some(((row, col), Matrix3::from(block).transpose().into_data()))
                        } else {
                            None
                        }
                    }),
            )
            .unzip();

        Ok(BlockMatrix3Triplets {
            block_indices: indices,
            blocks: Chunked3::from_flat(Chunked3::from_array_vec(
                Chunked3::from_array_vec(blocks).into_storage(),
            )),
            num_rows: num_vertices,
            num_cols: num_vertices,
        })
    }

    // Compute the product of friction matrix `mu J'B H(B'Jv)` with constraint Jacobian term `-k ddb dd/dq` and `dq/dv`.
    // constraint_jac is column major, but contact_gradient is row-major.
    // Output blocks are row-major.
    fn contact_constraint_jac_product_map<'a>(
        lambda: &'a [T],
        dist: &'a [T],
        contact_basis: &'a ContactBasis<T>,
        // Relative velocity in contact space: `vc = J(x)v`
        mut vc: Chunked3<Vec<T>>,
        contact_gradient: ContactGradientView<'a, T>, // G
        constraint_jac: Tensor![T; &'a S S 1 3],      // dd/dq
        num_vertices: usize,
        params: FrictionalContactParams,
        dqdv: T,
    ) -> Vec<(usize, usize, Matrix3<T>)> {
        //assert_eq!(constraint_jac.len(), contact_jac.into_tensor().num_cols());

        let eta = params.friction_params.friction_profile;
        let kappa = T::from(params.stiffness).unwrap();
        let delta = T::from(params.tolerance).unwrap();
        let friction_params = &params.friction_params;

        assert_eq!(vc.len(), lambda.len());

        // Compute `vc <- mu B(x) H(B'(x) vc)` this is now a diagonal block matrix stored as the vector vc.
        vc.view_mut().iter_mut().enumerate().for_each(|(i, vc)| {
            let [_, v1, v2] = contact_basis.to_contact_coordinates(*vc, i);
            let vc_t = [v1, v2].into_tensor();
            let vc_t_smoothed = eta.profile(vc_t, T::one(), friction_params).into_data();
            //dbg!(eta(vc_t, T::one(), T::from(1e-5).unwrap()).into_data());
            *vc = contact_basis
                .from_contact_coordinates([T::zero(), vc_t_smoothed[0], vc_t_smoothed[1]], i)
        });

        let vc = vc.view();

        // Iterate over each column of the constraint Jacobian.
        contact_gradient
            .into_iter()
            .filter(move |(row_idx, _)| *row_idx < num_vertices)
            .flat_map(move |(row_idx, lhs_row)| {
                constraint_jac
                    .into_iter()
                    .filter(move |(col_idx, _)| *col_idx < num_vertices)
                    .flat_map(move |(col_idx, rhs_col)| {
                        let mut lhs_row_iter = lhs_row.into_iter().peekable();
                        rhs_col
                            .into_iter()
                            .filter_map(move |(rhs_constraint_idx, rhs_block)| {
                                // Find the next matching lhs entry.
                                while let Some(&(lhs_constraint_idx, _)) = lhs_row_iter.peek() {
                                    match rhs_constraint_idx.cmp(&lhs_constraint_idx) {
                                        Ordering::Less => {
                                            return None;
                                        } // Skips entry and advances rhs iterator.
                                        Ordering::Greater => {
                                            // Skip lhs entry and continue in the loop.
                                            let _ = lhs_row_iter.next().unwrap();
                                        }
                                        Ordering::Equal => {
                                            // Found entry that matches both rhs_col and lhs_row.
                                            let (lhs_constraint_idx, lhs_block) =
                                                lhs_row_iter.next().unwrap();
                                            let lhs_block = *lhs_block.into_arrays().as_tensor();
                                            let rhs_block = *rhs_block.into_arrays().as_tensor();
                                            let index = lhs_constraint_idx; // = rhs_constraint_idx
                                                                            //dbg!(rhs_block* (-dqdv
                                                                            //    * kappa
                                                                            //    * ContactPenalty::new(delta)
                                                                            //    .ddb(dist[rhs_constraint_idx])));
                                            let rhs = *vc[index].as_tensor()
                                                * (rhs_block
                                                    * (-dqdv
                                                        * kappa
                                                        * ContactPenalty::new(delta)
                                                            .ddb(dist[rhs_constraint_idx])));
                                            //dbg!(&rhs.into_data());
                                            return Some((row_idx, col_idx, lhs_block * rhs));
                                        }
                                    }
                                }
                                None
                            })
                    })
            })
            .collect()
    }

    // Assume all the state (including friction workspace) has been updated
    pub(crate) fn friction_jacobian_indexed_value_iter<'a>(
        &'a mut self,
        v: Chunked3<&'a [T]>,
        dqdv: T,
        max_index: usize,
    ) -> Option<impl Iterator<Item = (usize, usize, T)> + 'a> {
        let params = self.params.friction_params.into_option()?;

        let t_begin = Instant::now();
        let eta = params.friction_profile;

        let state = if params.lagged {
            &self.contact_state_prev
        } else {
            &self.contact_state
        };

        // TODO: Refactor this monstrosity of a function.

        // TODO: no need to update it ... again. should be already done during contact phase
        // if recompute_contact_jacobian {
        //     self.update_constraint_gradient();
        // }

        let t_constraint_gradient = Instant::now();

        let num_vertices = max_index;
        let num_constraints = state.constraint_size();

        // Compute `c <- H(T(x)'v)λ(x)`
        let c = Self::compute_constraint_friction_force(
            state,
            &state.constrained_collider_vertices,
            &self.implicit_surface_vertex_indices,
            &self.collider_vertex_indices,
            v,
            &self.params.friction_params.into_option()?,
        );
        assert_eq!(c.len(), num_constraints);

        let friction_params = &self.params.friction_params;

        let t_constraint_friction_force = Instant::now();

        // TODO: memoize the contact Jacobian:
        //       Store the Jacobian sparsity at the beginning of each step with all of potential contact points.
        //       At each residual or Jacobian function call, replace the sparse rows with those
        //       of just the active contact points creating an even more sparse Jacobian equivalent
        //       to this one but only a *view*. The values of the contact Jacobian can be updated
        //       also but this way we avoid doing any new allocations. This is left as a todo since
        //       we need to profile if this is actually worthwhile.
        // Contact Jacobian is defined for object vertices only. Contact Jacobian for collider vertices is trivial.
        let contact_jacobian = state.contact_jacobian.as_ref().unwrap();
        assert_eq!(contact_jacobian.num_rows(), num_constraints);
        let t_contact_jacobian = Instant::now();

        let contact_gradient = &state.contact_gradient.as_ref().unwrap().matrix;
        assert_eq!(
            contact_gradient.view().into_tensor().num_cols(),
            num_constraints
        );

        let t_contact_gradient = Instant::now();

        let MappedDistanceGradient { matrix: g, .. } = state
            .distance_gradient
            .borrow()
            .expect("Uninitialized constraint gradient.");
        let j_view = Self::constraint_gradient_column_major_transpose(g.view());
        // j_view is col-major, but tensors only understand row major, so here cols means rows.
        assert_eq!(j_view.view().into_tensor().num_cols(), num_constraints);

        let t_constraint_jacobian = Instant::now();

        let dqdv = T::from(dqdv).unwrap();

        let t_e;
        let t_f_lambda_jac;
        let t_a;
        let t_b;
        let t_c;
        let t_d_half;
        let t_d;

        let update_timings = |timings: &mut FrictionJacobianTimings,
                              t_e,
                              t_f_lambda_jac,
                              t_a,
                              t_b,
                              t_c,
                              t_d_half,
                              t_d| {
            timings.constraint_gradient += t_constraint_gradient - t_begin;
            timings.constraint_friction_force +=
                t_constraint_friction_force - t_constraint_gradient;
            timings.contact_jacobian += t_contact_jacobian - t_constraint_friction_force;
            timings.contact_gradient += t_contact_gradient - t_contact_jacobian;
            timings.constraint_jacobian += t_constraint_jacobian - t_contact_gradient;
            timings.e += t_e - t_constraint_jacobian;
            timings.f_lambda_jac += t_f_lambda_jac - t_e;
            timings.a += t_a - t_f_lambda_jac;
            timings.b += t_b - t_a;
            timings.c += t_c - t_b;
            timings.d += t_d - t_c;
            timings.d_half += t_d_half - t_c;
        };

        let (f_lambda_jac, jac_contact_gradient) = {
            // Compute relative velocity at the point of contact: `vc = J(x)v`
            // TODO: This is already compute in c, reuse that value.
            let vc = contact_jacobian.mul(
                v,
                &state.constrained_collider_vertices,
                &self.implicit_surface_vertex_indices,
                &self.collider_vertex_indices,
            );
            assert_eq!(vc.len(), num_constraints);

            let surf = &state.point_constraint.implicit_surface;
            let query_points = &state.point_constraint.collider_vertex_positions;

            // eprintln!("Amult = {:?}", &self.friction_jacobian_workspace.bc.as_arrays());

            // Construct full change of basis matrix B
            let b = state.contact_basis.full_basis_matrix();
            let bt = state.contact_basis.full_basis_matrix_transpose();

            let lambda = &state.lambda;

            // Construct the eta matrix dH
            let dh_blocks: Chunked3<Vec<_>> = vc
                .iter()
                .zip(lambda.iter())
                .enumerate()
                .flat_map(|(constraint_idx, (vc, lambda))| {
                    let [_, v1, v2] = state
                        .contact_basis
                        .to_contact_coordinates(*vc, constraint_idx);
                    let mtx = eta.jacobian([v1, v2].into(), *lambda, friction_params);
                    std::iter::once([T::zero(); 3])
                        .chain(std::iter::once([T::zero(), mtx[0][0], mtx[0][1]]))
                        .chain(std::iter::once([T::zero(), mtx[1][0], mtx[1][1]]))
                })
                .collect();
            let dh = BlockDiagonalMatrix3::new(Chunked3::from_flat(dh_blocks));

            // Compute (E)

            let mut contact_gradient_t = contact_gradient.clone().into_tensor().transpose(); // Make mut
            contact_gradient_t.premultiply_block_diagonal_mtx(bt.view());
            //for (row_idx, row) in contact_gradient_t.0.view().into_data().into_iter() {
            //    for (col_idx, block) in row.into_iter() {
            //        eprintln!("jbt2({},{}): {:?}", row_idx, col_idx, block);
            //    }
            //}
            contact_gradient_t.premultiply_block_diagonal_mtx(dh.view());
            //for (row_idx, row) in contact_gradient_t.0.view().into_data().into_iter() {
            //    for (col_idx, block) in row.into_iter() {
            //        eprintln!("dh-jbt2({},{}): {:?}", row_idx, col_idx, block);
            //    }
            //}
            contact_gradient_t.premultiply_block_diagonal_mtx(b.view());
            //for (row_idx, row) in contact_gradient_t.0.view().into_data().into_iter() {
            //    for (col_idx, block) in row.into_iter() {
            //        eprintln!("b-dh-jbt2({},{}): {:?}", row_idx, col_idx, block);
            //    }
            //}
            self.friction_jacobian_workspace
                .contact_gradient_basis_eta_jac_basis_contact_jac =
                (contact_gradient.view().into_tensor() * contact_gradient_t.view()).into_data(); // (E)

            t_e = Instant::now();

            if params.lagged {
                // Only need E here
                update_timings(
                    &mut *self.jac_timings.borrow_mut(),
                    t_e,
                    t_e,
                    t_e,
                    t_e,
                    t_e,
                    t_e,
                    t_e,
                );
                return Some(Either::Left(Either::Left(
                    self.friction_jacobian_workspace
                        .contact_gradient_basis_eta_jac_basis_contact_jac
                        .view()
                        .into_iter()
                        .flat_map(move |(row_idx, row)| {
                            // (E)
                            row.into_iter().map(move |(col_idx, block)| {
                                (row_idx, col_idx, *block.into_arrays().as_tensor())
                            })
                        })
                        // .inspect(|(i, j, m)| {
                        //     if *i == 9 && *j == 9 {
                        //         log::trace!("E:({},{}): {:?}", i, j, (*m).into_data())
                        //     }
                        // })
                        .filter(move |&(row, col, _)| row < max_index && col < max_index)
                        .flat_map(move |(row, col, block)| {
                            (0..3).flat_map(move |r| {
                                (0..3).map(move |c| (3 * row + r, 3 * col + c, block[r][c]))
                            })
                        }),
                )));
            }

            // Derivative of lambda term.
            let f_lambda_jac = Self::contact_constraint_jac_product_map(
                lambda,
                state.distance_potential.as_slice(),
                &state.contact_basis,
                vc.clone(),
                contact_gradient.view(),
                j_view,
                num_vertices,
                self.params,
                dqdv,
            );

            t_f_lambda_jac = Instant::now();

            if params.incomplete_jacobian {
                // Only need E here
                update_timings(
                    &mut *self.jac_timings.borrow_mut(),
                    t_e,
                    t_e,
                    t_e,
                    t_e,
                    t_e,
                    t_e,
                    t_e,
                );
                return Some(Either::Left(Either::Right(
                    f_lambda_jac
                        .into_iter()
                        .chain(
                            self.friction_jacobian_workspace
                                .contact_gradient_basis_eta_jac_basis_contact_jac
                                .view()
                                .into_iter()
                                .flat_map(move |(row_idx, row)| {
                                    // (E)
                                    row.into_iter().map(move |(col_idx, block)| {
                                        (row_idx, col_idx, *block.into_arrays().as_tensor())
                                    })
                                }), // .inspect(|(i, j, m)| {
                                    //     if *i == 9 && *j == 9 {
                                    //         log::trace!("E:({},{}): {:?}", i, j, (*m).into_data())
                                    //     }
                                    // })
                        )
                        .filter(move |&(row, col, _)| row < max_index && col < max_index)
                        .flat_map(move |(row, col, block)| {
                            (0..3).flat_map(move |r| {
                                (0..3).map(move |c| (3 * row + r, 3 * col + c, block[r][c]))
                            })
                        }),
                )));
            }

            {
                let Self {
                    friction_jacobian_workspace: FrictionJacobianWorkspace { bc, .. },
                    contact_state:
                        ContactState {
                            ref contact_basis, ..
                        },
                    ..
                } = self;

                // Compute `bc(x,v) = mu * H(T(x)'v)λ(x) * dq/dv`, friction force in physical space.
                *bc = c
                    .iter()
                    .enumerate()
                    .map(|(i, &v)| {
                        (contact_basis.from_contact_coordinates(v, i).into_tensor() * dqdv)
                            .into_data()
                    })
                    .collect();
            }

            // Compute (A)
            // if recompute_contact_jacobian {
            Self::update_jac_contact_gradient(
                self.jac_contact_gradient.as_mut().unwrap(),
                surf,
                query_points.view(),
                &self.implicit_surface_vertex_indices,
                &self.collider_vertex_indices,
                self.friction_jacobian_workspace.bc.as_arrays(),
                num_vertices,
            );
            // }
            let jac_contact_gradient = self.jac_contact_gradient.as_ref().unwrap();

            t_a = Instant::now();

            // Compute (B)
            let jac_basis = Self::build_contact_basis_gradient_product_from_selection(
                surf,
                &state.contact_basis,
                query_points.view(),
                self.collider_vertex_constraints.view(),
                c.view().into_arrays(),
                &self.implicit_surface_vertex_indices,
                &self.collider_vertex_indices,
                num_vertices,
                num_constraints,
                false,
            )
            .ok()?
            .into_tensor()
            .transpose();

            // for (row_idx, row) in jac_basis.0.view().into_data().into_iter() {
            //     if row_idx == 0 {
            //         for (col_idx, block) in row.into_iter() {
            //             let v = Vector3::from([block.data[0], block.data[1], block.data[2]]);
            //             eprintln!("jb({},{}): {:?}", row_idx, col_idx, (v * dqdv).into_data());
            //         }
            //     }
            // }

            self.friction_jacobian_workspace.contact_gradient_jac_basis =
                (contact_gradient.view().into_tensor() * jac_basis.view()).into_data(); // (B)

            t_b = Instant::now();

            // Compute (C)

            let mut jac_basis_t = Self::build_contact_basis_gradient_product_from_selection(
                surf,
                &state.contact_basis,
                query_points.view(),
                self.collider_vertex_constraints.view(),
                vc.view().into_arrays(),
                &self.implicit_surface_vertex_indices,
                &self.collider_vertex_indices,
                num_vertices,
                num_constraints,
                true,
            )
            .ok()?
            .into_tensor()
            .transpose();

            // for (row_idx, row) in jac_basis_t.0.view().into_data().into_iter() {
            //     for (col_idx, block) in row.into_iter() {
            //         eprintln!("jbt({},{}): {:?}", row_idx, col_idx, block);
            //     }
            // }

            jac_basis_t.premultiply_block_diagonal_mtx(dh.view());

            // for (row_idx, row) in jac_basis_t.0.view().into_data().into_iter() {
            //     for (col_idx, block) in row.into_iter() {
            //         eprintln!("dh-jbt({},{}): {:?}", row_idx, col_idx, block);
            //     }
            // }

            jac_basis_t.premultiply_block_diagonal_mtx(b.view());
            // for (row_idx, row) in jac_basis_t.0.view().into_data().into_iter() {
            //     for (col_idx, block) in row.into_iter() {
            //         eprintln!("b-dh-jbt({},{}): {:?}", row_idx, col_idx, block);
            //     }
            // }
            self.friction_jacobian_workspace
                .contact_gradient_basis_eta_jac_basis_jac =
                (contact_gradient.view().into_tensor() * jac_basis_t.view()).into_data(); // (C)

            t_c = Instant::now();

            // Compute (D)

            let v0 =
                SubsetView::from_unique_ordered_indices(&self.implicit_surface_vertex_indices, v);
            let v0vec: Vec<[T; 3]> = v0.iter().cloned().collect();

            // eprintln!("Dmult = {v0vec:?}");

            // if recompute_contact_jacobian {
            Self::update_jac_contact_jacobian(
                self.jac_contact_jacobian.as_mut().unwrap(),
                surf,
                query_points.view(),
                self.collider_vertex_constraints.view(),
                &self.implicit_surface_vertex_indices,
                &self.collider_vertex_indices,
                v0vec.view(),
                num_vertices,
                num_constraints,
            );
            // }
            let mut jac_contact_jacobian = self
                .jac_contact_jacobian
                .clone()
                .unwrap()
                .matrix
                .into_tensor()
                .transpose();

            t_d_half = Instant::now();

            jac_contact_jacobian.premultiply_block_diagonal_mtx(bt.view());
            jac_contact_jacobian.premultiply_block_diagonal_mtx(dh.view());
            jac_contact_jacobian.premultiply_block_diagonal_mtx(b.view());
            self.friction_jacobian_workspace
                .contact_gradient_basis_eta_jac_basis_jac_contact_jac =
                (contact_gradient.view().into_tensor() * jac_contact_jacobian.view()).into_data(); // (D)

            t_d = Instant::now();

            (f_lambda_jac, jac_contact_gradient)
        };

        update_timings(
            &mut *self.jac_timings.borrow_mut(),
            t_e,
            t_f_lambda_jac,
            t_a,
            t_b,
            t_c,
            t_d_half,
            t_d,
        );

        // Combine all matrices.

        Some(Either::Right(
            f_lambda_jac
                .into_iter()
                // .inspect(|(i,j,m)| log::trace!("dL: ({},{}):{:?}", i, j, (*m).into_data()))
                .chain(
                    jac_contact_gradient
                        .block_indices
                        .iter()
                        .zip(jac_contact_gradient.blocks.iter())
                        .map(|((row, col), mtx)| (*row, *col, *mtx.into_arrays().as_tensor())), // .inspect(move |(i, j, m)| {
                                                                                                //     if !m.is_zero() && *i < max_index && *j < max_index {
                                                                                                //          log::debug!("A: {:?}", m);
                                                                                                //     }
                                                                                                // })
                ) // (A)
                .chain(
                    self.friction_jacobian_workspace
                        .contact_gradient_jac_basis
                        .view()
                        .into_iter()
                        .flat_map(move |(row_idx, row)| {
                            // (B)
                            row.into_iter().map(move |(col_idx, block)| {
                                (row_idx, col_idx, *block.into_arrays().as_tensor() * dqdv)
                            })
                        }), //.inspect(|(i,j,m)| log::trace!("B:({},{}): {:?}", i,j,(*m).into_data())) ,
                )
                .chain(
                    self.friction_jacobian_workspace
                        .contact_gradient_basis_eta_jac_basis_jac
                        .view()
                        .into_iter()
                        .flat_map(move |(row_idx, row)| {
                            // (C)
                            row.into_iter().map(move |(col_idx, block)| {
                                (row_idx, col_idx, *block.into_arrays().as_tensor() * dqdv)
                            })
                        }), //.inspect(|(i,j,m)| log::trace!("C:({},{}): {:?}", i,j,(*m).into_data())) ,
                )
                .chain(
                    self.friction_jacobian_workspace
                        .contact_gradient_basis_eta_jac_basis_jac_contact_jac
                        .view()
                        .into_iter()
                        .flat_map(move |(row_idx, row)| {
                            // (D)
                            row.into_iter().map(move |(col_idx, block)| {
                                (row_idx, col_idx, *block.into_arrays().as_tensor() * dqdv)
                            })
                        }), //.inspect(|(i,j,m)| log::trace!("D:({},{}): {:?}", i,j,(*m).into_data())) ,
                )
                .chain(
                    self.friction_jacobian_workspace
                        .contact_gradient_basis_eta_jac_basis_contact_jac
                        .view()
                        .into_iter()
                        .flat_map(move |(row_idx, row)| {
                            // (E)
                            row.into_iter().map(move |(col_idx, block)| {
                                (row_idx, col_idx, *block.into_arrays().as_tensor())
                            })
                        }), // .inspect(|(i, j, m)| {
                            //     if *i == 4 && *j == 4 {
                            //         log::trace!("E:({},{}): {:?}", i, j, (*m).into_data());
                            //     }
                            // }),
                )
                .filter(move |&(row, col, _)| row < max_index && col < max_index)
                .flat_map(move |(row, col, block)| {
                    (0..3).flat_map(move |r| {
                        (0..3).map(move |c| (3 * row + r, 3 * col + c, block[r][c]))
                    })
                }),
        ))
    }

    #[inline]
    fn distance_jacobian_blocks_iter<'a>(
        &'a self,
    ) -> impl Iterator<Item = (usize, usize, [T; 3])> + 'a {
        distance_jacobian_blocks_iter_fn(
            &self.contact_state.point_constraint,
            &self.implicit_surface_vertex_indices,
            &self.collider_vertex_indices,
        )
    }

    #[inline]
    pub(crate) fn constraint_hessian_size(&self, max_index: usize) -> usize {
        self.constraint_hessian_indices_iter(max_index).count()
    }

    /// Construct a transpose of the constraint gradient (constraint Jacobian).
    ///
    /// The structure is preserved but the inner blocks are transposed.
    pub(crate) fn constraint_gradient_column_major_transpose<'a>(
        matrix: Tensor![T; &'a S S 3 1],
    ) -> Tensor![T; &'a S S 1 3] {
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
            if self.contact_state.point_constraint.object_is_fixed() {
                None
            } else {
                let surf = &self.contact_state.point_constraint.implicit_surface;
                surf.surface_hessian_product_indexed_blocks_iter(
                    self.contact_state
                        .point_constraint
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
                (row, col, mtx)
            }),
        )
    }

    // Assumes surface and contact points are upto date.
    pub(crate) fn object_collider_distance_potential_hessian_indexed_blocks_iter<'a>(
        &'a self,
        lambda: &'a [T],
    ) -> Box<dyn Iterator<Item = (usize, usize, [[T; 3]; 3])> + 'a> {
        Box::new(
            if self.contact_state.point_constraint.object_is_fixed() {
                None
            } else {
                let surf = &self.contact_state.point_constraint.implicit_surface;
                surf.sample_query_hessian_product_indexed_blocks_iter(
                    self.contact_state
                        .point_constraint
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
                let row = self.collider_vertex_indices[row];
                let col = self.implicit_surface_vertex_indices[col];
                // if col > row {
                //     (col, row, Matrix3::from(mtx).transpose().into_data())
                // } else {
                //     (row, col, mtx)
                // }
                (row, col, mtx)
            }),
        )
    }

    // Assumes surface and contact points are upto date.
    #[inline]
    pub(crate) fn collider_distance_potential_hessian_indexed_blocks_iter<'a>(
        &'a self,
        lambda: &'a [T],
    ) -> impl Iterator<Item = (usize, [[T; 3]; 3])> + 'a {
        if self.contact_state.point_constraint.collider_is_fixed() {
            None
        } else {
            let surf = &self.contact_state.point_constraint.implicit_surface;
            Some(
                surf.query_hessian_product_indexed_blocks_iter(
                    self.contact_state
                        .point_constraint
                        .collider_vertex_positions
                        .view()
                        .into(),
                    lambda.iter().cloned(),
                )
                .map(move |(idx, mtx)| (self.collider_vertex_indices[idx], mtx)),
            )
        }
        .into_iter()
        .flatten()
    }

    #[inline]
    pub(crate) fn object_distance_potential_hessian_block_indices_iter(
        &self,
    ) -> impl Iterator<Item = (usize, usize)> + '_ {
        if self.contact_state.point_constraint.object_is_fixed() {
            None
        } else {
            self.contact_state
                .point_constraint
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

    pub(crate) fn object_collider_distance_potential_hessian_block_indices_iter(
        &self,
    ) -> impl Iterator<Item = (usize, usize)> + '_ {
        if self.contact_state.point_constraint.object_is_fixed() {
            None
        } else {
            self.contact_state
                .point_constraint
                .implicit_surface
                .sample_query_hessian_product_block_indices_iter()
                .ok()
        }
        .into_iter()
        .flatten()
        .map(move |(row, col)| {
            let row = self.collider_vertex_indices[row];
            let col = self.implicit_surface_vertex_indices[col];
            // if col > row {
            //     (col, row)
            // } else {
            (row, col)
            // }
        })
    }

    pub(crate) fn collider_distance_potential_hessian_block_indices_iter(
        &self,
    ) -> impl Iterator<Item = usize> + '_ {
        if self.contact_state.point_constraint.collider_is_fixed() {
            None
        } else {
            Some(
                self.contact_state
                    .point_constraint
                    .implicit_surface
                    .query_hessian_product_block_indices_iter()
                    .map(move |idx| self.collider_vertex_indices[idx]),
            )
        }
        .into_iter()
        .flatten()
    }

    pub(crate) fn constraint_hessian_indices_iter(
        &self,
        max_index: usize,
    ) -> impl Iterator<Item = MatrixElementIndex> + '_ {
        let obj_indices_iter = self.object_distance_potential_hessian_block_indices_iter();
        let obj_col_indices_iter =
            self.object_collider_distance_potential_hessian_block_indices_iter();
        let coll_indices_iter = self
            .collider_distance_potential_hessian_block_indices_iter()
            .map(|idx| (idx, idx));
        let hessian = obj_indices_iter
            .chain(obj_col_indices_iter)
            .flat_map(|(row, col)| {
                // Fill upper triangular portion.
                std::iter::once((row, col)).chain(if row == col {
                    Either::Left(std::iter::empty())
                } else {
                    Either::Right(std::iter::once((col, row)))
                })
            })
            .chain(coll_indices_iter);

        let MappedDistanceGradient { matrix: g, .. } = self
            .contact_state
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
                    .filter(move |(col_idx, _)| *col_idx < max_index)
                    .flat_map(move |(col_idx, rhs_col)| {
                        // Produce an iterator for the row-col block inner product.
                        MulExpr::with_op(lhs_row.into_expr(), rhs_col.into_expr(), Multiplication)
                            .map(move |_| (row_idx, col_idx))
                    })
            });

        hessian
            .filter(move |(row, col)| *row < max_index && *col < max_index)
            .chain(gj_iter)
            .flat_map(move |(row, col)| {
                //if row == col {
                //    // Only lower triangular part
                //    Either::Left(
                //        (0..3).flat_map(move |r| {
                //            (0..=r).map(move |c| (3 * row + r, 3 * col + c).into())
                //        }),
                //    )
                //} else {
                // Entire matrix
                //Either::Right(
                (0..3).flat_map(move |r| (0..3).map(move |c| (3 * row + r, 3 * col + c).into()))
                //)
                //}
            })
    }

    pub(crate) fn constraint_hessian_indexed_values_iter<'a>(
        &'a self,
        max_index: usize,
    ) -> impl Iterator<Item = (MatrixElementIndex, T)> + 'a {
        let lambda = self.contact_state.lambda.as_slice();
        let dist = self.contact_state.distance_potential.as_slice();

        let hessian = self
            .object_distance_potential_hessian_indexed_blocks_iter(lambda)
            .chain(self.object_collider_distance_potential_hessian_indexed_blocks_iter(lambda))
            .flat_map(|(row, col, mtx)| {
                // Fill upper triangular portion.
                std::iter::once((row, col, mtx)).chain(if row == col {
                    Either::Left(std::iter::empty())
                } else {
                    Either::Right(std::iter::once((
                        col,
                        row,
                        Matrix3::from(mtx).transpose().into_data(),
                    )))
                })
            })
            .chain(
                self.collider_distance_potential_hessian_indexed_blocks_iter(lambda)
                    .map(|(idx, mtx)| (idx, idx, mtx)),
            );

        let delta = self.params.tolerance;
        let kappa = T::from(self.params.stiffness).unwrap();

        let MappedDistanceGradient { matrix: g, .. } = self
            .contact_state
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
                    .filter(move |(col_idx, _)| *col_idx < max_index)
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
                // if !Matrix3::from(mtx).is_zero() {
                //     eprintln!("c({row},{col}): {mtx:?}");
                // }

                //if row == col {
                //    Either::Left((0..3).flat_map(move |r| {
                //        (0..=r).map(move |c| ((3 * row + r, 3 * col + c).into(), mtx[r][c]))
                //    }))
                //} else {
                //Either::Right(
                (0..3).flat_map(move |r| {
                    (0..3).map(move |c| ((3 * row + r, 3 * col + c).into(), mtx[r][c]))
                })
                // )
                //}
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
    use crate::attrib_defines::*;
    use crate::nl_fem::state::{ResidualState, State};
    use crate::nl_fem::{NonLinearProblem, SimParams, SolverBuilder};
    use crate::test_utils::{default_solid, static_nl_params, static_configs};
    use crate::{Elasticity, MaterialIdType};
    use ad::F1;
    use approx::assert_relative_eq;
    use flatk::zip;
    use geo::algo::Merge;
    use geo::mesh::builder::{AxisPlaneOrientation, GridBuilder, PlatonicSolidBuilder};
    use geo::mesh::Mesh;
    use geo::ops::{Rotate, Scale, Translate};
    use geo::topology::{CellIndex, FaceIndex, NumCells, NumFaces};

    #[test]
    fn contact_penalty_derivative() {
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

    #[test]
    fn eta_potential_derivative() {
        let eta_jac_tester = |friction_profile: FrictionProfile, x: [f64; 2]| {
            let factor = 1.0;
            let friction_params = FrictionParams {
                dynamic_friction: 1.0,
                static_friction: 1.0,
                epsilon: 0.1,
                ..Default::default()
            };
            let x = x.into_tensor();
            let df_dv = friction_profile.profile(x, factor, &friction_params);
            let mut v = x.mapd(|x| F1::cst(x));
            v[0] = F1::var(v[0]);
            let f = friction_profile.potential(v, factor.into(), &friction_params);
            let df_dv0_ad = f.deriv();
            assert_relative_eq!(df_dv0_ad, df_dv[0], max_relative = 1e-8);
            v[0] = F1::cst(v[0]);
            v[1] = F1::var(v[1]);
            let f = friction_profile.potential(v, factor.into(), &friction_params);
            let df_dv1_ad = f.deriv();
            assert_relative_eq!(df_dv1_ad, df_dv[1], max_relative = 1e-8);
        };

        let x = [-0.1, 0.9];
        eta_jac_tester(FrictionProfile::Stabilized, x);
        eta_jac_tester(FrictionProfile::Quadratic, x);

        // Near zero
        for i in 0..10 {
            let x = [0.0 + 0.0001 * i as f64, 0.0];
            eta_jac_tester(FrictionProfile::Stabilized, x);
            eta_jac_tester(FrictionProfile::Quadratic, x);
        }
    }

    #[test]
    fn eta_derivative() {
        let eta_jac_tester = |friction_profile: FrictionProfile| {
            let factor = 1.0;
            let friction_params = FrictionParams {
                dynamic_friction: 1.0,
                static_friction: 1.0,
                epsilon: 0.1,
                ..Default::default()
            };
            let x = [-0.1, 0.9].into_tensor();
            let df_dv = friction_profile.jacobian(x, factor, &friction_params);
            let mut v = x.mapd(|x| F1::cst(x));
            v[0] = F1::var(v[0]);
            let f = friction_profile.profile(v, factor.into(), &friction_params);
            let df_dv0_ad = f.mapd(|x| x.deriv());
            assert_relative_eq!(df_dv0_ad[0], df_dv[0][0]);
            assert_relative_eq!(df_dv0_ad[1], df_dv[1][0]);
            v[0] = F1::cst(v[0]);
            v[1] = F1::var(v[1]);
            let f = friction_profile.profile(v, factor.into(), &friction_params);
            let df_dv1_ad = f.mapd(|x| x.deriv());
            assert_relative_eq!(df_dv1_ad[0], df_dv[0][1]);
            assert_relative_eq!(df_dv1_ad[1], df_dv[1][1]);
        };

        eta_jac_tester(FrictionProfile::Stabilized);
        eta_jac_tester(FrictionProfile::Quadratic);
    }

    #[test]
    fn eta_derivative_near_zero() {
        let factor = 1.0;
        let friction_params = FrictionParams {
            dynamic_friction: 1.0,
            static_friction: 1.0,
            epsilon: 0.1,
            ..Default::default()
        };
        let eta_jac_tester = |fp: FrictionProfile| {
            for i in 0..10 {
                let x = [0.0 + 0.0001 * i as f64, 0.0].into_tensor();
                let df_dv = fp.jacobian(x, factor, &friction_params);
                let mut v = x.mapd(|x| F1::cst(x));
                v[0] = F1::var(v[0]);
                let f = fp.profile(v, factor.into(), &friction_params);
                let df_dv0_ad = f.mapd(|x| x.deriv());
                assert_relative_eq!(df_dv0_ad[0], df_dv[0][0]);
                assert_relative_eq!(df_dv0_ad[1], df_dv[1][0]);
                v[0] = F1::cst(v[0]);
                v[1] = F1::var(v[1]);
                let f = fp.profile(v, factor.into(), &friction_params);
                let df_dv1_ad = f.mapd(|x| x.deriv());
                assert_relative_eq!(df_dv1_ad[0], df_dv[0][1]);
                assert_relative_eq!(df_dv1_ad[1], df_dv[1][1]);
            }
        };
        eta_jac_tester(FrictionProfile::Stabilized);
        eta_jac_tester(FrictionProfile::Quadratic);
    }

    #[test]
    fn sliding_profile() {
        let h = 0.0000001;
        let q_near_zero = h * quadratic_sliding_profile(h, 0.001);
        assert!(q_near_zero < 0.5);
        let s_near_zero = h * stabilized_sliding_profile(h, 0.001);
        assert!(s_near_zero < 0.5);
        let h = 0.1;
        let q_large = h * quadratic_sliding_profile(h, 0.001);
        assert!(q_large > 0.9);
        let s_large = h * stabilized_sliding_profile(h, 0.001);
        assert!(s_large > 0.9);
    }

    // Validate that the friction Jacobian is correct.
    //
    // Note that the general purpose problem level jacobian checker may not catch problems with the
    // friction jacobian if it is not dominating. This test is designed to exhibit deliberately large
    // friction forces.
    #[test]
    fn friction_jacobian() -> Result<(), Error> {
        friction_jacobian_tester(false)?;
        friction_jacobian_tester(true)?;
        Ok(())
    }

    fn friction_jacobian_tester(lagged: bool) -> Result<(), Error> {
        use geo::mesh::VertexPositions;

        crate::test_utils::init_logger();
        // Using the sliding tet on implicit test we will check the friction derivative directly.
        let material = default_solid().with_elasticity(Elasticity::from_young_poisson(1e5, 0.4));

        let mut tetmesh = PlatonicSolidBuilder::new().build_tetrahedron();
        tetmesh.translate([0.0, 1.0 / 3.0, 0.0]);
        tetmesh.rotate([0.0, 0.0, 1.0], std::f64::consts::PI / 16.0);
        //geo::io::save_tetmesh(&tetmesh, "./out/tetmesh.vtk")?;

        let mut surface = GridBuilder {
            rows: 1,
            cols: 1,
            orientation: AxisPlaneOrientation::ZX,
        }
        .build();

        surface.vertex_positions_mut()[0][1] += 0.1;
        surface.rotate([0.0, 0.0, 1.0], std::f64::consts::PI / 16.0);
        surface.uniform_scale(2.0);
        //geo::io::save_polymesh(&surface, "./out/polymesh.vtk")?;

        tetmesh.insert_attrib_data::<MaterialIdType, CellIndex>(
            MATERIAL_ID_ATTRIB,
            vec![1; tetmesh.num_cells()],
        )?;
        tetmesh.insert_attrib_data::<ObjectIdType, CellIndex>(
            OBJECT_ID_ATTRIB,
            vec![1; tetmesh.num_cells()],
        )?;
        tetmesh.insert_attrib_data::<VelType, VertexIndex>(
            VELOCITY_ATTRIB,
            vec![[-0.98, -0.296, 0.0]; 4],
        )?;
        surface.insert_attrib_data::<ObjectIdType, FaceIndex>(
            OBJECT_ID_ATTRIB,
            vec![0; surface.num_faces()],
        )?;
        surface.insert_attrib_data::<FixedIntType, VertexIndex>(
            FIXED_ATTRIB,
            vec![1; surface.num_vertices()],
        )?;

        for config_idx in static_configs() {
            let params = SimParams {
                max_iterations: 50,
                gravity: [0.0f32, -9.81, 0.0],
                time_step: Some(0.01),
                derivative_test: 3,
                residual_tolerance: 1e-8.into(),
                velocity_tolerance: 1e-5.into(),
                ..static_nl_params(config_idx)
            };

            let surface = surface.clone();
            let tetmesh = tetmesh.clone();

            let mut mesh = Mesh::from(tetmesh);
            mesh.merge(Mesh::from(TriMesh::from(surface)));

            let fc_params = FrictionalContactParams {
                kernel: implicits::KernelType::Approximate {
                    radius_multiplier: 1.5,
                    tolerance: 1e-5,
                },
                tolerance: 0.0001,
                friction_params: FrictionParams {
                    dynamic_friction: 0.18,
                    lagged,
                    epsilon: 0.0001,
                    ..Default::default()
                },
                ..Default::default()
            };

            //let mesh: Mesh<f64> = geo::io::load_mesh("./out/problem.vtk")?;

            let mut solver = SolverBuilder::new(params.clone())
                .set_mesh(mesh)
                .set_materials(vec![material.with_id(1).into()])
                .add_frictional_contact(fc_params, (0, 1))
                .build::<f64>()?;

            //solver.step()?;
            //geo::io::save_mesh(&solver.mesh(), "./out/result.vtk");

            let problem = solver.problem_mut();

            problem.update_constraint_set(true, true);

            // Preliminary Jacobian check.
            // This probably will not catch errors in the friction Jacobian.
            //problem.check_jacobian(true)?;

            // Update the current vertex data using the current dof state.
            problem.update_cur_vertices();

            // Prepare variables
            let fc_ad = problem.frictional_contact_constraints_ad.clone();
            let fc = problem.frictional_contact_constraints.clone();

            let n = problem.num_variables();
            let dt = problem.time_step();

            let state = &mut *problem.state.borrow_mut();
            let dq = state.vtx.next.vel.storage().to_vec();
            let cur_pos_ad = Chunked3::from_flat(
                state
                    .vtx
                    .cur
                    .pos
                    .storage()
                    .iter()
                    .map(|&x| ad::F1::cst(x))
                    .collect::<Vec<_>>(),
            );

            State::be_step(state.step_state(&dq), dt);
            state.update_vertices(&dq);

            let State { vtx, .. } = state;
            let num_vertices = vtx.next.vel.len();

            // Column output row major jacobians.
            let mut jac_ad = vec![vec![0.0; n]; n];
            let mut jac = vec![vec![0.0; n]; n];

            if lagged {
                // Prepare previous state for lagged friction
                for fc in fc_ad.iter() {
                    let mut fc = fc.constraint.borrow_mut();
                    fc.update_state(cur_pos_ad.view());
                    fc.update_distance_potential();
                    fc.update_constraint_gradient();
                    fc.update_multipliers();
                    fc.update_sliding_basis(true, false, num_vertices);
                    fc.advance_state();
                }
                for fc in fc.iter() {
                    let mut fc = fc.constraint.borrow_mut();
                    fc.update_state(vtx.cur.pos.view());
                    fc.update_distance_potential();
                    fc.update_constraint_gradient();
                    fc.update_multipliers();
                    fc.update_sliding_basis(true, false, num_vertices);
                    fc.advance_state();
                }
            }

            // Compute jacobian
            {
                let ResidualState { next, .. } = vtx.residual_state().into_storage();

                // Update constraint state.
                for fc in fc.iter() {
                    let mut fc = fc.constraint.borrow_mut();
                    fc.update_state(Chunked3::from_flat(next.pos));
                    fc.update_distance_potential();
                    fc.update_constraint_gradient();
                    fc.update_multipliers();
                    fc.update_sliding_basis(true, false, num_vertices);
                }

                for fc in fc.iter() {
                    let mut constraint = fc.constraint.borrow_mut();
                    // Compute friction hessian second term (multipliers held constant)
                    let jac_iter = constraint
                        .friction_jacobian_indexed_value_iter(
                            Chunked3::from_flat(next.vel.view()),
                            dt.into(),
                            n / 3,
                        )
                        .unwrap();
                    jac_iter.for_each(|(row, col, value)| {
                        jac[row][col] += value;
                    });
                }
            }

            let ResidualState { cur, next, r } = vtx.residual_state_ad().into_storage();
            let mut vel = next.vel.to_vec(); // Need to change autodiff variable.
            let cur_pos = cur.pos.to_vec();
            let mut next_pos = next.pos.to_vec();
            for (next_pos, &cur_pos, &vel) in
                zip!(next_pos.iter_mut(), cur_pos.iter(), next.vel.iter())
            {
                *next_pos = cur_pos + vel * dt;
            }

            let mut success = true;
            for col in 0..n {
                //eprintln!("DERIVATIVE WRT {}", col);
                vel[col] = F1::var(vel[col]);
                // Update pos with backward euler.
                for (next_pos, &cur_pos, &vel) in
                    zip!(next_pos.iter_mut(), cur_pos.iter(), vel.iter())
                {
                    *next_pos = cur_pos + vel * dt;
                }
                r.iter_mut().for_each(|r| *r = F1::zero());
                for fc in fc_ad.iter() {
                    let mut fc = fc.constraint.borrow_mut();
                    fc.update_state(Chunked3::from_flat(&next_pos));
                    fc.update_distance_potential();
                    fc.update_constraint_gradient();
                    fc.update_multipliers();
                    fc.update_sliding_basis(true, false, num_vertices);
                    fc.subtract_friction_force(
                        Chunked3::from_flat(r),
                        Chunked3::from_flat(&vel),
                        false,
                    );
                }

                for row in 0..n {
                    let res = approx::relative_eq!(
                        jac[row][col],
                        r[row].deriv(),
                        max_relative = 1e-4,
                        epsilon = 1e-5
                    );
                    jac_ad[row][col] = r[row].deriv();
                    if !res {
                        success = false;
                        log::debug!(
                            "({}, {}): {} vs. {}",
                            row,
                            col,
                            jac[row][col],
                            r[row].deriv()
                        );
                    }
                }
                vel[col] = F1::cst(vel[col]);
            }

            // Print dense hessian if its small
            eprintln!("Actual:");
            for row in 0..n {
                for col in 0..n {
                    eprint!("{:10.2e} ", jac[row][col]);
                }
                eprintln!();
            }

            eprintln!("Expected:");
            for row in 0..n {
                for col in 0..n {
                    eprint!("{:10.2e} ", jac_ad[row][col]);
                }
                eprintln!();
            }

            if success {
                eprintln!("No errors during friction Jacobian check for config {config_idx}.");
            } else {
                return Err(crate::Error::DerivativeCheckFailure);
            }
        }
        Ok(())
    }
}
