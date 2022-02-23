use super::point_contact::PointContactConstraint;
use crate::constraints::{ContactConstraint, ContactSurface};
use crate::matrix::MatrixElementIndex;
use crate::{
    ContactBasis, ContactGradient, ContactGradientView, ContactJacobian, Error, FrictionParams,
    FrictionWorkspace, Index, Real, TriMesh, TripletContactJacobian, ORIGINAL_VERTEX_INDEX_ATTRIB,
};
use autodiff as ad;
use flatk::{
    Chunked, Chunked1, Chunked3, CloneWithStorage, Isolate, Offsets, Select, Set, Sparse, Storage,
    StorageMut, SubsetView, UniChunked, View, U1, U3,
};
use geo::attrib::Attrib;
use geo::index::CheckedIndex;
use geo::mesh::VertexMesh;
use geo::topology::{NumVertices, VertexIndex};
use implicits::{KernelType, QueryTopo};
use lazycell::LazyCell;
use num_traits::Zero;
use rayon::iter::Either;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::cell::RefCell;
use std::ops::AddAssign;
use std::time::{Duration, Instant};
use tensr::{
    AsMutTensor, AsTensor, BlockDiagonalMatrix3, CwiseBinExprImpl, Expr, ExprMut, IndexedExpr,
    IntoData, IntoExpr, IntoTensor, Matrix, Matrix2, Matrix3, MulExpr, Multiplication, Scalar,
    Tensor, Vector2, Vector3,
};

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
        debug_assert!(triplet_jac
            .block_indices
            .windows(2)
            .all(|w| w[0].0 <= w[1].0 || w[0].1 <= w[1].1));
        let blocks =
            Chunked3::from_flat(triplet_jac.blocks.data.view().into_arrays()).into_arrays();

        let mut mapping = vec![Index::INVALID; blocks.len()];

        let triplet_iter = triplet_jac
            .block_indices
            .iter()
            .zip(blocks.iter())
            .map(|((row, col), block)| (*row, *col, *block));

        let uncompressed = tensr::SSBlockMatrix3::from_block_triplets_iter_uncompressed(
            triplet_iter,
            triplet_jac.num_rows,
            triplet_jac.num_cols,
        );

        // Compress the CSR matrix.
        let matrix = uncompressed.pruned_with(
            |_, _, _| true,
            |src, dst| {
                mapping[src] = Index::new(dst);
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

    fn mul(
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

    fn transpose_mul(
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
            v[i] = [-v0, -v1, -v2];
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
fn stabilized_sliding_potential<T: Real>(x: T, epsilon: T) -> T {
    //   x - εlog(ε + x) + εlog(2ε)
    // = x + εlog(2ε / (ε + x))
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

/// The sliding potential.
///
/// This is the antiderivative of the function eta in the paper.
#[inline]
pub fn eta_int<T: Real>(v: Vector2<T>, factor: T, epsilon: T, is: impl FnOnce(T, T) -> T) -> T {
    factor * is(v.norm(), epsilon)
}

/// The full sliding profile including 1D direction.
///
/// This is the function eta in the paper.
#[inline]
pub fn eta<T: Real>(v: Vector2<T>, factor: T, epsilon: T, s: impl FnOnce(T, T) -> T) -> Vector2<T> {
    // This is similar to function s but with the norm of v multiplied through to avoid
    // degeneracies.
    // let s = |x| stabilized_sliding_profile(x, epsilon);
    v * (factor * s(v.norm(), epsilon))
}

/// Jacobian of the full directional 1D sliding profile
#[inline]
pub fn eta_jac<T: Real>(
    v: Vector2<T>,
    factor: T,
    epsilon: T,
    s: impl FnOnce(T, T) -> T,
    ds: impl FnOnce(T, T) -> T,
) -> Matrix2<T> {
    // let s = |x| stabilized_sliding_profile(x, epsilon);
    // let ds = |x| stabilized_sliding_profile_derivative(x, epsilon);
    let s = |x| s(x, epsilon);
    let ds = |x| ds(x, epsilon);
    let norm_v = v.norm();
    let mut out = Matrix2::identity() * (s(norm_v) * factor);
    if norm_v > T::zero() {
        out += v * (ds(norm_v) / norm_v) * (v.transpose() * factor);
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
    pub fn potential<T: Real>(self, v: Vector2<T>, factor: T, epsilon: T) -> T {
        match self {
            FrictionProfile::Stabilized => {
                eta_int(v, factor, epsilon, stabilized_sliding_potential::<T>)
            }
            FrictionProfile::Quadratic => {
                eta_int(v, factor, epsilon, quadratic_sliding_potential::<T>)
            }
        }
    }

    /// The full sliding profile including 1D direction.
    ///
    /// This is the function eta in the paper.
    #[inline]
    pub fn profile<T: Real>(self, v: Vector2<T>, factor: T, epsilon: T) -> Vector2<T> {
        match self {
            FrictionProfile::Stabilized => eta(v, factor, epsilon, stabilized_sliding_profile::<T>),
            FrictionProfile::Quadratic => eta(v, factor, epsilon, quadratic_sliding_profile::<T>),
        }
    }

    /// Jacobian of the full directional 1D sliding profile
    #[inline]
    pub fn jacobian<T: Real>(self, v: Vector2<T>, factor: T, epsilon: T) -> Matrix2<T> {
        match self {
            FrictionProfile::Stabilized => eta_jac(
                v,
                factor,
                epsilon,
                stabilized_sliding_profile::<T>,
                stabilized_sliding_profile_derivative::<T>,
            ),
            FrictionProfile::Quadratic => eta_jac(
                v,
                factor,
                epsilon,
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
    lambda: Vec<T>,
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
            lambda: Vec::new(),
            contact_gradient_basis_eta_jac_basis_contact_jac: mtx.clone(),
            contact_gradient_basis_eta_jac_basis_jac_contact_jac: mtx.clone(),
            contact_gradient_basis_eta_jac_basis_jac: mtx.clone(),
            contact_gradient_jac_basis: mtx,
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Default)]
pub struct ContactVelocityTimings {
    pub prep: Duration,
    pub contact_jac: Duration,
    pub velocity: Duration,
}

impl ContactVelocityTimings {
    pub fn clear(&mut self) {
        self.prep = Duration::new(0, 0);
        self.contact_jac = Duration::new(0, 0);
        self.velocity = Duration::new(0, 0);
    }
}

impl AddAssign<ContactVelocityTimings> for ContactVelocityTimings {
    fn add_assign(&mut self, rhs: ContactVelocityTimings) {
        self.prep += rhs.prep;
        self.contact_jac += rhs.contact_jac;
        self.velocity += rhs.velocity;
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Default)]
pub struct FrictionTimings {
    pub total: Duration,
    pub contact_velocity: ContactVelocityTimings,
    pub jac_basis_mul: Duration,
}

impl AddAssign<FrictionTimings> for FrictionTimings {
    fn add_assign(&mut self, rhs: FrictionTimings) {
        self.total += rhs.total;
        self.contact_velocity += rhs.contact_velocity;
        self.jac_basis_mul += rhs.jac_basis_mul;
    }
}

impl FrictionTimings {
    pub fn clear(&mut self) {
        self.total = Duration::new(0, 0);
        self.jac_basis_mul = Duration::new(0, 0);
        self.contact_velocity.clear();
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
        self.e = Duration::new(0, 0);
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

    pub eta: FrictionProfile,

    pub(crate) lambda: Vec<T>,
    pub distance_potential: Vec<T>,
    pub distance_potential_alt: Vec<T>,
    force_workspace: std::cell::RefCell<Vec<Vec<T>>>,

    contact_jacobian: Option<MappedContactJacobian<T>>,
    contact_gradient: Option<MappedContactGradient<T>>,
    distance_gradient: LazyCell<MappedDistanceGradient<T>>,

    friction_jacobian_workspace: FrictionJacobianWorkspace<T>,

    /// Collider vertex indices for each active constraint.
    constrained_collider_vertices: Vec<usize>,
    /// Constraint indices for each collider vertex.
    ///
    /// Unconstrained vertices are set to `INVALID`.
    collider_vertex_constraints: Vec<Index>,

    pub timings: std::cell::RefCell<FrictionTimings>,
    pub jac_timings: std::cell::RefCell<FrictionJacobianTimings>,
}

impl<T: Real> PenaltyPointContactConstraint<T> {
    pub fn clone_cast<S: Real>(&self) -> PenaltyPointContactConstraint<S> {
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
        // let mut contact_gradient = LazyCell::new();
        // if self.contact_gradient.filled() {
        //     contact_gradient.replace(clone_cast_ssblock_mtx::<T, S>(
        //         self.contact_gradient.borrow().unwrap(),
        //     ));
        // }
        let mut contact_gradient = None;
        if let Some(self_contact_gradient) = self.contact_gradient.as_ref() {
            contact_gradient.replace(MappedContactGradient {
                mapping: self_contact_gradient.mapping.clone(),
                matrix: clone_cast_ssblock_mtx::<T, S>(&self_contact_gradient.matrix),
            });
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
            distance_potential_alt: self
                .distance_potential_alt
                .iter()
                .map(|&x| S::from(x).unwrap())
                .collect(),
            contact_jacobian,
            contact_gradient,
            friction_jacobian_workspace: FrictionJacobianWorkspace::default(),
            force_workspace: std::cell::RefCell::new(Vec::new()),
            constrained_collider_vertices: self.constrained_collider_vertices.clone(),
            collider_vertex_constraints: self.collider_vertex_constraints.clone(),
            eta: self.eta,
            timings: RefCell::new(FrictionTimings::default()),
            jac_timings: RefCell::new(FrictionJacobianTimings::default()),
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
            contact_jacobian: None,
            // contact_gradient: LazyCell::new(),
            contact_gradient: None,
            lambda: Vec::new(),
            distance_potential: Vec::new(),
            distance_potential_alt: Vec::new(),
            force_workspace: RefCell::new(Vec::new()),
            friction_jacobian_workspace: FrictionJacobianWorkspace::default(),
            constrained_collider_vertices: Vec::new(),
            collider_vertex_constraints: Vec::new(),
            eta: friction_params
                .map(|x| x.friction_profile)
                .unwrap_or(FrictionProfile::Stabilized),
            timings: RefCell::new(FrictionTimings::default()),
            jac_timings: RefCell::new(FrictionJacobianTimings::default()),
        };

        penalty_constraint.precompute_contact_jacobian(num_vertices);
        penalty_constraint.update_distance_potential();
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

    // Same as `update_collider_vertex_positions` but without knowledge about original vertex indices.
    pub fn update_surface_vertex_positions_cast<S: Real>(&mut self, x: Chunked3<&[S]>) -> usize {
        let x = SubsetView::from_unique_ordered_indices(&self.implicit_surface_vertex_indices, x);
        self.point_constraint.update_surface_with_mesh_pos_cast(x)
    }

    // Same as `update_collider_vertex_positions` but without knowledge about original vertex indices.
    pub fn update_collider_vertex_positions_cast<S: Real>(&mut self, x: Chunked3<&[S]>) {
        let x = SubsetView::from_unique_ordered_indices(&self.collider_vertex_indices, x);
        self.point_constraint
            .update_collider_vertex_positions_cast(x);
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

    pub fn max_step_violated(&self, vel: Chunked3<&[T]>, dt: f64) -> bool {
        if self.contact_jacobian.is_none() {
            // If contact_jacobian hasn't yet been initialized, this is impossible to answer, so assume everything is ok.
            return false;
        }
        let jac = self.contact_jacobian.as_ref().unwrap();
        let vc = jac.mul(
            vel,
            &self.constrained_collider_vertices,
            &self.implicit_surface_vertex_indices,
            &self.collider_vertex_indices,
        );
        let max_vel = vc
            .iter()
            .map(|v| v.as_tensor().norm())
            .max_by(|x, y| x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Less))
            .unwrap_or(T::zero())
            .to_f64()
            .unwrap();

        let radius = self.point_constraint.implicit_surface.radius();
        let cur_max_step = self.point_constraint.implicit_surface.max_step() + radius;
        max_vel * dt > cur_max_step
    }

    pub fn update_max_step(&mut self, vel: Chunked3<&[T]>, dt: f64) {
        if self.contact_jacobian.is_none() {
            // Contact jacobian not yet initialized, just skip this step.
            // No need to increase max_step since we don't even know the velocity.
            return;
        }
        // Compute maximum relative velocity.
        let jac = self.contact_jacobian.as_ref().unwrap();
        let vc = jac.mul(
            vel,
            &self.constrained_collider_vertices,
            &self.implicit_surface_vertex_indices,
            &self.collider_vertex_indices,
        );
        let max_vel = vc
            .iter()
            .map(|v| v.as_tensor().norm())
            .max_by(|x, y| x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Less))
            .unwrap_or(T::zero())
            .to_f64()
            .unwrap();

        let radius = self.point_constraint.implicit_surface.radius();
        // The 1.5 factor ensures that the velocity has room to grow without triggering a recomputation.
        let max_step = 0.0_f64.max(1.5 * (max_vel * dt) - radius);
        self.point_constraint
            .implicit_surface
            .update_max_step(T::from(max_step).unwrap());
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

    pub(crate) fn precompute_contact_jacobian<'a>(&mut self, num_vertices: usize) {
        let pc = &mut self.point_constraint;

        if pc.friction_workspace.is_none() {
            return;
        }

        self.constrained_collider_vertices = self.point_constraint.active_constraint_indices();

        self.collider_vertex_constraints =
            vec![Index::invalid(); self.point_constraint.collider_vertex_positions.len()];
        for (constraint_idx, &query_idx) in self.constrained_collider_vertices.iter().enumerate() {
            self.collider_vertex_constraints[query_idx] = constraint_idx.into();
        }

        let constrained_collider_vertices = &self.constrained_collider_vertices;
        let constrained_collider_vertex_positions = Select::new(
            constrained_collider_vertices.as_slice(),
            self.point_constraint.collider_vertex_positions.view(),
        );

        let jac_triplets = TripletContactJacobian::from_selection(
            &self.point_constraint.implicit_surface,
            constrained_collider_vertex_positions.view(),
        );

        self.contact_jacobian = Some(MappedContactJacobian::from_triplets(jac_triplets));

        let jac_triplets = TripletContactJacobian::from_selection_reindexed_full(
            &self.point_constraint.implicit_surface,
            constrained_collider_vertex_positions.view(),
            &self.collider_vertex_constraints,
            &self.implicit_surface_vertex_indices,
            &self.collider_vertex_indices,
            num_vertices,
        );

        self.contact_gradient = Some(MappedContactGradient::from_triplets(&jac_triplets));
    }

    pub(crate) fn update_contact_jacobian<'a>(
        contact_jacobian: &'a mut MappedContactJacobian<T>,
        surf: &QueryTopo<T>,
        query_points: Chunked3<&[T]>,
        constrained_collider_vertices: &'a [usize],
    ) {
        let constrained_collider_vertex_positions =
            Select::new(constrained_collider_vertices, query_points.view());

        let jac_triplets =
            TripletContactJacobian::from_selection(&surf, constrained_collider_vertex_positions);

        contact_jacobian.update_values(&jac_triplets);
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
            &surf,
            constrained_collider_vertex_positions,
            collider_vertex_constraints,
            implicit_surface_vertex_indices,
            collider_vertex_indices,
            num_vertices,
        );

        contact_gradient.update_values(&triplets);
    }

    /// Initializes the constraint gradient sparsity pattern.
    pub fn reset_distance_gradient<'a>(&mut self, num_vertices: usize) {
        let (indices, blocks): (Vec<_>, Vec<_>) = self
            .distance_jacobian_blocks_par_iter()
            .map(|(row, col, block)| (MatrixElementIndex { row: col, col: row }, block))
            .unzip();
        let num_constraints = self.constraint_size();
        self.distance_gradient
            .replace(Self::build_distance_gradient(
                indices.as_slice(),
                Chunked3::from_array_slice(blocks.as_slice()),
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
            self.precompute_contact_jacobian(x.len());
            self.update_distance_potential();
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
    }

    pub fn update_state_cast<S: Real>(&mut self, x: Chunked3<&[S]>) {
        let num_vertices_updated = self.update_surface_vertex_positions_cast(x);
        assert_eq!(
            num_vertices_updated,
            self.point_constraint
                .implicit_surface
                .surface_vertex_positions()
                .len()
        );
        self.update_collider_vertex_positions_cast(x);
    }

    /// Update the cached constraint gradient for efficient future derivative computations.
    ///
    /// This function assumes that the `constraint_gradient` field sparsity has already been
    /// initialized.
    pub fn update_constraint_gradient(&mut self) {
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

    pub(crate) fn assist_line_search_for_contact(
        &mut self,
        mut alpha: T,
        pos_cur: Chunked3<&[T]>,
        pos_next: Chunked3<&[T]>,
        delta: f32,
    ) -> T {
        let delta = T::from(delta).unwrap();
        self.update_state(pos_cur);
        self.update_distance_potential();
        {
            let d1 = &mut self.distance_potential_alt;
            d1.resize(self.distance_potential.len(), T::zero());
            d1.copy_from_slice(&self.distance_potential);
        }

        self.update_state(pos_next);
        self.update_distance_potential();
        let d1 = &self.distance_potential_alt;
        let d2 = &self.distance_potential;

        let alpha_min = T::from(1e-4).unwrap();

        for (&d1, &d2) in d1.iter().zip(d2.iter()) {
            if d1 > delta && d2 <= delta {
                alpha = num_traits::Float::min(
                    alpha,
                    num_traits::Float::max(
                        alpha_min,
                        (d1 - T::from(0.5).unwrap() * delta) / (d1 - d2),
                    ),
                );
            }
        }
        alpha
    }

    pub(crate) fn assist_line_search_for_friction(
        &mut self,
        mut alpha: T,
        p: Chunked3<&[T]>,
        vel: Chunked3<&[T]>,
        f1: Chunked3<&[T]>,
        f2: Chunked3<&[T]>,
        _pos_cur: Chunked3<&[T]>,
        pos_next: Chunked3<&[T]>,
        delta: f32,
    ) -> T {
        if self.point_constraint.friction_workspace.is_none() {
            return alpha;
        }
        let delta = T::from(delta).unwrap();
        // self.update_state(pos_cur);
        // self.update_distance_potential();
        // {
        //     let d1 = &mut self.distance_potential_alt;
        //     d1.resize(self.distance_potential.len(), T::zero());
        //     d1.copy_from_slice(&self.distance_potential);
        // }

        self.update_state(pos_next);
        self.update_distance_potential();

        let alpha_min = T::from(1e-4).unwrap();

        // let mut f1 = Chunked3::from_flat(vec![T::zero(); vel.len() * 3]);
        // let mut f2 = Chunked3::from_flat(vec![T::zero(); vel.len() * 3]);
        // self.subtract_friction_force(f1.view_mut(), vel, epsilon);
        // self.subtract_friction_force(f2.view_mut(), vel_next, epsilon);
        // let epsilon = T::from(epsilon).unwrap();

        let constrained_collider_vertices = self.constrained_collider_vertices.as_slice();

        let jac = self.contact_jacobian.as_ref().unwrap();

        let contact_basis = &self
            .point_constraint
            .friction_workspace
            .as_mut()
            .unwrap()
            .contact_basis;

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
        let pc = jac.mul(
            p,
            constrained_collider_vertices,
            &self.implicit_surface_vertex_indices,
            &self.collider_vertex_indices,
        );

        // let d1 = &self.distance_potential_alt;
        let d2 = &self.distance_potential;
        // eprintln!("alpha before: {alpha}");

        for (i, (&d2, (p, v))) in d2.iter().zip(pc.iter().zip(vc.iter())).enumerate() {
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
                    // eprintln!("p = {:?}; v = {:?}", [p1,p2], [v1,v2]);
                    let v = Vector2::from([v1, v2]);
                    let p = Vector2::from([p1, p2]);
                    let p_dot_v = p.dot(v);
                    let v_dot_v = v.dot(v);
                    if alpha * p_dot_v + v_dot_v <= T::zero() {
                        // eprintln!("direction switched");
                        alpha = num_traits::Float::min(
                            alpha,
                            num_traits::Float::max(alpha_min, -v_dot_v / p_dot_v),
                            // num_traits::Float::max(alpha_min, -p_dot_v/p.norm_squared()),
                        );
                        // } else {
                        //     eprintln!("direction not switched: {}", alpha * p_dot_v + v_dot_v);
                        //     eprintln!("new alpha would be {}", -v_dot_v/p_dot_v);
                    }
                    // } else {
                    //     eprintln!("dry run");
                    //     let [_v0, v1, v2] = contact_basis.to_contact_coordinates(*v, i);
                    //     let [_p0, p1, p2] = contact_basis.to_contact_coordinates(*p, i);
                    //     eprintln!("p = {:?}; v = {:?}", [p1,p2], [v1,v2]);
                    //     let v = Vector2::from([v1, v2]);
                    //     let p = Vector2::from([p1, p2]);
                    //     let p_dot_v = p.dot(v);
                    //     let v_dot_v = v.dot(v);
                    //     if alpha * p_dot_v + v_dot_v <= T::zero()  {
                    //         eprintln!("new alpha would be {}", -v_dot_v/p_dot_v);
                    //     } else {
                    //         eprintln!("not switched");
                    //     }
                }
            }
        }
        // eprintln!("alpha after: {alpha}");
        alpha
    }

    pub(crate) fn constraint_size(&self) -> usize {
        self.point_constraint.constraint_size()
    }

    pub fn subtract_constraint_force(&self, mut f: Chunked3<&mut [T]>) {
        self.distance_jacobian_blocks_iter()
            .for_each(|(row, col, j)| {
                *f[col].as_mut_tensor() -= *j.as_tensor() * self.lambda[row];
            });
    }

    pub fn contact_constraint(&self, delta: f32, kappa: f32) -> T {
        let dist = self.distance_potential.as_slice();
        let kappa = T::from(kappa).unwrap();
        dist.iter()
            .map(|&d| kappa * ContactPenalty::new(delta).b(d))
            .sum()
    }

    pub fn lagged_friction_potential(&mut self, v: Chunked3<&[T]>, epsilon: f32) -> T {
        // Compute friction potential.
        let lambda = &self.lambda;
        let constrained_collider_vertices = self.constrained_collider_vertices.as_slice();
        let pc = &mut self.point_constraint;

        if pc.friction_workspace.is_none() {
            return T::infinity();
        }

        // Construct contact (or "sliding") basis.
        let normals = pc.contact_normals();
        assert_eq!(normals.len(), lambda.len());

        // Contact Jacobian is defined for object vertices only. Contact Jacobian for collider vertices is trivial.
        let jac = self.contact_jacobian.as_ref().unwrap();

        assert_eq!(jac.matrix.len(), lambda.len());

        let FrictionWorkspace {
            contact_basis,
            // TODO: use collider_force in place of vc to reduce additional allocations.
            object_force: _,
            collider_force: _, // for active point contacts
            ..
        } = pc.friction_workspace.as_mut().unwrap();

        contact_basis.update_from_normals(normals.into());

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

        let eta = self.eta;
        let friction_potential = vc
            .iter()
            .zip(lambda.iter())
            .enumerate()
            .map(|(i, (vc, lambda))| {
                let [_v0, v1, v2] = contact_basis.to_contact_coordinates(*vc, i);
                //dbg!(i, [v0, v1, v2]);
                let vc_t = [v1, v2].into_tensor();
                //dbg!(&lambda);
                eta.potential(vc_t, *lambda, T::from(epsilon).unwrap())
            })
            .sum();

        let FrictionWorkspace { params, .. } = pc.friction_workspace.as_ref().unwrap();
        let mu = T::from(params.dynamic_friction).unwrap();
        mu * friction_potential
    }

    pub fn subtract_constraint_force_par(&self, mut f: Chunked3<&mut [T]>) {
        let Self {
            point_constraint,
            implicit_surface_vertex_indices,
            collider_vertex_indices,
            lambda,
            force_workspace,
            ..
        } = self;
        let fws = &mut *force_workspace.borrow_mut();
        let ncpus = num_cpus::get();
        fws.resize(ncpus, Vec::new());
        for fws_vec in fws.iter_mut() {
            fws_vec.resize(f.storage().len(), T::zero());
            fws_vec.fill(T::zero());
        }

        Self::distance_jacobian_blocks_par_chunks_fn(
            &point_constraint,
            &implicit_surface_vertex_indices,
            &collider_vertex_indices,
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
        epsilon: f32,
    ) {
        if let Some(f_f) = self.compute_friction_force(v, epsilon) {
            assert_eq!(f.len(), v.len());
            *&mut f.expr_mut() -= f_f.expr();
        }
    }

    // Compute `b(x,v) = H(T(x)'v)λ(x)`, friction force in contact space.
    //
    // This function uses current state. To get an upto date friction impulse call update_state.
    // This version of computing friction force works in the space of all active constraints
    // as opposed to all active contacts.
    pub fn compute_constraint_friction_force(
        &mut self,
        // Velocity
        v: Chunked3<&[T]>,
        epsilon: f32,
        recompute_jacobian: bool,
    ) -> Option<Chunked3<Vec<T>>> {
        let t_begin = Instant::now();
        let lambda = &self.lambda;
        let constrained_collider_vertices = &self.constrained_collider_vertices;
        let pc = &mut self.point_constraint;

        if pc.friction_workspace.is_none() {
            return None;
        }

        // Construct contact (or "sliding") basis.
        let normals = pc.contact_normals();
        assert_eq!(normals.len(), lambda.len());
        let t_prep = Instant::now();

        // Contact Jacobian is defined for object vertices only. Contact Jacobian for collider vertices is trivial.
        if recompute_jacobian {
            Self::update_contact_jacobian(
                self.contact_jacobian.as_mut().unwrap(),
                &pc.implicit_surface,
                pc.collider_vertex_positions.view(),
                constrained_collider_vertices,
            );
        }

        let jac = self.contact_jacobian.as_ref().unwrap();

        let t_contact_jac = Instant::now();

        assert_eq!(jac.num_rows(), lambda.len());

        let FrictionWorkspace {
            contact_basis,
            // TODO: use collider_force in place of vc to reduce additional allocations.
            object_force: _,
            collider_force: _, // for active point contacts
            ..
        } = pc.friction_workspace.as_mut().unwrap();

        contact_basis.update_from_normals(normals.into());

        // Compute relative velocity at the point of contact: `vc = J(x)v`
        //assert_eq!(jac.num_cols() + pc.collider_vertex_positions.len(), v.len());
        let mut vc = jac.mul(
            v,
            constrained_collider_vertices,
            &self.implicit_surface_vertex_indices,
            &self.collider_vertex_indices,
        );

        assert_eq!(lambda.len(), vc.len());

        // Compute sliding bases velocity product.

        // Compute `vc <- H(B'(x) vc) λ(x)`.
        let eta = self.eta;
        vc.view_mut()
            .into_iter()
            .zip(lambda.iter())
            .enumerate()
            .for_each(|(i, (vc, lambda))| {
                let [_v0, v1, v2] = contact_basis.to_contact_coordinates(*vc, i);
                //dbg!(i, [v0, v1, v2]);
                let vc_t = [v1, v2].into_tensor();
                //dbg!(&lambda);
                let vc_t_smoothed = eta
                    .profile(vc_t, *lambda, T::from(epsilon).unwrap())
                    .into_data();
                *vc = [T::zero(), vc_t_smoothed[0], vc_t_smoothed[1]];
            });
        let t_velocity = Instant::now();
        let timings = &mut *self.timings.borrow_mut();
        timings.contact_velocity.prep += t_prep - t_begin;
        timings.contact_velocity.contact_jac += t_contact_jac - t_prep;
        timings.contact_velocity.velocity += t_velocity - t_contact_jac;
        Some(vc)
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
        epsilon: f32,
    ) -> Option<Chunked3<Vec<T>>> {
        let t_begin = Instant::now();

        //dbg!(&v);

        // Computes `b(x,v) = H(T(x)'v)λ(x)`.
        let mut vc = self.compute_constraint_friction_force(v, epsilon, true)?;

        let t_vc = Instant::now();
        //dbg!(&vc);

        let jac = self.contact_jacobian.as_ref().unwrap();

        let FrictionWorkspace {
            contact_basis,
            params,
            ..
        } = self.point_constraint.friction_workspace.as_ref()?;
        let mu = T::from(params.dynamic_friction).unwrap();

        // Compute `vc <- -mu B(x) H(B'(x) vc) λ(x)`.
        vc.view_mut().into_iter().enumerate().for_each(|(i, v)| {
            let vc = v.as_tensor();
            *v = contact_basis.from_contact_coordinates(*vc * (-mu), i);
        });

        //dbg!(&vc);

        // Compute object force (compute `f = J'(x)vc`)
        let f = jac.transpose_mul(
            vc.view(),
            &self.constrained_collider_vertices,
            &self.implicit_surface_vertex_indices,
            &self.collider_vertex_indices,
            v.len() * 3,
        );
        let t_vc_jac = Instant::now();
        let timings = &mut *self.timings.borrow_mut();
        timings.total += t_vc_jac - t_begin;
        timings.jac_basis_mul += t_vc_jac - t_vc;
        //dbg!(&f.view().storage());
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

                let nml_jac_row = (nml_jac[tangent_axis]
                    - (n.transpose() * nml_jac)[0] * n[tangent_axis])
                    * norm_n_inv;

                let neg_t_proj = t * t.transpose() - Matrix3::identity();

                let tangent_jac = neg_t_proj
                    * (n * nml_jac_row.transpose() + nml_jac * n[tangent_axis])
                    * norm_t_inv;

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

    /// Builds a contact jacobian gradient product.
    ///
    /// Computes `grad_q (J(q) b)'` where `b` is the multiplier.
    ///
    pub fn build_contact_jacobian_gradient_product<'a>(
        surf: &implicits::QueryTopo<T>,
        query_points: &[[T; 3]],
        collider_vertex_constraints: &[Index],
        multipliers: &[[T; 3]],
        num_vertices: usize,
        num_contacts: usize,
    ) -> Result<Tensor![T; S S 3 3], Error> {
        let (indices, blocks): (Vec<_>, Vec<_>) = surf
            .contact_jacobian_jacobian_product_indexed_blocks_iter(
                query_points.view(),
                multipliers,
            )?
            .filter(|(_, col, _)| col < &num_vertices)
            .filter_map(|(row, col, mtx)| {
                collider_vertex_constraints[row]
                    .into_option()
                    .map(|contact_idx| (contact_idx, col, mtx))
            })
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
            num_contacts,
        )
        .into_data())
    }

    // Compute the product of friction matrix `mu J'B H(B'Jv)` with constraint Jacobian term `-k ddb dd/dq` and `dq/dv`.
    // constraint_jac is column major, but contact_gradient is row-major.
    // Output blocks are row-major.
    fn contact_constraint_jac_product_map<'a>(
        lambda: &'a [T],
        dist: &'a [T],
        pc: &'a PointContactConstraint<T>,
        // Relative velocity in contact space: `vc = J(x)v`
        mut vc: Chunked3<Vec<T>>,
        contact_gradient: ContactGradientView<'a, T>, // G
        constraint_jac: Tensor![T; &'a S S 1 3],      // dd/dq
        num_vertices: usize,
        delta: f32,
        kappa: f32,
        epsilon: f32,
        dqdv: T,
        eta: FrictionProfile,
    ) -> impl Iterator<Item = (usize, usize, Matrix3<T>)> + 'a {
        //assert_eq!(constraint_jac.len(), contact_jac.into_tensor().num_cols());

        let kappa = T::from(kappa).unwrap();
        let delta = T::from(delta).unwrap();

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
            let vc_t_smoothed = eta.profile(vc_t, mu, T::from(epsilon).unwrap()).into_data();
            //dbg!(eta(vc_t, T::one(), T::from(1e-5).unwrap()).into_data());
            *vc = contact_basis
                .from_contact_coordinates([T::zero(), vc_t_smoothed[0], vc_t_smoothed[1]], i)
        });

        let vc = vc.view();

        // Iterate over each column of the constraint Jacobian.
        let res: Vec<_> = contact_gradient
            .into_iter()
            .filter(move |(row_idx, _)| *row_idx < num_vertices)
            .flat_map(move |(row_idx, lhs_row)| {
                constraint_jac
                    .into_iter()
                    .filter(move |(col_idx, _)| *col_idx < num_vertices)
                    .flat_map(move |(col_idx, rhs_col)| {
                        let mut lhs_row_iter = lhs_row.into_iter().peekable();
                        let out = rhs_col.into_iter().filter_map(
                            move |(rhs_constraint_idx, rhs_block)| {
                                // Find the next matching lhs entry.
                                loop {
                                    let lhs_constraint_idx = if let Some(&(lhs_constraint_idx, _)) =
                                        lhs_row_iter.peek()
                                    {
                                        lhs_constraint_idx
                                    } else {
                                        break;
                                    };

                                    if rhs_constraint_idx < lhs_constraint_idx {
                                        return None; // Skips entry and advances rhs iterator.
                                    } else if lhs_constraint_idx < rhs_constraint_idx {
                                        // Skip lhs entry and continue in the loop.
                                        let _ = lhs_row_iter.next().unwrap();
                                    } else {
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
                                None
                            },
                        );

                        out
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
        epsilon: f32,
        dqdv: T,
        max_index: usize,
        recompute_contact_jacobian: bool,
    ) -> Option<impl Iterator<Item = (usize, usize, T)> + 'a> {
        let t_begin = Instant::now();
        let eta = self.eta;

        // TODO: Refactor this monstrosity of a function.
        self.update_constraint_gradient();

        let t_constraint_gradient = Instant::now();

        if self.point_constraint.friction_workspace.is_none() {
            return None;
        }

        let num_vertices = max_index;
        let num_constraints = self.constraint_size();

        self.friction_jacobian_workspace.lambda = self.lambda.clone();

        // Compute `c <- H(T(x)'v)λ(x)`
        let c = self.compute_constraint_friction_force(v, epsilon, recompute_contact_jacobian)?;
        assert_eq!(c.len(), num_constraints);

        let t_constraint_friction_force = Instant::now();

        // TODO: memoize the contact Jacobian:
        //       Store the Jacobian sparsity at the beginning of each step with all of potential contact points.
        //       At each residual or Jacobian function call, replace the sparse rows with those
        //       of just the active contact points creating an even more sparse Jacobian equivalent
        //       to this one but only a *view*. The values of the contact Jacobian can be updated
        //       also but this way we avoid doing any new allocations. This is left as a todo since
        //       we need to profile if this is actually worthwhile.
        // Contact Jacobian is defined for object vertices only. Contact Jacobian for collider vertices is trivial.
        let contact_jacobian = self.contact_jacobian.as_ref().unwrap();
        assert_eq!(contact_jacobian.num_rows(), num_constraints);
        let t_contact_jacobian = Instant::now();

        if recompute_contact_jacobian {
            Self::update_contact_gradient(
                self.contact_gradient.as_mut().unwrap(),
                &self.point_constraint.implicit_surface,
                self.point_constraint.collider_vertex_positions.view(),
                &self.constrained_collider_vertices,
                &self.collider_vertex_constraints,
                &self.implicit_surface_vertex_indices,
                &self.collider_vertex_indices,
                v.len(),
            );
        }
        let contact_gradient = &self.contact_gradient.as_ref().unwrap().matrix;
        // Self::contact_gradient(
        //     &mut self.contact_gradient,
        //     &self.point_constraint.implicit_surface,
        //     self.point_constraint.collider_vertex_positions.view(),
        //     &self.constrained_collider_vertices,
        //     &self.collider_vertex_constraints,
        //     &self.implicit_surface_vertex_indices,
        //     &self.collider_vertex_indices,
        //     v.len(),
        //     recompute_contact_jacobian,
        // );
        assert_eq!(
            contact_gradient.view().into_tensor().num_cols(),
            num_constraints
        );
        let t_contact_gradient = Instant::now();

        let mu = T::from(
            self.point_constraint
                .friction_workspace
                .as_ref()
                .unwrap()
                .params
                .dynamic_friction,
        )
        .unwrap();

        let MappedDistanceGradient { matrix: g, .. } = self
            .distance_gradient
            .borrow()
            .expect("Uninitialized constraint gradient.");
        let j_view = Self::constraint_gradient_column_major_transpose(g.view());
        // j_view is col-major, but tensors only understand row major, so here cols means rows.
        assert_eq!(j_view.view().into_tensor().num_cols(), num_constraints);

        let t_constraint_jacobian = Instant::now();

        let dqdv = T::from(dqdv).unwrap();

        let t_f_lambda_jac;
        let t_a;
        let t_b;
        let t_c;
        let t_d;
        let t_e;

        let (f_lambda_jac, jac_contact_gradient) = {
            {
                let Self {
                    friction_jacobian_workspace: FrictionJacobianWorkspace { bc, .. },
                    ref point_constraint,
                    ..
                } = self;

                // Compute `c(x,v) = H(T(x)'v)λ(x) * dq/dv`, friction force in contact space.
                *bc = c
                    .iter()
                    .enumerate()
                    .map(|(i, &v)| {
                        (point_constraint
                            .friction_workspace
                            .as_ref()
                            .unwrap()
                            .contact_basis
                            .from_contact_coordinates(v, i)
                            .into_tensor()
                            * dqdv)
                            .into_data()
                    })
                    .collect();
            }

            // Compute relative velocity at the point of contact: `vc = J(x)v`
            let vc = contact_jacobian.mul(
                v,
                &self.constrained_collider_vertices,
                &self.implicit_surface_vertex_indices,
                &self.collider_vertex_indices,
            );
            assert_eq!(vc.len(), num_constraints);

            // Derivative of lambda term.
            let f_lambda_jac = Self::contact_constraint_jac_product_map(
                &self.friction_jacobian_workspace.lambda,
                self.distance_potential.as_slice(),
                &self.point_constraint,
                vc.clone(),
                contact_gradient.view(),
                j_view,
                num_vertices,
                delta,
                kappa,
                epsilon,
                dqdv,
                self.eta,
            );

            t_f_lambda_jac = Instant::now();

            let surf = &self.point_constraint.implicit_surface;
            let query_points = &self.point_constraint.collider_vertex_positions;

            // The following should be multiplied by mu.

            // Compute (A)
            let implicit_idxs = self.implicit_surface_vertex_indices.clone();
            let jac_contact_gradient = surf
                .contact_hessian_product_indexed_blocks_iter(
                    query_points.view().into_arrays(),
                    self.friction_jacobian_workspace.bc.as_arrays(),
                )
                .ok()?
                // Reindex
                .map(move |(row, col, block)| {
                    (
                        implicit_idxs[row],
                        implicit_idxs[col],
                        Matrix3::from(block).transpose(),
                    )
                })
                .filter(move |&(row, col, _)| row < num_vertices && col < num_vertices);

            t_a = Instant::now();

            // Compute (B)

            let jac_basis = Self::build_contact_basis_gradient_product_from_selection(
                surf,
                &self
                    .point_constraint
                    .friction_workspace
                    .as_ref()
                    .unwrap()
                    .contact_basis,
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
            //     for (col_idx, block) in row.into_iter() {
            //         eprintln!("jb({},{}): {:?}", row_idx, col_idx, block);
            //     }
            // }

            self.friction_jacobian_workspace.contact_gradient_jac_basis =
                (contact_gradient.view().into_tensor() * jac_basis.view()).into_data(); // (B)

            t_b = Instant::now();

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

            let pc_ws = self.point_constraint.friction_workspace.as_ref().unwrap();
            let lambda = &self.friction_jacobian_workspace.lambda;

            // Construct the eta matrix dH
            let dh_blocks: Chunked3<Vec<_>> = (0..num_constraints)
                .flat_map(|constraint_idx| {
                    let [_, v1, v2] = pc_ws
                        .contact_basis
                        .to_contact_coordinates(vc[constraint_idx], constraint_idx);
                    let mtx = eta.jacobian([v1, v2].into(), T::one(), T::from(epsilon).unwrap())
                        * lambda[constraint_idx];
                    std::iter::once([T::zero(); 3])
                        .chain(std::iter::once([T::zero(), mtx[0][0], mtx[0][1]]))
                        .chain(std::iter::once([T::zero(), mtx[1][0], mtx[1][1]]))
                })
                .collect();
            let dh = BlockDiagonalMatrix3::new(Chunked3::from_flat(dh_blocks));

            // Compute (C)

            let mut jac_basis_t = Self::build_contact_basis_gradient_product_from_selection(
                surf,
                &pc_ws.contact_basis,
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

            let mut jac_contact_jacobian = Self::build_contact_jacobian_gradient_product(
                surf,
                query_points.view().into_arrays(),
                self.collider_vertex_constraints.view(),
                v0vec.view(),
                self.implicit_surface_vertex_indices.len(),
                num_constraints,
            )
            .ok()?
            .into_tensor()
            .transpose();
            jac_contact_jacobian.premultiply_block_diagonal_mtx(bt.view());
            jac_contact_jacobian.premultiply_block_diagonal_mtx(dh.view());
            jac_contact_jacobian.premultiply_block_diagonal_mtx(b.view());
            self.friction_jacobian_workspace
                .contact_gradient_basis_eta_jac_basis_jac_contact_jac =
                (contact_gradient.view().into_tensor() * jac_contact_jacobian.view()).into_data(); // (D)

            t_d = Instant::now();

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

            (f_lambda_jac, jac_contact_gradient)
        };

        let timings = &mut *self.jac_timings.borrow_mut();
        timings.constraint_gradient += t_constraint_gradient - t_begin;
        timings.constraint_friction_force += t_constraint_friction_force - t_constraint_gradient;
        timings.contact_jacobian += t_contact_jacobian - t_constraint_friction_force;
        timings.contact_gradient += t_contact_gradient - t_contact_jacobian;
        timings.constraint_jacobian += t_constraint_jacobian - t_contact_gradient;
        timings.f_lambda_jac += t_f_lambda_jac - t_constraint_jacobian;
        timings.a += t_a - t_f_lambda_jac;
        timings.b += t_b - t_a;
        timings.c += t_c - t_b;
        timings.d += t_d - t_c;
        timings.e += t_e - t_d;

        // Combine all matrices.

        Some(
            f_lambda_jac
                //.inspect(|(i,j,m)| log::trace!("dL: ({},{}):{:?}", i, j, (*m).into_data()))
                //.chain(
                //    jac_contact_gradient, //.inspect(|(_,_,m)| log::trace!("A: {:?}", m)) ,
                //) // (A)
                .chain(
                    self.friction_jacobian_workspace
                        .contact_gradient_jac_basis
                        .view()
                        .into_iter()
                        .flat_map(move |(row_idx, row)| {
                            // (B)
                            row.into_iter().map(move |(col_idx, block)| {
                                (
                                    row_idx,
                                    col_idx,
                                    *block.into_arrays().as_tensor() * (mu * dqdv),
                                )
                            })
                        }), //        .inspect(|(i,j,m)| log::trace!("B:({},{}): {:?}", i,j,(*m).into_data())) ,
                )
                .chain(
                    self.friction_jacobian_workspace
                        .contact_gradient_basis_eta_jac_basis_jac
                        .view()
                        .into_iter()
                        .flat_map(move |(row_idx, row)| {
                            // (C)
                            row.into_iter().map(move |(col_idx, block)| {
                                (
                                    row_idx,
                                    col_idx,
                                    *block.into_arrays().as_tensor() * (mu * dqdv),
                                )
                            })
                        }), //.inspect(|(i,j,m)| log::trace!("C:({},{}): {:?}", i,j,(*m).into_data())) ,
                )
                //.chain(
                //    self.friction_jacobian_workspace
                //        .contact_gradient_basis_eta_jac_basis_jac_contact_jac
                //        .view()
                //        .into_iter()
                //        .flat_map(move |(row_idx, row)| {
                //            // (D)
                //            row.into_iter().map(move |(col_idx, block)| {
                //                (
                //                    row_idx,
                //                    col_idx,
                //                    *block.into_arrays().as_tensor() * (mu * dqdv),
                //                )
                //            })
                //        }),
                //)
                .chain(
                    self.friction_jacobian_workspace
                        .contact_gradient_basis_eta_jac_basis_contact_jac
                        .view()
                        .into_iter()
                        .flat_map(move |(row_idx, row)| {
                            // (E)
                            row.into_iter().map(move |(col_idx, block)| {
                                (row_idx, col_idx, *block.into_arrays().as_tensor() * mu)
                            })
                        }), //.inspect(|(i,j,m)| log::trace!("E:({},{}): {:?}", i,j,(*m).into_data()))
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

    fn distance_jacobian_blocks_par_iter<'a>(
        &'a self,
    ) -> impl ParallelIterator<Item = (usize, usize, [T; 3])> + 'a {
        Self::distance_jacobian_blocks_par_iter_fn(
            &self.point_constraint,
            &self.implicit_surface_vertex_indices,
            &self.collider_vertex_indices,
        )
    }

    fn distance_jacobian_blocks_par_iter_fn<'a>(
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

    fn distance_jacobian_blocks_par_chunks_fn<'a, OP, TWS>(
        point_constraint: &'a PointContactConstraint<T>,
        implicit_surface_vertex_indices: &'a [usize],
        collider_vertex_indices: &'a [usize],
        ws: &mut [TWS],
        op: OP,
    ) where
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
    pub(crate) fn object_collider_distance_potential_hessian_indexed_blocks_iter<'a>(
        &'a self,
        lambda: &'a [T],
    ) -> Box<dyn Iterator<Item = (usize, usize, [[T; 3]; 3])> + 'a> {
        Box::new(
            if self.point_constraint.object_is_fixed() {
                None
            } else {
                let surf = &self.point_constraint.implicit_surface;
                surf.sample_query_hessian_product_indexed_blocks_iter(
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
                let row = self.collider_vertex_indices[row];
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
                    lambda.iter().cloned(),
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
                    .filter(move |(col_idx, _)| *col_idx < max_index)
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
        delta: f32,
        kappa: f32,
        max_index: usize,
    ) -> impl Iterator<Item = (MatrixElementIndex, T)> + 'a {
        let lambda = self.lambda.as_slice();
        let dist = self.distance_potential.as_slice();

        let hessian = self
            .object_distance_potential_hessian_indexed_blocks_iter(lambda)
            .chain(self.object_collider_distance_potential_hessian_indexed_blocks_iter(lambda))
            .flat_map(|(row, col, mtx)| {
                // Fill upper triangular portion.
                std::iter::once((row, col, mtx)).chain(if row == col {
                    Either::Left(std::iter::empty())
                } else {
                    Either::Right(std::iter::once((col, row, mtx)))
                })
            })
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
    use crate::test_utils::{default_solid, static_nl_params};
    use crate::{ContactType, Elasticity, FrictionalContactParams, MaterialIdType};
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
    fn eta_derivative() {
        let eta_jac_tester = |friction_profile: FrictionProfile| {
            let factor = 1.0;
            let epsilon = 0.1;
            let x = [-0.1, 0.9].into_tensor();
            let df_dv = friction_profile.jacobian(x, factor, epsilon);
            let mut v = x.mapd(|x| F1::cst(x));
            v[0] = F1::var(v[0]);
            let f = friction_profile.profile(v, factor.into(), epsilon.into());
            let df_dv0_ad = f.mapd(|x| x.deriv());
            assert_relative_eq!(df_dv0_ad[0], df_dv[0][0]);
            assert_relative_eq!(df_dv0_ad[1], df_dv[1][0]);
            v[0] = F1::cst(v[0]);
            v[1] = F1::var(v[1]);
            let f = friction_profile.profile(v, factor.into(), epsilon.into());
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
        let epsilon = 0.1;
        let eta_jac_tester = |fp: FrictionProfile| {
            for i in 0..10 {
                let x = [0.0 + 0.0001 * i as f64, 0.0].into_tensor();
                let df_dv = fp.jacobian(x, factor, epsilon);
                let mut v = x.mapd(|x| F1::cst(x));
                v[0] = F1::var(v[0]);
                let f = fp.profile(v, factor.into(), epsilon.into());
                let df_dv0_ad = f.mapd(|x| x.deriv());
                assert_relative_eq!(df_dv0_ad[0], df_dv[0][0]);
                assert_relative_eq!(df_dv0_ad[1], df_dv[1][0]);
                v[0] = F1::cst(v[0]);
                v[1] = F1::var(v[1]);
                let f = fp.profile(v, factor.into(), epsilon.into());
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
        use flatk::Subset;
        use geo::mesh::VertexPositions;
        use tensr::IntoStorage;

        crate::test_utils::init_logger();
        // Using the sliding tet on implicit test we will check the friction derivative directly.
        let material = default_solid().with_elasticity(Elasticity::from_young_poisson(1e5, 0.4));

        let mut tetmesh = PlatonicSolidBuilder::build_tetrahedron();
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

        let params = SimParams {
            max_iterations: 50,
            gravity: [0.0f32, -9.81, 0.0],
            time_step: Some(0.01),
            derivative_test: 2,
            residual_tolerance: 1e-8.into(),
            velocity_tolerance: 1e-5.into(),
            contact_tolerance: 0.0001,
            friction_tolerance: 0.0001,
            ..static_nl_params()
        };

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

        let mut mesh = Mesh::from(tetmesh);
        mesh.merge(Mesh::from(TriMesh::from(surface)));

        let fc_params = FrictionalContactParams {
            contact_type: ContactType::Point,
            kernel: KernelType::Approximate {
                radius_multiplier: 1.5,
                tolerance: 1e-5,
            },
            contact_offset: 0.0,
            use_fixed: false,
            friction_params: Some(FrictionParams {
                smoothing_weight: 0.0,
                friction_forwarding: 1.0,
                dynamic_friction: 0.18,
                inner_iterations: 50,
                tolerance: 1e-10,
                print_level: 0,
                friction_profile: FrictionProfile::default(),
            }),
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

        problem.update_constraint_set();

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

        State::be_step(state.step_state(&dq), dt);
        state.update_vertices(&dq);

        let State { vtx, .. } = state;

        // Column output row major jacobians.
        let mut jac_ad = vec![vec![0.0; n]; n];
        let mut jac = vec![vec![0.0; n]; n];

        let delta = problem.delta as f32;
        let kappa = problem.kappa as f32;
        let epsilon = problem.epsilon as f32;

        // Compute jacobian
        {
            let ResidualState { next, .. } = vtx.residual_state().into_storage();

            // Update constraint state.
            for fc in fc.iter() {
                let mut fc = fc.constraint.borrow_mut();
                fc.update_state(Chunked3::from_flat(next.pos));
                fc.update_distance_potential();
                fc.update_constraint_gradient();
                fc.update_multipliers(delta, kappa);

                // TODO: temp debug code to remove sliding basis derivative.
                let lambda = fc.lambda.as_slice();
                let dist = fc.distance_potential.as_slice();
                let (active_constraint_subset, _, _, _, _) =
                    fc.point_constraint.in_contact_indices(lambda, dist);
                let normals = fc.point_constraint.contact_normals();
                let normals_subset = Subset::from_unique_ordered_indices(
                    active_constraint_subset,
                    normals.as_slice(),
                );
                let mut normals = Chunked3::from_array_vec(vec![[0.0; 3]; normals_subset.len()]);
                normals_subset.clone_into_other(&mut normals);
                let FrictionWorkspace { contact_basis, .. } =
                    fc.point_constraint.friction_workspace.as_mut().unwrap();
                contact_basis.update_from_normals(normals.into());
            }

            for fc in fc.iter() {
                let mut constraint = fc.constraint.borrow_mut();
                // Compute friction hessian second term (multipliers held constant)
                let jac_iter = constraint
                    .friction_jacobian_indexed_value_iter(
                        Chunked3::from_flat(next.vel.view()),
                        delta,
                        kappa,
                        epsilon,
                        dt.into(),
                        n / 3,
                        false,
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
        for (next_pos, &cur_pos, &vel) in zip!(next_pos.iter_mut(), cur_pos.iter(), next.vel.iter())
        {
            *next_pos = cur_pos + vel * dt;
        }

        // Update constraint state.
        for fc in fc_ad.iter() {
            let mut fc = fc.constraint.borrow_mut();
            fc.update_state(Chunked3::from_flat(&next_pos));
            fc.update_distance_potential();
            fc.update_constraint_gradient();
            fc.update_multipliers(delta, kappa);

            // TODO: temp debug code to remove sliding basis derivative.
            //let lambda = fc.lambda.as_slice();
            //let dist = fc.distance_potential.as_slice();
            //let (active_constraint_subset, _, _, _, _) =
            //    fc.point_constraint.in_contact_indices(lambda, dist);
            //let normals = fc.point_constraint.contact_normals();
            //let normals_subset =
            //    Subset::from_unique_ordered_indices(active_constraint_subset, normals.as_slice());
            //let mut normals = Chunked3::from_array_vec(vec![[F1::zero(); 3]; normals_subset.len()]);
            //normals_subset.clone_into_other(&mut normals);
            //let FrictionWorkspace {
            //    contact_basis,
            //    ..
            //} = fc.point_constraint.friction_workspace.as_mut().unwrap();
            //contact_basis.update_from_normals(normals.into());
        }

        let mut success = true;
        for col in 0..n {
            //eprintln!("DERIVATIVE WRT {}", col);
            vel[col] = F1::var(vel[col]);
            // Update pos with backward euler.
            for (next_pos, &cur_pos, &vel) in zip!(next_pos.iter_mut(), cur_pos.iter(), vel.iter())
            {
                *next_pos = cur_pos + vel * dt;
            }
            r.iter_mut().for_each(|r| *r = F1::zero());
            for fc in fc_ad.iter() {
                let mut fc = fc.constraint.borrow_mut();
                fc.update_state(Chunked3::from_flat(&next_pos));
                fc.update_distance_potential();
                fc.update_constraint_gradient();
                fc.update_multipliers(delta, kappa);
                fc.subtract_friction_force(
                    Chunked3::from_flat(r),
                    Chunked3::from_flat(&vel),
                    epsilon,
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
            eprintln!("");
        }

        eprintln!("Expected:");
        for row in 0..n {
            for col in 0..n {
                eprint!("{:10.2e} ", jac_ad[row][col]);
            }
            eprintln!("");
        }

        if success {
            eprintln!("No errors during friction Jacobian check.");
            Ok(())
        } else {
            Err(crate::Error::DerivativeCheckFailure)
        }
    }
}
