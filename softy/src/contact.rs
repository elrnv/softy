mod solver;

use rayon::prelude::*;

use crate::Real;
use num_traits::Float;
use tensr::*;

use crate::friction::FrictionParams;
use crate::Index;
pub use solver::ContactSolver;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum ContactType {
    LinearizedPoint,
    Point,
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct FrictionalContactParams {
    pub kernel: implicits::KernelType,
    pub contact_type: ContactType,
    pub contact_offset: f64,
    /// Use fixed elements when building the contact surface.
    pub use_fixed: bool,
    pub friction_params: Option<FrictionParams>,
}

impl Default for FrictionalContactParams {
    fn default() -> Self {
        FrictionalContactParams {
            kernel: implicits::KernelType::Approximate {
                radius_multiplier: 1.0,
                tolerance: 1.0e-5,
            },
            contact_type: ContactType::LinearizedPoint,
            contact_offset: 0.0,
            use_fixed: true,
            friction_params: None, // Frictionless by default
        }
    }
}

/// A two dimensional vector in polar coordinates.
#[derive(Copy, Clone, Debug, PartialEq)]
#[repr(C)]
pub struct Polar2<T> {
    pub radius: T,
    pub angle: T,
}

unsafe impl bytemuck::Pod for Polar2<f64> {}
unsafe impl bytemuck::Zeroable for Polar2<f64> {}

impl<T: Real> Polar2<T> {
    pub fn from_euclidean<V2: Into<[T; 2]>>(v: V2) -> Polar2<T> {
        let v = v.into();
        Polar2 {
            radius: Vector2::new([v[0], v[1]]).norm(),
            angle: Float::atan2(v[1], v[0]),
        }
    }

    pub fn to_euclidean(self) -> [T; 2] {
        let Polar2 { radius, angle } = self;
        [radius * Float::cos(angle), radius * Float::sin(angle)]
    }
}

/// An annotated set of Cylindrical coordinates. The standard Vector3 struct is not applicable here
/// because arithmetic is different in cylindrical coordinates.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct VectorCyl<T> {
    pub normal: T,
    pub tangent: Polar2<T>,
}

impl<T: Real> VectorCyl<T> {
    pub fn new(normal: T, radius: T, angle: T) -> VectorCyl<T> {
        VectorCyl {
            normal,
            tangent: Polar2 { radius, angle },
        }
    }

    pub fn from_polar_tangent(tangent: Polar2<T>) -> VectorCyl<T> {
        VectorCyl {
            normal: T::zero(),
            tangent,
        }
    }

    pub fn from_euclidean<V3: Into<[T; 3]>>(v: V3) -> VectorCyl<T> {
        let v = v.into();
        VectorCyl {
            normal: v[0],
            tangent: Polar2 {
                radius: Vector2::new([v[1], v[2]]).norm(),
                angle: Float::atan2(v[2], v[1]),
            },
        }
    }

    pub fn to_euclidean(self) -> [T; 3] {
        let VectorCyl {
            normal,
            tangent: Polar2 { radius, angle },
        } = self;
        [
            normal,
            radius * Float::cos(angle),
            radius * Float::sin(angle),
        ]
    }
}

impl<T: Real> From<Polar2<T>> for VectorCyl<T> {
    fn from(v: Polar2<T>) -> Self {
        Self::from_polar_tangent(v)
    }
}

impl<T: Real> From<[T; 3]> for VectorCyl<T> {
    fn from(v: [T; 3]) -> Self {
        Self::from_euclidean(v)
    }
}

impl<T: Real> Into<[T; 3]> for VectorCyl<T> {
    fn into(self) -> [T; 3] {
        Self::to_euclidean(self)
    }
}

/// This struct defines the basis frame at the set of contact points.
#[derive(Clone, Debug, PartialEq)]
pub struct ContactBasis<T> {
    pub normals: Vec<[T; 3]>,
    pub tangents: Vec<[T; 3]>,
}

impl<T: Real> ContactBasis<T> {
    pub fn new() -> ContactBasis<T> {
        ContactBasis {
            normals: Vec::new(),
            tangents: Vec::new(),
        }
    }

    /// Converts this contact basis to `ContactBasis<f64>`.
    ///
    /// # Panics
    /// This function panics if the conversion to `f64` fails.
    #[inline]
    pub fn to_f64(&self) -> ContactBasis<f64> {
        self.clone_cast()
    }
    /// Clone this basis with all variables cast to `S`.
    #[inline]
    pub fn clone_cast<S: Real>(&self) -> ContactBasis<S> {
        ContactBasis {
            normals: self
                .normals
                .iter()
                .map(|x| x.as_tensor().cast::<S>().into())
                .collect(),
            tangents: self
                .tangents
                .iter()
                .map(|x| x.as_tensor().cast::<S>().into())
                .collect(),
        }
    }

    /// Check if the basis is empty. It may be empty if uninitialized or when there are no
    /// contacts.
    pub fn is_empty(&self) -> bool {
        self.normals.is_empty()
    }

    /// Remap values in the contact basis when the set of contacts change.
    pub fn remap(&mut self, old_set: &[usize], new_set: &[usize]) {
        // TODO: In addition to remapping this basis, we should just rebuild the missing parts.
        self.normals = crate::constraints::remap_values(
            self.normals.iter().cloned(),
            [T::zero(); 3],
            old_set.iter().cloned(),
            new_set.iter().cloned(),
        );
        self.tangents = crate::constraints::remap_values(
            self.tangents.iter().cloned(),
            [T::zero(); 3],
            old_set.iter().cloned(),
            new_set.iter().cloned(),
        );
    }

    pub fn to_cylindrical_contact_coordinates<V3>(
        &self,
        v: V3,
        contact_index: usize,
    ) -> VectorCyl<T>
    where
        V3: Into<[T; 3]>,
    {
        VectorCyl::from(self.to_contact_coordinates(v, contact_index))
    }

    /// Row-major basis matrix that transforms vectors from physical space to contact space.
    pub fn contact_basis_matrix(&self, contact_index: usize) -> Matrix3<T> {
        let n = self.normals[contact_index];
        let t = self.tangents[contact_index];
        let b = Vector3::new(n).cross(Vector3::new(t)).into();
        //b = b/b.norm(); // b may need to be renormalized here.
        Matrix3::new([n, t, b])
    }

    /// Transform a vector at the given contact point index to contact coordinates. The index
    /// determines which local contact coordinates to use.
    pub fn to_contact_coordinates<V3>(&self, v: V3, contact_index: usize) -> [T; 3]
    where
        V3: Into<[T; 3]>,
    {
        (self.contact_basis_matrix(contact_index) * Vector3::new(v.into())).into()
    }

    pub fn from_cylindrical_contact_coordinates(
        &self,
        v: VectorCyl<T>,
        contact_index: usize,
    ) -> [T; 3] {
        self.from_contact_coordinates(v, contact_index)
    }

    /// Transform a vector at the given contact point index to physical coordinates. The index
    /// determines which local contact coordinates to use.
    pub fn from_contact_coordinates<V3>(&self, v: V3, contact_index: usize) -> [T; 3]
    where
        V3: Into<[T; 3]>,
    {
        (self.contact_basis_matrix(contact_index).transpose() * Vector3::new(v.into())).into()
    }

    /// Transform a given stacked vector of vectors in physical space
    /// to stacked 2D vectors in the tangent space of the contact point.
    pub fn to_tangent_space<'a>(
        &'a self,
        physical: &'a [[T; 3]],
    ) -> impl Iterator<Item = [T; 2]> + 'a {
        physical.iter().enumerate().map(move |(i, &v)| {
            let [_, v1, v2] = self.to_contact_coordinates(v, i);
            [v1, v2]
        })
    }

    /// Transform a given stacked vector of vectors in contact space to vectors in physical space.
    pub fn from_tangent_space<'a>(
        &'a self,
        contact: &'a [[T; 2]],
    ) -> impl Iterator<Item = [T; 3]> + 'a {
        contact
            .iter()
            .enumerate()
            .map(move |(i, &v)| self.from_contact_coordinates([T::zero(), v[0], v[1]], i))
    }

    /// Transform a given vector of vectors in physical space to values in normal direction.
    pub fn to_normal_space<'a>(&'a self, physical: &'a [[T; 3]]) -> impl Iterator<Item = T> + 'a {
        physical
            .iter()
            .enumerate()
            .map(move |(i, &v)| self.to_contact_coordinates(v, i)[0])
    }
    /// Transform a given vector of normal coordinates in contact space to vectors in physical space.
    pub fn from_normal_space<'a>(&'a self, contact: &'a [T]) -> impl Iterator<Item = [T; 3]> + 'a {
        contact
            .iter()
            .enumerate()
            .map(move |(i, &n)| self.from_contact_coordinates([n, T::zero(), T::zero()], i))
    }

    pub fn to_polar_tangent_space(&self, physical: &[[T; 3]]) -> Vec<Polar2<T>> {
        physical
            .iter()
            .enumerate()
            .map(|(i, &v)| self.to_cylindrical_contact_coordinates(v, i).tangent)
            .collect()
    }

    pub fn from_polar_tangent_space(&self, contact: &[Polar2<T>]) -> Vec<[T; 3]> {
        contact
            .iter()
            .enumerate()
            .map(|(i, &v)| {
                self.from_cylindrical_contact_coordinates(v.into(), i)
                    .into()
            })
            .collect()
    }

    pub fn normal_basis_matrix_sprs(&self) -> sprs::CsMat<T> {
        let n = self.normals.len();

        // A vector of column major change of basis "matrices"
        let row_mtx = Vector3::new([0usize, 1, 2]);
        let col_mtx = Vector3::new([0usize, 0, 0]);
        let mut rows = vec![[0; 3]; n];
        let mut cols = vec![[0; 3]; n];
        let mut bases = vec![[T::zero(); 3]; n];
        for (contact_idx, (m, r, c)) in
            zip!(bases.iter_mut(), rows.iter_mut(), cols.iter_mut()).enumerate()
        {
            let mtx = self.contact_basis_matrix(contact_idx);
            *m = mtx[0].into_data();

            *r = row_mtx.mapd(|x| x + 3 * contact_idx).into();
            *c = col_mtx.mapd(|x| x + contact_idx).into();
        }

        let num_rows = 3 * n;
        let num_cols = n;
        sprs::TriMat::from_triplets(
            (num_rows, num_cols),
            Chunked3::from_array_vec(rows).into_storage(),
            Chunked3::from_array_vec(cols).into_storage(),
            Chunked3::from_array_vec(bases).into_storage(),
        )
        .to_csr()
    }

    pub fn tangent_basis_matrix_sprs(&self) -> sprs::CsMat<T> {
        let n = self.normals.len();

        // A vector of column major change of basis matrices
        let row_stencil = Matrix2x3::new([[0usize, 0, 1], [1, 2, 2]]);
        let col_stencil = Matrix2x3::new([[0usize, 1, 0], [1, 0, 1]]);
        let mut rows = vec![[[0; 3]; 2]; n];
        let mut cols = vec![[[0; 3]; 2]; n];
        let mut bases = vec![[[T::zero(); 3]; 2]; n];
        for (contact_idx, (m, r, c)) in
            zip!(bases.iter_mut(), rows.iter_mut(), cols.iter_mut()).enumerate()
        {
            let mtx = self.contact_basis_matrix(contact_idx);
            m[0] = mtx[1].into();
            m[1] = mtx[2].into();

            *r = row_stencil.mapd_inner(|x| x + 3 * contact_idx).into();
            *c = col_stencil.mapd_inner(|x| x + 2 * contact_idx).into();
        }

        let num_rows = 3 * n;
        let num_cols = 2 * n;
        sprs::TriMat::from_triplets(
            (num_rows, num_cols),
            Chunked3::from_array_vec(Chunked2::from_array_vec(rows).into_storage()).into_storage(),
            Chunked3::from_array_vec(Chunked2::from_array_vec(cols).into_storage()).into_storage(),
            Chunked3::from_array_vec(Chunked2::from_array_vec(bases).into_storage()).into_storage(),
        )
        .to_csr()
    }

    pub fn normal_basis_matrix(&self) -> BlockDiagonalMatrix3x1<T> {
        let n = self.normals.len();

        // A vector of column major change of basis matrices
        let bases: Chunked3<Vec<_>> = (0..n)
            .map(|contact_idx| {
                let mtx = self.contact_basis_matrix(contact_idx);
                // Take the transpose of the basis [nml] (ignoring the tangential components).
                [[mtx[0][0]], [mtx[0][1]], [mtx[0][2]]]
            })
            .collect();

        BlockDiagonalMatrix::new(Chunked3::from_flat(Chunked1::from_array_vec(
            bases.into_storage(),
        )))
    }

    pub fn tangent_basis_matrix(&self) -> BlockDiagonalMatrix3x2<T> {
        let n = self.normals.len();

        // A vector of column major change of basis matrices
        let bases: Chunked3<Vec<_>> = (0..n)
            .map(|contact_idx| {
                let mtx = self.contact_basis_matrix(contact_idx);
                // Take the transpose of the basis [t, b] (ignoring the 0th normal component).
                [
                    [mtx[1][0], mtx[2][0]],
                    [mtx[1][1], mtx[2][1]],
                    [mtx[1][2], mtx[2][2]],
                ]
            })
            .collect();

        BlockDiagonalMatrix::new(Chunked3::from_flat(Chunked2::from_array_vec(
            bases.into_storage(),
        )))
    }

    /// Update the basis for the contact space at each contact point given the specified set of
    /// normals. The tangent space is chosen arbitrarily.
    pub fn update_from_normals(&mut self, normals: Vec<[T; 3]>) {
        self.tangents.clear();
        self.tangents.resize(normals.len(), [T::zero(); 3]);
        self.normals = normals;

        for (&n, t) in self.normals.iter().zip(self.tangents.iter_mut()) {
            // Find the axis that is most aligned with the normal, then use the next axis for the
            // tangent.
            let tangent_axis = (Vector3::new(n).iamax() + 1) % 3;
            t[tangent_axis] = T::one();

            // Project out the normal component.
            *t.as_mut_tensor() -= Vector3::new(n) * n[tangent_axis];

            t.as_mut_tensor().normalize(); // Normalize in-place.
        }
    }

    /// A convenience function for projecting onto the tangent space of this basis.
    ///
    /// This is a faster and more accurate way to compute:
    /// ```
    /// # let vecs = vec![[1.0;3];1];
    /// # let normals = vecs.clone();
    /// # let tangents = vecs.clone();
    /// # let basis = softy::ContactBasis { normals, tangents };
    /// let vecs2d: Vec<_> = basis.to_tangent_space(&vecs).collect();
    /// let projected_vecs: Vec<_> = basis.from_tangent_space(&vecs2d).collect();
    /// ```
    /// given
    pub fn project_to_tangent_space<'a, V>(&self, vecs: V)
    where
        V: Iterator,
        V::Item: Into<&'a mut Vector3<T>>,
    {
        Self::project_out_normal_component(self.normals.iter().cloned(), vecs);
    }

    /// A basis independent version of `project_to_tangent_space`.
    ///
    /// This function projects out the normal component from the given iterator of vectors in-place.
    pub fn project_out_normal_component<'a, N, V>(normals: N, vecs: V)
    where
        N: Iterator,
        V: Iterator,
        N::Item: Into<Vector3<T>>,
        V::Item: Into<&'a mut Vector3<T>>,
    {
        for (n, v) in normals.zip(vecs) {
            let n = n.into();
            let v = v.into();
            let nml_component = n.dot(*v);
            *v -= n * nml_component;
        }
    }
}

/// An intermediate representation of a contact jacobian that makes it easy to
/// convert to other sparse representations.
pub(crate) struct TripletContactJacobian<T> {
    pub block_indices: Vec<(usize, usize)>,
    pub blocks: Chunked3<Chunked3<Vec<T>>>,
    pub num_rows: usize,
    pub num_cols: usize,
}

impl<T: Real> TripletContactJacobian<T> {
    #[cfg(feature = "optsolver")]
    pub fn new() -> TripletContactJacobian<T> {
        TripletContactJacobian {
            block_indices: Vec::new(),
            blocks: Chunked3::from_flat(Chunked3::from_flat(Vec::new())),
            num_rows: 0,
            num_cols: 0,
        }
    }

    pub fn from_selection<'a>(
        surf: &implicits::QueryTopo<T>,
        active_contact_points: SelectView<'a, Chunked3<&'a [T]>>,
    ) -> TripletContactJacobian<T> {
        //let mut orig_cj_matrices = vec![[[T::zero(); 3]; 3]; surf.num_contact_jacobian_matrices()];
        let query_points = active_contact_points.target;
        let orig_cj_matrix_iter = surf.contact_jacobian_matrices_par_iter(query_points.into());

        // Build a reverse map for mapping to active contacts
        let mut active_contact_point_indices = vec![Index::INVALID; query_points.len()];
        for (i, &acpi) in active_contact_points.index_iter().enumerate() {
            active_contact_point_indices[acpi] = Index::new(i);
        }

        // Remap rows to match active constraints. This means that some entries of the raw jacobian
        // will not have a valid entry in the pruned jacobian.
        let (cj_indices, cj_matrices): (Vec<_>, Vec<_>) = orig_cj_matrix_iter
            .filter_map(|(row, col, matrix)| {
                active_contact_point_indices[row]
                    .into_option()
                    .map(|idx| ((idx, col), matrix))
            })
            .unzip();

        let blocks = Chunked3::from_flat(Chunked3::from_array_vec(
            Chunked3::from_array_vec(cj_matrices).into_storage(),
        ));

        TripletContactJacobian {
            block_indices: cj_indices,
            blocks,
            num_rows: active_contact_points.len(),
            num_cols: surf.surface_vertex_positions().len(),
        }
    }

    /// Append a set of triplets to the current contact Jacobian set of triplets.
    ///
    /// This includes the Jacobian with respect to object vertices as well as collider vertices.
    ///
    /// Note that this function does *not* update `num_rows` and `num_cols` automatically.
    ///
    /// Along with the necessary data for constructing the contact jacobian, this function accepts
    /// three offsets intended to position this jacobian inside the global contact jacobian matrix.
    ///
    /// The columns of the contact jacobian are indexed by surface vertex indices in the entire
    /// surface (as opposed to a subset) for objects and colliders.
    #[cfg(feature = "optsolver")]
    pub fn append_selection<'a>(
        &mut self,
        surf: &implicits::QueryTopo<T>,
        active_contact_points: SelectView<'a, Chunked3<&'a [T]>>,
        contact_offset: usize,
        object_offset: usize,
        collider_offset: usize,
    ) {
        let TripletContactJacobian {
            block_indices,
            blocks,
            ..
        } = self;

        // First we append the contact jacobian with respect to object vertices.

        let mut orig_cj_matrices = vec![[[T::zero(); 3]; 3]; surf.num_contact_jacobian_matrices()];
        let query_points = active_contact_points.target;
        surf.contact_jacobian_matrices(query_points.into(), &mut orig_cj_matrices);

        // Build a reverse map for mapping to active contacts
        let mut active_contact_point_indices = vec![Index::INVALID; query_points.len()];
        for (i, &acpi) in active_contact_points.index_iter().enumerate() {
            active_contact_point_indices[acpi] = Index::new(i);
        }

        let orig_cj_indices_iter = surf.contact_jacobian_matrix_indices_iter();

        let mut cj_matrices: Vec<_> = orig_cj_indices_iter
            .clone()
            .zip(orig_cj_matrices.into_iter())
            .filter_map(|((row, _), matrix)| {
                active_contact_point_indices[row]
                    .into_option()
                    .map(|_| matrix)
            })
            .collect();

        // Remap rows to match active constraints. This means that some entries of the raw jacobian
        // will not have a valid entry in the pruned jacobian.
        block_indices.extend(orig_cj_indices_iter.filter_map(move |(row, col)| {
            active_contact_point_indices[row]
                .into_option()
                .map(|idx| (contact_offset + idx, object_offset + col))
        }));

        // Lastly we append the contact Jacobian with respect to collider vertices.

        block_indices.extend(
            active_contact_points
                .index_iter()
                .enumerate()
                .map(|(i, &acp)| (i, collider_offset + acp)),
        );

        cj_matrices
            .extend((0..active_contact_points.len()).map(|_| Matrix3::<T>::identity().into_data()));

        // Convert blocks into Chunked types.
        // TODO: make this type of conversion more ergonomic in flatk.

        let chunked_matrices = Chunked3::from_flat(Chunked3::from_array_vec(
            Chunked3::from_array_vec(cj_matrices).into_storage(),
        ));

        blocks
            .storage_mut()
            .extend(chunked_matrices.into_storage().into_iter());
    }
}

/// Contact jacobian maps values at the surface vertex positions (of the object)
/// to query points (contact points).
///
/// Not all contact points are active, so rows are sparse, and not all surface vertex positions are
/// affected by each query point, so columns are also sparse.
pub type ContactJacobian<T = f64> = Tensor![T; S S 3 3]; // SSBlockMatrix3<T>;
pub type ContactJacobianView<'a, T = f64> = Tensor![T; &'a S S 3 3]; //SSBlockMatrix3View<'a, T>;

/// The global contact Jacobian maps values at the surface vertex positions (of all objects) to
/// query points (contact points).
///
/// # Notes about sparsity
///
/// Not all objects are in contact (even if there is a constraint between them), so outer dimension
/// is sparse.
///
/// Only two objects participate in the contact jacobian block row, so second dimension is sparse.
///
/// Not all contact points are active, so third dimension is sparse.
///
/// Finally not all surface vertex positions are affected by each query point, so the fourth
/// dimension is also sparse.
pub type GlobalContactJacobian<T = f64> = Tensor![T; S S S S 3 3];
pub type GlobalContactJacobianView<'a, T = f64> = Tensor![T; &'a S S S S 3 3];

impl<T: Real> Into<ContactJacobian<T>> for TripletContactJacobian<T> {
    fn into(self) -> ContactJacobian<T> {
        SSBlockMatrix3::from_index_iter_and_data(
            self.block_indices.iter().cloned(),
            self.num_rows,
            self.num_cols,
            self.blocks,
        )
        .into_data()
    }
}
//
//impl<'a, I> std::ops::Mul<Tensor<SubsetView<'a, Chunked3<&'a [f64]>>>>
//    for &TripletContactJacobian<I>
//where
//    I: Clone + Iterator<Item = (usize, usize)>,
//{
//    type Output = Chunked3<Vec<f64>>;
//    fn mul(self, rhs: Tensor<SubsetView<'a, Chunked3<&'a [f64]>>>) -> Self::Output {
//        assert_eq!(rhs.data.len(), self.num_cols);
//
//        let mut res = Chunked3::from_array_vec(vec![[0.0; 3]; self.num_rows]);
//        for ((r, c), &block) in self.iter.clone().zip(self.blocks.iter()) {
//            let out = geo::math::Vector3(res[r]) + block * geo::math::Vector3(rhs[c]);
//            res[r] = out.into();
//        }
//
//        res
//    }
//}

impl Into<sprs::CsMat<f64>> for TripletContactJacobian<f64> {
    // Compute contact jacobian
    fn into(self) -> sprs::CsMat<f64> {
        let (rows, cols) = self
            .block_indices
            .iter()
            .flat_map(move |(row_mtx, col_mtx)| {
                (0..3).flat_map(move |i| (0..3).map(move |j| (3 * row_mtx + i, 3 * col_mtx + j)))
            })
            .unzip();

        let values = self.blocks.into_storage();

        sprs::TriMat::from_triplets((3 * self.num_rows, 3 * self.num_cols), rows, cols, values)
            .to_csr()
    }
}

//impl<I> TripletContactJacobian<I> {
//pub(crate) fn transpose(&self) -> Transpose<&Self> {
//    Transpose(&self)
//}
//}

//impl<'a, I> Transpose<&TripletContactJacobian<I>>
//where
//    I: Clone + Iterator<Item = (usize, usize)>,
//{
//    fn mul_vector(self, rhs: Tensor<SubsetView<'a, Chunked3<&'a [f64]>>>) -> Chunked3<Vec<f64>> {
//        assert_eq!(rhs.data.len(), self.0.num_rows);
//
//        let mut res = Chunked3::from_array_vec(vec![[0.0; 3]; self.0.num_cols]);
//
//        for ((r, c), &block) in self.0.iter.clone().zip(self.0.blocks.iter()) {
//            let out = geo::math::Vector3(res[c]) + block.transpose() * geo::math::Vector3(rhs[r]);
//            res[c] = out.into();
//        }
//
//        res
//    }
//}

/// A diagonal mass matrix chunked by triplet blocks (one triplet for each vertex).
pub type MassMatrix<T = f64> = DiagonalBlockMatrix3<T>;
pub type MassMatrixView<'a, T = f64> = DiagonalBlockMatrix3View<'a, T>;

pub type Delassus<T = f64> = DSBlockMatrix3<T>;
pub type DelassusView<'a, T = f64> = DSBlockMatrix3View<'a, T>;

#[cfg(feature = "optsolver")]
pub(crate) type EffectiveMassInv<T = f64> = DSBlockMatrix3<T>;
#[cfg(feature = "optsolver")]
pub(crate) type EffectiveMassInvView<'a, T = f64> = DSBlockMatrix3View<'a, T>;

/// Global effective mass inverse.
///
/// The dimensions in order are as follows
/// D D -> Dense object contact coupling (only active couplings are included)
/// D D -> For each coupling only active contacts are considered.
/// 3 3 -> Each vertex mass is a 3x3 block.
#[cfg(feature = "optsolver")]
pub(crate) type GlobalEffectiveMassInv<T = f64> = Tensor![T; D D D D 3 3];
#[cfg(feature = "optsolver")]
pub(crate) type GlobalEffectiveMassInvView<'a, T = f64> = Tensor![T; &'a D D D D 3 3];

#[cfg(test)]
mod tests {
    use super::*;
    use approx::*;
    use geo::mesh::builder::*;
    use geo::mesh::{topology::*, TriMesh, VertexPositions};

    /// We pass a contact basis here by mutable reference. This tests that a basis can be reused
    /// safely.
    fn contact_basis_from_trimesh(trimesh: &TriMesh<f64>, basis: &mut ContactBasis<f64>) {
        use tensr::*;
        let mut normals = vec![[0.0; 3]; trimesh.num_vertices()];
        geo::algo::compute_vertex_area_weighted_normals(
            trimesh.vertex_positions(),
            trimesh.indices.as_slice(),
            &mut normals,
        );

        for n in normals.iter_mut() {
            let norm = (*n).as_tensor().norm();
            *n.as_mut_tensor() /= norm;
        }

        basis.update_from_normals(normals);
    }

    // Verify that converting to contact space and back to physical space produces the same
    // vectors.
    #[test]
    fn contact_physical_space_conversion_test() -> Result<(), crate::Error> {
        // TODO: upon migrating from nalgebra to tensr. The relative comparisons below required a
        //       larger max_relative tolerance.
        //       Investigate why this is so.
        let mut basis = ContactBasis::new();
        let mut run = |trimesh: TriMesh<f64>| -> Result<(), crate::Error> {
            contact_basis_from_trimesh(&trimesh, &mut basis);

            let vecs = utils::random_vectors(trimesh.num_vertices());

            // Test euclidean basis
            let contact_vecs: Vec<_> = basis.to_tangent_space(vecs.as_slice()).collect();
            let projected_physical_vecs: Vec<_> =
                basis.from_tangent_space(contact_vecs.as_slice()).collect();

            // Manually project out normal component for another point of comparison:
            let mut vecs_t = vecs.clone();
            for (v_out, &n) in vecs_t.iter_mut().zip(basis.normals.iter()) {
                let n = Vector3::new(n);
                let v = Vector3::new(*v_out);
                *v_out.as_mut_tensor() -= n * v.dot(n);
            }

            let projected_contact_vecs: Vec<_> = basis
                .to_tangent_space(projected_physical_vecs.as_slice())
                .collect();

            for (a, &b) in contact_vecs.into_iter().zip(projected_contact_vecs.iter()) {
                for i in 0..2 {
                    assert_relative_eq!(a[i], b[i], max_relative = 1e-9);
                }
            }
            for (a, &b) in vecs_t.into_iter().zip(projected_physical_vecs.iter()) {
                assert_relative_eq!(a.as_tensor(), b.as_tensor(), max_relative = 1e-9);
            }

            let reprojected_physical_vecs =
                basis.from_tangent_space(projected_contact_vecs.as_slice());

            for (a, b) in projected_physical_vecs
                .into_iter()
                .zip(reprojected_physical_vecs.into_iter())
            {
                for i in 0..3 {
                    assert_relative_eq!(a[i], b[i], max_relative = 1e-9);
                }
            }

            // Test cylindrical coordinates
            let contact_vecs = basis.to_polar_tangent_space(vecs.as_slice());
            let projected_physical_vecs = basis.from_polar_tangent_space(contact_vecs.as_slice());
            let projected_contact_vecs =
                basis.to_polar_tangent_space(projected_physical_vecs.as_slice());

            for (a, &b) in contact_vecs.into_iter().zip(projected_contact_vecs.iter()) {
                assert_relative_eq!(a.radius, b.radius, max_relative = 1e-9);
                assert_relative_eq!(a.angle, b.angle, max_relative = 1e-9);
            }

            let reprojected_physical_vecs = basis.from_polar_tangent_space(&projected_contact_vecs);

            for (a, b) in projected_physical_vecs
                .into_iter()
                .zip(reprojected_physical_vecs.into_iter())
            {
                for i in 0..3 {
                    assert_relative_eq!(a[i], b[i], max_relative = 1e-9);
                }
            }
            Ok(())
        };

        let trimesh = PlatonicSolidBuilder::build_octahedron();
        run(trimesh)?;
        let trimesh = PlatonicSolidBuilder::build_icosahedron();
        run(trimesh)
    }

    // Verify that multiplying by the basis matrix has the same effect as converting a vector using
    // an iterator function like `from_tangent_space`.
    #[test]
    fn contact_basis_matrix_test() {
        let trimesh = PlatonicSolidBuilder::build_octahedron();
        let mut basis = ContactBasis::new();
        contact_basis_from_trimesh(&trimesh, &mut basis);

        let vecs = Chunked3::from_array_vec(utils::random_vectors(trimesh.num_vertices()));

        let nml_basis_mtx = basis.normal_basis_matrix();

        let exp_contact_vecs: Vec<_> = basis.to_normal_space(vecs.view().into_arrays()).collect();
        let contact_vecs = nml_basis_mtx.view().transpose() * UTensor3::new(vecs.view());

        assert_eq!(exp_contact_vecs, contact_vecs.into_data());
    }
}
