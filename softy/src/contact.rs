mod solver;

use crate::friction::FrictionParams;
use implicits::ImplicitSurface;
use na::{Matrix3, Matrix3x2, RealField, Vector2, Vector3};
use reinterpret::*;
pub use solver::ContactSolver;
use utils::soap::*;
use utils::zip;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum ContactType {
    SPImplicit,
    Implicit,
    Point,
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct FrictionalContactParams {
    pub kernel: implicits::KernelType,
    pub contact_type: ContactType,
    pub friction_params: Option<FrictionParams>,
}

/// A two dimensional vector in polar coordinates.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Polar2<T: RealField> {
    pub radius: T,
    pub angle: T,
}

/// An annotated set of Cylindrical coordinates. The standard Vector3 struct is not applicable here
/// because arithmetic is different in cylindrical coordinates.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct VectorCyl<T: RealField> {
    pub normal: T,
    pub tangent: Polar2<T>,
}

impl<T: RealField> VectorCyl<T> {
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
                radius: Vector2::new(v[1], v[2]).norm(),
                angle: T::atan2(v[2], v[1]),
            },
        }
    }

    pub fn to_euclidean(self) -> Vector3<T> {
        let VectorCyl {
            normal,
            tangent: Polar2 { radius, angle },
        } = self;
        Vector3::new(normal, radius * angle.cos(), radius * angle.sin())
    }
}

impl<T: RealField> From<Polar2<T>> for VectorCyl<T> {
    fn from(v: Polar2<T>) -> Self {
        Self::from_polar_tangent(v)
    }
}

impl<T: RealField> From<Vector3<T>> for VectorCyl<T> {
    fn from(v: Vector3<T>) -> Self {
        Self::from_euclidean(v)
    }
}

impl<T: RealField> Into<Vector3<T>> for VectorCyl<T> {
    fn into(self) -> Vector3<T> {
        Self::to_euclidean(self)
    }
}

/// This struct defines the basis frame at the set of contact points.
#[derive(Clone, Debug, PartialEq)]
pub struct ContactBasis {
    normals: Vec<Vector3<f64>>,
    tangents: Vec<Vector3<f64>>,
}

impl ContactBasis {
    pub fn new() -> ContactBasis {
        ContactBasis {
            normals: Vec::new(),
            tangents: Vec::new(),
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
        let new_normals = crate::constraints::remap_values(
            self.normals.iter().cloned(),
            Vector3::zeros(),
            old_set.iter().cloned(),
            new_set.iter().cloned(),
        );
        let new_tangents = crate::constraints::remap_values(
            self.tangents.iter().cloned(),
            Vector3::zeros(),
            old_set.iter().cloned(),
            new_set.iter().cloned(),
        );
        std::mem::replace(&mut self.normals, new_normals);
        std::mem::replace(&mut self.tangents, new_tangents);
    }

    pub fn to_cylindrical_contact_coordinates<V3>(
        &self,
        v: V3,
        contact_index: usize,
    ) -> VectorCyl<f64>
    where
        V3: Into<Vector3<f64>>,
    {
        VectorCyl::from(self.to_contact_coordinates(v, contact_index))
    }

    pub fn contact_basis_matrix(&self, contact_index: usize) -> Matrix3<f64> {
        let n = self.normals[contact_index];
        let t = self.tangents[contact_index];
        let b = n.cross(&t);
        //b = b/b.norm(); // b may need to be renormalized here.
        Matrix3::from_columns(&[n, t, b])
    }

    /// Transform a vector at the given contact point index to contact coordinates. The index
    /// determines which local contact coordinates to use.
    pub fn to_contact_coordinates<V3>(&self, v: V3, contact_index: usize) -> Vector3<f64>
    where
        V3: Into<Vector3<f64>>,
    {
        self.contact_basis_matrix(contact_index).transpose() * v.into()
    }

    pub fn from_cylindrical_contact_coordinates(
        &self,
        v: VectorCyl<f64>,
        contact_index: usize,
    ) -> Vector3<f64> {
        self.from_contact_coordinates(v, contact_index)
    }

    /// Transform a vector at the given contact point index to physical coordinates. The index
    /// determines which local contact coordinates to use.
    pub fn from_contact_coordinates<V3>(&self, v: V3, contact_index: usize) -> Vector3<f64>
    where
        V3: Into<Vector3<f64>>,
    {
        self.contact_basis_matrix(contact_index) * v.into()
    }

    /// Transform a given stacked vector of vectors in physical space to values in normal direction
    /// to each contact point and stacked 2D vectors in the tangent space of the contact point.
    pub fn to_tangent_space(&self, physical: Vec<[f64; 3]>) -> Vec<[f64; 2]> {
        physical
            .iter()
            .enumerate()
            .map(|(i, &v)| {
                let new_v: Vector3<f64> = self.to_contact_coordinates(v, i).into();
                [new_v[1], new_v[2]]
            })
            .collect()
    }

    /// Transform a given stacked vector of vectors in contact space to vectors in physical space.
    pub fn from_tangent_space(&self, contact: Vec<[f64; 2]>) -> Vec<[f64; 3]> {
        contact
            .iter()
            .enumerate()
            .map(|(i, &v)| self.from_contact_coordinates([0.0, v[0], v[1]], i).into())
            .collect()
    }

    pub fn to_polar_tangent_space(&self, physical: Vec<[f64; 3]>) -> Vec<Polar2<f64>> {
        physical
            .iter()
            .enumerate()
            .map(|(i, &v)| self.to_cylindrical_contact_coordinates(v, i).tangent)
            .collect()
    }

    pub fn from_polar_tangent_space(&self, contact: Vec<Polar2<f64>>) -> Vec<[f64; 3]> {
        contact
            .iter()
            .enumerate()
            .map(|(i, &v)| {
                self.from_cylindrical_contact_coordinates(v.into(), i)
                    .into()
            })
            .collect()
    }

    pub fn normal_basis_matrix_sprs(&self) -> sprs::CsMat<f64> {
        let n = self.normals.len();

        // A vector of column major change of basis matrices
        let row_mtx = Vector3::new(0, 1, 2);
        let col_mtx = Vector3::new(0, 0, 0);
        let mut rows = vec![[0; 3]; n];
        let mut cols = vec![[0; 3]; n];
        let mut bases = vec![[0.0; 3]; n];
        for (contact_idx, (m, r, c)) in
            zip!(bases.iter_mut(), rows.iter_mut(), cols.iter_mut()).enumerate()
        {
            let mtx = self.contact_basis_matrix(contact_idx);
            *m = mtx.column(0).into();

            *r = row_mtx.add_scalar(3 * contact_idx).into();
            *c = col_mtx.add_scalar(contact_idx).into();
        }

        let num_rows = 3 * n;
        let num_cols = n;
        sprs::TriMat::from_triplets(
            (num_rows, num_cols),
            reinterpret_vec(rows),
            reinterpret_vec(cols),
            reinterpret_vec(bases),
        )
        .to_csr()
    }

    pub fn tangent_basis_matrix_sprs(&self) -> sprs::CsMat<f64> {
        let n = self.normals.len();

        // A vector of column major change of basis matrices
        let row_mtx = Matrix3x2::new(0, 0, 1, 1, 2, 2);
        let col_mtx = Matrix3x2::new(0, 1, 0, 1, 0, 1);
        let mut rows = vec![[[0; 3]; 2]; n];
        let mut cols = vec![[[0; 3]; 2]; n];
        let mut bases = vec![[[0.0; 3]; 2]; n];
        for (contact_idx, (m, r, c)) in
            zip!(bases.iter_mut(), rows.iter_mut(), cols.iter_mut()).enumerate()
        {
            let mtx = self.contact_basis_matrix(contact_idx);
            m[0] = mtx.column(1).into();
            m[1] = mtx.column(2).into();

            *r = row_mtx.add_scalar(3 * contact_idx).into();
            *c = col_mtx.add_scalar(2 * contact_idx).into();
        }

        let num_rows = 3 * n;
        let num_cols = 2 * n;
        sprs::TriMat::from_triplets(
            (num_rows, num_cols),
            reinterpret_vec(rows),
            reinterpret_vec(cols),
            reinterpret_vec(bases),
        )
        .to_csr()
    }

    /// Update the basis for the contact space at each contact point given the specified set of
    /// normals. The tangent space is chosen arbitrarily
    pub fn update_from_normals(&mut self, normals: Vec<[f64; 3]>) {
        self.tangents.resize(normals.len(), Vector3::zeros());
        self.normals = reinterpret::reinterpret_vec(normals);

        for (&n, t) in self.normals.iter().zip(self.tangents.iter_mut()) {
            // Find the axis that is most aligned with the normal, then use the next axis for the
            // tangent.
            let tangent_axis = (n.iamax() + 1) % 3;
            t[tangent_axis] = 1.0;

            // Project out the normal component.
            *t -= n * n[tangent_axis];

            // Normalize in-place.
            t.normalize_mut();
        }
    }
}

/// An intermediate representation of a contact jacobian that makes it easy to
/// convert to other sparse representations.
pub(crate) struct TripletContactJacobian<I> {
    pub iter: I,
    pub blocks: Vec<geo::math::Matrix3<f64>>,
    pub num_rows: usize,
    pub num_cols: usize,
}

pub(crate) fn build_triplet_contact_jacobian(
    surf: &ImplicitSurface,
    query_points: Chunked3<&[f64]>,
) -> TripletContactJacobian<impl Iterator<Item = (usize, usize)> + Clone> {
    let mut cj_matrices = vec![
        geo::math::Matrix3::zeros();
        surf.num_contact_jacobian_matrices()
            .expect("Failed to get contact Jacobian size.")
    ];
    surf.contact_jacobian_matrices(
        query_points.view().into(),
        reinterpret_mut_slice(&mut cj_matrices),
    )
    .expect("Failed to compute contact Jacobian.");
    let cj_indices_iter = surf
        .contact_jacobian_matrix_indices_iter()
        .expect("Failed to get contact Jacobian indices.");

    TripletContactJacobian {
        iter: cj_indices_iter,
        blocks: cj_matrices,
        num_rows: query_points.len(),
        num_cols: surf.surface_vertex_positions().len(),
    }
}

/// Contact jacobian maps values at the surface vertex positions (of the object)
/// to query points (contact points). Not all contact points are active, so rows
/// are sparse, and not all surface vertex positions are affected by each query
/// point, so columns are also sparse.
pub type ContactJacobian<S = Vec<f64>, I = Vec<usize>> = SSBlockMatrix3<S, I>;
pub type ContactJacobianView<'a> = ContactJacobian<&'a [f64], &'a [usize]>;

impl<I: Iterator<Item = (usize, usize)>> Into<ContactJacobian> for TripletContactJacobian<I> {
    fn into(self) -> ContactJacobian {
        let blocks = Chunked3::from_flat(Chunked3::from_flat(reinterpret_vec(self.blocks)));
        ContactJacobian::from_triplets(self.iter, self.num_rows, self.num_cols, blocks)
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

impl<I> Into<sprs::CsMat<f64>> for TripletContactJacobian<I>
where
    I: Iterator<Item = (usize, usize)>,
{
    // Compute contact jacobian
    fn into(self) -> sprs::CsMat<f64> {
        let (rows, cols) = self
            .iter
            .flat_map(move |(row_mtx, col_mtx)| {
                (0..3).flat_map(move |j| (0..3).map(move |i| (3 * row_mtx + i, 3 * col_mtx + j)))
            })
            .unzip();

        let values = reinterpret::reinterpret_vec(self.blocks);

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
pub type MassMatrix<S = Vec<f64>> = DiagonalBlockMatrix3<S>;
pub type MassMatrixView<'a> = MassMatrix<&'a [f64]>;

pub type Delassus<S = Vec<f64>, I = Vec<usize>> = DSBlockMatrix3<S, I>;
pub type DelassusView<'a> = Delassus<&'a [f64], &'a [usize]>;

pub(crate) type EffectiveMassInv<S = Vec<f64>, I = Vec<usize>> = DSBlockMatrix3<S, I>;
pub(crate) type EffectiveMassInvView<'a> = EffectiveMassInv<&'a [f64], &'a [usize]>;

#[cfg(test)]
mod tests {
    use super::*;
    use approx::*;

    // Verify that converting to contact space and back to physical space produces the same
    // vectors.
    #[test]
    fn contact_physical_space_conversion_test() -> Result<(), crate::Error> {
        use geo::mesh::{topology::*, TriMesh, VertexPositions};
        use reinterpret::*;

        let run = |trimesh: TriMesh<f64>| -> Result<(), crate::Error> {
            let mut normals = vec![geo::math::Vector3::zeros(); trimesh.num_vertices()];
            geo::algo::compute_vertex_area_weighted_normals(
                trimesh.vertex_positions(),
                reinterpret_slice(trimesh.indices.as_slice()),
                &mut normals,
            );

            for n in normals.iter_mut() {
                *n /= n.norm();
            }

            let mut basis = ContactBasis::new();
            basis.update_from_normals(reinterpret_vec(normals));

            let vecs = utils::random_vectors(trimesh.num_vertices());

            // Test euclidean basis
            let contact_vecs = basis.to_tangent_space(reinterpret_vec(vecs.clone()));
            let projected_physical_vecs = basis.from_tangent_space(contact_vecs.clone());
            let projected_contact_vecs =
                basis.to_tangent_space(reinterpret_vec(projected_physical_vecs.clone()));

            for (a, &b) in contact_vecs.into_iter().zip(projected_contact_vecs.iter()) {
                for i in 0..2 {
                    assert_relative_eq!(a[i], b[i]);
                }
            }

            let reprojected_physical_vecs = basis.from_tangent_space(projected_contact_vecs);

            for (a, b) in projected_physical_vecs
                .into_iter()
                .zip(reprojected_physical_vecs.into_iter())
            {
                for i in 0..3 {
                    assert_relative_eq!(a[i], b[i]);
                }
            }

            // Test cylindrical coordinates
            let contact_vecs = basis.to_polar_tangent_space(reinterpret_vec(vecs.clone()));
            let projected_physical_vecs = basis.from_polar_tangent_space(contact_vecs.clone());
            let projected_contact_vecs =
                basis.to_polar_tangent_space(reinterpret_vec(projected_physical_vecs.clone()));

            for (a, &b) in contact_vecs.into_iter().zip(projected_contact_vecs.iter()) {
                assert_relative_eq!(a.radius, b.radius);
                assert_relative_eq!(a.angle, b.angle);
            }

            let reprojected_physical_vecs = basis.from_polar_tangent_space(projected_contact_vecs);

            for (a, b) in projected_physical_vecs
                .into_iter()
                .zip(reprojected_physical_vecs.into_iter())
            {
                for i in 0..3 {
                    assert_relative_eq!(a[i], b[i]);
                }
            }
            Ok(())
        };

        let trimesh = utils::make_sample_octahedron();
        run(trimesh)?;
        let trimesh = utils::make_regular_icosahedron();
        run(trimesh)
    }
}
