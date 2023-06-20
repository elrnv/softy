use crate::attrib_defines::*;
use crate::matrix::*;
use crate::Error;
use crate::Material;
use crate::Real;
use crate::{Mesh, TetMesh};
use geo::{attrib::*, mesh::topology::*, mesh::CellType, ops::Volume};
use tensr::{Matrix3, Vector3};

#[derive(Clone, Debug, PartialEq)]
pub struct VolumeConstraint {
    /// The topology of the surface of a volumetric mesh.
    ///
    /// This is a vector of triplets of indices of mesh vertices. Each
    /// triplet corresponds to a triangle on the surface of the mesh.
    pub surface_topo: Vec<[usize; 3]>,

    /// The volume of the solid at rest. The contraint is equated to this value.
    pub rest_volume: f64,
}

impl VolumeConstraint {
    pub fn new(tetmesh: &TetMesh) -> Self {
        let surface_topo = tetmesh.surface_topo();
        VolumeConstraint {
            surface_topo,
            rest_volume: Self::compute_tetmesh_volume(tetmesh),
        }
    }

    /// Constructs a volume constraint per zone id in a given `Mesh`.
    pub fn try_from_mesh(mesh: &Mesh, materials: &[Material]) -> Result<Vec<Self>, Error> {
        if materials.is_empty() {
            return Ok(vec![]);
        }

        let ref_pos =
            mesh.attrib_as_slice::<RefPosType, CellVertexIndex>(REFERENCE_CELL_VERTEX_POS_ATTRIB)?;

        let mut unique_zones = mesh
            .attrib_clone_into_vec::<VolumeZoneIdType, CellIndex>(VOLUME_ZONE_ID_ATTRIB)
            .unwrap_or_else(|_| vec![0; 1]);
        unique_zones.sort_unstable();
        unique_zones.dedup();
        Ok(unique_zones
            .iter()
            .filter_map(|&zone| {
                let zone_cells: Vec<_> = mesh
                    .cell_iter()
                    .zip(mesh.cell_type_iter())
                    .zip(
                        mesh.attrib_iter::<VolumeZoneIdType, CellIndex>(VOLUME_ZONE_ID_ATTRIB)
                            .unwrap_or_else(|_| Box::new(std::iter::repeat(&0))),
                    )
                    .zip(
                        mesh.attrib_iter::<MaterialIdType, CellIndex>(MATERIAL_ID_ATTRIB)
                            .unwrap_or_else(|_| Box::new(std::iter::repeat(&0))),
                    )
                    .filter_map(|(((cell, cell_type), &cell_zone), &mtl_id)| {
                        if let Material::Solid(mtl) = materials[mtl_id as usize] {
                            if cell_zone == zone
                                && mtl.volume_preservation()
                                && cell.len() == 4
                                && cell_type == CellType::Tetrahedron
                            {
                                return Some([cell[0], cell[1], cell[2], cell[3]]);
                            }
                        }
                        None
                    })
                    .collect();

                if zone_cells.is_empty() {
                    return None;
                }

                let rest_volume = zone_cells
                    .iter()
                    .map(|cell| {
                        let tet = [
                            ref_pos[cell[0]],
                            ref_pos[cell[1]],
                            ref_pos[cell[2]],
                            ref_pos[cell[3]],
                        ];
                        crate::fem::ref_tet(&tet).signed_volume()
                    })
                    .sum();

                let surface_topo = TetMesh::surface_topo_from_tets(zone_cells.iter());
                Some(VolumeConstraint {
                    surface_topo,
                    rest_volume,
                })
            })
            .collect())
    }

    /// Computes the volume of a tetmesh given its vertex reference positions (one per tet vertex).
    pub fn compute_volume(ref_pos: &[RefPosType]) -> f64 {
        ref_pos
            .chunks_exact(4)
            .map(|tet| crate::fem::ref_tet(tet).volume())
            .sum()
    }

    pub fn compute_tetmesh_volume(tetmesh: &TetMesh) -> f64 {
        let ref_pos = tetmesh
            .attrib_as_slice::<RefPosType, CellVertexIndex>(REFERENCE_CELL_VERTEX_POS_ATTRIB)
            .unwrap();

        Self::compute_volume(ref_pos)
    }

    #[inline]
    pub fn constraint_size(&self) -> usize {
        1
    }

    #[inline]
    pub fn constraint_bounds<T: Real>(&self) -> (Vec<T>, Vec<T>) {
        (vec![T::zero()], vec![T::zero()])
    }

    pub fn constraint<T: Real>(&mut self, _x0: &[T], x1: &[T], value: &mut [T]) {
        debug_assert_eq!(value.len(), self.constraint_size());
        let pos1: &[[T; 3]] = bytemuck::cast_slice(x1);
        let mut total_volume = T::zero();
        for tri in self.surface_topo.iter() {
            let p = Matrix3::new(tri_at(pos1, tri));
            let signed_volume = p[0].dot(p[1].cross(p[2]));
            total_volume += signed_volume;
        }
        value[0] = total_volume - T::from(6.0 * self.rest_volume).unwrap();
    }

    #[inline]
    pub fn constraint_jacobian_size(&self) -> usize {
        3 * 3 * self.surface_topo.len()
    }
    pub fn constraint_jacobian_values<T: Real>(
        &mut self,
        x0: &[T],
        x1: &[T],
        values: &mut [T],
    ) -> Result<(), Error> {
        debug_assert_eq!(values.len(), self.constraint_jacobian_size());
        for (out, val) in values
            .iter_mut()
            .zip(self.constraint_jacobian_values_iter(x0, x1))
        {
            *out = val;
        }
        Ok(())
    }

    #[inline]
    pub fn constraint_hessian_size(&self) -> usize {
        6 * 3 * self.surface_topo.len()
    }
    #[inline]
    pub fn num_hessian_diagonal_nnz(&self) -> usize {
        0
    }
    pub fn constraint_hessian_values<T: Real>(
        &mut self,
        x0: &[T],
        x1: &[T],
        lambda: &[T],
        scale: T,
        values: &mut [T],
    ) -> Result<(), Error> {
        debug_assert_eq!(values.len(), self.constraint_hessian_size());
        for (out, val) in values
            .iter_mut()
            .zip(self.constraint_hessian_values_iter(x0, x1, lambda))
        {
            *out = T::from(val).unwrap() * scale;
        }
        Ok(())
    }
}

/// A utility function to index a slice using three indices, creating a new array of 3
/// corresponding entries of the slice.
fn tri_at<T: Copy>(slice: &[T], tri: &[usize; 3]) -> [T; 3] {
    [slice[tri[0]], slice[tri[1]], slice[tri[2]]]
}

impl VolumeConstraint {
    /// Compute the values of the constraint Jacobian.
    fn constraint_jacobian_values_iter<'a, T: Real>(
        &'a self,
        _x0: &'a [T],
        x1: &'a [T],
    ) -> impl Iterator<Item = T> + 'a {
        let pos1: &[[T; 3]] = bytemuck::cast_slice(x1);

        self.surface_topo.iter().flat_map(move |tri| {
            let p = Matrix3::new(tri_at(pos1, tri));
            let c = [p[1].cross(p[2]), p[2].cross(p[0]), p[0].cross(p[1])];

            (0..3).flat_map(move |vi| (0..3).map(move |j| c[vi][j]))
        })
    }
}

// TODO: move to geo::math
/// Convert a 3 dimensional vector `v = (x,y,z)` into a skew matrix `[v]×` given by
/// ```verbatim
/// ⎡ 0 -z  y⎤
/// ⎢ z  0 -x⎥
/// ⎣-y  x  0⎦
/// ```
#[inline]
fn skew<T: Real>(x: Vector3<T>) -> Matrix3<T> {
    Matrix3::new([
        [T::zero(), x[2], -x[1]],
        [-x[2], T::zero(), x[0]],
        [x[1], -x[0], T::zero()],
    ])
}

impl VolumeConstraint {
    /// A generic Hessian element iterator. This is used to implement iterators over indices and
    /// values of the sparse Hessian matrix enetries.
    /// Note: it is an attempt to code reuse. Ideally we should use generators here.
    fn constraint_hessian_iter(
        tri: &[usize; 3],
    ) -> impl Iterator<Item = ((usize, usize), (usize, usize), usize, usize)> + '_ {
        (0..3).flat_map(move |vi| {
            let col_v = tri[vi];
            let row_v = move |off| tri[(vi + off) % 3];
            (1..=2)
                .filter(move |&off| row_v(off) > col_v)
                .flat_map(move |off| {
                    (0..3).flat_map(move |c| {
                        (0..3)
                            .filter(move |&r| r != c)
                            .map(move |r| ((row_v(off), col_v), (r, c), vi, off))
                    })
                })
        })
    }

    pub fn constraint_hessian_indices_iter(&self) -> impl Iterator<Item = MatrixElementIndex> + '_ {
        self.surface_topo.iter().flat_map(move |tri| {
            Self::constraint_hessian_iter(tri).map(|((row_v, col_v), (r, c), _, _)| {
                MatrixElementIndex {
                    row: 3 * row_v + r,
                    col: 3 * col_v + c,
                }
            })
        })
    }

    pub fn constraint_hessian_values_iter<'a, T: Real>(
        &'a self,
        _x0: &'a [T],
        x1: &'a [T],
        lambda: &'a [T],
    ) -> impl Iterator<Item = T> + 'a {
        let pos1: &[[T; 3]] = bytemuck::cast_slice(x1);

        self.surface_topo.iter().flat_map(move |tri| {
            let p = Matrix3::new(tri_at(pos1, tri));
            let local_hess = [skew(p[0]), skew(p[1]), skew(p[2])];
            Self::constraint_hessian_iter(tri).map(move |(_, (r, c), vi, off)| {
                let vjn = (vi + off + off) % 3;
                let factor = if off == 1 { T::one() } else { -T::one() };
                factor * lambda[0] * local_hess[vjn][c][r]
            })
        })
    }
}
