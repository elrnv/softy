use crate::attrib_defines::*;
use crate::matrix::*;
use crate::Error;
use crate::Material;
use crate::Real;
use crate::{Mesh, TetMesh};
use geo::index::CheckedIndex;
use geo::VertexPositions;
use geo::{attrib::*, mesh::topology::*, mesh::CellType, ops::Volume, Index};
use rayon::iter::Either;
use tensr::{Matrix3, Vector3};
use crate::nl_fem::ZoneParams;

/// One standard atmospheric pressure in `Pa`.
const PRESSURE_ATM: f64 = 101325.0;

#[derive(Clone, Debug, PartialEq)]
pub struct VolumeChangePenalty {
    /// The topology of the surface of a volumetric mesh.
    ///
    /// This is a vector of triplets of indices of mesh vertices. Each
    /// triplet corresponds to a triangle on the surface of the mesh.
    pub surface_topo: Vec<[usize; 3]>,

    /// The volume of the solid at rest. Volumes different than this value are penalized.
    pub rest_volume: f64,

    /// Rest pressurization (in `atm`) for this penalty.
    pub pressurization: f32,

    /// Isothermal compression coefficient (in `atm^-1`).
    ///
    /// Smaller values will cause stronger restorative forces.
    /// Water at room temperature has a compression coefficient of `4.6 x 10^-5 atm^{-1}`.
    /// The compression coefficient for air is around `1 atm^{-1}`.
    pub compression: f32,

    /// Approximate the Hessian using a sparse matrix.
    ///
    /// The true Hessian for the volume change penalty is dense, however under small enough
    /// time steps and small compression ratios, the Hessian can be sparsely approximated.
    /// In zones with a large number of triangles, the dense Hessian can be very expensive to compute.
    pub hessian_approximation: bool,
}

impl VolumeChangePenalty {
    pub fn new(tetmesh: &TetMesh) -> Self {
        let surface_topo = tetmesh.surface_topo();
        VolumeChangePenalty {
            surface_topo,
            rest_volume: Self::compute_tetmesh_volume(tetmesh),
            pressurization: 1.0,
            compression: 1.0,
            hessian_approximation: true,
        }
    }

    /// Constructs a volume change penalty per zone id in a given `Mesh`.
    pub fn try_from_mesh(
        mesh: &Mesh,
        materials: &[Material],
        zone_params: &ZoneParams,
    ) -> Result<Vec<Self>, Error> {
        if materials.is_empty() {
            return Ok(vec![]);
        }

        let ref_pos =
            mesh.attrib_as_slice::<RefPosType, CellVertexIndex>(REFERENCE_CELL_VERTEX_POS_ATTRIB)?;

        let pos = mesh.vertex_positions();

        let mut unique_zones = mesh
            .attrib_clone_into_vec::<VolumeZoneIdType, CellIndex>(VOLUME_ZONE_ID_ATTRIB)
            .unwrap_or_else(|_| vec![0]);
        unique_zones.sort_unstable();
        unique_zones.dedup();
        Ok(unique_zones
            .iter()
            .filter_map(|&zone| {
                if zone == 0 {
                    return None; // Skip zone 0 as the deactivated zone.
                }

                // Use ref_pos to compute the rest volume for tetrahedral cells.
                // The remaining volume is computed from the current positions.
                let mut rest_volume = 0.0;

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
                    .enumerate()
                    .filter_map(|(cell_idx, (((cell, cell_type), &cell_zone), &mtl_id))| {
                        match materials[mtl_id as usize] {
                            Material::Solid(_) => {
                                if cell_zone == zone
                                    && cell.len() == 4
                                    && cell_type == CellType::Tetrahedron
                                {
                                    // Compute volume
                                    let tet = [
                                        ref_pos
                                            [mesh.cell_vertex(cell_idx, 0).unwrap().into_inner()],
                                        ref_pos
                                            [mesh.cell_vertex(cell_idx, 1).unwrap().into_inner()],
                                        ref_pos
                                            [mesh.cell_vertex(cell_idx, 2).unwrap().into_inner()],
                                        ref_pos
                                            [mesh.cell_vertex(cell_idx, 3).unwrap().into_inner()],
                                    ];
                                    rest_volume += crate::fem::ref_tet(&tet).signed_volume();
                                    // Return 4 vertex indices representing the tetrahedron cell.
                                    return Some((
                                        Index::from(cell[0]),
                                        [cell[1], cell[2], cell[3]],
                                    ));
                                }
                            }
                            Material::SoftShell(_) => {
                                if cell_zone == zone
                                    && cell.len() == 3
                                    && cell_type == CellType::Triangle
                                {
                                    // Compute volume
                                    let tet = [[0.0; 3], pos[cell[0]], pos[cell[1]], pos[cell[2]]];
                                    rest_volume -= crate::fem::ref_tet(&tet).signed_volume();
                                    return Some((Index::invalid(), [cell[0], cell[1], cell[2]]));
                                }
                            }
                            _ => {}
                        }
                        None
                    })
                    .collect();

                if zone_cells.is_empty() {
                    return None;
                }

                let mut surface_topo = TetMesh::surface_topo_from_tets(
                    zone_cells
                        .iter()
                        .filter_map(|(vtx0, [vtx1, vtx2, vtx3])| {
                            vtx0.into_option().map(|vtx0| [vtx0, *vtx1, *vtx2, *vtx3])
                        })
                        .collect::<Vec<_>>()
                        .iter(),
                );
                surface_topo.extend(zone_cells.iter().filter_map(|(vtx0, [vtx1, vtx2, vtx3])| {
                    if !vtx0.is_valid() {
                        Some([*vtx1, *vtx2, *vtx3])
                    } else {
                        None
                    }
                }));

                Some(VolumeChangePenalty {
                    surface_topo,
                    rest_volume: rest_volume.into(),
                    pressurization: *zone_params.zone_pressurizations.get(zone as usize - 1).unwrap_or(&1.0),
                    compression: *zone_params.compression_coefficients
                        .get(zone as usize - 1)
                        .unwrap_or(&1.0),
                    hessian_approximation: *zone_params.hessian_approximation
                        .get(zone as usize - 1)
                        .unwrap_or(&true),
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

    /// Compute the signed volume of the stored topology according to the given set of positions.
    pub fn compute_signed_volume<T: Real>(&self, pos: &[[T; 3]]) -> T {
        Self::compute_signed_volume_from_topo(&self.surface_topo, pos)
    }

    /// Compute the signed volume of the stored topology according to the given set of positions.
    pub fn compute_signed_volume_from_topo<T: Real>(topo: &[[usize; 3]], pos: &[[T; 3]]) -> T {
        topo.iter()
            .map(|tri| {
                let p = Matrix3::new(tri_at(pos, tri));
                p[0].dot(p[1].cross(p[2]))
            })
            .sum::<T>()
            / T::from(6.0).unwrap()
    }

    #[inline]
    pub fn penalty_size(&self) -> usize {
        1
    }

    #[inline]
    pub fn penalty_bounds<T: Real>(&self) -> (Vec<T>, Vec<T>) {
        (vec![T::zero()], vec![T::zero()])
    }

    /// Computes a quadratic approximation of an isothermic pressurized fluid energy.
    ///
    /// W = Pa ( V0 ( b ( exp(a) - 1 ) - exp(a) ) + (1/k)((1-a)exp(a) - 1)) - (V - V0 exp(a))^2/(2kV0exp(a))
    /// where Pa is atmospheric pressure, V0 is rest volume, k is coefficient of isothermal compression, b is initial pressurization and V is current volume.
    #[inline]
    pub fn compute_penalty<T: Real>(&self, _x0: &[T], x1: &[T]) -> T {
        let pos1: &[[T; 3]] = bytemuck::cast_slice(x1);
        let v = self.compute_signed_volume(pos1);
        let v0 = T::from(self.rest_volume).unwrap();
        let b = T::from(self.pressurization).unwrap();
        let k = T::from(self.compression).unwrap();
        let p0 = T::from(PRESSURE_ATM).unwrap();
        let dv = (v / (b * v0)) - T::one();
        let dv2 = dv * dv;
        p0 * v0 * (dv2 * b / (k * T::from(2.0).unwrap()) + T::one())
    }

    pub fn penalty<T: Real>(&mut self, x0: &[T], x1: &[T], value: &mut [T]) {
        debug_assert_eq!(value.len(), self.penalty_size());
        value[0] = self.compute_penalty(x0, x1);
    }

    pub fn subtract_pressure_force<T: Real, S: Real>(&self, x0: &[S], x1: &[T], force: &mut [T]) {
        for (MatrixElementIndex { col, .. }, val) in self
            .penalty_jacobian_indices_iter()
            .zip(self.penalty_jacobian_values_iter(x0, x1))
        {
            // Adding potential energy gradient is equivalent to subtracting force here.
            force[col] += val;
        }
    }

    #[inline]
    pub fn penalty_jacobian_size(&self) -> usize {
        3 * 3 * self.surface_topo.len()
    }
    pub fn penalty_jacobian_values<T: Real>(
        &mut self,
        x0: &[T],
        x1: &[T],
        values: &mut [T],
    ) -> Result<(), Error> {
        debug_assert_eq!(values.len(), self.penalty_jacobian_size());
        for (out, val) in values
            .iter_mut()
            .zip(self.penalty_jacobian_values_iter(x0, x1))
        {
            *out = val;
        }
        Ok(())
    }

    #[inline]
    pub fn penalty_hessian_size(&self) -> usize {
        let mut size = 6 * 2 * 3 * self.surface_topo.len();
        if !self.hessian_approximation {
            size += self.penalty_jacobian_size() * self.penalty_jacobian_size();
        }
        size
    }

    #[inline]
    pub fn num_hessian_diagonal_nnz(&self) -> usize {
        0
    }
    pub fn penalty_hessian_values<T: Real>(
        &mut self,
        x0: &[T],
        x1: &[T],
        lambda: &[T],
        scale: T,
        values: &mut [T],
    ) -> Result<(), Error> {
        debug_assert_eq!(values.len(), self.penalty_hessian_size());
        for (out, val) in values
            .iter_mut()
            .zip(self.penalty_hessian_values_iter(x0, x1, lambda))
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

impl VolumeChangePenalty {
    /// Compute the indices of the sparse matrix entries of the constraint Jacobian.
    fn penalty_jacobian_indices_iter<'a>(
        &'a self,
    ) -> impl Iterator<Item = MatrixElementIndex> + 'a {
        self.surface_topo.iter().flat_map(|tri| {
            (0..3).flat_map(move |vi| {
                (0..3).map(move |j| MatrixElementIndex {
                    row: 0,
                    col: 3 * tri[vi] + j,
                })
            })
        })
    }

    /// Compute the values of the constraint Jacobian.
    fn penalty_jacobian_values_iter<'a, T: Real, S: Real>(
        &'a self,
        _x0: &'a [S],
        x1: &'a [T],
    ) -> impl Iterator<Item = T> + 'a {
        let pos1: &[[T; 3]] = bytemuck::cast_slice(x1);
        let v = self.compute_signed_volume(pos1);
        let v0 = T::from(self.rest_volume).unwrap();
        let b = T::from(self.pressurization).unwrap();
        let k = T::from(self.compression).unwrap();
        let p0 = T::from(PRESSURE_ATM).unwrap();
        let dv = (v / (b * v0)) - T::one();
        let factor = p0 * dv / (k * T::from(6.0).unwrap());

        self.surface_topo.iter().flat_map(move |tri| {
            let p = Matrix3::new(tri_at(pos1, tri));
            let c = [p[1].cross(p[2]), p[2].cross(p[0]), p[0].cross(p[1])];

            (0..3).flat_map(move |vi| (0..3).map(move |j| c[vi][j] * factor))
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

impl VolumeChangePenalty {
    /// A generic Hessian element iterator. This is used to implement iterators over indices and
    /// values of the sparse Hessian matrix enetries.
    /// Note: it is an attempt to code reuse. Ideally we should use generators here.
    fn penalty_hessian_iter(
        tri: &[usize; 3],
    ) -> impl Iterator<Item = ((usize, usize), (usize, usize), usize, usize)> + '_ {
        (0..3).flat_map(move |vi| {
            let col_v = tri[vi];
            let row_v = move |off| tri[(vi + off) % 3];
            (1..=2)
                // .filter(move |&off| row_v(off) > col_v)
                .flat_map(move |off| {
                    (0..3).flat_map(move |c| {
                        (0..3)
                            .filter(move |&r| r != c)
                            .map(move |r| ((row_v(off), col_v), (r, c), vi, off))
                    })
                })
        })
    }

    pub fn penalty_hessian_indices_iter(&self) -> impl Iterator<Item = MatrixElementIndex> + '_ {
        self.surface_topo.iter().flat_map(move |tri| {
            Self::penalty_hessian_iter(tri)
                .map(|((row_v, col_v), (r, c), _, _)| MatrixElementIndex {
                    row: 3 * row_v + r,
                    col: 3 * col_v + c,
                })
                .chain(if !self.hessian_approximation {
                    Either::Left(self.surface_topo.iter().flat_map(move |tri_rhs| {
                        (0..3).flat_map(move |vi| {
                            let row_v = tri[vi];
                            (0..3).flat_map(move |vj| {
                                let col_v = tri_rhs[vj];
                                (0..3).flat_map(move |i| {
                                    (0..3).map(move |j| MatrixElementIndex {
                                        row: 3 * row_v + i,
                                        col: 3 * col_v + j,
                                    })
                                })
                            })
                        })
                    }))
                } else {
                    Either::Right(std::iter::empty())
                })
        })
    }

    pub fn penalty_hessian_values_iter<'a, T: Real>(
        &'a self,
        _x0: &'a [T],
        x1: &'a [T],
        lambda: &'a [T],
    ) -> impl Iterator<Item = T> + 'a {
        let pos1: &[[T; 3]] = bytemuck::cast_slice(x1);
        let v = self.compute_signed_volume(pos1);
        let v0 = T::from(self.rest_volume).unwrap();
        let b = T::from(self.pressurization).unwrap();
        let k = T::from(self.compression).unwrap();
        let p0 = T::from(PRESSURE_ATM).unwrap();
        let dv = (v / (b * v0)) - T::one();
        let factor = p0 / (k * T::from(6.0).unwrap());

        self.surface_topo.iter().flat_map(move |tri| {
            let p = Matrix3::new(tri_at(pos1, tri));
            let lhs_grad = [p[1].cross(p[2]), p[2].cross(p[0]), p[0].cross(p[1])];
            let local_hess = [skew(p[0]), skew(p[1]), skew(p[2])];
            Self::penalty_hessian_iter(tri)
                .map(move |((_, _), (r, c), vi, off)| {
                    let vjn = (vi + off + off) % 3;
                    let factor = if off == 1 { factor } else { -factor };
                    factor * lambda[0] * local_hess[vjn][c][r] * dv
                })
                .chain(if !self.hessian_approximation {
                    Either::Left(self.surface_topo.iter().flat_map(move |tri_rhs| {
                        let p = Matrix3::new(tri_at(pos1, tri_rhs));
                        let rhs_grad = [p[1].cross(p[2]), p[2].cross(p[0]), p[0].cross(p[1])];
                        (0..3).flat_map(move |vi| {
                            (0..3).flat_map(move |vj| {
                                (0..3).flat_map(move |i| {
                                    (0..3).map(move |j| {
                                        lhs_grad[vi][i] * rhs_grad[vj][j] * lambda[0] * factor
                                            / (b * v0 * T::from(6.0).unwrap())
                                    })
                                })
                            })
                        })
                    }))
                } else {
                    Either::Right(std::iter::empty())
                })
        })
    }
}
