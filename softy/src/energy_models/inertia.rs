use num_traits::FromPrimitive;
//use rayon::prelude::*;
use reinterpret::*;

use geo::mesh::{Attrib, topology::*};
use geo::prim::{Tetrahedron, Triangle};
use utils::soap::{AsTensor, IntoTensor, Real, Vector3};
use utils::zip;

use crate::attrib_defines::*;
use crate::energy::*;
use crate::matrix::*;
use crate::objects::shell::*;
use crate::objects::solid::*;

const NUM_HESSIAN_TRIPLETS_PER_TET: usize = 12;
const NUM_HESSIAN_TRIPLETS_PER_TRI: usize = 9;

pub trait Inertia<'a, E> {
    fn inertia(&'a self) -> E;
}

pub struct TetMeshInertia<'a>(pub &'a TetMeshSolid);

impl<T: Real> Energy<T> for TetMeshInertia<'_> {
    #[allow(non_snake_case)]
    fn energy(&self, v0: &[T], v1: &[T]) -> T {
        let tetmesh = &self.0.tetmesh;

        let vel0: &[Vector3<T>] = reinterpret_slice(v0);
        let vel1: &[Vector3<T>] = reinterpret_slice(v1);

        zip!(
            tetmesh
                .attrib_iter::<RefVolType, CellIndex>(REFERENCE_VOLUME_ATTRIB)
                .unwrap(),
            tetmesh
                .attrib_iter::<DensityType, CellIndex>(DENSITY_ATTRIB)
                .unwrap()
                .map(|&x| f64::from(x)),
            tetmesh.cell_iter(),
        )
        .map(|(&vol, density, cell)| {
            let tet_v0 = Tetrahedron::from_indexed_slice(cell, vel0);
            let tet_v1 = Tetrahedron::from_indexed_slice(cell, vel1);
            let tet_dv = tet_v1 - tet_v0;

            T::from(0.5).unwrap() * {
                let dvTdv: T = tet_dv
                    .into_array()
                    .iter()
                    .map(|&dv| Vector3::new(dv).norm_squared())
                    .sum();
                // momentum
                T::from(0.25 * vol * density).unwrap() * dvTdv
            }
        })
        .sum()
    }
}

impl<T: Real> EnergyGradient<T> for TetMeshInertia<'_> {
    #[allow(non_snake_case)]
    fn add_energy_gradient(&self, v0: &[T], v1: &[T], grad_f: &mut [T]) {
        let tetmesh = &self.0.tetmesh;

        let vel0: &[Vector3<T>] = reinterpret_slice(v0);
        let vel1: &[Vector3<T>] = reinterpret_slice(v1);

        debug_assert_eq!(grad_f.len(), v0.len());

        let gradient: &mut [Vector3<T>] = reinterpret_mut_slice(grad_f);

        // Transfer forces from cell-vertices to vertices themselves
        for (&vol, density, cell) in zip!(
            tetmesh
                .attrib_iter::<RefVolType, CellIndex>(REFERENCE_VOLUME_ATTRIB)
                .unwrap(),
            tetmesh
                .attrib_iter::<DensityType, CellIndex>(DENSITY_ATTRIB)
                .unwrap()
                .map(|&x| f64::from(x)),
            tetmesh.cell_iter()
        ) {
            let tet_v0 = Tetrahedron::from_indexed_slice(cell, vel0);
            let tet_v1 = Tetrahedron::from_indexed_slice(cell, vel1);
            let tet_dv = (tet_v1 - tet_v0).into_array();

            for i in 0..4 {
                gradient[cell[i]] +=
                    Vector3::new(tet_dv[i]) * (T::from(0.25 * vol * density).unwrap());
            }
        }
    }
}

impl EnergyHessianTopology for TetMeshInertia<'_> {
    fn energy_hessian_size(&self) -> usize {
        self.0.tetmesh.num_cells() * NUM_HESSIAN_TRIPLETS_PER_TET
    }

    fn energy_hessian_rows_cols_offset<I: FromPrimitive + Send>(
        &self,
        offset: MatrixElementIndex,
        rows: &mut [I],
        cols: &mut [I],
    ) {
        assert_eq!(rows.len(), self.energy_hessian_size());
        assert_eq!(cols.len(), self.energy_hessian_size());

        let tetmesh = &self.0.tetmesh;

        // Break up the hessian triplets into chunks of elements for each tet.
        let hess_row_chunks: &mut [[I; NUM_HESSIAN_TRIPLETS_PER_TET]] =
            reinterpret_mut_slice(rows);
        let hess_col_chunks: &mut [[I; NUM_HESSIAN_TRIPLETS_PER_TET]] =
            reinterpret_mut_slice(cols);

        // The momentum hessian is a diagonal matrix.
        hess_row_chunks
            .iter_mut()
            .zip(hess_col_chunks.iter_mut())
            .zip(tetmesh.cells().iter())
            .for_each(|((tet_hess_rows, tet_hess_cols), cell)| {
                for vi in 0..4 {
                    // vertex index
                    for j in 0..3 {
                        // vector component
                        tet_hess_rows[3 * vi + j] =
                            I::from_usize(3 * cell[vi] + j + offset.row).unwrap();
                        tet_hess_cols[3 * vi + j] =
                            I::from_usize(3 * cell[vi] + j + offset.col).unwrap();
                    }
                }
            });
    }

    fn energy_hessian_indices_offset(
        &self,
        offset: MatrixElementIndex,
        indices: &mut [MatrixElementIndex],
    ) {
        assert_eq!(indices.len(), self.energy_hessian_size());

        let tetmesh = &self.0.tetmesh;

        // Break up the hessian triplets into chunks of elements for each tet.
        let hess_chunks: &mut [[MatrixElementIndex; NUM_HESSIAN_TRIPLETS_PER_TET]] =
            reinterpret_mut_slice(indices);

        // The momentum hessian is a diagonal matrix.
        hess_chunks
            .iter_mut()
            .zip(tetmesh.cells().iter())
            .for_each(|(tet_hess, cell)| {
                for vi in 0..4 {
                    // vertex index
                    for j in 0..3 {
                        // vector component
                        tet_hess[3 * vi + j] = MatrixElementIndex {
                            row: 3 * cell[vi] + j + offset.row,
                            col: 3 * cell[vi] + j + offset.col,
                        };
                    }
                }
            });
    }
}

impl<T: Real> EnergyHessian<T> for TetMeshInertia<'_> {
    #[allow(non_snake_case)]
    fn energy_hessian_values(
        &self,
        _v0: &[T],
        _v1: &[T],
        scale: T,
        values: &mut [T],
    ) {
        assert_eq!(values.len(), self.energy_hessian_size());

        let TetMeshInertia(ref solid) = *self;

        // Break up the hessian triplets into chunks of elements for each tet.
        let hess_chunks: &mut [[T; NUM_HESSIAN_TRIPLETS_PER_TET]] =
            reinterpret_mut_slice(values);

        let vol_iter = solid
            .tetmesh
            .attrib_iter::<RefVolType, CellIndex>(REFERENCE_VOLUME_ATTRIB)
            .unwrap();

        let density_iter = solid
            .tetmesh
            .attrib_iter::<DensityType, CellIndex>(DENSITY_ATTRIB)
            .unwrap()
            .map(|&x| f64::from(x));

        // The momentum hessian is a diagonal matrix.
        hess_chunks
            .iter_mut()
            .zip(vol_iter.zip(density_iter))
            .for_each(|(tet_hess, (&vol, density))| {
                for vi in 0..4 {
                    // vertex index
                    for j in 0..3 {
                        // vector component
                        tet_hess[3 * vi + j] = T::from(0.25 * vol * density).unwrap() * scale;
                    }
                }
            });
    }
}

pub struct TriMeshInertia<'a>(pub &'a TriMeshShell);

impl<T: Real> Energy<T> for TriMeshInertia<'_> {
    #[allow(non_snake_case)]
    fn energy(&self, v0: &[T], v1: &[T]) -> T {
        let trimesh = &self.0.trimesh;

        let vel0: &[Vector3<T>] = reinterpret_slice(v0);
        let vel1: &[Vector3<T>] = reinterpret_slice(v1);

        zip!(
            trimesh
                .attrib_iter::<RefAreaType, FaceIndex>(REFERENCE_AREA_ATTRIB)
                .unwrap(),
            trimesh
                .attrib_iter::<DensityType, FaceIndex>(DENSITY_ATTRIB)
                .unwrap()
                .map(|&x| f64::from(x)),
            trimesh.face_iter(),
        )
        .map(|(&area, density, face)| {
            let tri_v0 = Triangle::from_indexed_slice(face, vel0);
            let tri_v1 = Triangle::from_indexed_slice(face, vel1);
            let dv = tri_v1.into_array().into_tensor() - tri_v0.into_array().into_tensor();

            let third = 1.0 / 3.0;
            T::from(0.5).unwrap() * {
                let dvTdv: T = dv
                    .map(|dv| dv.norm_squared().into_tensor())
                    .sum();
                // momentum
                T::from(third * area * density).unwrap() * dvTdv
            }
        })
        .sum()
    }
}

impl<T: Real> EnergyGradient<T> for TriMeshInertia<'_> {
    #[allow(non_snake_case)]
    fn add_energy_gradient(&self, v0: &[T], v1: &[T], grad_f: &mut [T]) {
        let trimesh = &self.0.trimesh;

        let vel0: &[Vector3<T>] = reinterpret_slice(v0);
        let vel1: &[Vector3<T>] = reinterpret_slice(v1);

        debug_assert_eq!(grad_f.len(), v0.len());

        let gradient: &mut [Vector3<T>] = reinterpret_mut_slice(grad_f);

        // Transfer forces from cell-vertices to vertices themselves
        for (&area, density, face) in zip!(
            trimesh
                .attrib_iter::<RefAreaType, FaceIndex>(REFERENCE_AREA_ATTRIB)
                .unwrap(),
            trimesh
                .attrib_iter::<DensityType, FaceIndex>(DENSITY_ATTRIB)
                .unwrap()
                .map(|&x| f64::from(x)),
            trimesh.face_iter()
        ) {
            let tri_v0 = Triangle::from_indexed_slice(face, vel0);
            let tri_v1 = Triangle::from_indexed_slice(face, vel1);
            let dv = *tri_v1.as_array().as_tensor() - *tri_v0.as_array().as_tensor();

            let third = 1.0 / 3.0;
            for i in 0..3 {
                gradient[face[i]] +=
                    dv[i] * (T::from(third * area * density).unwrap());
            }
        }
    }
}

impl EnergyHessianTopology for TriMeshInertia<'_> {
    fn energy_hessian_size(&self) -> usize {
        self.0.trimesh.num_faces() * NUM_HESSIAN_TRIPLETS_PER_TRI
    }

    fn energy_hessian_rows_cols_offset<I: FromPrimitive + Send>(
        &self,
        offset: MatrixElementIndex,
        rows: &mut [I],
        cols: &mut [I],
    ) {
        assert_eq!(rows.len(), self.energy_hessian_size());
        assert_eq!(cols.len(), self.energy_hessian_size());

        let trimesh = &self.0.trimesh;

        // Break up the hessian triplets into chunks of elements for each triangle.
        let hess_row_chunks: &mut [[I; NUM_HESSIAN_TRIPLETS_PER_TRI]] =
            reinterpret_mut_slice(rows);
        let hess_col_chunks: &mut [[I; NUM_HESSIAN_TRIPLETS_PER_TRI]] =
            reinterpret_mut_slice(cols);

        // The momentum hessian is a diagonal matrix.
        hess_row_chunks
            .iter_mut()
            .zip(hess_col_chunks.iter_mut())
            .zip(trimesh.faces().iter())
            .for_each(|((tri_hess_rows, tri_hess_cols), cell)| {
                for vi in 0..3 {
                    // vertex index
                    for j in 0..3 {
                        // vector component
                        tri_hess_rows[3 * vi + j] =
                            I::from_usize(3 * cell[vi] + j + offset.row).unwrap();
                        tri_hess_cols[3 * vi + j] =
                            I::from_usize(3 * cell[vi] + j + offset.col).unwrap();
                    }
                }
            });
    }

    fn energy_hessian_indices_offset(
        &self,
        offset: MatrixElementIndex,
        indices: &mut [MatrixElementIndex],
    ) {
        assert_eq!(indices.len(), self.energy_hessian_size());

        let trimesh = &self.0.trimesh;

        // Break up the hessian triplets into chunks of elements for each triangle.
        let hess_chunks: &mut [[MatrixElementIndex; NUM_HESSIAN_TRIPLETS_PER_TRI]] =
            reinterpret_mut_slice(indices);

        // The momentum hessian is a diagonal matrix.
        hess_chunks
            .iter_mut()
            .zip(trimesh.faces().iter())
            .for_each(|(tri_hess, face)| {
                for vi in 0..3 {
                    // vertex index
                    for j in 0..3 {
                        // vector component
                        tri_hess[3 * vi + j] = MatrixElementIndex {
                            row: 3 * face[vi] + j + offset.row,
                            col: 3 * face[vi] + j + offset.col,
                        };
                    }
                }
            });
    }
}

impl<T: Real> EnergyHessian<T> for TriMeshInertia<'_> {
    #[allow(non_snake_case)]
    fn energy_hessian_values(
        &self,
        _v0: &[T],
        _v1: &[T],
        scale: T,
        values: &mut [T],
    ) {
        assert_eq!(values.len(), self.energy_hessian_size());

        let TriMeshInertia(ref shell) = *self;

        // Break up the hessian triplets into chunks of elements for each triangle.
        let hess_chunks: &mut [[T; NUM_HESSIAN_TRIPLETS_PER_TRI]] =
            reinterpret_mut_slice(values);

        let vol_iter = shell
            .trimesh
            .attrib_iter::<RefAreaType, FaceIndex>(REFERENCE_AREA_ATTRIB)
            .unwrap();

        let density_iter = shell
            .trimesh
            .attrib_iter::<DensityType, FaceIndex>(DENSITY_ATTRIB)
            .unwrap()
            .map(|&x| f64::from(x));

        let third = 1.0 / 3.0;
        // The momentum hessian is a diagonal matrix.
        hess_chunks
            .iter_mut()
            .zip(vol_iter.zip(density_iter))
            .for_each(|(tri_hess, (&area, density))| {
                for vi in 0..3 {
                    // vertex index
                    for j in 0..3 {
                        // vector component
                        tri_hess[3 * vi + j] = T::from(third * area * density).unwrap() * scale;
                    }
                }
            });
    }
}

#[cfg(test)]
mod tests {
    use geo::mesh::VertexPositions;

    use crate::energy_models::test_utils::*;
    use crate::fem::SolverBuilder;
    use crate::objects::{Object, material::*};

    use super::*;

    mod solid {
        use super::*;

        fn solid_material() -> SolidMaterial {
            SolidMaterial::new(0).with_density(1000.0)
        }

        fn test_solids() -> Vec<TetMeshSolid> {
            let material = solid_material();

            test_tetmeshes()
                .into_iter()
                .map(|mut tetmesh| {
                    // Prepare attributes relevant for elasticity computations.
                    SolverBuilder::prepare_deformable_tetmesh_attributes(&mut tetmesh).unwrap();
                    let mut solid = TetMeshSolid::new(tetmesh, material);
                    solid.init_density_attribute().unwrap();
                    solid
                })
                .collect()
        }

        fn build_energies(solids: &[TetMeshSolid]) -> Vec<(TetMeshInertia, Vec<[f64; 3]>)> {
            solids
                .iter()
                .map(|solid| (solid.inertia(), solid.tetmesh.vertex_positions().to_vec()))
                .collect()
        }

        #[test]
        fn gradient() {
            let solids = test_solids();
            gradient_tester(build_energies(&solids), EnergyType::Velocity);
        }

        #[test]
        fn hessian() {
            let solids = test_solids();
            hessian_tester(build_energies(&solids), EnergyType::Velocity);
        }
    }

    mod shell {
        use super::*;

        fn shell_material() -> ShellMaterial {
            ShellMaterial::new(0).with_density(1000.0)
        }

        fn test_shells() -> Vec<TriMeshShell> {
            let material = shell_material();

            test_trimeshes()
                .into_iter()
                .map(|trimesh| {
                    // Prepare attributes relevant for elasticity computations.
                    let mut shell = TriMeshShell::new(trimesh, material);
                    shell.init_deformable_attributes().unwrap();
                    shell.init_density_attribute().unwrap();
                    shell
                })
                .collect()
        }

        fn build_energies(shells: &[TriMeshShell]) -> Vec<(TriMeshInertia, Vec<[f64; 3]>)> {
            shells
                .iter()
                .map(|shell| (shell.inertia().unwrap(), shell.trimesh.vertex_positions().to_vec()))
                .collect()
        }

        #[test]
        fn gradient() {
            let shells = test_shells();
            gradient_tester(build_energies(&shells), EnergyType::Velocity);
        }

        #[test]
        fn hessian() {
            let shells = test_shells();
            hessian_tester(build_energies(&shells), EnergyType::Velocity);
        }
    }
}
