use crate::attrib_defines::*;
use crate::energy::*;
use crate::matrix::*;
use crate::objects::*;
use geo::mesh::{topology::*, Attrib};
use geo::ops::*;
use geo::prim::{Tetrahedron, Triangle};
use reinterpret::*;
use utils::soap::{Real, Vector3};
use utils::zip;

/// This trait defines a convenient accessor for the specific gravity implementation for a given
/// object.
pub trait Gravity<'a, E> {
    fn gravity(&'a self, g: [f64; 3]) -> E;
}

/// A constant directional force.
pub struct TetMeshGravity<'a> {
    solid: &'a TetMeshSolid,
    g: Vector3<f64>,
}

impl<'a> TetMeshGravity<'a> {
    pub fn new(solid: &'a TetMeshSolid, gravity: [f64; 3]) -> TetMeshGravity<'a> {
        TetMeshGravity {
            solid,
            g: gravity.into(),
        }
    }
}

/// Define energy for gravity.
/// Gravity is a position based energy.
impl<T: Real> Energy<T> for TetMeshGravity<'_> {
    /// Since gravity depends on position, `x` is expected to be a position quantity.
    fn energy(&self, _x0: &[T], x1: &[T]) -> T {
        let pos1: &[Vector3<T>] = reinterpret_slice(x1);
        let tetmesh = &self.solid.tetmesh;
        let tet_iter = tetmesh
            .cell_iter()
            .map(|cell| Tetrahedron::from_indexed_slice(cell, pos1));

        let g = self.g.cast::<T>();

        zip!(
            tetmesh
                .attrib_iter::<RefVolType, CellIndex>(REFERENCE_VOLUME_ATTRIB)
                .unwrap(),
            tetmesh
                .attrib_iter::<DensityType, CellIndex>(DENSITY_ATTRIB)
                .unwrap()
                .map(|&x| f64::from(x)),
            tet_iter
        )
        .map(|(&vol, density, tet)| {
            // We really want mass here. Since mass is conserved we can rely on reference
            // volume and density.
            g.dot(Vector3::new(tet.centroid())) * T::from(-vol * density).unwrap()
        })
        .sum()
    }
}

impl<T: Real> EnergyGradient<T> for TetMeshGravity<'_> {
    /// Add the gravity gradient to the given global vector.
    fn add_energy_gradient(&self, _x0: &[T], _x1: &[T], grad: &mut [T]) {
        debug_assert_eq!(grad.len(), _x0.len());

        let tetmesh = &self.solid.tetmesh;
        let gradient: &mut [Vector3<T>] = reinterpret_mut_slice(grad);

        let g = self.g.cast::<T>();

        // Transfer forces from cell-vertices to vertices themeselves
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
            for i in 0..4 {
                // Energy gradient is in opposite direction to the force hence minus here.
                gradient[cell[i]] -= g * T::from(0.25 * vol * density).unwrap();
            }
        }
    }
}

impl EnergyHessianTopology for TetMeshGravity<'_> {
    fn energy_hessian_size(&self) -> usize {
        0
    }
    fn energy_hessian_indices_offset(&self, _: MatrixElementIndex, _: &mut [MatrixElementIndex]) {}
}
impl<T: Real> EnergyHessian<T> for TetMeshGravity<'_> {
    fn energy_hessian_values(&self, _x0: &[T], _x1: &[T], _scale: T, _vals: &mut [T]) {}
}

/*
 * Gravity for TriMeshShell
 */

/// A constant directional force.
pub struct TriMeshGravity<'a> {
    shell: &'a TriMeshShell,
    g: Vector3<f64>,
}

impl<'a> TriMeshGravity<'a> {
    pub fn new(shell: &'a TriMeshShell, gravity: [f64; 3]) -> TriMeshGravity<'a> {
        TriMeshGravity {
            shell,
            g: gravity.into(),
        }
    }
}

/// Define energy for gravity.
/// Gravity is a position based energy.
impl<T: Real> Energy<T> for TriMeshGravity<'_> {
    /// Since gravity depends on position, `x` is expected to be a position quantity.
    fn energy(&self, _x0: &[T], x1: &[T]) -> T {
        let pos1: &[Vector3<T>] = reinterpret_slice(x1);
        let trimesh = &self.shell.trimesh;
        let tri_iter = trimesh
            .face_iter()
            .map(|face| Triangle::from_indexed_slice(face, pos1));

        let g = self.g.cast::<T>();

        zip!(
            trimesh
                .attrib_iter::<RefAreaType, FaceIndex>(REFERENCE_AREA_ATTRIB)
                .unwrap(),
            trimesh
                .attrib_iter::<DensityType, FaceIndex>(DENSITY_ATTRIB)
                .unwrap()
                .map(|&x| f64::from(x)),
            tri_iter
        )
        .map(|(&area, density, tri)| {
            // We really want mass here. Since mass is conserved we can rely on reference
            // volume and density.
            g.dot(Vector3::new(tri.centroid())) * T::from(-area * density).unwrap()
        })
        .sum()
    }
}

impl<T: Real> EnergyGradient<T> for TriMeshGravity<'_> {
    /// Add the gravity gradient to the given global vector.
    fn add_energy_gradient(&self, _x0: &[T], _x1: &[T], grad: &mut [T]) {
        debug_assert_eq!(grad.len(), _x0.len());

        let trimesh = &self.shell.trimesh;
        let gradient: &mut [Vector3<T>] = reinterpret_mut_slice(grad);

        let g = self.g.cast::<T>();

        let third = 1.0 / 3.0;

        // Transfer forces from cell-vertices to vertices themeselves
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
            for i in 0..3 {
                // Energy gradient is in opposite direction to the force hence minus here.
                gradient[face[i]] -= g * T::from(third * area * density).unwrap();
            }
        }
    }
}

impl EnergyHessianTopology for TriMeshGravity<'_> {
    fn energy_hessian_size(&self) -> usize {
        0
    }
    fn energy_hessian_indices_offset(&self, _: MatrixElementIndex, _: &mut [MatrixElementIndex]) {}
}

impl<T: Real> EnergyHessian<T> for TriMeshGravity<'_> {
    fn energy_hessian_values(&self, _x0: &[T], _x1: &[T], _scale: T, _vals: &mut [T]) {}
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::energy_models::test_utils::*;
    use crate::fem::SolverBuilder;
    use geo::mesh::VertexPositions;

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

        fn build_energies(solids: &[TetMeshSolid]) -> Vec<(TetMeshGravity, Vec<[f64; 3]>)> {
            solids
                .iter()
                .map(|solid| {
                    (
                        solid.gravity([0.0, -9.81, 0.0]),
                        solid.tetmesh.vertex_positions().to_vec(),
                    )
                })
                .collect()
        }

        #[test]
        fn gradient() {
            let solids = test_solids();
            gradient_tester(build_energies(&solids), EnergyType::Position);
        }

        #[test]
        fn hessian() {
            let solids = test_solids();
            hessian_tester(build_energies(&solids), EnergyType::Position);
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

        fn build_energies(shells: &[TriMeshShell]) -> Vec<(TriMeshGravity, Vec<[f64; 3]>)> {
            shells
                .iter()
                .map(|shell| {
                    (
                        shell.gravity([0.0, -9.81, 0.0]).unwrap(),
                        shell.trimesh.vertex_positions().to_vec(),
                    )
                })
                .collect()
        }

        #[test]
        fn gradient() {
            let shells = test_shells();
            gradient_tester(build_energies(&shells), EnergyType::Position);
        }

        #[test]
        fn hessian() {
            let shells = test_shells();
            hessian_tester(build_energies(&shells), EnergyType::Position);
        }
    }
}
