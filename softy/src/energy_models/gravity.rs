use crate::attrib_defines::*;
use crate::energy::*;
use crate::matrix::*;
use crate::objects::*;
use flatk::zip;
use geo::mesh::{topology::*, Attrib};
use geo::ops::*;
use geo::prim::{Tetrahedron, Triangle};
use tensr::{Real, Vector3};

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
        let pos1: &[Vector3<T>] = bytemuck::cast_slice(x1);
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
        let gradient: &mut [Vector3<T>] = bytemuck::cast_slice_mut(grad);

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
 * Gravity for trimesh shells
 */

/// A constant directional force.
pub struct SoftShellGravity<'a> {
    shell: &'a TriMeshShell,
    g: Vector3<f64>,
}

impl<'a> SoftShellGravity<'a> {
    pub fn new(shell: &'a TriMeshShell, gravity: [f64; 3]) -> SoftShellGravity<'a> {
        SoftShellGravity {
            shell,
            g: gravity.into(),
        }
    }
}

/// Define energy for gravity.
/// Gravity is a position based energy.
impl<T: Real> Energy<T> for SoftShellGravity<'_> {
    /// Since gravity depends on position, `x` is expected to be a position quantity.
    fn energy(&self, _x0: &[T], x1: &[T]) -> T {
        let pos1: &[Vector3<T>] = bytemuck::cast_slice(x1);
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

impl<T: Real> EnergyGradient<T> for SoftShellGravity<'_> {
    /// Add the gravity gradient to the given global vector.
    fn add_energy_gradient(&self, _x0: &[T], _x1: &[T], grad: &mut [T]) {
        debug_assert_eq!(grad.len(), _x0.len());

        let trimesh = &self.shell.trimesh;
        let gradient: &mut [Vector3<T>] = bytemuck::cast_slice_mut(grad);

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

impl EnergyHessianTopology for SoftShellGravity<'_> {
    fn energy_hessian_size(&self) -> usize {
        0
    }
    fn energy_hessian_indices_offset(&self, _: MatrixElementIndex, _: &mut [MatrixElementIndex]) {}
}

impl<T: Real> EnergyHessian<T> for SoftShellGravity<'_> {
    fn energy_hessian_values(&self, _x0: &[T], _x1: &[T], _scale: T, _vals: &mut [T]) {}
}

/// A constant directional force.
pub struct RigidShellGravity {
    mass: f64,
    g: Vector3<f64>,
}

impl RigidShellGravity {
    pub fn new(gravity: [f64; 3], mass: f64) -> RigidShellGravity {
        RigidShellGravity {
            g: gravity.into(),
            mass,
        }
    }
}

/// Define energy for gravity.
/// Gravity is a position based energy.
impl<T: Real> Energy<T> for RigidShellGravity {
    /// Since gravity depends on position, `x` is expected to be a position quantity.
    fn energy(&self, _x0: &[T], x1: &[T]) -> T {
        let pos = Vector3::new([x1[0], x1[1], x1[2]]);
        T::from(-self.mass).unwrap() * self.g.cast::<T>().dot(pos)
    }
}

impl<T: Real> EnergyGradient<T> for RigidShellGravity {
    /// Add the gravity gradient to the given global vector.
    fn add_energy_gradient(&self, _x0: &[T], _x1: &[T], grad: &mut [T]) {
        use tensr::AsMutTensor;
        debug_assert_eq!(grad.len(), _x0.len());

        *grad[0..3].as_mut_tensor() -= self.g.cast::<T>() * T::from(self.mass).unwrap();
    }
}

impl EnergyHessianTopology for RigidShellGravity {
    fn energy_hessian_size(&self) -> usize {
        0
    }
    fn energy_hessian_indices_offset(&self, _: MatrixElementIndex, _: &mut [MatrixElementIndex]) {}
}

impl<T: Real> EnergyHessian<T> for RigidShellGravity {
    fn energy_hessian_values(&self, _x0: &[T], _x1: &[T], _scale: T, _vals: &mut [T]) {}
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::energy_models::test_utils::*;
    use crate::fem::opt::SolverBuilder;
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

    mod soft_shell {
        use super::*;
        use crate::energy_models::Either;

        fn soft_shell_material() -> SoftShellMaterial {
            SoftShellMaterial::new(0).with_density(1000.0)
        }

        fn test_shells() -> Vec<TriMeshShell> {
            let material = soft_shell_material();

            test_trimeshes()
                .into_iter()
                .map(|trimesh| {
                    // Prepare attributes relevant for elasticity computations.
                    let mut shell = TriMeshShell::soft(trimesh, material);
                    shell.init_deformable_attributes().unwrap();
                    shell.init_density_attribute().unwrap();
                    shell
                })
                .collect()
        }

        fn build_energies(
            shells: &[TriMeshShell],
        ) -> Vec<(Either<SoftShellGravity, RigidShellGravity>, Vec<[f64; 3]>)> {
            shells
                .iter()
                .map(|shell| {
                    let g = shell.gravity([0.0, -9.81, 0.0]).unwrap();
                    let pos = match &g {
                        Either::Left(_) => shell.trimesh.vertex_positions().to_vec(),
                        Either::Right(_) => vec![[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
                    };
                    (g, pos)
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
