use crate::attrib_defines::*;
use crate::energy::*;
use crate::matrix::*;
use crate::objects::*;
use geo::math::Vector3;
use geo::mesh::{topology::*, Attrib};
use geo::ops::*;
use geo::prim::{Tetrahedron, Triangle};
use geo::Real;
use reinterpret::*;

/// This trait defines a convenient accessor for the specific gravity implementation for a given
/// object.
pub trait Gravity<E> {
    fn gravity(&self, g: [f64; 3]) -> E;
}

/// A constant directional force.
pub struct TetMeshGravity<'a> {
    solid: &'a TetMeshSolid,
    g: Vector3<f64>,
}

impl<'a> TetMeshGravity<'a> {
    pub fn new(solid: &TetMeshSolid, gravity: &[f64; 3]) -> Gravity<TetMeshGravity<'a>> {
        TetMeshGravity {
            solid,
            g: (*gravity).into(),
        }
    }
}

/// Define energy for gravity.
/// Gravity is a position based energy.
impl<T: Real> Energy<T> for TetMeshGravity<'_> {
    /// Since gravity depends on position, `x` is expected to be a position quantity.
    fn energy(&self, _x0: &[T], x1: &[T]) -> T {
        let pos1: &[Vector3<T>] = reinterpret_slice(x1);
        let tetmesh = self.solid.tetmesh;
        let tet_iter = tetmesh
            .cell_iter()
            .map(|cell| Tetrahedron::from_indexed_slice(cell, pos1));

        let g = self.g.map(|x| T::from(x).unwrap());

        tetmesh
            .attrib_iter::<RefVolType, CellIndex>(REFERENCE_VOLUME_ATTRIB)
            .unwrap()
            .zip(tet_iter)
            .map(|(&vol, tet)| {
                // We really want mass here. Since mass is conserved we can rely on reference
                // volume and density.
                g.dot(tet.centroid()) * T::from(-vol * self.density).unwrap()
            })
            .sum()
    }
}

impl<T: Real> EnergyGradient<T> for TetMeshGravity<'_> {
    /// Add the gravity gradient to the given global vector.
    fn add_energy_gradient(&self, _x0: &[T], _x1: &[T], grad: &mut [T]) {
        debug_assert_eq!(grad.len(), _x0.len());

        let tetmesh = self.solid.tetmesh;
        let gradient: &mut [Vector3<T>] = reinterpret_mut_slice(grad);

        let g = self.g.map(|x| T::from(x).unwrap());

        // Transfer forces from cell-vertices to vertices themeselves
        for (&vol, cell) in tetmesh
            .attrib_iter::<RefVolType, CellIndex>(REFERENCE_VOLUME_ATTRIB)
            .unwrap()
            .zip(tetmesh.cell_iter())
        {
            for i in 0..4 {
                // Energy gradient is in opposite direction to the force hence minus here.
                gradient[cell[i]] -= g * T::from(0.25 * vol * self.density).unwrap();
            }
        }
    }
}

impl EnergyHessian for TetMeshGravity<'_> {
    fn energy_hessian_size(&self) -> usize {
        0
    }
    fn energy_hessian_indices_offset(&self, _: MatrixElementIndex, _: &mut [MatrixElementIndex]) {}
    fn energy_hessian_values<T: Real>(&self, _x0: &[T], _x1: &[T], _scale: T, _vals: &mut [T]) {}
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
    pub fn new(shell: &TriMeshShell, gravity: &[f64; 3]) -> Gravity<TriMeshGravity<'a>> {
        TriMeshGravity {
            shell,
            g: (*gravity).into(),
        }
    }
}

/// Define energy for gravity.
/// Gravity is a position based energy.
impl<T: Real> Energy<T> for TriMeshGravity<'_> {
    /// Since gravity depends on position, `x` is expected to be a position quantity.
    fn energy(&self, _x0: &[T], x1: &[T]) -> T {
        let pos1: &[Vector3<T>] = reinterpret_slice(x1);
        let trimesh = self.shell.trimesh;
        let tri_iter = trimesh
            .face_iter()
            .map(|face| Triangle::from_indexed_slice(face, pos1));

        let g = self.g.map(|x| T::from(x).unwrap());

        trimesh
            .attrib_iter::<RefAreaType, FaceIndex>(REFERENCE_AREA_ATTRIB)
            .unwrap()
            .zip(tri_iter)
            .map(|(&area, tri)| {
                // We really want mass here. Since mass is conserved we can rely on reference
                // volume and density.
                g.dot(tri.centroid()) * T::from(-area * self.density).unwrap()
            })
            .sum()
    }
}

impl<T: Real> EnergyGradient<T> for TriMeshGravity<'_> {
    /// Add the gravity gradient to the given global vector.
    fn add_energy_gradient(&self, _x0: &[T], _x1: &[T], grad: &mut [T]) {
        debug_assert_eq!(grad.len(), _x0.len());

        let trimesh = self.shell.trimesh;
        let gradient: &mut [Vector3<T>] = reinterpret_mut_slice(grad);

        let g = self.g.map(|x| T::from(x).unwrap());

        // Transfer forces from cell-vertices to vertices themeselves
        for (&area, face) in trimesh
            .attrib_iter::<RefAreaType, FaceIndex>(REFERENCE_AREA_ATTRIB)
            .unwrap()
            .zip(trimesh.face_iter())
        {
            for i in 0..3 {
                // Energy gradient is in opposite direction to the force hence minus here.
                gradient[face[i]] -= g * T::from(0.25 * area * self.density).unwrap();
            }
        }
    }
}

impl EnergyHessian for TriMeshGravity<'_> {
    fn energy_hessian_size(&self) -> usize {
        0
    }
    fn energy_hessian_indices_offset(&self, _: MatrixElementIndex, _: &mut [MatrixElementIndex]) {}
    fn energy_hessian_values<T: Real>(&self, _x0: &[T], _x1: &[T], _scale: T, _vals: &mut [T]) {}
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::energy_models::test_utils::*;
    use crate::objects::*;
    use crate::TetMesh;

    fn make_tetmesh_solid(tetmesh: TetMesh) -> TetMeshSolid {
        TetMeshSolid {
            tetmesh,
            material: Material::solid(0, DeformableProperties::default().density(1000.0), false),
        }
    }

    #[test]
    fn gradient() {
        gradient_tester(
            |mesh| TetMeshGravity::new(&make_tetmesh_solid(mesh), &[0.0, -9.81, 0.0]),
            EnergyType::Position,
        );
    }

    #[test]
    fn hessian() {
        hessian_tester(
            |mesh| TetMeshGravity::new(&make_tetmesh_solid(mesh), &[0.0, -9.81, 0.0]),
            EnergyType::Position,
        );
    }
}
