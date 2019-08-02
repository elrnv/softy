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

        let g = self.g.map(|x| T::from(x).unwrap());

        zip!(tetmesh
            .attrib_iter::<RefVolType, CellIndex>(REFERENCE_VOLUME_ATTRIB)
            .unwrap(),
        tetmesh
            .attrib_iter::<DensityType, CellIndex>(DENSITY_ATTRIB)
            .unwrap(),
            tet_iter)
            .map(|(&vol, &density, tet)| {
                // We really want mass here. Since mass is conserved we can rely on reference
                // volume and density.
                g.dot(tet.centroid()) * T::from(-vol * density).unwrap()
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

        let g = self.g.map(|x| T::from(x).unwrap());

        // Transfer forces from cell-vertices to vertices themeselves
        for (&vol, &density, cell) in zip!(tetmesh
            .attrib_iter::<RefVolType, CellIndex>(REFERENCE_VOLUME_ATTRIB)
            .unwrap(),
            tetmesh
            .attrib_iter::<DensityType, CellIndex>(DENSITY_ATTRIB)
            .unwrap(),
            tetmesh.cell_iter())
        {
            for i in 0..4 {
                // Energy gradient is in opposite direction to the force hence minus here.
                gradient[cell[i]] -= g * T::from(0.25 * vol * density).unwrap();
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

        let g = self.g.map(|x| T::from(x).unwrap());

        zip!(trimesh
            .attrib_iter::<RefAreaType, FaceIndex>(REFERENCE_AREA_ATTRIB)
            .unwrap(),
            trimesh
            .attrib_iter::<DensityType, FaceIndex>(DENSITY_ATTRIB)
            .unwrap(),
            tri_iter)
            .map(|(&area, &density, tri)| {
                // We really want mass here. Since mass is conserved we can rely on reference
                // volume and density.
                g.dot(tri.centroid()) * T::from(-area * density).unwrap()
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

        let g = self.g.map(|x| T::from(x).unwrap());

        // Transfer forces from cell-vertices to vertices themeselves
        for (&area, &density, face) in zip!(trimesh
            .attrib_iter::<RefAreaType, FaceIndex>(REFERENCE_AREA_ATTRIB)
            .unwrap(),
            trimesh
            .attrib_iter::<DensityType, FaceIndex>(DENSITY_ATTRIB)
            .unwrap(),
            trimesh.face_iter())
        {
            for i in 0..3 {
                // Energy gradient is in opposite direction to the force hence minus here.
                gradient[face[i]] -= g * T::from(0.25 * area * density).unwrap();
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
    use crate::energy_models::{gravity::*, test_utils::*};
    use geo::mesh::VertexPositions;

    fn material() -> SolidMaterial {
        SolidMaterial::new(0).with_density(1000.0)
    }

    fn build_energies(solids: &[TetMeshSolid]) -> Vec<(TetMeshGravity, Vec<[f64; 3]>)> {
        solids.iter().map(|solid| {
            (solid.gravity([0.0, -9.81, 0.0]), solid.tetmesh.vertex_positions().to_vec())
        }).collect()
    }

    #[test]
    fn gradient() {
        let solids = test_solids(material());
        gradient_tester(
            build_energies(&solids),
            EnergyType::Position,
        );
    }

    #[test]
    fn hessian() {
        let solids = test_solids(material());
        hessian_tester(
            build_energies(&solids),
            EnergyType::Position,
        );
    }
}
