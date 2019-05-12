use crate::attrib_defines::*;
use crate::energy::*;
use crate::matrix::*;
use crate::TetMesh;
use geo::math::Vector3;
use geo::mesh::{topology::*, Attrib};
use geo::ops::*;
use geo::prim::Tetrahedron;
use geo::Real;
use reinterpret::*;
use std::{cell::RefCell, rc::Rc};

/// A constant directional force.
#[derive(Clone, Debug, PartialEq)]
pub struct Gravity {
    pub tetmesh: Rc<RefCell<TetMesh>>,
    density: f64,
    g: Vector3<f64>,
}

impl Gravity {
    pub fn new(tetmesh: Rc<RefCell<TetMesh>>, density: f64, gravity: &[f64; 3]) -> Gravity {
        Gravity {
            tetmesh,
            density,
            g: (*gravity).into(),
        }
    }
}

/// Define energy for gravity.
/// Gravity is a position based energy.
impl<T: Real> Energy<T> for Gravity {
    /// Since gravity depends on position, `x` is expected to be a position quantity.
    fn energy(&self, _x0: &[T], x1: &[T]) -> T {
        let pos1: &[Vector3<T>] = reinterpret_slice(x1);
        let tetmesh = self.tetmesh.borrow();
        let tet_iter = tetmesh.cell_iter().map(|cell| {
            Tetrahedron::from_indexed_slice(cell.get(), pos1)
        });

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

impl<T: Real> EnergyGradient<T> for Gravity {
    /// Add the gravity gradient to the given global vector.
    fn add_energy_gradient(&self, _x0: &[T], _x1: &[T], grad: &mut [T]) {
        debug_assert_eq!(grad.len(), _x0.len());

        let tetmesh = self.tetmesh.borrow();
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

impl EnergyHessian for Gravity {
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

    #[test]
    fn gradient() {
        gradient_tester(
            |mesh| Gravity::new(Rc::new(RefCell::new(mesh)), 1000.0, &[0.0, -9.81, 0.0]),
            EnergyType::Position,
        );
    }

    #[test]
    fn hessian() {
        hessian_tester(
            |mesh| Gravity::new(Rc::new(RefCell::new(mesh)), 1000.0, &[0.0, -9.81, 0.0]),
            EnergyType::Position,
        );
    }
}
