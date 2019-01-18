use crate::attrib_defines::*;
use crate::energy::*;
use geo::math::Vector3;
use geo::mesh::{topology::*, Attrib};
use geo::ops::*;
use geo::prim::Tetrahedron;
use crate::matrix::*;
use crate::TetMesh;
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
impl Energy<f64> for Gravity {
    /// Since gravity depends on position, `x` is expected to be a position quantity.
    fn energy(&self, x: &[f64], dx: &[f64]) -> f64 {
        let prev_pos: &[Vector3<f64>] = reinterpret_slice(x);
        let disp: &[Vector3<f64>] = reinterpret_slice(dx);
        let tetmesh = self.tetmesh.borrow();
        let tet_iter = tetmesh.cell_iter().map(|cell| {
            Tetrahedron::from_indexed_slice(cell.get(), prev_pos)
                + Tetrahedron::from_indexed_slice(cell.get(), disp)
        });

        tetmesh
            .attrib_iter::<RefVolType, CellIndex>(REFERENCE_VOLUME_ATTRIB)
            .unwrap()
            .zip(tet_iter)
            .map(|(&vol, tet)| {
                // We really want mass here. Since mass is conserved we can rely on reference
                // volume and density.
                -vol * self.density * self.g.dot(tet.centroid())
            })
            .sum()
    }
}

impl EnergyGradient<f64> for Gravity {
    /// Add the gravity gradient to the given global vector.
    fn add_energy_gradient(&self, _x: &[f64], _dx: &[f64], grad: &mut [f64]) {
        debug_assert_eq!(grad.len(), _x.len());

        let tetmesh = self.tetmesh.borrow();
        let gradient: &mut [Vector3<f64>] = reinterpret_mut_slice(grad);

        // Transfer forces from cell-vertices to vertices themeselves
        for (&vol, cell) in tetmesh
            .attrib_iter::<RefVolType, CellIndex>(REFERENCE_VOLUME_ATTRIB)
            .unwrap()
            .zip(tetmesh.cell_iter())
        {
            for i in 0..4 {
                // Energy gradient is in opposite direction to the force hence minus here.
                gradient[cell[i]] -= 0.25 * vol * self.density * self.g;
            }
        }
    }
}

impl EnergyHessian<f64> for Gravity {
    fn energy_hessian_size(&self) -> usize {
        0
    }
    fn energy_hessian_indices_offset(&self, _: MatrixElementIndex, _: &mut [MatrixElementIndex]) {}
    fn energy_hessian_values(&self, _x: &[f64], _dx: &[f64], _: &mut [f64]) {}
}
