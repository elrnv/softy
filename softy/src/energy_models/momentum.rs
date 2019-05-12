use crate::attrib_defines::*;
use crate::energy::*;
use crate::matrix::*;
use crate::TetMesh;
use geo::math::Vector3;
use geo::mesh::{topology::*, Attrib};
use geo::prim::Tetrahedron;
use geo::Real;
use num_traits::FromPrimitive;
use rayon::prelude::*;
use reinterpret::*;
use std::{cell::RefCell, rc::Rc};

/// The potential energy responsible for maintaining the intertial motion of an object.
#[derive(Debug, PartialEq)]
pub struct MomentumPotential {
    /// The discretization of the solid domain using a tetrahedral mesh. This field should never be
    /// borrowed mutably.
    pub tetmesh: Rc<RefCell<TetMesh>>,
    /// The density of the material. Typically measured in kg/m³, however the actual value stored
    /// in this struct may be normalized.
    pub density: f64,
}

impl MomentumPotential {
    const NUM_HESSIAN_TRIPLETS_PER_TET: usize = 12;

    /// Create a new Neo-Hookean energy model defining a non-linear problem that can be solved
    /// using a non-linear solver like Ipopt.
    /// This function takes a tetrahedron mesh specifying a discretization of the solid domain
    pub fn new(tetmesh: Rc<RefCell<TetMesh>>, density: f64) -> Self {
        MomentumPotential {
            tetmesh,
            density,
        }
    }
}

impl<T: Real> Energy<T> for MomentumPotential {
    #[allow(non_snake_case)]
    fn energy(&self, v0: &[T], v1: &[T]) -> T {
        let MomentumPotential {
            ref tetmesh,
            density,
            ..
        } = *self;

        let tetmesh = tetmesh.borrow();

        let vel0: &[Vector3<T>] = reinterpret_slice(v0);
        let vel1: &[Vector3<T>] = reinterpret_slice(v1);

        tetmesh
            .attrib_iter::<RefVolType, CellIndex>(REFERENCE_VOLUME_ATTRIB)
            .unwrap()
            .zip(tetmesh.cell_iter())
            .map(|(&vol, cell)| {
                let tet_v0 = Tetrahedron::from_indexed_slice(cell.get(), vel0);
                let tet_v1 = Tetrahedron::from_indexed_slice(cell.get(), vel1);
                let tet_dv = tet_v1 - tet_v0;

                T::from(0.5).unwrap() * {
                    let dvTdv: T = tet_dv.into_array().iter().map(|&dv| dv.dot(dv)).sum();
                    // momentum
                    T::from(0.25 * vol * density).unwrap() * dvTdv
                }
            })
            .sum()
    }
}

impl<T: Real> EnergyGradient<T> for MomentumPotential {
    #[allow(non_snake_case)]
    fn add_energy_gradient(&self, v0: &[T], v1: &[T], grad_f: &mut [T]) {
        let MomentumPotential {
            ref tetmesh,
            density,
            ..
        } = *self;

        let tetmesh = tetmesh.borrow();

        let vel0: &[Vector3<T>] = reinterpret_slice(v0);
        let vel1: &[Vector3<T>] = reinterpret_slice(v1);

        debug_assert_eq!(grad_f.len(), v0.len());

        let gradient: &mut [Vector3<T>] = reinterpret_mut_slice(grad_f);

        // Transfer forces from cell-vertices to vertices themeselves
        for (&vol, cell) in tetmesh
            .attrib_iter::<RefVolType, CellIndex>(REFERENCE_VOLUME_ATTRIB)
            .unwrap()
            .zip(tetmesh.cell_iter())
        {
            let tet_v0 = Tetrahedron::from_indexed_slice(cell.get(), vel0);
            let tet_v1 = Tetrahedron::from_indexed_slice(cell.get(), vel1);
            let tet_dv = (tet_v1 - tet_v0).into_array();

            for i in 0..4 {
                gradient[cell[i]] += tet_dv[i] * (T::from(0.25 * vol * density).unwrap());
            }
        }
    }
}

impl EnergyHessian for MomentumPotential {
    fn energy_hessian_size(&self) -> usize {
        self.tetmesh.borrow().num_cells() * Self::NUM_HESSIAN_TRIPLETS_PER_TET
    }

    fn energy_hessian_rows_cols_offset<I: FromPrimitive + Send>(
        &self,
        offset: MatrixElementIndex,
        rows: &mut [I],
        cols: &mut [I],
    ) {
        assert_eq!(rows.len(), self.energy_hessian_size());
        assert_eq!(cols.len(), self.energy_hessian_size());

        let tetmesh = self.tetmesh.borrow();

        // Break up the hessian triplets into chunks of elements for each tet.
        let hess_row_chunks: &mut [[I; Self::NUM_HESSIAN_TRIPLETS_PER_TET]] =
            reinterpret_mut_slice(rows);
        let hess_col_chunks: &mut [[I; Self::NUM_HESSIAN_TRIPLETS_PER_TET]] =
            reinterpret_mut_slice(cols);

        // The momentum hessian is a diagonal matrix.
        hess_row_chunks
            .par_iter_mut()
            .zip(hess_col_chunks.par_iter_mut())
            .zip(tetmesh.cells().par_iter())
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

        let tetmesh = self.tetmesh.borrow();

        // Break up the hessian triplets into chunks of elements for each tet.
        let hess_chunks: &mut [[MatrixElementIndex; Self::NUM_HESSIAN_TRIPLETS_PER_TET]] =
            reinterpret_mut_slice(indices);

        // The momentum hessian is a diagonal matrix.
        hess_chunks
            .par_iter_mut()
            .zip(tetmesh.cells().par_iter())
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

    #[allow(non_snake_case)]
    fn energy_hessian_values<T: Real + Send + Sync>(&self, _v0: &[T], _v1: &[T], scale: T, values: &mut [T]) {
        assert_eq!(values.len(), self.energy_hessian_size());

        let MomentumPotential {
            ref tetmesh,
            density,
            ..
        } = *self;

        let tetmesh = &*tetmesh.borrow();

        // Break up the hessian triplets into chunks of elements for each tet.
        let hess_chunks: &mut [[T; Self::NUM_HESSIAN_TRIPLETS_PER_TET]] =
            reinterpret_mut_slice(values);

        let vol_iter = tetmesh
            .attrib_as_slice::<RefVolType, CellIndex>(REFERENCE_VOLUME_ATTRIB)
            .unwrap()
            .par_iter();

        // The momentum hessian is a diagonal matrix.
        hess_chunks
            .par_iter_mut()
            .zip(vol_iter)
            .for_each(|(tet_hess, &vol)| {
                for vi in 0..4 {
                    // vertex index
                    for j in 0..3 {
                        // vector component
                        tet_hess[3 * vi + j] =
                            T::from(0.25 * vol * density).unwrap() * scale;
                    }
                }
            });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::energy_models::test_utils::*;

    #[test]
    fn gradient() {
        gradient_tester(
            |mesh| MomentumPotential::new(Rc::new(RefCell::new(mesh)), 1000.0),
            EnergyType::Velocity,
        );
    }

    #[test]
    fn hessian() {
        hessian_tester(
            |mesh| MomentumPotential::new(Rc::new(RefCell::new(mesh)), 1000.0),
            EnergyType::Velocity,
        );
    }
}
