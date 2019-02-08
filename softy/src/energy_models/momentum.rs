use crate::attrib_defines::*;
use crate::energy::*;
use geo::Real;
use geo::math::{Vector3};
use geo::mesh::{topology::*, Attrib};
use geo::prim::Tetrahedron;
use crate::matrix::*;
use crate::TetMesh;
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

    /// Step size in seconds for dynamics time integration scheme. If the time step is zero
    /// then the simulation becomes effectively quasi-static.
    time_step: f64,
}

impl MomentumPotential {
    const NUM_HESSIAN_TRIPLETS_PER_TET: usize = 12;

    /// Create a new Neo-Hookean energy model defining a non-linear problem that can be solved
    /// using a non-linear solver like Ipopt.
    /// This function takes a tetrahedron mesh specifying a discretization of the solid domain
    pub fn new(tetmesh: Rc<RefCell<TetMesh>>, density: f64, time_step: f64) -> Self {
        MomentumPotential {
            tetmesh,
            density,
            time_step,
        }
    }
}

impl<T: Real> Energy<T> for MomentumPotential {
    #[allow(non_snake_case)]
    fn energy(&self, v: &[T], dx: &[T]) -> T {
        let MomentumPotential {
            ref tetmesh,
            density,
            time_step: dt,
            ..
        } = *self;

        let dt = T::from(dt).unwrap();
        let density = T::from(density).unwrap();
        let dt_inv = T::one() / dt;

        let tetmesh = tetmesh.borrow();

        let vel: &[Vector3<T>] = reinterpret_slice(v);
        let disp: &[Vector3<T>] = reinterpret_slice(dx);

        tetmesh
            .attrib_iter::<RefVolType, CellIndex>(REFERENCE_VOLUME_ATTRIB)
            .unwrap()
            .zip(tetmesh.cell_iter())
            .map(|(&vol, cell)| {
                let tet_v = Tetrahedron::from_indexed_slice(cell.get(), vel);
                let tet_dx = Tetrahedron::from_indexed_slice(cell.get(), disp);
                let tet_dv = tet_dx * dt_inv - tet_v;

                T::from(0.5).unwrap() * dt_inv * {
                    let dvTdv: T = tet_dv.into_array().into_iter().map(|&dv| dv.dot(dv)).sum();
                    // momentum
                    T::from(0.25).unwrap() * T::from(vol).unwrap() * density * dvTdv * dt
                }
            })
            .sum()
    }
}

impl<T: Real> EnergyGradient<T> for MomentumPotential {
    #[allow(non_snake_case)]
    fn add_energy_gradient(&self, v: &[T], dx: &[T], grad_f: &mut [T]) {
        let MomentumPotential {
            ref tetmesh,
            density,
            time_step: dt,
            ..
        } = *self;

        let density = T::from(density).unwrap();
        let dt_inv = T::one() / T::from(dt).unwrap();

        let tetmesh = tetmesh.borrow();

        let vel: &[Vector3<T>] = reinterpret_slice(v);
        let disp: &[Vector3<T>] = reinterpret_slice(dx);

        debug_assert_eq!(grad_f.len(), dx.len());

        let gradient: &mut [Vector3<T>] = reinterpret_mut_slice(grad_f);

        // Transfer forces from cell-vertices to vertices themeselves
        for (&vol, cell) in tetmesh
            .attrib_iter::<RefVolType, CellIndex>(REFERENCE_VOLUME_ATTRIB)
            .unwrap()
            .zip(tetmesh.cell_iter())
        {
            let tet_v = Tetrahedron::from_indexed_slice(cell.get(), vel);
            let tet_dx = Tetrahedron::from_indexed_slice(cell.get(), disp);
            let tet_dv = (tet_dx * dt_inv - tet_v).into_array();

            let vol = T::from(vol).unwrap();

            for i in 0..4 {
                gradient[cell[i]] += tet_dv[i] * (dt_inv * T::from(0.25).unwrap() * vol * density);
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
    fn energy_hessian_values<T: Real + std::iter::Sum>(&self, _x: &[T], _dx: &[T], values: &mut [T]) {
        assert_eq!(values.len(), self.energy_hessian_size());

        let MomentumPotential {
            ref tetmesh,
            density,
            time_step: dt,
            ..
        } = *self;

        let dt_inv = T::one() / T::from(dt).unwrap();

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
                        tet_hess[3 * vi + j] = T::from(0.25 * vol * density).unwrap() * dt_inv * dt_inv;
                    }
                }
            });
    }
}
