use crate::attrib_defines::*;
use crate::energy::*;
use crate::geo::math::Vector3;
use crate::geo::mesh::{topology::*, Attrib};
use crate::matrix::*;
use crate::TetMesh;
use rayon::prelude::*;
use reinterpret::*;
use std::{cell::RefCell, rc::Rc};

/// The potential energy responsible for maintaining the intertial motion of an object.
#[derive(Debug, PartialEq)]
pub struct MomentumPotential {
    /// The discretization of the solid domain using a tetrahedral mesh.
    pub tetmesh: Rc<RefCell<TetMesh>>,
    /// The density of the material. Typically measured in kg/m³, however the actual value stored
    /// in this struct may be normalized.
    pub density: f64,

    /// Step size in seconds for dynamics time integration scheme. If the time step is zero (meaning
    /// the reciprocal would be infinite) then this value is set to zero and the simulation becomes
    /// effectively quasi-static.
    /// This field stores the reciprocal of the time step since this is what we compute with and it
    /// naturally determines if the simulation is dynamic.
    time_step_inv: f64,

    // Workspace fields for storing intermediate results
    /// Energy hessian indices. Keep this memory around and reuse it to avoid unnecessary allocations.
    energy_hessian_indices: Vec<MatrixElementIndex>,
    /// Energy hessian values. Keep this memory around and reuse it to avoid unnecessary allocations.
    energy_hessian_values: Vec<f64>,
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
            time_step_inv: 1.0 / time_step,
            energy_hessian_indices: Vec::new(),
            energy_hessian_values: Vec::new(),
        }
    }
}

impl Energy<f64> for MomentumPotential {
    #[allow(non_snake_case)]
    fn energy(&mut self, dx: &[f64]) -> f64 {
        let MomentumPotential {
            ref tetmesh,
            density,
            time_step_inv: dt_inv,
            ..
        } = *self;

        let tetmesh = tetmesh.borrow();

        let disp: &[Vector3<f64>] = reinterpret_slice(dx);

        let prev_disp: &[Vector3<f64>] = reinterpret_slice(
            tetmesh
                .attrib_as_slice::<DispType, VertexIndex>(DISPLACEMENT_ATTRIB)
                .unwrap(),
        );

        tetmesh
            .attrib_iter::<RefVolType, CellIndex>(REFERENCE_VOLUME_ATTRIB)
            .unwrap()
            .zip(tetmesh.cell_iter())
            .map(|(&vol, cell)| {
                0.5 * dt_inv * {
                    let dvTdv: f64 = [
                        disp[cell[0]] - prev_disp[cell[0]],
                        disp[cell[1]] - prev_disp[cell[1]],
                        disp[cell[2]] - prev_disp[cell[2]],
                        disp[cell[3]] - prev_disp[cell[3]],
                    ]
                    .into_iter()
                    .map(|&dv| dv.dot(dv))
                    .sum();
                    // momentum
                    0.25 * vol * density * dvTdv * dt_inv
                }
            })
            .sum()
    }
}

impl EnergyGradient<f64> for MomentumPotential {
    #[allow(non_snake_case)]
    fn add_energy_gradient(&self, dx: &[f64], grad_f: &mut [f64]) {
        let MomentumPotential {
            ref tetmesh,
            density,
            time_step_inv: dt_inv,
            ..
        } = *self;

        let tetmesh = tetmesh.borrow();

        let disp: &[Vector3<f64>] = reinterpret_slice(dx);

        let prev_disp: &[Vector3<f64>] = reinterpret_slice(
            tetmesh
                .attrib_as_slice::<DispType, VertexIndex>(DISPLACEMENT_ATTRIB)
                .unwrap(),
        );

        debug_assert_eq!(grad_f.len(), dx.len());

        let gradient: &mut [Vector3<f64>] = reinterpret_mut_slice(grad_f);

        // Transfer forces from cell-vertices to vertices themeselves
        for (&vol, cell) in tetmesh
            .attrib_iter::<RefVolType, CellIndex>(REFERENCE_VOLUME_ATTRIB)
            .unwrap()
            .zip(tetmesh.cell_iter())
        {
            let dv = [
                // current displacement - previous displacement
                disp[cell[0]] - prev_disp[cell[0]],
                disp[cell[1]] - prev_disp[cell[1]],
                disp[cell[2]] - prev_disp[cell[2]],
                disp[cell[3]] - prev_disp[cell[3]],
            ];

            for i in 0..4 {
                gradient[cell[i]] += dt_inv * dt_inv * 0.25 * vol * density * dv[i];
            }
        }
    }
}

impl EnergyHessian<f64> for MomentumPotential {
    fn energy_hessian_size(&self) -> usize {
        self.tetmesh.borrow().num_cells() * Self::NUM_HESSIAN_TRIPLETS_PER_TET
    }

    fn energy_hessian_indices(&mut self) -> &[MatrixElementIndex] {
        let num_hess_triplets = self.energy_hessian_size();
        let MomentumPotential {
            ref tetmesh,
            energy_hessian_indices: ref mut hess,
            ..
        } = *self;

        let tetmesh = tetmesh.borrow();

        // Ensure there are enough entries in our hessian index buffer.
        hess.resize(num_hess_triplets, MatrixElementIndex { row: 0, col: 0 });

        {
            // Break up the hessian triplets into chunks of elements for each tet.
            let hess_chunks: &mut [[MatrixElementIndex;
                                       Self::NUM_HESSIAN_TRIPLETS_PER_TET]] =
                reinterpret_mut_slice(hess);

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
                                row: 3 * cell[vi] + j,
                                col: 3 * cell[vi] + j,
                            };
                        }
                    }
                });
        }
        hess
    }

    #[allow(non_snake_case)]
    fn energy_hessian_values(&mut self, _dx: &[f64]) -> &[f64] {
        let num_hess_triplets = self.energy_hessian_size();
        let MomentumPotential {
            ref tetmesh,
            density,
            time_step_inv: dt_inv,
            energy_hessian_values: ref mut hess,
            ..
        } = *self;

        let tetmesh = &*tetmesh.borrow();

        // Ensure there are enough entries in our hessian index buffer.
        hess.resize(num_hess_triplets, 0.0);

        {
            // Break up the hessian triplets into chunks of elements for each tet.
            let hess_chunks: &mut [[f64; Self::NUM_HESSIAN_TRIPLETS_PER_TET]] =
                reinterpret_mut_slice(hess);

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
                            tet_hess[3 * vi + j] = 0.25 * vol * density * dt_inv * dt_inv;
                        }
                    }
                });
        }
        hess
    }
}
