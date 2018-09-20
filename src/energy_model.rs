//use std::path::Path;
//use geo::io::save_tetmesh;
use energy::*;
use geo::math::{Matrix3, Vector3};
use geo::mesh::{topology::*, Attrib};
use geo::ops::*;
use rayon::prelude::*;
use reinterpret::*;
use TetMesh;

/// Per-tetrahedron Neo-Hookean energy model. This struct stores conveniently precomputed values
/// for tet energy computation. It encapsulates tet specific energy computation.
#[allow(non_snake_case)]
pub struct NeoHookeanTetEnergy {
    Dx: Matrix3<f64>,
    DX_inv: Matrix3<f64>,
    volume: f64,
    lambda: f64,
    mu: f64,
}

impl NeoHookeanTetEnergy {
    pub const NUM_HESSIAN_TRIPLETS: usize = 78; // There are 4*6 + 3*9*4/2 = 78 triplets per tet (overestimate)

    #[allow(non_snake_case)]
    pub fn new(Dx: Matrix3<f64>, DX_inv: Matrix3<f64>, volume: f64, lambda: f64, mu: f64) -> Self {
        NeoHookeanTetEnergy {
            Dx,
            DX_inv,
            volume,
            lambda,
            mu,
        }
    }

    /// Compute the deformation gradient `F` for this tet.
    #[allow(non_snake_case)]
    #[inline]
    fn deformation_gradient(&self) -> Matrix3<f64> {
        self.Dx * self.DX_inv
    }

    /// Compute the deformation gradient differential `dF` for this tet.
    #[allow(non_snake_case)]
    #[inline]
    fn deformation_gradient_differential(&self, dx: &[Vector3<f64>; 4]) -> Matrix3<f64> {
        // Build differential dDx
        let dDx = Matrix3([
            (dx[0] - dx[3]).into(),
            (dx[1] - dx[3]).into(),
            (dx[2] - dx[3]).into(),
        ]);
        dDx * self.DX_inv
    }

    /// Elastic strain energy per element.
    /// This is a helper function that computes the strain energy given shape matrices, which can
    /// be obtained from a tet and its reference configuration.
    #[allow(non_snake_case)]
    #[inline]
    pub fn elastic_energy(&self) -> f64 {
        let NeoHookeanTetEnergy {
            volume, mu, lambda, ..
        } = *self;
        let F = self.deformation_gradient();
        let I = F.norm_squared(); // tr(F^TF)
        let J = F.determinant();
        if J <= 0.0 {
            ::std::f64::INFINITY
        } else {
            let logJ = J.ln();
            volume * (0.5 * mu * (I - 3.0) - mu * logJ + 0.5 * lambda * logJ * logJ)
        }
    }

    /// Elastic energy gradient per element vertex.
    /// This is a helper function that computes the energy gradient given shape matrices, which can
    /// be obtained from a tet and its reference configuration.
    #[allow(non_snake_case)]
    #[inline]
    pub fn elastic_energy_gradient(&self) -> [Vector3<f64>; 4] {
        let NeoHookeanTetEnergy {
            DX_inv,
            volume,
            mu,
            lambda,
            ..
        } = *self;
        let F = self.deformation_gradient();
        let J = F.determinant();
        if J <= 0.0 {
            [Vector3::zeros(); 4]
        } else {
            let F_inv_tr = F.inverse_transpose().unwrap();
            let logJ = J.ln();
            let H = volume * (mu * F + (lambda * logJ - mu) * F_inv_tr) * DX_inv.transpose();
            [H[0], H[1], H[2], -H[0] - H[1] - H[2]]
        }
    }

    /// Elasticity Hessian*displacement product per element. Respresented by a 3x3 matrix where column `i`
    /// produces the hessian product contribution for the vertex `i` within the current element.
    /// The contribution to the last vertex is given by the negative sum of all the columns.
    #[allow(non_snake_case)]
    #[inline]
    fn elastic_energy_hessian_product(&self, dx: &[Vector3<f64>; 4]) -> Matrix3<f64> {
        let NeoHookeanTetEnergy {
            DX_inv,
            volume,
            lambda,
            mu,
            ..
        } = *self;
        let F = self.deformation_gradient();
        let dF = self.deformation_gradient_differential(dx);
        let J = F.determinant();
        if J > 0.0 {
            let alpha = mu - lambda * J.ln();

            let F_inv_tr = F.inverse_transpose().unwrap();
            let F_inv = F_inv_tr.transpose();

            let dP = mu * dF
                + alpha * F_inv_tr * dF.transpose() * F_inv_tr
                + lambda * (F_inv * dF).trace() * F_inv_tr;

            volume * dP * DX_inv.transpose()
        } else {
            Matrix3::zeros()
        }
    }
}

/// A possibly non-linear elastic energy for tetrahedral meshes.
pub struct ElasticTetMeshEnergy<'a> {
    /// The discretization of the solid domain using a tetrahedral mesh.
    pub solid: &'a mut TetMesh,
    /// Position from the previous time step.
    pub prev_pos: Vec<Vector3<f64>>,
    /// Material parameters.
    material: MaterialModel,
    /// Step size in seconds for dynamics time integration scheme. If the time step is zero (meaning
    /// the reciprocal would be infinite) then this value is set to zero and the simulation becomes
    /// effectively quasi-static.
    /// This field stores the reciprocal of the time step since this is what we compute with and it
    /// naturally determines if the simulation is dynamic.
    time_step_inv: f64,
    /// Gravitational acceleration in m/s².
    gravity: Vector3<f64>,

    // Workspace fields for storing intermediate results
    /// Energy gradient.
    energy_gradient: Vec<Vector3<f64>>,
    /// Energy hessian triplets. Keep this memory around and reuse it to avoid unnecessary allocations.
    energy_hessian_triplets: Vec<MatrixElementTriplet<f64>>,
}

/// The material model including elasticity Lame parameters as well as dynamics specific parameters
/// like material density and damping coefficient.
#[derive(Copy, Clone, Debug, PartialEq)]
struct MaterialModel {
    /// First Lame parameter. Measured in Pa = N/m² = kg/(ms²).
    pub lambda: f64,
    /// Second Lame parameter. Measured in Pa = N/m² = kg/(ms²).
    pub mu: f64,
    /// The density of the material. Measured in kg/m³
    pub density: f64,
    /// Coefficient measuring the amount of artificial viscosity as dictated by the Rayleigh
    /// damping model.
    pub damping: f64,
}

impl Default for MaterialModel {
    fn default() -> Self {
        MaterialModel {
            lambda: 5.4,
            mu: 263.1,
            density: 1000.0,
            damping: 0.0,
        }
    }
}

impl<'a> ElasticTetMeshEnergy<'a> {
    /// Create a new Neo-Hookean energy model defining a non-linear problem that can be solved
    /// using a non-linear solver like Ipopt.
    /// This function takes a tetrahedron mesh specifying a discretization of the solid domain and
    /// a closure that interrupts the solver if it returns `false`.
    pub fn new(tetmesh: &'a mut TetMesh) -> Self {
        let prev_pos = reinterpret_slice(tetmesh.vertex_positions()).to_vec();

        ElasticTetMeshEnergy {
            solid: tetmesh,
            prev_pos,
            material: MaterialModel::default(),
            time_step_inv: 0.0,
            gravity: Vector3::zeros(),
            energy_gradient: Vec::new(),
            energy_hessian_triplets: Vec::new(),
        }
    }

    // Builder routines.

    /// Set the elastic material properties of the volumetric solid discretized by the tetmesh.
    pub fn material(mut self, lambda: f64, mu: f64, density: f64, damping: f64) -> Self {
        self.material = MaterialModel {
            lambda,
            mu,
            density,
            damping,
        };
        self
    }

    /// Set the gravity for the simulation.
    pub fn gravity(mut self, gravity: [f64; 3]) -> Self {
        self.gravity = gravity.into();
        self
    }

    /// Set the time step making this simulation dynamic.
    /// Without the time step the simulation will assume an infinite time-step making it a
    /// quasi-static simulator.
    pub fn time_step(mut self, time_step: f64) -> Self {
        self.time_step_inv = if time_step > 0.0 {
            1.0 / time_step
        } else {
            0.0
        };
        self
    }

    // Solver specific functions

    /// Update the tetmesh vertex positions.
    pub fn update(&mut self, x: &[f64]) {
        let x_slice: &[[f64; 3]] = reinterpret_slice(x);
        let verts = self.solid.vertex_positions_mut();
        verts.copy_from_slice(x_slice);
    }
}

/// Define energy for ElasticTetMeshEnergy materials.
impl<'a> Energy<f64> for ElasticTetMeshEnergy<'a> {
    #[allow(non_snake_case)]
    fn energy(&mut self) -> f64 {
        let ElasticTetMeshEnergy {
            ref solid,
            ref prev_pos,
            material:
                MaterialModel {
                    lambda,
                    mu,
                    density,
                    damping,
                },
            time_step_inv: dt_inv,
            gravity,
            ..
        } = *self;

        let prev_vel: &[Vector3<f64>] = reinterpret_slice(
            solid
                .attrib_as_slice::<[f64; 3], VertexIndex>("vel")
                .unwrap(),
        );

        solid
            .attrib_iter::<f64, CellIndex>("ref_volume")
            .unwrap()
            .zip(
                solid
                    .attrib_iter::<Matrix3<f64>, CellIndex>("ref_shape_mtx_inv")
                    .unwrap(),
            ).zip(solid.cell_iter())
            .zip(solid.tet_iter())
            .map(|(((&vol, &DX_inv), cell), tet)| {
                let Dx = tet.shape_matrix();
                let tet_energy = NeoHookeanTetEnergy::new(Dx, DX_inv, vol, lambda, mu);
                // elasticity
                tet_energy.elastic_energy()
                // gravity (external forces)
                - vol * density * gravity.dot(tet.centroid())
                    // dynamics (including damping)
                    + 0.5 * dt_inv * {
                        let dx = [
                            tet.0 - prev_pos[cell[0]],
                            tet.1 - prev_pos[cell[1]],
                            tet.2 - prev_pos[cell[2]],
                            tet.3 - prev_pos[cell[3]],
                        ];
                        let dH = tet_energy.elastic_energy_hessian_product(&dx);
                        let dvTdv: f64 = [
                            dx[0] - prev_vel[cell[0]],
                            dx[1] - prev_vel[cell[1]],
                            dx[2] - prev_vel[cell[2]],
                            dx[3] - prev_vel[cell[3]],
                        ].into_iter()
                            .map(|&dv| dv.dot(dv))
                            .sum();
                        // momentum
                        0.25 * vol * density * dvTdv * dt_inv
                            // damping (viscosity)
                            + damping
                            * (dH[0].dot(dx[0]) + dH[1].dot(dx[1]) + dH[2].dot(dx[2]) -
                               (dx[3].transpose()*dH).sum())
                    }
            }).sum()
    }

    #[allow(non_snake_case)]
    fn energy_gradient(&mut self) -> &[Vector3<f64>] {
        let ElasticTetMeshEnergy {
            ref solid,
            ref prev_pos,
            material:
                MaterialModel {
                    lambda,
                    mu,
                    density,
                    damping,
                },
            time_step_inv: dt_inv,
            gravity,
            energy_gradient: ref mut gradient,
            ..
        } = *self;

        let prev_vel: &[Vector3<f64>] = reinterpret_slice(
            solid
                .attrib_as_slice::<[f64; 3], VertexIndex>("vel")
                .unwrap(),
        );

        gradient.resize(solid.num_vertices(), Vector3::zeros());

        let force_iter = solid
            .attrib_iter::<f64, CellIndex>("ref_volume")
            .unwrap()
            .zip(
                solid
                    .attrib_iter::<Matrix3<f64>, CellIndex>("ref_shape_mtx_inv")
                    .unwrap(),
            ).zip(solid.tet_iter())
            .map(|((&vol, &DX_inv), tet)| {
                let tet_energy =
                    NeoHookeanTetEnergy::new(tet.shape_matrix(), DX_inv, vol, lambda, mu);
                tet_energy.elastic_energy_gradient()
            });

        // Clear gradient vector.
        for grad in gradient.iter_mut() {
            *grad = Vector3::zeros();
        }

        // Transfer forces from cell-vertices to vertices themeselves
        for ((((&vol, &DX_inv), tet), cell), grad) in solid
            .attrib_iter::<f64, CellIndex>("ref_volume")
            .unwrap()
            .zip(
                solid
                    .attrib_iter::<Matrix3<f64>, CellIndex>("ref_shape_mtx_inv")
                    .unwrap(),
            ).zip(solid.tet_iter())
            .zip(solid.cell_iter())
            .zip(force_iter)
        {
            let dx = [
                // current displacement
                tet.0 - prev_pos[cell[0]],
                tet.1 - prev_pos[cell[1]],
                tet.2 - prev_pos[cell[2]],
                tet.3 - prev_pos[cell[3]],
            ];

            let dv = [
                // current displacement - previous displacement
                dx[0] - prev_vel[cell[0]],
                dx[1] - prev_vel[cell[1]],
                dx[2] - prev_vel[cell[2]],
                dx[3] - prev_vel[cell[3]],
            ];

            for i in 0..4 {
                gradient[cell[i]] += dt_inv * dt_inv * 0.25 * vol * density * dv[i];

                // Energy gradient is in opposite direction to the force hence minus here.
                gradient[cell[i]] -= 0.25 * vol * density * gravity;
                gradient[cell[i]] += grad[i];
            }

            // Needed for damping.
            let tet_energy = NeoHookeanTetEnergy::new(tet.shape_matrix(), DX_inv, vol, lambda, mu);
            let dH = tet_energy.elastic_energy_hessian_product(&dx);
            for i in 0..3 {
                // Damping
                gradient[cell[i]] += dt_inv * damping * dH[i];
                gradient[cell[3]] -= dt_inv * damping * dH[i];
            }
        }
        gradient
    }

    fn energy_hessian_size(&self) -> usize {
        NeoHookeanTetEnergy::NUM_HESSIAN_TRIPLETS * self.solid.num_cells()
    }

    #[allow(non_snake_case)]
    fn energy_hessian(&mut self) -> &[MatrixElementTriplet<f64>] {
        let num_hess_triplets = self.energy_hessian_size();
        let ElasticTetMeshEnergy {
            ref solid,
            material:
                MaterialModel {
                    lambda,
                    mu,
                    density,
                    damping,
                },
            time_step_inv: dt_inv,
            energy_hessian_triplets: ref mut hess,
            ..
        } = *self;

        // Ensure there are enough entries in our hessian triplet buffer.
        hess.resize(num_hess_triplets, MatrixElementTriplet::new(0, 0, 0.0));

        {
            // Break up the hessian triplets into chunks of elements for each tet.
            let hess_chunks: &mut [[MatrixElementTriplet<f64>;
                                       NeoHookeanTetEnergy::NUM_HESSIAN_TRIPLETS]] =
                reinterpret_mut_slice(hess);

            let hess_iter = hess_chunks
                .par_iter_mut()
                .zip(
                    solid
                        .attrib_as_slice::<f64, CellIndex>("ref_volume")
                        .unwrap()
                        .par_iter(),
                ).zip(
                    solid
                        .attrib_as_slice::<Matrix3<f64>, CellIndex>("ref_shape_mtx_inv")
                        .unwrap()
                        .par_iter(),
                ).zip(solid.cells().par_iter())
                .zip(
                    solid
                        .cells()
                        .par_iter()
                        .map(|tet| solid.tet_from_indices(tet).shape_matrix()),
                );

            hess_iter.for_each(|((((tet_hess, &vol), &DX_inv), cell), Dx)| {
                //let tet_energy = NeoHookeanTetEnergy::new(tet.shape_matrix(), DX_inv, vol, lambda, mu);
                let F = Dx * DX_inv;
                let J = F.determinant();
                if J <= 0.0 {
                    return;
                }
                let A = DX_inv * DX_inv.transpose();
                // Theoretically we known Dx is invertible since F is, but it could have
                // numerical differences, so we check anyways.
                let Dx_inv_tr = match Dx.inverse_transpose() {
                    Some(inv) => inv,
                    None => return,
                };

                let alpha = mu - lambda * J.ln();
                let factor = 1.0 + damping * dt_inv;

                let mut i = 0; // triplet index for the tet. there should be 78 in total

                // Off-diagonal elements
                for col in 0..3 {
                    for row in 0..3 {
                        let mut last_hess = [0.0; 4];
                        for k in 0..3 {
                            // which vertex
                            let mut last_wrt_hess = 0.0;
                            for n in 0..3 {
                                // with respect to which vertex
                                let c_lambda = lambda * Dx_inv_tr[n][row] * Dx_inv_tr[k][col];
                                let c_alpha = alpha * Dx_inv_tr[n][col] * Dx_inv_tr[k][row];
                                let mut h = factor * vol * (c_alpha + c_lambda);
                                if col == row {
                                    h += factor * vol * mu * A[k][n];
                                }
                                last_wrt_hess -= h;
                                last_hess[n] -= h;

                                if col == row && k == n {
                                    h += 0.25 * vol * density * dt_inv * dt_inv;
                                }

                                // skip upper trianglar part of the global hessian.
                                if (cell[n] == cell[k] && row >= col) || cell[n] > cell[k] {
                                    tet_hess[i] = MatrixElementTriplet::new(
                                        3 * cell[n] + row,
                                        3 * cell[k] + col,
                                        h,
                                    );
                                    i += 1;
                                }
                            }

                            // with respect to last vertex
                            last_hess[3] -= last_wrt_hess;
                            if cell[3] > cell[k] {
                                tet_hess[i] = MatrixElementTriplet::new(
                                    3 * cell[3] + row,
                                    3 * cell[k] + col,
                                    last_wrt_hess,
                                );
                                i += 1;
                            }
                        }

                        // last vertex
                        for n in 0..4 {
                            // with respect to which vertex
                            if (cell[n] == cell[3] && row >= col) || cell[n] > cell[3] {
                                let mut h = last_hess[n];
                                if col == row && 3 == n {
                                    h += 0.25 * vol * density * dt_inv * dt_inv;
                                }
                                tet_hess[i] = MatrixElementTriplet::new(
                                    3 * cell[n] + row,
                                    3 * cell[3] + col,
                                    h,
                                );
                                i += 1;
                            }
                        }
                    }
                }
                assert_eq!(i, NeoHookeanTetEnergy::NUM_HESSIAN_TRIPLETS);
            });
        }
        hess
    }
}
