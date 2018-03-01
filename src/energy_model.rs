//use std::path::Path;
//use geo::io::save_tetmesh;
use energy::*;
use geo::topology::*;
use geo::mesh::Attrib;
use geo::ops::*;
use geo::math::{Matrix3, Vector3};
use TetMesh;
use ipopt::{self, Index, Number};
use util;

/// Non-linear problem.
pub struct NeohookeanEnergyModel<'a, F: Fn() -> bool> {
    pub body: &'a mut TetMesh,
    pub energy_count: u32,
    /// Position from the previous time step.
    pub prev_pos: Vec<Vector3<f64>>,
    /// Time step scaled velocity (delta of pos) from the previous time step.
    pub prev_vel: Vec<Vector3<f64>>,
    interrupt_checker: F,
    material: ElasticMaterial,
    dynamics: Option<DynamicsParams>,
    //energy_hessian: Vec<MatrixElementTriplet<f64>>,
}

#[derive(Copy, Clone, Debug, PartialEq)]
struct DynamicsParams {
    /// Delta in time. The dynamic simulation is advanced by this much time at every iteration.
    pub time_step: f64,
    /// Coefficient measuring the amount of artificial viscosity as dictated by the Rayleigh
    /// damping model.
    pub damping: f64,
}

#[derive(Copy, Clone, Debug, PartialEq)]
struct ElasticMaterial {
    /// First Lame coefficient.
    pub lambda: f64,
    /// Second Lame coefficient.
    pub mu: f64,
    /// The density of the material.
    pub density: f64,
}

impl<'a, F: Fn() -> bool> NeohookeanEnergyModel<'a, F> {
    pub fn new(tetmesh: &'a mut TetMesh, interrupt_checker: F) -> Self {
        let prev_pos = util::reinterpret_slice(tetmesh.vertex_positions()).to_vec();
        let prev_vel = util::reinterpret_slice(
            tetmesh
                .attrib_as_slice::<[f64; 3], VertexIndex>("vel")
                .unwrap(),
        ).to_vec();
        NeohookeanEnergyModel {
            body: tetmesh,
            energy_count: 0u32,
            prev_pos,
            prev_vel,
            material: ElasticMaterial {
                lambda: 5.4,
                mu: 263.1,
                density: 1000.0,
            },
            dynamics: None,
            interrupt_checker,
        }
    }

    // Builder routines.

    /// Set the elastic material properties of the volumetric body discretized by the tetmesh.
    pub fn material(mut self, lambda: f64, mu: f64, density: f64) -> Self {
        self.material = ElasticMaterial {
            lambda,
            mu,
            density,
        };
        self
    }

    /// Set the dynamics parameters making the simulation advance in time by the given time step.
    /// Without these parameters the simulation will assume an infinite time-step making it a
    /// quasi-static model.
    pub fn dynamics(mut self, time_step: f64, damping: f64) -> Self {
        self.dynamics = Some(DynamicsParams { time_step, damping });
        self
    }

    /// Intermediate callback for `Ipopt`.
    pub fn intermediate_cb(
        &mut self,
        _alg_mod: Index,
        _iter_count: Index,
        _obj_value: Number,
        _inf_pr: Number,
        _inf_du: Number,
        _mu: Number,
        _d_norm: Number,
        _regularization_size: Number,
        _alpha_du: Number,
        _alpha_pr: Number,
        _ls_trials: Index,
    ) -> bool {
        (self.interrupt_checker)()
    }

    /// Update the tetmesh vertex positions.
    pub fn update(&mut self, x: &[Number]) {
        let x_slice: &[[f64; 3]] = util::reinterpret_slice(x);
        let verts = self.body.vertex_positions_mut();
        verts.copy_from_slice(x_slice);
    }
}

impl<'a, F: Fn() -> bool> ipopt::BasicProblem for NeohookeanEnergyModel<'a, F> {
    fn num_variables(&self) -> usize {
        self.body.num_verts() * 3
    }

    fn bounds(&self) -> (Vec<Number>, Vec<Number>) {
        let n = self.num_variables();
        let mut lo = Vec::with_capacity(n);
        let mut hi = Vec::with_capacity(n);
        // unbounded
        lo.resize(n, -2e19);
        hi.resize(n, 2e19);
        (lo, hi)
    }

    fn initial_point(&self) -> Vec<Number> {
        util::reinterpret_slice(self.body.vertex_positions()).to_vec()
    }

    fn objective(&mut self, x: &[Number], obj: &mut Number) -> bool {
        self.update(x);

        //save_tetmesh(
        //    self.body,
        //    Path::new(format!("./tetmesh_{}.vtk", self.energy_count).as_str()),
        //);
        self.energy_count += 1;

        *obj = self.energy();
        true
    }

    fn objective_grad(&mut self, x: &[Number], grad_f: &mut [Number]) -> bool {
        self.update(x);
        self.energy_gradient(util::reinterpret_mut_slice(grad_f));

        true
    }
}

impl<'a, F: Fn() -> bool> ipopt::NewtonProblem for NeohookeanEnergyModel<'a, F> {
    fn num_hessian_non_zeros(&self) -> usize {
        self.energy_hessian_size()
    }
    fn hessian_indices(&mut self, rows: &mut [Index], cols: &mut [Index]) -> bool {
        let mut hess = Vec::new();
        hess.resize(
            self.energy_hessian_size(),
            MatrixElementTriplet::new(0, 0, 0.0),
        );

        self.energy_hessian(hess.as_mut_slice());
        for (i, &MatrixElementTriplet { ref idx, .. }) in hess.iter().enumerate() {
            rows[i] = idx.row as Index;
            cols[i] = idx.col as Index;
        }

        true
    }
    fn hessian_values(&mut self, x: &[Number], vals: &mut [Number]) -> bool {
        self.update(x);
        let mut hess = Vec::new();
        hess.resize(
            self.energy_hessian_size(),
            MatrixElementTriplet::new(0, 0, 0.0),
        );

        self.energy_hessian(hess.as_mut_slice());
        for (i, elem) in hess.iter().enumerate() {
            vals[i] = elem.val as Number;
        }

        true
    }
}

/// Define energy for Neohookean materials.
impl<'a, F: Fn() -> bool> Energy<f64> for NeohookeanEnergyModel<'a, F> {
    #[allow(non_snake_case)]
    fn energy(&self) -> f64 {
        let dynamics_params = self.dynamics;
        let ElasticMaterial {
            lambda,
            mu,
            density,
        } = self.material;
        self.body
            .attrib_iter::<f64, CellIndex>("ref_volume")
            .unwrap()
            .zip(
                self.body
                    .attrib_iter::<Matrix3<f64>, CellIndex>("ref_shape_mtx_inv")
                    .unwrap(),
            )
            .zip(self.body.cell_iter())
            .zip(self.body.tet_iter())
            .map(|(((&vol, &Dm_inv), cell), tet)| {
                let F = tet.shape_matrix() * Dm_inv;
                let I = F.clone().map(|x| x * x).sum(); // tr(F^TF)
                let J = F.determinant();
                if J <= 0.0 {
                    ::std::f64::INFINITY
                } else {
                    let logJ = J.ln();
                    vol * (0.5 * mu * (I - 3.0) - mu * logJ + 0.5 * lambda * logJ * logJ)
                        + if let Some(DynamicsParams { time_step, .. }) = dynamics_params {
                            let dxTdx: f64 = [
                                tet.a - self.prev_pos[cell[0]] - self.prev_vel[cell[0]],
                                tet.b - self.prev_pos[cell[1]] - self.prev_vel[cell[1]],
                                tet.c - self.prev_pos[cell[2]] - self.prev_vel[cell[2]],
                                tet.d - self.prev_pos[cell[3]] - self.prev_vel[cell[3]],
                            ].into_iter()
                                .map(|&x| x.dot(x))
                                .sum();
                            0.5 * 0.25 * vol * density * dxTdx / (time_step * time_step)
                        } else {
                            0.0
                        }
                }
            })
            .sum()
    }

    #[allow(non_snake_case)]
    fn energy_gradient(&self, vtx_grad: &mut [Vector3<Number>]) {
        let dynamics_params = self.dynamics;
        let ElasticMaterial {
            lambda,
            mu,
            density,
        } = self.material;
        let force_iter = self.body
            .attrib_iter::<f64, CellIndex>("ref_volume")
            .unwrap()
            .zip(
                self.body
                    .attrib_iter::<Matrix3<f64>, CellIndex>("ref_shape_mtx_inv")
                    .unwrap(),
            )
            .zip(self.body.tet_iter())
            .map(|((&vol, &Dm_inv), tet)| {
                let F = tet.shape_matrix() * Dm_inv;
                let J = F.determinant();
                if J <= 0.0 {
                    Matrix3::zeros()
                } else {
                    let F_inv_tr = F.inverse_transpose().unwrap();
                    let logJ = J.ln();
                    vol * (mu * F + (lambda * logJ - mu) * F_inv_tr) * Dm_inv.transpose()
                }
            });

        // Clear gradient vector.
        for v in vtx_grad.iter_mut() {
            *v = Vector3::zeros();
        }

        // Transfer forces from cell-vertices to vertices themeselves
        for (((&vol, tet), cell), grad) in self.body
            .attrib_iter::<f64, CellIndex>("ref_volume")
            .unwrap()
            .zip(self.body.tet_iter())
            .zip(self.body.cell_iter())
            .zip(force_iter)
        {
            if let Some(DynamicsParams { time_step, .. }) = dynamics_params {
                let dx_tet = [
                    tet.a - self.prev_pos[cell[0]] - self.prev_vel[cell[0]],
                    tet.b - self.prev_pos[cell[1]] - self.prev_vel[cell[1]],
                    tet.c - self.prev_pos[cell[2]] - self.prev_vel[cell[2]],
                    tet.d - self.prev_pos[cell[3]] - self.prev_vel[cell[3]],
                ];

                for i in 0..4 {
                    vtx_grad[cell[i]] += 0.25 * vol * density * dx_tet[i] / (time_step * time_step);
                }
            }
            for i in 0..3 {
                vtx_grad[cell[i]] += grad[i];
                vtx_grad[cell[3]] -= grad[i];
            }
        }
    }

    fn energy_hessian_size(&self) -> usize {
        78 * self.body.num_cells() // There are 4*6 + 3*9*4/2 = 78 triplets per tet (overestimate)
    }

    #[allow(non_snake_case)]
    fn energy_hessian(&self, hess: &mut [MatrixElementTriplet<f64>]) {
        let dynamics_params = self.dynamics;
        let ElasticMaterial {
            lambda,
            mu,
            density,
        } = self.material;
        let hess_iter = self.body
            .attrib_iter::<f64, CellIndex>("ref_volume")
            .unwrap()
            .zip(
                self.body
                    .attrib_iter::<Matrix3<f64>, CellIndex>("ref_shape_mtx_inv")
                    .unwrap(),
            )
            .zip(self.body.cell_iter())
            .zip(self.body.tet_iter());

        let mut triplet_idx = 0;
        let mut push_elem = |row, col, val| {
            hess[triplet_idx] = MatrixElementTriplet::new(row, col, val);
            triplet_idx += 1;
        };

        for (((&vol, &Dm_inv), cell), tet) in hess_iter {
            let Ds = tet.shape_matrix();
            let F = Ds * Dm_inv;
            let J = F.determinant();
            if J > 0.0 {
                let A = Dm_inv * Dm_inv.transpose();
                // Theoretically we known Ds is invertible since F is, but it could have
                // numerical differences.
                let Ds_inv_tr = match Ds.inverse_transpose() {
                    Some(inv) => inv,
                    None => break,
                };

                let alpha = mu - lambda * J.ln();

                // Off-diagonal elements
                for col in 0..3 {
                    for row in 0..3 {
                        let mut last_hess = [0.0; 4];
                        for k in 0..3 {
                            // which vertex
                            let mut last_wrt_hess = 0.0;
                            for n in 0..3 {
                                // with respect to which vertex
                                let c_lambda = lambda * Ds_inv_tr[n][row] * Ds_inv_tr[k][col];
                                let c_alpha = alpha * Ds_inv_tr[n][col] * Ds_inv_tr[k][row];
                                let mut h = vol * (c_alpha + c_lambda);
                                if col == row {
                                    h += vol * mu * A[k][n];
                                }
                                last_wrt_hess -= h;
                                last_hess[n] -= h;

                                if col == row && k == n {
                                    if let Some(DynamicsParams { time_step, .. }) = dynamics_params
                                    {
                                        h += 0.25 * vol * density / (time_step * time_step);
                                    }
                                }

                                // skip upper trianglar part of the global hessian.
                                if (cell[n] == cell[k] && row >= col) || cell[n] > cell[k] {
                                    push_elem(3 * cell[n] + row, 3 * cell[k] + col, h);
                                }
                            }

                            // with respect to last vertex
                            last_hess[3] -= last_wrt_hess;
                            if cell[3] > cell[k] {
                                push_elem(3 * cell[3] + row, 3 * cell[k] + col, last_wrt_hess);
                            }
                        }

                        // last vertex
                        for n in 0..4 {
                            // with respect to which vertex
                            if (cell[n] == cell[3] && row >= col) || cell[n] > cell[3] {
                                let mut h = last_hess[n];
                                if col == row && 3 == n {
                                    if let Some(DynamicsParams { time_step, .. }) = dynamics_params
                                    {
                                        h += 0.25 * vol * density / (time_step * time_step);
                                    }
                                }
                                push_elem(3 * cell[n] + row, 3 * cell[3] + col, h);
                            }
                        }
                    }
                }
            }
        }
    }
}
