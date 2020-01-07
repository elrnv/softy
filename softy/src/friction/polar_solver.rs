#![allow(dead_code)]
use super::FrictionParams;
use super::FrictionSolveResult;
use ipopt::{self, Ipopt, Index, Number};
use reinterpret::*;
use num_traits::Zero;

use utils::soap::*;
use utils::zip;

use crate::contact::*;
use crate::Error;

/// Friction solver.
pub struct FrictionPolarSolver<'a> {
    /// Non-linear solver.
    solver: Ipopt<SemiImplicitFrictionPolarProblem<'a>>,
}

impl<'a> FrictionPolarSolver<'a> {
    /// Build a new solver for the friction problem. The given `predictor_impulse` is a stacked
    /// vector of tangential velocities for each contact point in contact space. `contact_impulse` is
    /// the normal component of the predictor frictional contact impulse at each contact point.
    /// Finally, `mu` is the friction coefficient.
    pub(crate) fn new(
        predictor_impulse: &'a [[f64; 3]],
        contact_impulse: &'a [f64],
        contact_basis: &'a ContactBasis,
        mass_inv_mtx: EffectiveMassInvView<'a>,
        params: FrictionParams,
    ) -> Result<FrictionPolarSolver<'a>, Error> {
        let basis_mtx = contact_basis.tangent_basis_matrix();
        let hessian = mass_inv_mtx.clone()
            .diagonal_congruence_transform(basis_mtx.view());

        let mu = params.dynamic_friction;

        let scale = predictor_impulse.iter().map(|&p| Vector3::new(p).norm_squared()).max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Less)).unwrap_or(1.0);
        let scale = 1.0 / scale.sqrt();

        let predictor_impulse: Chunked3<Vec<f64>> = 
            predictor_impulse.iter()
            .map(|&p| (Vector3::new(p) * scale).into_data())
            .collect();

        let problem = SemiImplicitFrictionPolarProblem(FrictionPolarProblem {
            predictor_impulse,
            contact_impulse,
            contact_basis,
            mu,
            mass_inv_mtx,
            hessian,
            iterations: 0,
            scale, 
        });

        let mut ipopt = Ipopt::new_newton(problem)?;
        ipopt.set_option("print_level", params.print_level as i32);
        ipopt.set_option("tol", params.tolerance);
        ipopt.set_option("sb", "yes");
        ipopt.set_option("nlp_scaling_max_gradient", 1.0);
        ipopt.set_option("mu_strategy", "adaptive");
        //ipopt.set_option("derivative_test", "second-order");
        //ipopt.set_option("print_timing_statistics", "yes");
        ipopt.set_option("max_iter", params.inner_iterations as i32);
        ipopt.set_intermediate_callback(Some(SemiImplicitFrictionPolarProblem::intermediate_cb));

        Ok(FrictionPolarSolver { solver: ipopt })
    }

    /// Solve one step without updating the mesh. This function is useful for testing and
    /// benchmarking. Otherwise it is intended to be used internally.
    pub fn step(&mut self) -> Result<FrictionSolveResult, Error> {
        // Solve non-linear problem
        let ipopt::SolveResult {
            // unpack ipopt result
            solver_data,
            objective_value,
            status,
            ..
        } = self.solver.solve();

        let polar_solution: &[Polar2<f64>] = reinterpret_slice(solver_data.solution.primal_variables);
        let tangent_sol: Vec<[f64; 2]> = polar_solution.iter().map(|&p| {
            let mut scaled_p = p;
            scaled_p.radius /= solver_data.problem.0.scale;
            scaled_p.to_euclidean()
        }).collect();

        let result = FrictionSolveResult {
            objective_value,
            solution: tangent_sol,
            iterations: solver_data.problem.0.iterations
        };

        match status {
            ipopt::SolveStatus::SolveSucceeded | ipopt::SolveStatus::SolvedToAcceptableLevel => {
                Ok(result)
            }
            e => Err(Error::FrictionSolveError { status: e }),
        }
    }
}

#[inline]
pub(crate) fn polar_gradient(p: Polar2<f64>) -> Matrix2<f64> {
    Matrix2::new([[p.angle.cos(), p.angle.sin()],
                 [-p.radius * p.angle.sin(), p.radius * p.angle.cos()]])
}

#[inline]
pub(crate) fn polar_hessian_product(p: Polar2<f64>, mult: Vector2<f64>) -> Matrix2<f64> {
    Matrix2::new(
        [[0.0, mult[1] * p.angle.cos() - mult[0] * p.angle.sin()],
        [mult[1] * p.angle.cos() - mult[0] * p.angle.sin(),
        -mult[0] * p.radius * p.angle.cos() - mult[1] * p.radius * p.angle.sin()]])
}


pub(crate) struct FrictionPolarProblem<'a> {
    predictor_impulse: Chunked3<Vec<f64>>,
    /// A set of contact impulses for each contact point.
    contact_impulse: &'a [f64],
    /// Basis defining the normal and tangent space at each point of contact.
    contact_basis: &'a ContactBasis,
    /// Coefficient of dynamic friction.
    mu: f64,
    /// Effective mass matrix.
    mass_inv_mtx: EffectiveMassInvView<'a>,
    /// Worspace hessian.
    hessian: DSBlockMatrix2,
    /// Iteration count,
    iterations: u32,
    scale: f64,
}

impl FrictionPolarProblem<'_> {
    pub fn num_contacts(&self) -> usize {
        self.predictor_impulse.len()
    }
    pub fn num_variables(&self) -> usize {
        2 * self.num_contacts()
    }

    pub fn bounds(&self, x_l: &mut [Number], x_u: &mut [Number]) -> bool {
        let x_l: &mut [Polar2<f64>] = reinterpret_mut_slice(x_l);
        let x_u: &mut [Polar2<f64>] = reinterpret_mut_slice(x_u);
        for (l, u, &cr) in zip!(x_l.iter_mut(), x_u.iter_mut(), self.contact_impulse.iter()) {
            // First coordinate is the radius and second is the angle.
            l.radius = 0.0; // radius is never negative
            u.radius = self.mu * cr.abs() * self.scale;
            if u.radius == 0.0 {
                l.angle = 0.0;
                u.angle = 0.0;
            } else {
                // Angle is unconstrained.
                l.angle = -2e19;
                u.angle = 2e19;
            }
        }
        true
    }

    pub fn initial_point(&self, r: &mut [Number]) -> bool {
        let impulses: &mut [Polar2<f64>] = reinterpret_mut_slice(r);
        for (i, (r, &cr, &p)) in zip!(
            impulses.iter_mut(),
            self.contact_impulse.iter(),
            self.predictor_impulse.iter()
        ).enumerate() {
            let p = self.contact_basis.to_cylindrical_contact_coordinates(p, i);
            let rad = self.mu * cr.abs() * self.scale;
            if rad > 0.0 {
                if p.tangent.radius > rad {
                    r.radius = rad;
                } else {
                    r.radius = p.tangent.radius;
                }
                r.angle = p.tangent.angle;
            } else {
                r.radius = 0.0;
                r.angle = 0.0;
            }
        }
        true
    }
}

/// Return an angle in [-PI, PI] at 90 degrees to the given angle. The input angle is given in
/// radians.
pub fn negate_angle(angle: f64) -> f64 {
    normalize_angle(angle + std::f64::consts::PI)
}

/// Given an angle in radians, translate it to the canonical [-PI, PI] range. We choose this range
/// because this what atan2 maps to. This function doesn't distinguish between -PI and PI, so the
/// normalized angles overlap at PI and -PI.
fn normalize_angle(angle: f64) -> f64 {
    use std::f64::consts::PI;
    (angle + PI) % (2.0 * PI) + if angle < 0.0 { PI } else { -PI }
}

/// Semi-implicit Friction problem is one step more accurate than the explicit one.
pub(crate) struct SemiImplicitFrictionPolarProblem<'a>(FrictionPolarProblem<'a>);

impl SemiImplicitFrictionPolarProblem<'_> {
    pub fn intermediate_cb(&mut self, _: ipopt::IntermediateCallbackData) -> bool {
        self.0.iterations += 1;
        true
    }
}

/// Prepare the problem for Newton iterations.
impl ipopt::BasicProblem for SemiImplicitFrictionPolarProblem<'_> {
    fn num_variables(&self) -> usize {
        self.0.num_variables()
    }

    fn bounds(&self, x_l: &mut [Number], x_u: &mut [Number]) -> bool {
        self.0.bounds(x_l, x_u)
    }

    fn initial_point(&self, r: &mut [Number]) -> bool {
        self.0.initial_point(r)
    }

    fn objective(&self, r_p: &[Number], obj: &mut Number) -> bool {
        let polar_impulses: &[Polar2<f64>] = reinterpret_slice(r_p);
        assert_eq!(self.0.predictor_impulse.len(), polar_impulses.len());

        let mut r: Chunked3<Vec<f64>> = Chunked3::from_array_vec(self.0.contact_basis.from_polar_tangent_space(polar_impulses));
        *&mut r.expr_mut() -= self.0.predictor_impulse.expr();

        let rhs = self.0.mass_inv_mtx.view() * *r.view().as_tensor();

        *obj = 0.5 * r.expr().dot::<f64, _>(rhs.expr());

        true
    }

    fn objective_grad(&self, r_p: &[Number], grad_f: &mut [Number]) -> bool {
        let polar_impulses: &[Polar2<Number>] = reinterpret_slice(r_p);
        let polar_grad: &mut [[Number; 2]] = reinterpret_mut_slice(grad_f);
        assert_eq!(self.0.predictor_impulse.len(), polar_grad.len());

        let mut r: Chunked3<Vec<f64>> = Chunked3::from_array_vec(self.0.contact_basis.from_polar_tangent_space(polar_impulses));
        *&mut r.expr_mut() -= self.0.predictor_impulse.expr();

        let grad = self.0.mass_inv_mtx.view() * *r.view().as_tensor();
        self.0.contact_basis
            .to_tangent_space(grad.view().into_data().into())
            .zip(polar_impulses.iter())
            .zip(polar_grad.iter_mut())
            .for_each(|((g, &r_p), out_g)| {
                let g = Vector2::new(g);
                let d = polar_gradient(r_p);
                *out_g = (d*g).into();
            });

        true
    }
}

impl ipopt::NewtonProblem for SemiImplicitFrictionPolarProblem<'_> {
    fn num_hessian_non_zeros(&self) -> usize {
        let mut idx = 0;
        for (row_idx, row) in self.0.hessian.as_data().iter().enumerate() {
            for col_idx in row.index_iter() {
                if row_idx > col_idx {
                    idx += 4;
                } else if row_idx == col_idx {
                    idx += 3;
                }
            }
        }
        idx
    }

    fn hessian_indices(&self, rows: &mut [Index], cols: &mut [Index]) -> bool {
        let mut idx = 0;
        for (row_idx, row) in self.0.hessian.as_data().iter().enumerate() {
            for col_idx in row.index_iter() {
                if row_idx < col_idx {
                    continue;
                }
                rows[idx] = (2 * row_idx) as Index;
                cols[idx] = (2 * col_idx) as Index;
                idx += 1;

                if row_idx > col_idx {
                    rows[idx] = (2 * row_idx) as Index;
                    cols[idx] = (2 * col_idx + 1) as Index;
                    idx += 1;
                }

                rows[idx] = (2 * row_idx + 1) as Index;
                cols[idx] = (2 * col_idx) as Index;
                idx += 1;

                rows[idx] = (2 * row_idx + 1) as Index;
                cols[idx] = (2 * col_idx + 1) as Index;
                idx += 1;
            }
        }
        assert_eq!(idx, rows.len());
        assert_eq!(idx, cols.len());

        true
    }
    fn hessian_values(&self, r: &[Number], vals: &mut [Number]) -> bool {
        let r_p: &[Polar2<Number>] = reinterpret_slice(r);

        let mut r: Chunked3<Vec<f64>> = Chunked3::from_array_vec(self.0.contact_basis.from_polar_tangent_space(r_p));
        *&mut r.expr_mut() -= self.0.predictor_impulse.expr();

        let grad = (self.0.mass_inv_mtx.view() * *r.view().as_tensor()).into_data();
        let grad_t: Vec<[f64; 2]> = self.0.contact_basis
            .to_tangent_space(grad.view().into_data().into()).collect();

        let mut idx = 0;
        for (row_idx, row) in self.0.hessian.as_data().iter().enumerate() {
            for (col_idx, block) in row.indexed_source_iter() {
                if row_idx < col_idx {
                    continue;
                }

                let row_g = polar_gradient(r_p[row_idx]);
                let col_g_tr = polar_gradient(r_p[col_idx]).transpose();
                let polar_hess = if row_idx == col_idx {
                    polar_hessian_product(r_p[row_idx], grad_t[row_idx].into_tensor())
                } else {
                    Matrix2::zero()
                };

                let hess_block =
                    (row_g * *block.into_arrays().as_tensor() * col_g_tr + polar_hess)
                        .into_data();

                vals[idx] = hess_block[0][0];
                idx += 1;

                if row_idx > col_idx {
                    vals[idx] = hess_block[0][1];
                    idx += 1;
                }

                vals[idx] = hess_block[1][0];
                idx += 1;

                vals[idx] = hess_block[1][1];
                idx += 1;
            }
        }

        assert_eq!(idx, vals.len());

        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::*;
    //use crate::test_utils::*;
    //use crate::*;
    //use std::path::PathBuf;
    
    #[test]
    fn sliding_point() -> Result<(), Error> {
        let (velocity, impulse) = sliding_point_tester(0.000001)?;
        assert!(velocity[1] > 0.8);

        assert_relative_eq!(velocity[0], 0.0, max_relative = 1e-5, epsilon = 1e-8);
        assert_relative_eq!(impulse[0], 0.0, max_relative = 1e-5, epsilon = 1e-8);
        Ok(())
    }

    #[test]
    fn sticking_point() -> Result<(), Error> {
        let (velocity, impulse) = sliding_point_tester(1.5)?;
        assert_relative_eq!(velocity[1], 0.0, max_relative = 1e-5, epsilon = 1e-8);

        assert_relative_eq!(velocity[0], 0.0, max_relative = 1e-5, epsilon = 1e-8);
        assert_relative_eq!(impulse[0], 0.0, max_relative = 1e-5, epsilon = 1e-8);
        Ok(())
    }

    /// A point mass slides across a 2D surface in the positive x direction.
    fn sliding_point_tester(mu: f64) -> Result<(Vector2<f64>, Vector2<f64>), Error> {
        let params = FrictionParams {
            smoothing_weight: 0.0,
            friction_forwarding: 1.0,
            dynamic_friction: mu,
            inner_iterations: 30,
            tolerance: 1e-15,
            print_level: 0,
        };
        let mass = 1.0;

        let predictor_impulse = vec![[-1.0 * mass, 0.0, 0.0]]; // one point across
        let contact_force = vec![10.0 * mass];
        let mass_inv_mtx: DSBlockMatrix3 =
            DiagonalBlockMatrix::new(Chunked3::from_flat(vec![1.0 / mass; 3])).into();

        let mut contact_basis = ContactBasis::new();
        contact_basis.update_from_normals(vec![[0.0, 1.0, 0.0]]);

        let mut solver = FrictionPolarSolver::new(
            &predictor_impulse,
            &contact_force,
            &contact_basis,
            mass_inv_mtx.view(),
            params,
        )?;
        let result = solver.step()?;
        let FrictionSolveResult {
            solution,
            ..
        } = result;

        let impulse = Vector2::new(solution[0]);
        let new_vel = impulse / mass;
        let p_imp_t: Vec<_> = contact_basis.to_tangent_space(&predictor_impulse).collect();
        let prev_vel = -Vector2::new(p_imp_t[0]) / mass;
        let velocity = prev_vel + new_vel;

        Ok((velocity, impulse))
    }

    #[test]
    fn normalize_angle_test() {
        use std::f64::consts::PI;
        let inputs = vec![
            6.0 * PI,
            5.0 * PI,
            -5.0 * PI,
            3.0 * PI / 4.0,
            9.0 * PI / 4.0,
            3.0 * PI / 2.0,
            9.0 * PI / 2.0,
            10.0 * PI / 3.0,
            -9.0 * PI / 4.0,
            -9.0 * PI / 2.0,
            -10.0 * PI / 3.0,
        ];
        let expected = vec![
            0.0,
            -PI,
            PI,
            3.0 * PI / 4.0,
            PI / 4.0,
            -PI / 2.0,
            PI / 2.0,
            -2.0 * PI / 3.0,
            -PI / 4.0,
            -PI / 2.0,
            2.0 * PI / 3.0,
        ];

        for (i, o) in inputs.into_iter().zip(expected.into_iter()) {
            assert_relative_eq!(normalize_angle(i), o, max_relative = 1e-5, epsilon = 1e-8);
        }
    }

    #[test]
    fn negate_angle_test() {
        use std::f64::consts::PI;
        let inputs = vec![PI, 0.0, PI / 3.0, -PI / 4.0, 2.0 * PI / 3.0];
        let expected = vec![0.0, -PI, -2.0 * PI / 3.0, 3.0 * PI / 4.0, -PI / 3.0];

        for (i, o) in inputs.into_iter().zip(expected.into_iter()) {
            assert_relative_eq!(negate_angle(i), o, max_relative = 1e-5, epsilon = 1e-8);
        }
    }

    ///// Pinch a box between two probes.
    ///// Given sufficient friction, the box should not fall.
    //fn pinch_tester(
    //    sc_params: FrictionalContactParams,
    //) -> Result<(), Error> {
    //    use geo::mesh::attrib::Attrib;
    //    use geo::mesh::VertexPositions;
    //    use geo::mesh::topology::*;

    //    let params = SimParams {
    //        max_iterations: 200,
    //        max_outer_iterations: 20,
    //        gravity: [0.0f32, -9.81, 0.0],
    //        time_step: Some(0.01),
    //        print_level: 5,
    //        friction_iterations: 1,
    //        ..DYNAMIC_PARAMS
    //    };

    //    let material = Material {
    //        elasticity: ElasticityParameters::from_young_poisson(1e6, 0.45),
    //        ..SOLID_MATERIAL
    //    };

    //    let clamps = geo::io::load_polymesh(&PathBuf::from("assets/clamps.vtk"))?;
    //    let mut box_mesh = geo::io::load_tetmesh(&PathBuf::from("assets/box.vtk"))?;
    //    box_mesh.remove_attrib::<VertexIndex>("fixed")?;

    //    let mut solver = fem::SolverBuilder::new(params.clone())
    //        .solid_material(material)
    //        .add_solid(box_mesh)
    //        .add_shell(clamps)
    //        .smooth_contact_params(sc_params)
    //        .build()?;

    //    for _ in 0..50 {
    //        let res = solver.step()?;
    //
    //        //println!("res = {:?}", res);
    //        assert!(
    //            res.iterations <= params.max_outer_iterations,
    //            "Exceeded max outer iterations."
    //        );

    //        // Check that the mesh hasn't fallen.
    //        let tetmesh = solver.borrow_mesh();
    //
    //        for v in tetmesh.vertex_position_iter() {
    //            assert!(v[1] > -0.6);
    //        }
    //    }

    //    Ok(())
    //}

    ///// Pinch a box against a couple of implicit surfaces.
    //#[test]
    //fn pinch_against_implicit() -> Result<(), Error> {
    //    let sc_params = FrictionalContactParams {
    //        contact_type: ContactType::Point,
    //        kernel: KernelType::Cubic {
    //            radius_multiplier: 1.5,
    //        },
    //        friction_params: Some(FrictionParams {
    //            dynamic_friction: 0.4,
    //            inner_iterations: 40,
    //            tolerance: 1e-5,
    //            print_level: 5,
    //        }),
    //    };

    //    pinch_tester(sc_params)
    //}
}
