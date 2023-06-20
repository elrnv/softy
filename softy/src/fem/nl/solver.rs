use num_traits::ToPrimitive;
use std::io::Write;

use super::newton::*;
use super::problem::{NLProblem, NonLinearProblem};
use super::{NLSolver, SimParams, SolveResult, Status};
use crate::inf_norm;
use crate::nl_fem::{ContactViolation, ProblemInfo, StageResult, StepResult};
use crate::objects::tetsolid::*;
use crate::objects::trishell::*;
use crate::Real64;
use crate::{Error, Mesh, PointCloud};

mod builder;

pub use builder::*;

#[derive(Debug, Error)]
pub enum LogError {
    #[error("Log format: {}", .0)]
    Format(#[from] std::fmt::Error),
    #[error("Log IO: {}", .0)]
    IO(#[from] std::io::Error),
}

/// Finite element engine.
pub struct Solver<S, T> {
    /// Non-linear solver.
    solver: S,
    /// Simulation parameters. This is kept around for convenience.
    sim_params: SimParams,
    /// Maximal displacement length.
    ///
    /// Used to limit displacement which is necessary in contact scenarios
    /// because it defines how far a step we can take before the constraint
    /// Jacobian sparsity pattern changes. If zero, then no limit is applied but
    /// constraint Jacobian is kept sparse.
    max_step: f64,
    /// Structured solution to the problem in the solver.
    ///
    /// This is also used as warm start for subsequent steps.
    solution: Vec<T>,
    /// Step workspace vector to remember what the initial condition was.
    initial_point: Vec<T>,
    /// Counts the number of times `step` is called.
    iteration_count: u32,
}

impl<S, T> Solver<S, T>
where
    T: Real64 + na::RealField,
    S: NLSolver<NLProblem<T>, T>,
{
    /// Sets the interrupt checker to be called at every outer iteration.
    pub fn set_coarse_interrupter(&self, mut interrupted: impl FnMut() -> bool + Send + 'static) {
        *self.solver.outer_callback().borrow_mut() = Box::new(move |_| !interrupted());
    }

    /// Sets the interrupt checker to be called at every inner iteration.
    pub fn set_fine_interrupter(&self, mut interrupted: impl FnMut() -> bool + Send + 'static) {
        *self.solver.inner_callback().borrow_mut() = Box::new(move |_| !interrupted());
    }

    /// If the time step was not specified or specified to be zero, then this function will return
    /// zero.
    pub fn time_step(&self) -> f64 {
        self.sim_params.time_step.unwrap_or(0.0).into()
    }
    /// Get an immutable reference to the underlying problem.
    fn problem(&self) -> &NLProblem<T> {
        self.solver.problem()
    }

    /// Get a mutable reference to the underlying problem.
    pub(crate) fn problem_mut(&mut self) -> &mut NLProblem<T> {
        self.solver.problem_mut()
    }

    /// Get a slice of solid objects represented in this solver
    pub fn solid(&self) -> std::cell::Ref<TetSolid> {
        std::cell::Ref::map(self.problem().state.borrow(), |state| &state.solid)
    }

    /// Get a slice of shell objects represented in this solver.
    pub fn shell(&self) -> std::cell::Ref<TriShell> {
        std::cell::Ref::map(self.problem().state.borrow(), |state| &state.shell)
    }

    pub fn mesh(&self) -> Mesh {
        self.problem().mesh()
    }

    /// Get simulation parameters.
    pub fn params(&self) -> &SimParams {
        &self.sim_params
    }

    /// Update the maximal displacement allowed. If zero, no limit is applied.
    pub fn update_max_step(&mut self, step: f64) {
        self.max_step = step;
        self.problem_mut().update_max_step(step);
    }

    pub fn update_radius_multiplier(&mut self, rad: f64) {
        self.problem_mut().update_radius_multiplier(rad);
    }

    /// Update the solid meshes with the given points.
    pub fn update_vertices(&mut self, pts: &PointCloud) -> Result<(), Error> {
        self.problem_mut().update_vertices(pts)
    }

    /// Update the solid meshes with the given points.
    pub fn update_vertex_positions(&mut self, pos: impl AsRef<[[f64; 3]]>) -> Result<(), Error> {
        self.problem_mut().update_vertex_positions(pos.as_ref())
    }

    /// Returns the solved positions of the vertices in original order.
    pub fn vertex_positions(&self) -> Vec<[T; 3]> {
        self.problem().vertex_positions()
    }

    /// Update the `mesh` and `prev_pos` with the current solution.
    fn commit_solution(&mut self, relax_max_step: bool) {
        // Save as warm start for next step.
        self.initial_point.copy_from_slice(&self.solution);
        {
            let Self {
                solver, solution, ..
            } = self;

            // Advance internal state (positions and velocities) of the problem.
            solver.problem_mut().advance(solution);
            //TODO Remove debug code:
            // solution.iter_mut().for_each(|x| *x = T::zero());
        }

        // Reduce max_step for next iteration if the solution was a good one.
        if relax_max_step {
            let dt = self.time_step();
            if let Some(radius) = self.problem().min_contact_radius() {
                let step =
                    inf_norm(self.solution.iter().cloned()) * if dt > 0.0 { dt } else { 1.0 };
                let new_max_step = (step.to_f64().unwrap() - radius).max(self.max_step * 0.5);
                if self.max_step != new_max_step {
                    log::info!(
                        "Relaxing max step from {} to {}",
                        self.max_step,
                        new_max_step
                    );
                    self.max_step = new_max_step;
                    self.problem_mut().update_max_step(new_max_step);
                }
            }
        }
    }

    ///// Revert previously committed solution. We just advance in the opposite direction.
    //fn revert_solution(&mut self) {
    //    self.problem_mut().retreat();
    //}

    //fn initial_residual_error(&self) -> f64 {
    //    self.problem().initial_residual_error
    //}

    //fn save_current_active_constraint_set(&mut self) {
    //    let Solver {
    //        ref solver,
    //        ref mut old_active_constraint_set,
    //        ..
    //    } = self;
    //    old_active_constraint_set.clear();
    //    solver
    //        .solver_data()
    //        .problem
    //        .compute_active_constraint_set(old_active_constraint_set);
    //}

    //fn remap_warm_start(&mut self) {
    //    let Solver {
    //        solver,
    //        old_active_constraint_set,
    //        ..
    //    } = self;

    //    solver
    //        .solver_data_mut()
    //        .problem
    //        .remap_warm_start(old_active_constraint_set.view());
    //}

    //fn all_contacts_linear(&self) -> bool {
    //    self.problem().all_contacts_linear()
    //}

    /// Writes result to log if file is specified, otherwise does nothing.
    pub fn log_result(&self, step_result: &StepResult) -> Result<(), LogError> {
        if let Some(log_file) = self.sim_params.log_file.as_ref() {
            let mut file = std::fs::File::options().append(true).open(log_file)?;
            writeln!(file, "\nFrame: {}:\n{}", self.iteration_count, step_result)?;
        }
        Ok(())
    }

    /// Run the non-linear solver on one time step.
    pub fn step(&mut self) -> Result<StepResult, Error> {
        let dt = self.time_step();
        self.iteration_count += 1;

        let update_autodiff_state =
            matches!(self.sim_params.linsolve, LinearSolver::Iterative { .. })
                || self.sim_params.derivative_test > 2;
        let update_jacobian = !matches!(self.sim_params.linsolve, LinearSolver::Iterative { .. })
            || self.sim_params.derivative_test > 0;

        log::debug!("Updating constraint set...");
        self.solver.problem_mut().update_constraint_set(
            self.sim_params.should_compute_jacobian_matrix(),
            update_autodiff_state,
        );

        let mut contact_iterations = self.sim_params.contact_iterations as i64;

        let velocity_clear_steps = if self.sim_params.velocity_clear_frequency > 0.0 {
            (1.0 / (f64::from(self.sim_params.velocity_clear_frequency) * dt))
                .round()
                .to_u32()
                .unwrap_or(u32::MAX)
        } else {
            u32::MAX
        }
        .max(1);

        // Prepare warm start.
        //solver.problem().compute_warm_start(solution.as_mut_slice());

        let true_time_step = self.solver.problem_mut().time_step;
        let time_integration = self.sim_params.time_integration;

        log::debug!("Begin main nonlinear solve.");
        let mut step_result = StepResult::default();
        for stage in 0..time_integration.num_stages() {
            let (step_integrator, factor) = time_integration.step_integrator(stage);
            log::debug!("Single step integration scheme: {step_integrator:?}");
            self.solver.problem_mut().time_integration = step_integrator;
            self.solver.problem_mut().time_step = true_time_step * factor as f64;

            // Update the current vertex data using the current dof state.
            self.solver.problem_mut().update_cur_vertices();

            // Updates constraint state according to latest solution and saves old state for
            // lagged solves.
            self.solver.problem_mut().update_constraint_state(
                self.initial_point.as_slice(),
                update_jacobian,
                update_autodiff_state,
            );

            // No need to do this every time.
            self.solver.problem().check_jacobian(
                self.sim_params.derivative_test,
                self.solution.as_slice(),
                false,
            )?;

            let mut stage_result = StageResult::new(step_integrator);
            // Loop to resolve all contacts.
            let mut update_jacobian_indices = true;
            loop {
                // Start from initial point inside this loop explicitly. initial_point is not
                // updated when a bad step is taken (contact violation or max step violation).
                self.solution.copy_from_slice(&self.initial_point);

                /****    Main solve step    ****/
                let solve_result = self
                    .solver
                    .solve_with(self.solution.as_mut_slice(), update_jacobian_indices);
                // if stage < time_integration.num_stages() - 1 {
                //     match result.status {
                //         Status::Success | Status::MaximumIterationsExceeded => {
                //             // Commit all stages except the last immediately.
                //             // The last stage is subject to contact correction.
                //             self.commit_solution(false);
                //         }
                //         _ => {
                //             return Err(Error::NLSolveError { result });
                //         }
                //     }
                // }
                update_jacobian_indices = false;
                /*******************************/

                match solve_result.status {
                    Status::Success | Status::MaximumIterationsExceeded => {
                        let ContactViolation {
                            bump_ratio,
                            violation: contact_violation,
                            largest_penalty,
                            ..
                        } = self
                            .solver
                            .problem()
                            .contact_violation(self.solution.as_slice());

                        log::debug!("Bump ratio: {}", bump_ratio);
                        log::debug!("Kappa: {}", self.solver.problem().kappa);
                        log::debug!("Contact violation: {}", contact_violation);

                        // Save results for future reporting and analysis.
                        let problem_info = ProblemInfo {
                            total_contacts: self.solver.problem().num_contacts() as u64,
                            total_in_proximity: self.solver.problem().num_in_proximity() as u64,
                        };

                        contact_iterations -= 1;

                        if contact_iterations < 0 {
                            // Technically in this case contacts are not yet resolved, but we treat
                            // this leniently to allow for applications that need to resolve contacts
                            // gradually.
                            stage_result.solve_results.push((
                                problem_info,
                                SolveResult {
                                    status: Status::MaximumContactIterationsExceeded,
                                    ..solve_result
                                },
                            ));
                            self.commit_solution(false);
                            // Kill velocities if needed.
                            if self.iteration_count % velocity_clear_steps == 0 {
                                self.solver.problem_mut().clear_velocities();
                            }
                            break;
                        }

                        if contact_violation > 0.0 {
                            stage_result.contact_violations += 1;
                            self.solver.problem_mut().kappa *= bump_ratio.max(2.0);
                        }

                        let max_step_violation = self.solver.problem().max_step_violation();
                        if max_step_violation {
                            stage_result.max_step_violations += 1;
                            //log::warn!("Max is step violated. Increase kernel radius to ensure all Jacobians are accurate.")
                            self.solver.problem_mut().update_constraint_set(
                                self.sim_params.should_compute_jacobian_matrix(),
                                false,
                            );
                            update_jacobian_indices = true;
                        }

                        stage_result
                            .solve_results
                            .push((problem_info, solve_result));

                        if contact_violation > 0.0 {
                            continue;
                        } else {
                            // Relax kappa
                            if largest_penalty == 0.0
                                && self.solver.problem().kappa
                                    > 1.0e2 / self.sim_params.contact_tolerance as f64
                            {
                                self.solver.problem_mut().kappa /= 2.0;
                            }
                            self.commit_solution(false);
                            // Kill velocities if needed.
                            if self.iteration_count % velocity_clear_steps == 0 {
                                self.solver.problem_mut().clear_velocities();
                            }
                            break;
                        }
                    }
                    _ => {
                        return Err(Error::NLSolveError {
                            result: solve_result,
                        });
                    }
                }
            }
            step_result.stage_solves.push(stage_result);
        }
        Ok(step_result)
    }
}
