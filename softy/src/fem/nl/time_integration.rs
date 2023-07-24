use serde::{Deserialize, Serialize};

/// Time integration method.
#[derive(Copy, Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum TimeIntegration {
    /// Backward Euler integration.
    BE,
    /// Trapezoid Rule integration.
    TR,
    /// Second order Backward Differentiation Formula.
    BDF2,
    /// TR followed by BDF2.
    ///
    /// The given float indicates the fraction of the TR step.
    TRBDF2(f32),
    /// BE followed by a linear interpolation between next and current forces.
    SDIRK2,
}

impl Default for TimeIntegration {
    fn default() -> Self {
        TimeIntegration::BE
    }
}

impl TimeIntegration {
    /// Given the stage index determines which single step integrator to use.
    pub fn step_integrator(&self, stage: u8) -> (SingleStepTimeIntegration, f64) {
        match *self {
            TimeIntegration::BE => (SingleStepTimeIntegration::BE, 1.0),
            TimeIntegration::TR => (SingleStepTimeIntegration::TR, 1.0),
            TimeIntegration::BDF2 => (SingleStepTimeIntegration::BDF2, 1.0),
            TimeIntegration::TRBDF2(t) => {
                if stage % 2 == 0 {
                    (SingleStepTimeIntegration::TR, t as f64)
                } else {
                    // The factor is builtin to the MixedBDF2 single step because it is unique to TRBDF2.
                    (SingleStepTimeIntegration::MixedBDF2(1.0 - t), 1.0)
                }
            }
            TimeIntegration::SDIRK2 => {
                if stage % 2 == 0 {
                    let factor = 1.0 - 0.5 * 2.0_f64.sqrt();
                    (SingleStepTimeIntegration::BE, factor)
                } else {
                    // The factor is builtin to the SDRIK2 single step because it is unique.
                    (SingleStepTimeIntegration::SDIRK2, 1.0)
                }
            }
        }
    }

    pub fn num_stages(&self) -> u8 {
        match self {
            TimeIntegration::BE | TimeIntegration::TR | TimeIntegration::BDF2 => 1,
            TimeIntegration::TRBDF2(_) | TimeIntegration::SDIRK2 => 2,
        }
    }
}

/// Time integration method.
#[derive(Copy, Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum SingleStepTimeIntegration {
    /// Backward Euler integration.
    BE,
    /// Trapezoid Rule integration.
    TR,
    /// Second order Backward Differentiation Formula.
    BDF2,
    /// Second order Backward Differentiation Formula mixed with another integrator within a single step.
    ///
    /// The specified float indicates how much of the BDF2 step should be taken.
    MixedBDF2(f32),
    /// Second stage of the SDIRK2 scheme.
    ///
    /// This interpolates between explicit and implicit steps, but uses the `prev` velocity and
    /// position during the advance step.
    SDIRK2,
}

impl SingleStepTimeIntegration {
    /// Returns the fraction of the implicit step represented by this single step integrator.
    pub fn implicit_factor(&self) -> f32 {
        match self {
            SingleStepTimeIntegration::BE => 1.0,
            SingleStepTimeIntegration::TR => 0.5,
            SingleStepTimeIntegration::BDF2 => 2.0 / 3.0,
            SingleStepTimeIntegration::MixedBDF2(t) => t / (1.0 + t),
            SingleStepTimeIntegration::SDIRK2 => 1.0 - 0.5 * 2.0_f32.sqrt(),
        }
    }
    /// Returns the fraction of the explicit step represented by this single step integrator.
    ///
    /// This is not always `1 - implicit_factor` (e.g. check BDF2).
    pub fn explicit_factor(&self) -> f32 {
        match self {
            SingleStepTimeIntegration::BE => 0.0,
            SingleStepTimeIntegration::TR => 0.5,
            // BDF2 objective is computed same as BE, but note that vtx.cur is different.
            // vtx.cur is set in update_cur_vertices at the beginning of the step.
            SingleStepTimeIntegration::BDF2 => 0.0,
            SingleStepTimeIntegration::MixedBDF2(_) => 0.0,
            SingleStepTimeIntegration::SDIRK2 => 0.5 * 2.0_f32.sqrt(),
        }
    }
}

impl Default for SingleStepTimeIntegration {
    fn default() -> Self {
        SingleStepTimeIntegration::BE
    }
}
