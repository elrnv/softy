use crate::constraints::{FrictionJacobianTimings, FrictionTimings, UpdateTimings};
use std::fmt::{Display, Formatter};
use std::time::Duration;

#[derive(Copy, Clone, Debug, PartialEq, Default)]
pub struct JacobianTimings {
    pub total: Duration,
    pub fem: Duration,
    pub diag: Duration,
    pub volume: Duration,
    pub contact: Duration,
    pub friction: Duration,
}

impl JacobianTimings {
    pub fn clear(&mut self) {
        self.fem = Duration::new(0, 0);
        self.diag = Duration::new(0, 0);
        self.volume = Duration::new(0, 0);
        self.contact = Duration::new(0, 0);
        self.friction = Duration::new(0, 0);
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Default)]
pub struct ResidualTimings {
    pub total: Duration,
    pub energy_gradient: Duration,
    pub update_state: Duration,
    pub update_distance_potential: Duration,
    pub update_constraint_gradient: Duration,
    pub update_multipliers: Duration,
    pub update_sliding_basis: Duration,
    pub contact_force: Duration,
    pub volume_force: Duration,
    pub contact_jacobian: Duration,
    pub jacobian: JacobianTimings,
    pub friction_force: FrictionTimings,
    pub update_constraint_details: UpdateTimings,
    pub force_ad: Duration,
    pub dof_to_vtx_ad: Duration,
    pub read_deriv_ad: Duration,
    pub product_ad: Duration,
    pub preconditioner: Duration,
}

impl ResidualTimings {
    pub fn clear(&mut self) {
        self.total = Duration::new(0, 0);
        self.energy_gradient = Duration::new(0, 0);
        self.update_state = Duration::new(0, 0);
        self.update_distance_potential = Duration::new(0, 0);
        self.update_constraint_gradient = Duration::new(0, 0);
        self.update_multipliers = Duration::new(0, 0);
        self.update_sliding_basis = Duration::new(0, 0);
        self.contact_force = Duration::new(0, 0);
        self.contact_jacobian = Duration::new(0, 0);
        self.volume_force = Duration::new(0, 0);
        self.friction_force.clear();
        self.update_constraint_details.clear();
        self.force_ad = Duration::new(0, 0);
        self.dof_to_vtx_ad = Duration::new(0, 0);
        self.read_deriv_ad = Duration::new(0, 0);
        self.product_ad = Duration::new(0, 0);
        self.preconditioner = Duration::new(0, 0);
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Default)]
pub struct Timings {
    pub line_search_assist: Duration,
    pub residual: ResidualTimings,
    pub friction_jacobian: FrictionJacobianTimings,
    pub linear_solve: Duration,
    pub direct_solve: Duration,
    pub jacobian_product: Duration,
    pub jacobian_values: Duration,
    pub jacobian_indices: Duration,
    pub linsolve_debug_info: Duration,
    pub line_search: Duration,
    pub total: Duration,
}

impl Display for Timings {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Timings (ms):")?;
        writeln!(
            f,
            "  Line search assist time:    {}",
            self.line_search_assist.as_millis()
        )?;
        writeln!(
            f,
            "  Balance equation time:      {}",
            self.residual.total.as_millis()
        )?;
        writeln!(
            f,
            "    Energy gradient time:     {}",
            self.residual.energy_gradient.as_millis()
        )?;
        writeln!(
            f,
            "    Volume force time:        {}",
            self.residual.volume_force.as_millis()
        )?;
        writeln!(
            f,
            "    Contact prep time:        {}",
            self.residual.update_state.as_millis()
                + self.residual.update_constraint_gradient.as_millis()
                + self.residual.update_distance_potential.as_millis()
                + self.residual.update_multipliers.as_millis()
                + self.residual.update_sliding_basis.as_millis()
        )?;
        writeln!(
            f,
            "      Update state time:      {}",
            self.residual.update_state.as_millis()
        )?;
        writeln!(
            f,
            "      Update distance:        {}",
            self.residual.update_distance_potential.as_millis()
        )?;
        writeln!(
            f,
            "      Update constraint grad: {}",
            self.residual.update_constraint_gradient.as_millis()
        )?;
        writeln!(
            f,
            "        Collect triplets:     {}",
            self.residual
                .update_constraint_details
                .collect_triplets
                .as_millis()
        )?;
        writeln!(
            f,
            "        Redistribute triplets:{}",
            self.residual
                .update_constraint_details
                .redistribute_triplets
                .as_millis()
        )?;
        writeln!(
            f,
            "      Update multipliers:     {}",
            self.residual.update_multipliers.as_millis()
        )?;
        writeln!(
            f,
            "      Update sliding basis:   {}",
            self.residual.update_sliding_basis.as_millis()
        )?;
        writeln!(
            f,
            "        Stash:                {}",
            self.residual.update_constraint_details.stash.as_millis()
        )?;
        writeln!(
            f,
            "        Contact jacobian:     {}",
            self.residual
                .update_constraint_details
                .contact_jac
                .as_millis()
        )?;
        writeln!(
            f,
            "          Collect triplets:   {}",
            self.residual
                .update_constraint_details
                .jac_collect_triplets
                .as_millis()
        )?;
        writeln!(
            f,
            "          Redistribute trip:  {}",
            self.residual
                .update_constraint_details
                .jac_redistribute_triplets
                .as_millis()
        )?;
        writeln!(
            f,
            "        Contact basis:        {}",
            self.residual
                .update_constraint_details
                .contact_basis
                .as_millis()
        )?;
        writeln!(
            f,
            "        Contact gradient:     {}",
            self.residual
                .update_constraint_details
                .contact_grad
                .as_millis()
        )?;
        writeln!(
            f,
            "    Contact force time:       {}",
            self.residual.contact_force.as_millis()
        )?;
        writeln!(
            f,
            "    Friction force time:      {}",
            self.residual.friction_force.total.as_millis()
        )?;
        writeln!(
            f,
            "      Jac + basis mul time:   {}",
            self.residual.friction_force.jac_basis_mul.as_millis()
        )?;
        writeln!(
            f,
            "    Jacobian values time:     {}",
            self.residual.jacobian.total.as_millis()
        )?;
        writeln!(
            f,
            "      Jacobian FEM:           {}",
            self.residual.jacobian.fem.as_millis()
        )?;
        writeln!(
            f,
            "      Jacobian Diagonal:      {}",
            self.residual.jacobian.diag.as_millis()
        )?;
        writeln!(
            f,
            "      Jacobian Volume:        {}",
            self.residual.jacobian.volume.as_millis()
        )?;
        writeln!(
            f,
            "      Jacobian Contact:       {}",
            self.residual.jacobian.contact.as_millis()
        )?;
        writeln!(
            f,
            "      Jacobian Friction:      {}",
            self.residual.jacobian.friction.as_millis()
        )?;
        writeln!(
            f,
            "        Friction constraint F:{}",
            self.friction_jacobian.constraint_friction_force.as_millis()
        )?;
        writeln!(
            f,
            "        Friction contact J:   {}",
            self.friction_jacobian.contact_jacobian.as_millis()
        )?;
        writeln!(
            f,
            "        Friction contact G:   {}",
            self.friction_jacobian.contact_gradient.as_millis()
        )?;
        writeln!(
            f,
            "        Friction constraint J:{}",
            self.friction_jacobian.constraint_jacobian.as_millis()
        )?;
        writeln!(
            f,
            "        Friction constraint G:{}",
            self.friction_jacobian.constraint_gradient.as_millis()
        )?;
        writeln!(
            f,
            "        Friction f lambda jac:{}",
            self.friction_jacobian.f_lambda_jac.as_millis()
        )?;
        writeln!(
            f,
            "        Friction A:           {}",
            self.friction_jacobian.a.as_millis()
        )?;
        writeln!(
            f,
            "        Friction B:           {}",
            self.friction_jacobian.b.as_millis()
        )?;
        writeln!(
            f,
            "        Friction C:           {}",
            self.friction_jacobian.c.as_millis()
        )?;
        writeln!(
            f,
            "        Friction D Half:      {}",
            self.friction_jacobian.d_half.as_millis()
        )?;
        writeln!(
            f,
            "        Friction D:           {}",
            self.friction_jacobian.d.as_millis()
        )?;
        writeln!(
            f,
            "        Friction E:           {}",
            self.friction_jacobian.e.as_millis()
        )?;
        writeln!(
            f,
            "  Force AD time:              {}",
            self.residual.force_ad.as_millis()
        )?;
        writeln!(
            f,
            "  DOF to VTX AD time:         {}",
            self.residual.dof_to_vtx_ad.as_millis()
        )?;
        writeln!(
            f,
            "  Read deriv AD time:         {}",
            self.residual.read_deriv_ad.as_millis()
        )?;
        writeln!(
            f,
            "  Product AD time:            {}",
            self.residual.product_ad.as_millis()
        )?;
        writeln!(
            f,
            "  Jacobian indices time:      {}",
            self.jacobian_indices.as_millis()
        )?;
        writeln!(
            f,
            "  Linear solve time:          {}",
            self.linear_solve.as_millis()
        )?;
        writeln!(
            f,
            "    Jacobian product time:    {}",
            self.jacobian_product.as_millis()
        )?;
        writeln!(
            f,
            "    Jacobian values time:     {}",
            self.jacobian_values.as_millis()
        )?;
        writeln!(
            f,
            "    Direct solve time:        {}",
            self.direct_solve.as_millis()
        )?;
        writeln!(
            f,
            "    Debug info time:          {}",
            self.linsolve_debug_info.as_millis()
        )?;
        writeln!(
            f,
            "  Line search time:           {}",
            self.line_search.as_millis()
        )?;
        writeln!(
            f,
            "  Preconditioner time:        {}",
            self.residual.preconditioner.as_millis()
        )?;
        writeln!(
            f,
            "  Total solve time            {}",
            self.total.as_millis()
        )
    }
}
