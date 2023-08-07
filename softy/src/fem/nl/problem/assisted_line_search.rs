use super::state;
use super::NonLinearProblem;
use crate::nl_fem::LineSearchWorkspace;
use crate::nl_fem::MixedComplementarityProblem;
use crate::nl_fem::NLProblem;
use crate::nl_fem::MCP;
use crate::{Real, Real64};
use flatk::Chunked;
use tensr::{Set, Storage, StorageMut};

pub trait AssistedNonLinearProblem<T: Real>: NonLinearProblem<T> {
    /// Returns a better alpha estimate according to problem priors.
    ///
    /// By default this function does nothing.
    fn assist_line_search_for_contact(&self, alpha: T, _x: &[T], _r_cur: &[T], _r_next: &[T]) -> T {
        alpha
    }

    /// Returns a better alpha estimate according to problem priors.
    ///
    /// By default this function does nothing.
    fn assist_line_search_for_friction(
        &self,
        alpha: T,
        _x: &[T],
        _p: &[T],
        _r_cur: &[T],
        _r_next: &[T],
    ) -> T;
}

impl<T: Real64> AssistedNonLinearProblem<T> for NLProblem<T> {
    fn assist_line_search_for_contact(
        &self,
        mut alpha: T,
        v: &[T],
        r_cur: &[T],
        r_next: &[T],
    ) -> T {
        let LineSearchWorkspace {
            pos_next,
            f1vtx,
            f2vtx,
            ..
        } = &mut *self.line_search_ws.borrow_mut();

        // Prepare positions
        let num_coords = {
            let state = &*self.state.borrow_mut();
            state.vtx.next.pos.len() * 3
        };

        pos_next.storage_mut().resize(num_coords, T::zero());
        f1vtx.storage_mut().resize(num_coords, T::zero());
        f2vtx.storage_mut().resize(num_coords, T::zero());

        {
            //self.integrate_step(v);
            let state = &*self.state.borrow();
            // state.update_vertices(v);
            pos_next
                .storage_mut()
                .copy_from_slice(state.vtx.next.pos.storage());

            state::to_vertex_velocity(
                Chunked::from_offsets(&[0, v.len()][..], r_cur),
                f1vtx.view_mut(),
            );
            state::to_vertex_velocity(
                Chunked::from_offsets(&[0, v.len()][..], r_next),
                f2vtx.view_mut(),
            );
        }

        // Contact assist
        for fc in self.frictional_contact_constraints.iter() {
            let fc_constraint = &mut *fc.constraint.borrow_mut();
            alpha = num_traits::Float::min(
                alpha,
                fc_constraint.assist_line_search_for_contact(
                    alpha,
                    pos_next.view(),
                    f1vtx.view(),
                    f2vtx.view(),
                ),
            );
        }
        alpha
    }

    fn assist_line_search_for_friction(
        &self,
        alpha: T,
        v: &[T],
        p: &[T],
        r_cur: &[T],
        r_next: &[T],
    ) -> T {
        let do_friction = self
            .frictional_contact_constraints
            .iter()
            .any(|fc| fc.constraint.borrow().params.friction_params.is_some());

        if !do_friction {
            return alpha;
        }

        let LineSearchWorkspace {
            vel,
            search_dir,
            f1vtx,
            f2vtx,
            ..
        } = &mut *self.line_search_ws.borrow_mut();

        // Prepare positions
        let num_coords = {
            let state = &*self.state.borrow_mut();
            state.vtx.next.pos.len() * 3
        };

        vel.storage_mut().resize(num_coords, T::zero());
        search_dir.storage_mut().resize(num_coords, T::zero());
        f1vtx.storage_mut().resize(num_coords, T::zero());
        f2vtx.storage_mut().resize(num_coords, T::zero());

        // Copy search direction to vertex degrees of freedom.
        // All fixed vertices will have a zero corresponding search direction.
        // TODO: This data layout state specific, and should be moved to the state module.
        state::to_vertex_velocity(
            Chunked::from_offsets(&[0, p.len()][..], p),
            search_dir.view_mut(),
        );
        state::to_vertex_velocity(Chunked::from_offsets(&[0, p.len()][..], v), vel.view_mut());
        state::to_vertex_velocity(
            Chunked::from_offsets(&[0, p.len()][..], r_cur),
            f1vtx.view_mut(),
        );
        state::to_vertex_velocity(
            Chunked::from_offsets(&[0, p.len()][..], r_next),
            f2vtx.view_mut(),
        );

        // Friction assist
        assert_eq!(vel.len(), search_dir.len());

        let mut total_sum_alpha = T::zero();
        let mut total_alphas = 0;

        for fc in self.frictional_contact_constraints.iter() {
            let mut fc_constraint = fc.constraint.borrow_mut();
            let (sum_alphas, num_alphas) = fc_constraint.assist_line_search_for_friction(
                alpha,
                search_dir.view(),
                vel.view(),
                f1vtx.view(),
                f2vtx.view(),
            );
            total_alphas += num_alphas;
            total_sum_alpha += sum_alphas;
        }

        if total_alphas > 0 {
            total_sum_alpha / total_alphas as f64
        } else {
            alpha
        }
    }
}

impl<T, P> AssistedNonLinearProblem<T> for MCP<P>
where
    T: Real,
    P: MixedComplementarityProblem<T> + AssistedNonLinearProblem<T>,
{
    #[inline]
    fn assist_line_search_for_contact(&self, alpha: T, x: &[T], r_cur: &[T], r_next: &[T]) -> T {
        self.problem
            .assist_line_search_for_contact(alpha, x, r_cur, r_next)
    }
    #[inline]
    fn assist_line_search_for_friction(
        &self,
        alpha: T,
        x: &[T],
        p: &[T],
        r_cur: &[T],
        r_next: &[T],
    ) -> T {
        self.problem
            .assist_line_search_for_friction(alpha, x, p, r_cur, r_next)
    }
}
