use super::{SparseIterExpr, SparseExpr, CwiseUnExpr};
use super::cwise_bin_expr::CwiseBinExpr;

/// A trait to retrieve the target type of a sparse expression.
pub trait Target {
    type Target;
    fn target(&self) -> &Self::Target;
}

impl<'a, S, T> Target for SparseIterExpr<'a, S, T> {
    type Target = T;

    fn target(&self) -> &Self::Target {
        &self.target
    }
}

impl<E: Target + Iterator> Target for SparseExpr<E> {
    type Target = E::Target;

    fn target(&self) -> &Self::Target {
        &self.expr.target()
    }
}

impl<L, R, T, F> Target for CwiseBinExpr<L, R, F>
where
    L: Target<Target = T>,
    R: Target<Target = T>,
    T: PartialEq + std::fmt::Debug,
{
    type Target = T;

    fn target(&self) -> &Self::Target {
        debug_assert_eq!(self.left.target(), self.right.target());
        &self.left.target()
    }
}

impl<E: Target, F> Target for CwiseUnExpr<E, F> {
    type Target = E::Target;

    fn target(&self) -> &Self::Target {
        self.expr.target()
    }
}
