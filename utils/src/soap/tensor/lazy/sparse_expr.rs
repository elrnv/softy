/**
 * `SparseExpr` wraps any sparse expr, this is partially used as a speialization workaround and
 * partially to simplify the iteration algorithms for binary operators.
 */
use super::*;

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct SparseExpr<E>
where
    E: Iterator,
{
    pub(crate) expr: E,
    /// Remember a peeked value, even if it was None.
    peeked: Option<Option<E::Item>>,
}

impl<I: Iterator> SparseExpr<I> {
    pub fn new(expr: I) -> SparseExpr<I> {
        SparseExpr { expr, peeked: None }
    }
}

impl<E: Expression + Iterator> Expression for SparseExpr<E> {}
impl<E: ExprSize + Iterator> ExprSize for SparseExpr<E> {
    #[inline]
    fn expr_size(&self) -> usize {
        self.expr.expr_size()
    }
}
impl<E: TotalExprSize + Iterator> TotalExprSize for SparseExpr<E> {
    #[inline]
    fn total_size_hint(&self, cwise_reduce: u32) -> Option<usize> {
        self.expr.total_size_hint(cwise_reduce)
    }
}

impl<I: Iterator> Iterator for SparseExpr<I> {
    type Item = I::Item;

    #[inline]
    fn next(&mut self) -> Option<I::Item> {
        match self.peeked.take() {
            Some(v) => v,
            None => self.expr.next(),
        }
    }

    #[inline]
    fn count(mut self) -> usize {
        match self.peeked.take() {
            Some(None) => 0,
            Some(Some(_)) => 1 + self.expr.count(),
            None => self.expr.count(),
        }
    }

    #[inline]
    fn nth(&mut self, n: usize) -> Option<I::Item> {
        match self.peeked.take() {
            Some(None) => None,
            Some(v @ Some(_)) if n == 0 => v,
            Some(Some(_)) => self.expr.nth(n - 1),
            None => self.expr.nth(n),
        }
    }

    #[inline]
    fn last(mut self) -> Option<I::Item> {
        let peek_opt = match self.peeked.take() {
            Some(None) => return None,
            Some(v) => v,
            None => None,
        };
        self.expr.last().or(peek_opt)
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let peek_len = match self.peeked {
            Some(None) => return (0, Some(0)),
            Some(Some(_)) => 1,
            None => 0,
        };
        let (lo, hi) = self.expr.size_hint();
        let lo = lo.saturating_add(peek_len);
        let hi = match hi {
            Some(x) => x.checked_add(peek_len),
            None => None,
        };
        (lo, hi)
    }

    #[inline]
    fn fold<Acc, Fold>(self, init: Acc, mut fold: Fold) -> Acc
    where
        Fold: FnMut(Acc, Self::Item) -> Acc,
    {
        let acc = match self.peeked {
            Some(None) => return init,
            Some(Some(v)) => fold(init, v),
            None => init,
        };
        self.expr.fold(acc, fold)
    }
}
impl<I> DoubleEndedIterator for SparseExpr<I>
where
    I: DoubleEndedIterator,
{
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        self.expr
            .next_back()
            .or_else(|| self.peeked.take().and_then(|x| x))
    }

    #[inline]
    fn rfold<Acc, Fold>(self, init: Acc, mut fold: Fold) -> Acc
    where
        Fold: FnMut(Acc, Self::Item) -> Acc,
    {
        match self.peeked {
            Some(None) => init,
            Some(Some(v)) => {
                let acc = self.expr.rfold(init, &mut fold);
                fold(acc, v)
            }
            None => self.expr.rfold(init, fold),
        }
    }
}

impl<I: ExactSizeIterator> ExactSizeIterator for SparseExpr<I> {}

impl<I: Iterator> SparseExpr<I> {
    /// This is identical to `peek` in `Peekable`.
    #[inline]
    pub fn peek(&mut self) -> Option<&I::Item> {
        let expr = &mut self.expr;
        self.peeked.get_or_insert_with(|| expr.next()).as_ref()
    }
}
