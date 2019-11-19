use super::*;

/// A lazy component-wise binary expression to be evaluated at a later time.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct CwiseBinExpr<L, R, F> {
    pub(crate) left: L,
    pub(crate) right: R,
    pub(crate) op: F,
}

impl<L, R, F: Default> CwiseBinExpr<L, R, F> {
    #[inline]
    pub fn new(left: L, right: R) -> Self {
        Self::with_op(left, right, Default::default())
    }
}

impl<L, R, F> CwiseBinExpr<L, R, F> {
    #[inline]
    pub fn with_op(left: L, right: R, op: F) -> Self {
        CwiseBinExpr { left, right, op }
    }
}

/// A lazy `Add` expression to be evaluated at a later time.
pub type AddExpr<L, R> = CwiseBinExpr<L, R, Addition>;

/// A lazy `Sub` expression to be evaluated at a later time.
pub type SubExpr<L, R> = CwiseBinExpr<L, R, Subtraction>;

/// A lazy component-wise multiply expression to be evaluated at a later time.
pub type CwiseMulExpr<L, R> = CwiseBinExpr<L, R, Multiplication>;

/*
 * CwiseBinExpr impls
 */

/*
 * Iterator * Tensor
 */

impl<L: Iterator, R, F, Out> Iterator for CwiseBinExpr<L, Tensor<R>, F>
where
    R: Copy,
    F: BinOp<L::Item, Tensor<R>, Output = Out>,
{
    type Item = Out;
    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.left.next().map(|l| self.op.apply(l, self.right))
    }
    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.left.size_hint()
    }
}

impl<L: Expression, R, F> Expression for CwiseBinExpr<L, Tensor<R>, F> {}
impl<L: Expression, R, F> ExprSize for CwiseBinExpr<L, Tensor<R>, F> {
    fn expr_size(&self) -> usize {
        self.left.expr_size()
    }
    fn total_size_hint(&self, cwise_reduce: u32) -> Option<usize> {
        self.left.total_size_hint(cwise_reduce)
    }
}

impl<L: ExactSizeIterator, R, F> ExactSizeIterator for CwiseBinExpr<L, Tensor<R>, F> where
    Self: Iterator
{
}

/*
 * Tensor * Iterator
 */

impl<L, R: Iterator, F, Out> Iterator for CwiseBinExpr<Tensor<L>, R, F>
where
    L: Copy,
    F: BinOp<Tensor<L>, R::Item, Output = Out>,
{
    type Item = Out;
    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.right.next().map(|r| self.op.apply(self.left, r))
    }
    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.right.size_hint()
    }
}

impl<L, R: Iterator + Expression, F> Expression for CwiseBinExpr<Tensor<L>, R, F> {}
impl<L, R: Iterator + Expression, F> ExprSize for CwiseBinExpr<Tensor<L>, R, F> {
    fn expr_size(&self) -> usize {
        self.right.expr_size()
    }
    fn total_size_hint(&self, cwise_reduce: u32) -> Option<usize> {
        self.right.total_size_hint(cwise_reduce)
    }
}

impl<L, R: ExactSizeIterator, F> ExactSizeIterator for CwiseBinExpr<Tensor<L>, R, F> where
    Self: Iterator
{
}

/*
 * Dense * Dense
 */

impl<L: Iterator + DenseExpr, R: Iterator + DenseExpr, F, Out> Iterator for CwiseBinExpr<L, R, F>
where
    F: BinOp<L::Item, R::Item, Output = Out>,
{
    type Item = Out;
    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.left
            .next()
            .and_then(|l| self.right.next().map(|r| self.op.apply(l, r)))
    }
    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let left = self.left.size_hint();
        let right = self.right.size_hint();
        (
            left.0.min(right.0),
            left.1.and_then(|l| right.1.map(|r| l.max(r))),
        )
    }
}

impl<L: DenseExpr + Iterator + Expression, R: DenseExpr + Iterator + Expression, F> Expression
    for CwiseBinExpr<L, R, F>
{
}
impl<L: DenseExpr + Iterator + Expression, R: DenseExpr + Iterator + Expression, F> ExprSize
    for CwiseBinExpr<L, R, F>
{
    #[inline]
    fn expr_size(&self) -> usize {
        self.left.expr_size().min(self.right.expr_size())
    }
    #[inline]
    fn total_size_hint(&self, cwise_reduce: u32) -> Option<usize> {
        if let Some(left) = self.left.total_size_hint(cwise_reduce) {
            if let Some(right) = self.right.total_size_hint(cwise_reduce) {
                Some(left.min(right))
            } else {
                Some(left)
            }
        } else {
            self.right.total_size_hint(cwise_reduce)
        }
    }
}

impl<L: DenseExpr + ExactSizeIterator, R: DenseExpr + ExactSizeIterator, F> ExactSizeIterator
    for CwiseBinExpr<L, R, F>
where
    Self: Iterator,
{
}


/**************
 * SPARSE OPS *
 **************/

/*
 * Dense * Sparse Additive
 */

impl<L, R, F, Out> Iterator for CwiseBinExpr<L, SparseExpr<R>, F>
where
    L: Iterator + DenseExpr,
    R: Iterator,
    F: Additive + BinOp<L::Item, R::Item, Output = Out>,
{
    type Item = Out;
    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        // Need to iterate over everything
        self.left
            .next()
            .and_then(|l| self.right.next().map(|r| self.op.apply(l, r)))
    }
    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.left.size_hint()
    }
}

impl<L: DenseExpr + Iterator + Expression, R: Iterator + Expression, F> Expression
    for CwiseBinExpr<L, SparseExpr<R>, F>
{
}
impl<L: DenseExpr + Iterator + Expression, R: Iterator + Expression, F> ExprSize
    for CwiseBinExpr<L, SparseExpr<R>, F>
{
    #[inline]
    fn expr_size(&self) -> usize {
        self.left.expr_size()
    }
    #[inline]
    fn total_size_hint(&self, cwise_reduce: u32) -> Option<usize> {
        self.left.total_size_hint(cwise_reduce)
    }
}

impl<L: DenseExpr + ExactSizeIterator, R: ExactSizeIterator, F> ExactSizeIterator
    for CwiseBinExpr<L, SparseExpr<R>, F>
where
    Self: Iterator,
{
}

/*
 * Sparse * Dense Additive
 */

impl<L, R, F, Out> Iterator for CwiseBinExpr<SparseExpr<L>, R, F>
where
    L: Iterator,
    R: Iterator + DenseExpr,
    F: Additive + BinOp<L::Item, R::Item, Output = Out>,
{
    type Item = Out;
    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        // Need to iterate over everything
        self.left
            .next()
            .and_then(|l| self.right.next().map(|r| self.op.apply(l, r)))
    }
    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.right.size_hint()
    }
}

impl<L: Iterator + Expression, R: DenseExpr + Iterator + Expression, F> Expression
    for CwiseBinExpr<SparseExpr<L>, R, F>
{
}
impl<L: Iterator + Expression, R: DenseExpr + Iterator + Expression, F> ExprSize
    for CwiseBinExpr<SparseExpr<L>, R, F>
{
    #[inline]
    fn expr_size(&self) -> usize {
        self.right.expr_size()
    }
    #[inline]
    fn total_size_hint(&self, cwise_reduce: u32) -> Option<usize> {
        self.right.total_size_hint(cwise_reduce)
    }
}

impl<L: ExactSizeIterator, R: DenseExpr + ExactSizeIterator, F> ExactSizeIterator
    for CwiseBinExpr<SparseExpr<L>, R, F>
where
    Self: Iterator,
{
}

/*
 * Dense * Sparse Multiplicative
 */

impl<L, R, A, F, Out> Iterator for CwiseBinExpr<Enumerate<L>, SparseExpr<R>, F>
where
    L: Iterator + DenseExpr,
    R: Iterator<Item = IndexedExpr<A>>,
    F: BinOp<L::Item, R::Item, Output = Out>,
{
    type Item = Out;
    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.right.next().and_then(|right| {
            while let Some((count, left)) = self.left.next() {
                if count == right.index {
                    return Some(self.op.apply(left, right));
                }
            }
            None
        })
    }
    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.right.size_hint()
    }
}

impl<L: DenseExpr + Iterator + Expression, R: Iterator + Expression, F> Expression
    for CwiseBinExpr<Enumerate<L>, SparseExpr<R>, F>
{
}
impl<L: DenseExpr + Iterator + Expression, R: Iterator + Expression, F> ExprSize
    for CwiseBinExpr<Enumerate<L>, SparseExpr<R>, F>
{
    #[inline]
    fn expr_size(&self) -> usize {
        self.right.expr_size()
    }
    #[inline]
    fn total_size_hint(&self, cwise_reduce: u32) -> Option<usize> {
        self.right.total_size_hint(cwise_reduce)
    }
}

/*
 * Sparse * Dense Multiplicative
 */

impl<L, R, A, F, Out> Iterator for CwiseBinExpr<SparseExpr<L>, Enumerate<R>, F>
where
    L: Iterator<Item = IndexedExpr<A>>,
    R: Iterator + DenseExpr,
    F: BinOp<L::Item, R::Item, Output = Out>,
{
    type Item = Out;
    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.left.next().and_then(|left| {
            while let Some((count, right)) = self.right.next() {
                if count == left.index {
                    return Some(self.op.apply(left, right));
                }
            }
            None // Really this means that left and right have different sizes
        })
    }
    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.left.size_hint()
    }
}
impl<L: Iterator + Expression, R: DenseExpr + Iterator + Expression, F> Expression
    for CwiseBinExpr<SparseExpr<L>, Enumerate<R>, F>
{
}
impl<L: Iterator + Expression, R: DenseExpr + Iterator + Expression, F> ExprSize
    for CwiseBinExpr<SparseExpr<L>, Enumerate<R>, F>
{
    #[inline]
    fn expr_size(&self) -> usize {
        self.left.expr_size()
    }
    #[inline]
    fn total_size_hint(&self, cwise_reduce: u32) -> Option<usize> {
        self.left.total_size_hint(cwise_reduce)
    }
}

/*
 * Sparse * Sparse Additive
 */

impl<L, R, A, B, F, Out> Iterator for CwiseBinExpr<SparseExpr<L>, SparseExpr<R>, F>
where
    L: Iterator<Item = IndexedExpr<A>>,
    R: Iterator<Item = IndexedExpr<B>>,
    F: Additive + BinOp<A, B, Output = Out>,
{
    type Item = IndexedExpr<SparseAddResult<A, B, Out>>;
    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        let left_index = self
            .left
            .peek()
            .map(|item| item.index)
            .unwrap_or(std::usize::MAX);
        let right_index = self
            .right
            .peek()
            .map(|item| item.index)
            .unwrap_or(std::usize::MAX);
        if left_index < right_index {
            self.left
                .next()
                .map(|IndexedExpr { index, expr }| IndexedExpr {
                    index,
                    expr: SparseAddResult::Left(expr),
                })
        } else if left_index > right_index {
            self.right
                .next()
                .map(|IndexedExpr { index, expr }| IndexedExpr {
                    index,
                    expr: SparseAddResult::Right(expr),
                })
        } else {
            if left_index == std::usize::MAX {
                return None;
            }
            Some(IndexedExpr {
                index: left_index,
                expr: SparseAddResult::Expr(self.op.apply(
                    self.left.next().unwrap().expr,
                    self.right.next().unwrap().expr,
                )),
            })
        }
    }
    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let left = self.left.size_hint();
        let right = self.right.size_hint();
        (
            left.0.min(right.0),
            left.1.and_then(|l| right.1.map(|r| l.max(r))),
        )
    }
}
impl<L: Iterator, R: Iterator, F> Expression for CwiseBinExpr<SparseExpr<L>, SparseExpr<R>, F>
where
    F: Additive,
    SparseExpr<L>: Iterator + Expression,
    SparseExpr<R>: Iterator + Expression,
{
}
impl<L: Iterator, R: Iterator, F> ExprSize for CwiseBinExpr<SparseExpr<L>, SparseExpr<R>, F>
where
    F: Additive,
    SparseExpr<L>: Iterator + Expression,
    SparseExpr<R>: Iterator + Expression,
{
    #[inline]
    fn expr_size(&self) -> usize {
        self.left.expr_size() + self.right.expr_size()
    }
    #[inline]
    fn total_size_hint(&self, cwise_reduce: u32) -> Option<usize> {
        if let Some(left) = self.left.total_size_hint(cwise_reduce) {
            if let Some(right) = self.right.total_size_hint(cwise_reduce) {
                Some(left + right)
            } else {
                Some(left)
            }
        } else {
            self.right.total_size_hint(cwise_reduce)
        }
    }
}

/*
 * Sparse * Sparse Multiply
 */

impl<L, R, A, B> Iterator for CwiseMulExpr<SparseExpr<L>, SparseExpr<R>>
where
    L: Iterator<Item = IndexedExpr<A>>,
    R: Iterator<Item = IndexedExpr<B>>,
    A: CwiseMulOp<B>,
{
    type Item = IndexedExpr<<A as CwiseMulOp<B>>::Output>;
    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.left.peek().is_none() || self.right.peek().is_none() {
            return None;
        }
        loop {
            let left = self.left.peek().unwrap();
            let right = self.right.peek().unwrap();
            if left.index < right.index {
                let next_left = self.left.next();
                if next_left.is_none() {
                    return None;
                }
            } else if left.index > right.index {
                let next_right = self.right.next();
                if next_right.is_none() {
                    return None;
                }
            } else {
                return Some(IndexedExpr {
                    index: left.index,
                    expr: self.op.apply(
                        self.left.next().unwrap().expr,
                        self.right.next().unwrap().expr,
                    ),
                });
            }
        }
    }
    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let left = self.left.size_hint();
        let right = self.right.size_hint();
        (
            left.0.min(right.0),
            // Yes min because excess elements must vanish when multiplied by zero.
            left.1.and_then(|l| right.1.map(|r| l.min(r))),
        )
    }
}

impl<L: Iterator, R: Iterator> Expression for CwiseMulExpr<SparseExpr<L>, SparseExpr<R>>
where
    SparseExpr<L>: Iterator + Expression,
    SparseExpr<R>: Iterator + Expression,
{
}
impl<L: Iterator, R: Iterator> ExprSize for CwiseMulExpr<SparseExpr<L>, SparseExpr<R>>
where
    SparseExpr<L>: Iterator + Expression,
    SparseExpr<R>: Iterator + Expression,
{
    #[inline]
    fn expr_size(&self) -> usize {
        self.left.expr_size().min(self.right.expr_size())
    }
    #[inline]
    fn total_size_hint(&self, cwise_reduce: u32) -> Option<usize> {
        if let Some(left) = self.left.total_size_hint(cwise_reduce) {
            if let Some(right) = self.right.total_size_hint(cwise_reduce) {
                Some(left.min(right))
            } else {
                Some(left)
            }
        } else {
            self.right.total_size_hint(cwise_reduce)
        }
    }
}
