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
pub type CwiseMulExpr<L, R> = CwiseBinExpr<L, R, CwiseMultiplication>;

/*
 * CwiseBinExpr impls
 */

/*
 * Dense * Tensor
 */

impl<L: Iterator + DenseExpr, R, Out> Iterator for CwiseBinExpr<L, Tensor<R>, Multiplication>
where
    R: Copy,
    L::Item: MulOp<Tensor<R>, Output = Out>,
{
    type Item = Out;
    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.left.next().map(|l| l.mul(self.right))
    }
    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.left.size_hint()
    }
}

impl<L: Expression, R> Expression for CwiseBinExpr<L, Tensor<R>, Multiplication> {}
impl<L: ExprSize, R> ExprSize for CwiseBinExpr<L, Tensor<R>, Multiplication> {
    #[inline]
    fn expr_size(&self) -> usize {
        self.left.expr_size()
    }
}
impl<L: TotalExprSize, R> TotalExprSize for CwiseBinExpr<L, Tensor<R>, Multiplication> {
    #[inline]
    fn total_size_hint(&self, cwise_reduce: u32) -> Option<usize> {
        self.left.total_size_hint(cwise_reduce)
    }
}

impl<L: ExactSizeIterator, R> ExactSizeIterator for CwiseBinExpr<L, Tensor<R>, Multiplication> where
    Self: Iterator
{
}

/*
 * Tensor * DenseExpr
 */

impl<L, R: Iterator + DenseExpr, Out> Iterator for CwiseBinExpr<Tensor<L>, R, Multiplication>
where
    L: Copy,
    Tensor<L>: MulOp<R::Item, Output = Out>,
{
    type Item = Out;
    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.right.next().map(|r| self.left.mul(r))
    }
    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.right.size_hint()
    }
}

impl<L, R: Iterator + Expression> Expression for CwiseBinExpr<Tensor<L>, R, Multiplication> {}
impl<L, R: Iterator + ExprSize> ExprSize for CwiseBinExpr<Tensor<L>, R, Multiplication> {
    #[inline]
    fn expr_size(&self) -> usize {
        self.right.expr_size()
    }
}
impl<L, R: Iterator + TotalExprSize> TotalExprSize for CwiseBinExpr<Tensor<L>, R, Multiplication> {
    #[inline]
    fn total_size_hint(&self, cwise_reduce: u32) -> Option<usize> {
        self.right.total_size_hint(cwise_reduce)
    }
}

impl<L, R: ExactSizeIterator> ExactSizeIterator for CwiseBinExpr<Tensor<L>, R, Multiplication> where
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
impl<L: DenseExpr + ExprSize, R: DenseExpr + ExprSize, F> ExprSize for CwiseBinExpr<L, R, F> {
    #[inline]
    fn expr_size(&self) -> usize {
        assert_eq!(self.left.expr_size(), self.right.expr_size());
        self.left.expr_size()
    }
}
impl<L: DenseExpr + TotalExprSize, R: DenseExpr + TotalExprSize, F> TotalExprSize
    for CwiseBinExpr<L, R, F>
{
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
 * Sparse * Tensor
 */

impl<L: Iterator<Item = IndexedExpr<A>>, A, R, Out> Iterator
    for CwiseBinExpr<SparseExpr<L>, Tensor<R>, Multiplication>
where
    R: Copy,
    A: MulOp<Tensor<R>, Output = Out>,
{
    type Item = IndexedExpr<Out>;
    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.left.next().map(|l| l.map_expr(|l| l.mul(self.right)))
    }
    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.left.size_hint()
    }
}

//impl<L: Expression, R> Expression for CwiseBinExpr<SparseExpr<L>, Tensor<R>, Multiplication> {}
//impl<L: Expression, R> ExprSize for CwiseBinExpr<SparseExpr<L>, Tensor<R>, Multiplication> {
//    fn expr_size(&self) -> usize {
//        self.left.expr_size()
//    }
//    fn total_size_hint(&self, cwise_reduce: u32) -> Option<usize> {
//        self.left.total_size_hint(cwise_reduce)
//    }
//}
//
//impl<L: ExactSizeIterator, R> ExactSizeIterator for CwiseBinExpr<SparseExpr<L>, Tensor<R>, Multiplication> where
//    Self: Iterator
//{
//}

/*
 * Tensor * Sparse
 */

impl<L, R: Iterator<Item = IndexedExpr<A>>, A, Out> Iterator
    for CwiseBinExpr<Tensor<L>, SparseExpr<R>, Multiplication>
where
    L: Copy,
    Tensor<L>: MulOp<A, Output = Out>,
{
    type Item = IndexedExpr<Out>;
    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.right.next().map(|r| r.map_expr(|r| self.left.mul(r)))
    }
    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.right.size_hint()
    }
}

//impl<L, R: Iterator + Expression> Expression for CwiseBinExpr<Tensor<L>, SparseExpr<R>, Multiplication> {}
//impl<L, R: Iterator + Expression> ExprSize for CwiseBinExpr<Tensor<L>, SparseExpr<R>, Multiplication> {
//    fn expr_size(&self) -> usize {
//        self.right.expr_size()
//    }
//    fn total_size_hint(&self, cwise_reduce: u32) -> Option<usize> {
//        self.right.total_size_hint(cwise_reduce)
//    }
//}
//
//impl<L, R: ExactSizeIterator> ExactSizeIterator for CwiseBinExpr<Tensor<L>, SparseExpr<R>, Multiplication> where
//    Self: Iterator
//{
//}

/*
 * Dense * Sparse Additive
 */

/* TODO: Fix this
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
*/

/*
 * Dense * Sparse Multiplicative
 */

impl<L, R, A, F, Out> Iterator for CwiseBinExpr<Enumerate<L>, SparseExpr<R>, F>
where
    L: Iterator + DenseExpr,
    R: Iterator<Item = IndexedExpr<A>>,
    F: BinOp<L::Item, A, Output = Out>,
{
    type Item = IndexedExpr<Out>;
    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.right.next().and_then(|right| {
            while let Some((count, left)) = self.left.next() {
                if count == right.index {
                    return Some(right.map_expr(|r| self.op.apply(left, r)));
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
impl<L: DenseExpr, R: Iterator + ExprSize, F> ExprSize
    for CwiseBinExpr<Enumerate<L>, SparseExpr<R>, F>
{
    #[inline]
    fn expr_size(&self) -> usize {
        self.right.expr_size()
    }
}
impl<L: DenseExpr, R: Iterator + TotalExprSize, F> TotalExprSize
    for CwiseBinExpr<Enumerate<L>, SparseExpr<R>, F>
{
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
    F: BinOp<A, R::Item, Output = Out>,
{
    type Item = IndexedExpr<Out>;
    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.left.next().and_then(|left| {
            while let Some((count, right)) = self.right.next() {
                if count == left.index {
                    return Some(left.map_expr(|l| self.op.apply(l, right)));
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
impl<L: Iterator + ExprSize, R: DenseExpr, F> ExprSize
    for CwiseBinExpr<SparseExpr<L>, Enumerate<R>, F>
{
    #[inline]
    fn expr_size(&self) -> usize {
        self.left.expr_size()
    }
}
impl<L: Iterator + TotalExprSize, R: DenseExpr, F> TotalExprSize
    for CwiseBinExpr<SparseExpr<L>, Enumerate<R>, F>
{
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
}
impl<L: Iterator, R: Iterator, F> TotalExprSize for CwiseBinExpr<SparseExpr<L>, SparseExpr<R>, F>
where
    F: Additive,
    SparseExpr<L>: Iterator + Expression,
    SparseExpr<R>: Iterator + Expression,
{
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

impl<L, R, A, B, Out> Iterator for CwiseMulExpr<SparseExpr<L>, SparseExpr<R>>
where
    L: Iterator<Item = IndexedExpr<A>>,
    R: Iterator<Item = IndexedExpr<B>>,
    A: CwiseMulOp<B, Output = Out>,
{
    type Item = IndexedExpr<Out>;
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
}
impl<L: Iterator, R: Iterator> TotalExprSize for CwiseMulExpr<SparseExpr<L>, SparseExpr<R>>
where
    SparseExpr<L>: Iterator + Expression,
    SparseExpr<R>: Iterator + Expression,
{
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
