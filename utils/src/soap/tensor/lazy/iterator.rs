pub trait IterExpr: Iterator {
    #[inline]
    fn unroll4(mut self, mut f: impl FnMut(Self::Item))
    where
        Self: Sized,
    {
        loop {
            if let Some(next) = self.next() {
                f(next);
            } else {
                break;
            }
            if let Some(next) = self.next() {
                f(next);
            } else {
                break;
            }
            if let Some(next) = self.next() {
                f(next);
            } else {
                break;
            }
            if let Some(next) = self.next() {
                f(next);
            } else {
                break;
            }
        }
    }
}

impl<I> IterExpr for I where I: Iterator {}
