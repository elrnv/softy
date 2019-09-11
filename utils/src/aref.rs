/// A reference abstracting over borrowed cell refs and plain old & style refs.
pub enum ARef<'a, T: ?Sized> {
    Plain(&'a T),
    Cell(std::cell::Ref<'a, T>),
}

impl<'a, T: ?Sized> std::ops::Deref for ARef<'a, T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        match self {
            ARef::Plain(r) => r,
            ARef::Cell(r) => r,
        }
    }
}
