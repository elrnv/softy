use super::*;

impl<T> Owned for Vec<T> {}

impl<T> Clear for Vec<T> {
    fn clear(&mut self) {
        Vec::<T>::clear(self);
    }
}

impl<T> Set for Vec<T> {
    type Elem = T;
    fn len(&self) -> usize {
        Vec::len(self)
    }
}

impl<'a, T: 'a> View<'a> for Vec<T> {
    type Type = &'a [T];

    fn view(&'a self) -> Self::Type {
        self.as_slice()
    }
}

impl<'a, T: 'a> ViewMut<'a> for Vec<T> {
    type Type = &'a mut [T];

    fn view_mut(&'a mut self) -> Self::Type {
        self.as_mut_slice()
    }
}

impl<T> Push<T> for Vec<T> {
    fn push(&mut self, element: T) {
        Vec::push(self, element);
    }
}

impl<T, N> SplitPrefix<N> for Vec<T>
where
    T: Grouped<N>,
    <T as Grouped<N>>::Array: Default,
    N: num::Unsigned,
{
    type Prefix = T::Array;

    fn split_prefix(mut self) -> Option<(Self::Prefix, Self)> {
        if self.len() < N::value() {
            return None;
        }
        // Note: This is inefficient ( as is the implementation for `remove_prefix` ).
        // As such it shouldn't be used when iterating over `Subset`s of
        // `Vec<T>` or `Subset`s of any other chunked collection that uses
        // `Vec<T>` for storage. We should be able to specialize the
        // implementation of subsets of `Vec<T>` types for better performance.
        self.rotate_left(N::value());
        let at = self.len() - N::value();
        let mut out: T::Array = Default::default();
        unsafe {
            self.set_len(at);
            std::ptr::copy_nonoverlapping(
                self.as_ptr().add(at),
                &mut out as *mut T::Array as *mut T,
                N::value(),
            );
        }
        Some((out, self))
    }
}

impl<T> IntoFlat for Vec<T> {
    type FlatType = Vec<T>;
    /// Since a `Vec` has no information about the structure of its underlying
    /// data, this is effectively a no-op.
    fn into_flat(self) -> Self::FlatType {
        self
    }
}

impl<T> CloneWithFlat<Vec<T>> for Vec<T> {
    type CloneType = Vec<T>;
    /// This function simply ignores self and returns flat since self is already
    /// a flat type.
    fn clone_with_flat(&self, flat: Vec<T>) -> Self::CloneType {
        assert_eq!(self.len(), flat.len());
        flat
    }
}

impl<T> SplitAt for Vec<T> {
    fn split_at(mut self, mid: usize) -> (Self, Self) {
        let r = self.split_off(mid);
        (self, r)
    }
}

impl<T> RemovePrefix for Vec<T> {
    fn remove_prefix(&mut self, n: usize) {
        self.rotate_left(n);
        self.truncate(self.len() - n);
    }
}

/// Since `Vec` already owns its data, this is simply a noop.
impl<T> ToOwned for Vec<T> {
    type Owned = Self;
    fn to_owned(self) -> Self::Owned {
        self
    }
}
