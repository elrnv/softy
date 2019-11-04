//! Tuples are useful for combining different storage types together in an
//! ad-hoc structure of arrays pattern. This module facilitates this functionality.

use super::*;

/// A tuple wraps its containers, so itself is a value type and not a borrow.
impl<S, T> ValueType for (S, T) {}
impl<S: Viewed, T: Viewed> Viewed for (S, T) {}

impl<A, B, S: Push<A>, T: Push<B>> Push<(A, B)> for (S, T) {
    fn push(&mut self, (a, b): (A, B)) {
        self.0.push(a);
        self.1.push(b);
    }
}

impl<S: Truncate, T: Truncate> Truncate for (S, T) {
    fn truncate(&mut self, len: usize) {
        self.0.truncate(len);
        self.1.truncate(len);
    }
}

impl<S: Clear, T: Clear> Clear for (S, T) {
    fn clear(&mut self) {
        self.0.clear();
        self.1.clear();
    }
}

impl<S: IntoFlat, T: IntoFlat> IntoFlat for (S, T) {
    type FlatType = (S::FlatType, T::FlatType);

    fn into_flat(self) -> Self::FlatType {
        (self.0.into_flat(), self.1.into_flat())
    }
}

impl<U, V, S: StorageInto<U>, T: StorageInto<V>> StorageInto<(U, V)> for (S, T) {
    type Output = (S::Output, T::Output);
    fn storage_into(self) -> Self::Output {
        (self.0.storage_into(), self.1.storage_into())
    }
}

impl<S, T, U> CloneWithStorage<U> for (S, T) {
    type CloneType = U;
    fn clone_with_storage(&self, storage: U) -> Self::CloneType {
        storage
    }
}

impl<S: IntoOwned, T: IntoOwned> IntoOwned for (S, T) {
    type Owned = (S::Owned, T::Owned);
    fn into_owned(self) -> Self::Owned {
        (self.0.into_owned(), self.1.into_owned())
    }
}

impl<S: IntoOwnedData, T: IntoOwnedData> IntoOwnedData for (S, T) {
    type OwnedData = (S::OwnedData, T::OwnedData);
    fn into_owned_data(self) -> Self::OwnedData {
        (self.0.into_owned_data(), self.1.into_owned_data())
    }
}

impl<'a, S, T> GetIndex<'a, (S, T)> for usize
where
    S: Get<'a, usize>,
    T: Get<'a, usize>,
{
    type Output = (S::Output, T::Output);
    fn get(self, (ref s, ref t): &(S, T)) -> Option<Self::Output> {
        s.get(self)
            .and_then(|s_item| t.get(self).map(|t_item| (s_item, t_item)))
    }
}

impl<'a, S, T> GetIndex<'a, (S, T)> for std::ops::Range<usize>
where
    S: Get<'a, std::ops::Range<usize>>,
    T: Get<'a, std::ops::Range<usize>>,
{
    type Output = (S::Output, T::Output);
    fn get(self, (ref s, ref t): &(S, T)) -> Option<Self::Output> {
        s.get(self.clone())
            .and_then(|s_item| t.get(self).map(|t_item| (s_item, t_item)))
    }
}

impl<'a, S, T> IsolateIndex<(S, T)> for usize
where
    S: Isolate<usize>,
    T: Isolate<usize>,
{
    type Output = (S::Output, T::Output);
    fn try_isolate(self, (s, t): (S, T)) -> Option<Self::Output> {
        s.try_isolate(self)
            .and_then(|s_item| t.try_isolate(self).map(|t_item| (s_item, t_item)))
    }
}

impl<'a, S, T> IsolateIndex<(S, T)> for std::ops::Range<usize>
where
    S: Isolate<std::ops::Range<usize>>,
    T: Isolate<std::ops::Range<usize>>,
{
    type Output = (S::Output, T::Output);
    fn try_isolate(self, (s, t): (S, T)) -> Option<Self::Output> {
        s.try_isolate(self.clone())
            .and_then(|s_item| t.try_isolate(self).map(|t_item| (s_item, t_item)))
    }
}

impl<S: Set, T: Set> Set for (S, T) {
    type Elem = (S::Elem, T::Elem);
    type Atom = (S::Atom, T::Atom);
    fn len(&self) -> usize {
        debug_assert_eq!(self.0.len(), self.1.len());
        self.0.len()
    }
}

impl<'a, S: View<'a>, T: View<'a>> View<'a> for (S, T) {
    type Type = (S::Type, T::Type);

    fn view(&'a self) -> Self::Type {
        (self.0.view(), self.1.view())
    }
}

impl<'a, S: ViewMut<'a>, T: ViewMut<'a>> ViewMut<'a> for (S, T) {
    type Type = (S::Type, T::Type);

    fn view_mut(&'a mut self) -> Self::Type {
        (self.0.view_mut(), self.1.view_mut())
    }
}

impl<S, T> SplitAt for (S, T)
where
    S: SplitAt,
    T: SplitAt,
{
    fn split_at(self, mid: usize) -> (Self, Self) {
        let (s, t) = self;
        let (s_l, s_r) = s.split_at(mid);
        let (t_l, t_r) = t.split_at(mid);
        ((s_l, t_l), (s_r, t_r))
    }
}

impl<S, T> SplitOff for (S, T)
where
    S: SplitOff,
    T: SplitOff,
{
    fn split_off(&mut self, mid: usize) -> Self {
        let (s, t) = self;
        let s_r = s.split_off(mid);
        let t_r = t.split_off(mid);
        (s_r, t_r)
    }
}

impl<S, T, N> SplitPrefix<N> for (S, T)
where
    S: SplitPrefix<N>,
    T: SplitPrefix<N>,
{
    type Prefix = (S::Prefix, T::Prefix);

    fn split_prefix(self) -> Option<(Self::Prefix, Self)> {
        let (s, t) = self;
        s.split_prefix().and_then(|(s_prefix, s_rest)| {
            t.split_prefix()
                .map(|(t_prefix, t_rest)| ((s_prefix, t_prefix), (s_rest, t_rest)))
        })
    }
}

impl<S, T> SplitFirst for (S, T)
where
    S: SplitFirst,
    T: SplitFirst,
{
    type First = (S::First, T::First);

    fn split_first(self) -> Option<(Self::First, Self)> {
        let (s, t) = self;
        s.split_first().and_then(|(s_first, s_rest)| {
            t.split_first()
                .map(|(t_first, t_rest)| ((s_first, t_first), (s_rest, t_rest)))
        })
    }
}

/// We can only provide a reference to the underlying storage of a tuple if the
/// tuple is made up of storage type collections itself. In this case the
/// storage is the tuple itself.
impl<S: Storage<Storage = S>, T: Storage<Storage = T>> Storage for (S, T) {
    type Storage = (S, T);
    fn storage(&self) -> &Self::Storage {
        self
    }
}

impl<S: StorageMut<Storage = S>, T: StorageMut<Storage = T>> StorageMut for (S, T) {
    fn storage_mut(&mut self) -> &mut Self::Storage {
        self
    }
}

impl<S: Dummy, T: Dummy> Dummy for (S, T) {
    unsafe fn dummy() -> Self {
        (S::dummy(), T::dummy())
    }
}

impl<S: RemovePrefix, T: RemovePrefix> RemovePrefix for (S, T) {
    fn remove_prefix(&mut self, n: usize) {
        self.0.remove_prefix(n);
        self.1.remove_prefix(n);
    }
}

impl<N, S, T> IntoStaticChunkIterator<N> for (S, T)
where
    S: IntoStaticChunkIterator<N>,
    T: IntoStaticChunkIterator<N>,
    N: Unsigned,
{
    type Item = (S::Item, T::Item);
    type IterType = std::iter::Zip<S::IterType, T::IterType>;

    fn into_static_chunk_iter(self) -> Self::IterType {
        self.0
            .into_static_chunk_iter()
            .zip(self.1.into_static_chunk_iter())
    }
}

impl<N, S: UniChunkable<N>, T: UniChunkable<N>> UniChunkable<N> for (S, T) {
    type Chunk = (S::Chunk, T::Chunk);
}
