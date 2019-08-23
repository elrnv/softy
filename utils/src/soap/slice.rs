use super::*;

impl<'a, T, N> GetIndex<'a, &'a [T]> for StaticRange<N>
where
    N: Unsigned + Array<T>,
    <N as Array<T>>::Array: 'a,
{
    type Output = &'a N::Array;
    fn get(self, set: &&'a [T]) -> Option<Self::Output> {
        if self.end() <= set.len() {
            let slice = *set;
            Some(unsafe { &*(slice.as_ptr().add(self.start()) as *const N::Array) })
        } else {
            None
        }
    }
}

impl<'a, T, N> IsolateIndex<&'a [T]> for StaticRange<N>
where
    N: Unsigned + Array<T>,
    <N as Array<T>>::Array: 'a,
{
    type Output = &'a N::Array;
    fn try_isolate(self, set: &'a [T]) -> Option<Self::Output> {
        if self.end() <= set.len() {
            Some(unsafe { &*(set.as_ptr().add(self.start()) as *const N::Array) })
        } else {
            None
        }
    }
}

impl<'a, T, N> IsolateIndex<&'a mut [T]> for StaticRange<N>
where
    N: Unsigned + Array<T>,
    <N as Array<T>>::Array: 'a,
{
    type Output = &'a mut N::Array;
    fn try_isolate(self, set: &'a mut [T]) -> Option<Self::Output> {
        if self.end() <= set.len() {
            Some(unsafe { &mut *(set.as_mut_ptr().add(self.start()) as *mut N::Array) })
        } else {
            None
        }
    }
}

impl<'a, T, I> GetIndex<'a, &'a [T]> for I
where
    I: std::slice::SliceIndex<[T]>,
    <[T] as std::ops::Index<I>>::Output: 'a,
{
    type Output = &'a <[T] as std::ops::Index<I>>::Output;
    fn get(self, set: &&'a [T]) -> Option<Self::Output> {
        Some(std::ops::Index::<I>::index(*set, self))
    }
}

impl<'a, T, I> IsolateIndex<&'a [T]> for I
where
    I: std::slice::SliceIndex<[T]>,
    <I as std::slice::SliceIndex<[T]>>::Output: 'a,
{
    type Output = &'a <[T] as std::ops::Index<I>>::Output;
    fn try_isolate(self, set: &'a [T]) -> Option<&'a <[T] as std::ops::Index<I>>::Output> {
        Some(std::ops::Index::<I>::index(set, self))
    }
}

impl<'a, T, I> IsolateIndex<&'a mut [T]> for I
where
    I: std::slice::SliceIndex<[T]>,
    <I as std::slice::SliceIndex<[T]>>::Output: 'a,
{
    type Output = &'a mut <[T] as std::ops::Index<I>>::Output;
    fn try_isolate(self, set: &'a mut [T]) -> Option<&'a mut <[T] as std::ops::Index<I>>::Output> {
        let slice = unsafe { std::slice::from_raw_parts_mut(set.as_mut_ptr(), set.len()) };
        Some(std::ops::IndexMut::<I>::index_mut(slice, self))
    }
}

//impl<'a, T, I> Isolate<I> for &'a [T]
//where
//    I: IsolateIndex<&'a [T]>,
//{
//    type Output = I::Output;
//    fn try_isolate(self, idx: I) -> Option<Self::Output> {
//        IsolateIndex::try_isolate(idx, self)
//    }
//}
//
//impl<'a, T, I> Isolate<I> for &'a mut [T]
//where
//    I: IsolateIndex<&'a mut [T]>,
//{
//    type Output = I::Output;
//    /// Mutably index into a standard slice `[T]` using the `GetMut` trait.
//    /// This function is not intended to be used directly, but it allows getters
//    /// for chunked collections to work.
//    ///
//    /// # Example
//    ///
//    /// ```rust
//    /// let mut v = vec![1,2,3,4,5];
//    /// *utils::soap::Isolate::try_isolate(v.as_mut_slice(), 2).unwrap() = 100;
//    /// assert_eq!(v, vec![1,2,100,4,5]);
//    /// ```
//    fn try_isolate(self, idx: I) -> Option<Self::Output> {
//        IsolateIndex::try_isolate(idx, self)
//    }
//}

impl<T> Set for [T] {
    type Elem = T;
    fn len(&self) -> usize {
        <[T]>::len(self)
    }
}

impl<'a, T: 'a> View<'a> for [T] {
    type Type = &'a [T];

    fn view(&'a self) -> Self::Type {
        self
    }
}

impl<'a, T: 'a> ViewMut<'a> for [T] {
    type Type = &'a mut [T];

    fn view_mut(&'a mut self) -> Self::Type {
        self
    }
}

impl<'a, T, N> SplitPrefix<N> for &'a [T]
where
    N: Unsigned + Array<T>,
    <N as Array<T>>::Array: 'a,
{
    type Prefix = &'a N::Array;

    fn split_prefix(self) -> Option<(Self::Prefix, Self)> {
        if self.len() < N::to_usize() {
            return None;
        }

        let (prefix, rest) = unsafe {
            let prefix = self.as_ptr() as *const N::Array;
            (&*prefix, self.get_unchecked(N::to_usize()..))
        };
        Some((prefix, rest))
    }
}

impl<'a, T, N> SplitPrefix<N> for &'a mut [T]
where
    N: Unsigned + Array<T>,
    <N as Array<T>>::Array: 'a,
{
    type Prefix = &'a mut N::Array;

    fn split_prefix(self) -> Option<(Self::Prefix, Self)> {
        if self.len() < N::to_usize() {
            return None;
        }

        let (prefix, rest) = unsafe {
            let prefix = self.as_mut_ptr() as *mut N::Array;
            (&mut *prefix, self.get_unchecked_mut(N::to_usize()..))
        };
        Some((prefix, rest))
    }
}

impl<'a, T, N> IntoStaticChunkIterator<N> for &'a [T]
where
    Self: SplitPrefix<N>,
    N: Unsigned,
{
    type Item = <Self as SplitPrefix<N>>::Prefix;
    type IterType = UniChunkedIter<Self, N>;
    fn into_static_chunk_iter(self) -> Self::IterType {
        self.into_generic_static_chunk_iter()
    }
}

impl<'a, T, N> IntoStaticChunkIterator<N> for &'a mut [T]
where
    Self: SplitPrefix<N>,
    N: Unsigned,
{
    type Item = <Self as SplitPrefix<N>>::Prefix;
    type IterType = UniChunkedIter<Self, N>;
    fn into_static_chunk_iter(self) -> Self::IterType {
        self.into_generic_static_chunk_iter()
    }
}

impl<'a, T> SplitFirst for &'a [T] {
    type First = &'a T;

    fn split_first(self) -> Option<(Self::First, Self)> {
        self.split_first()
    }
}

impl<'a, T> SplitFirst for &'a mut [T] {
    type First = &'a mut T;

    fn split_first(self) -> Option<(Self::First, Self)> {
        self.split_first_mut()
    }
}

impl<'a, T> IntoFlat for &'a [T] {
    type FlatType = &'a [T];
    fn into_flat(self) -> Self::FlatType {
        self
    }
}

impl<'a, T> IntoFlat for &'a mut [T] {
    type FlatType = &'a mut [T];
    fn into_flat(self) -> Self::FlatType {
        self
    }
}

impl<'a, T> Storage for &'a [T] {
    type Storage = &'a [T];
    /// A slice is a type of storage, simply return an immutable reference to self.
    fn storage(&self) -> &Self::Storage {
        self
    }
}

impl<'a, T> StorageMut for &'a [T] {
    /// A slice is a type of storage, simply return a mutable reference to self.
    fn storage_mut(&mut self) -> &mut Self::Storage {
        self
    }
}

impl<'a, T> Storage for &'a mut [T] {
    type Storage = &'a mut [T];
    /// A slice is a type of storage, simply return an immutable reference to self.
    fn storage(&self) -> &Self::Storage {
        self
    }
}

impl<'a, T> StorageMut for &'a mut [T] {
    /// A slice is a type of storage, simply return a mutable reference to self.
    fn storage_mut(&mut self) -> &mut Self::Storage {
        self
    }
}

impl<'a, T: 'a> CloneWithFlat<Vec<T>> for &'a [T] {
    type CloneType = Vec<T>;
    /// This function simply ignores self and returns flat since self is already
    /// a flat type.
    fn clone_with_flat(&self, flat: Vec<T>) -> Self::CloneType {
        assert_eq!(self.len(), flat.len());
        flat
    }
}

impl<'a, T: 'a> CloneWithFlat<&'a [T]> for &'a [T] {
    type CloneType = &'a [T];
    /// This function simply ignores self and returns flat since self is already
    /// a flat type.
    fn clone_with_flat(&self, flat: &'a [T]) -> Self::CloneType {
        assert_eq!(self.len(), flat.len());
        flat
    }
}

impl<'a, T: 'a> CloneWithFlat<&'a mut [T]> for &'a mut [T] {
    type CloneType = &'a mut [T];
    /// This function simply ignores self and returns flat since self is already
    /// a flat type.
    fn clone_with_flat(&self, flat: &'a mut [T]) -> Self::CloneType {
        assert_eq!(self.len(), flat.len());
        flat
    }
}

impl<'a, T> SplitAt for &mut [T] {
    fn split_at(self, mid: usize) -> (Self, Self) {
        self.split_at_mut(mid)
    }
}

impl<'a, T> SplitAt for &[T] {
    fn split_at(self, mid: usize) -> (Self, Self) {
        self.split_at(mid)
    }
}

impl<T> Dummy for &[T] {
    unsafe fn dummy() -> Self {
        &[]
    }
}

impl<T> Dummy for &mut [T] {
    unsafe fn dummy() -> Self {
        &mut []
    }
}

impl<T> RemovePrefix for &[T] {
    fn remove_prefix(&mut self, n: usize) {
        let (_, r) = self.split_at(n);
        *self = r;
    }
}

impl<T> RemovePrefix for &mut [T] {
    /// Remove a prefix of size `n` from this mutable slice.
    ///
    /// # Example
    ///
    /// ```rust
    /// use::utils::soap::*;
    /// let mut v = vec![1,2,3,4,5];
    /// let mut s = v.as_mut_slice();
    /// s.remove_prefix(2);
    /// assert_eq!(&[3,4,5], s);
    /// ```
    fn remove_prefix(&mut self, n: usize) {
        let data = std::mem::replace(self, &mut []);

        let (_, r) = data.split_at_mut(n);
        *self = r;
    }
}

impl<'a, T, N> ReinterpretAsGrouped<N> for &'a [T]
where
    N: Array<T>,
    <N as Array<T>>::Array: 'a,
{
    type Output = &'a [N::Array];
    #[inline]
    fn reinterpret_as_grouped(self) -> Self::Output {
        unsafe { reinterpret::reinterpret_slice(self) }
    }
}

impl<'a, T, N> ReinterpretAsGrouped<N> for &'a mut [T]
where
    N: Array<T>,
    <N as Array<T>>::Array: 'a,
{
    type Output = &'a mut [N::Array];
    #[inline]
    fn reinterpret_as_grouped(self) -> Self::Output {
        unsafe { reinterpret::reinterpret_mut_slice(self) }
    }
}

impl<T> Viewed for &[T] {}
impl<T> Viewed for &mut [T] {}

impl<T> Truncate for &[T] {
    fn truncate(&mut self, new_len: usize) {
        // Simply forget about the elements past new_len.
        *self = self.split_at(new_len).0;
    }
}

impl<T> Truncate for &mut [T] {
    fn truncate(&mut self, new_len: usize) {
        let data = std::mem::replace(self, &mut []);
        // Simply forget about the elements past new_len.
        *self = data.split_at_mut(new_len).0;
    }
}

/*
 * These are base cases for `ConvertStorage`. We apply the conversion at this point since slices
 * are storage types. The following are some common conversion behaviours.
 */

/// Convert a slice into an owned `Vec` type.
impl<'a, T: Clone> StorageInto<Vec<T>> for &'a [T] {
    type Output = Vec<T>;
    fn storage_into(self) -> Self::Output {
        self.to_vec()
    }
}

/// Convert a mutable slice into an owned `Vec` type.
impl<'a, T: Clone> StorageInto<Vec<T>> for &'a mut [T] {
    type Output = Vec<T>;
    fn storage_into(self) -> Self::Output {
        self.to_vec()
    }
}

/// Convert a mutable slice into an immutable borrow.
impl<'a, T: 'a> StorageInto<&'a [T]> for &'a mut [T] {
    type Output = &'a [T];
    fn storage_into(self) -> Self::Output {
        &*self
    }
}

/*
 * End of ConvertStorage impls
 */
