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

impl<'a, T: 'a> ViewIterator<'a> for [T] {
    type Item = &'a T;
    type Iter = std::slice::Iter<'a, T>;

    fn view_iter(&'a self) -> Self::Iter {
        self.iter()
    }
}
impl<'a, T: 'a> ViewMutIterator<'a> for [T] {
    type Item = &'a mut T;
    type Iter = std::slice::IterMut<'a, T>;

    fn view_mut_iter(&'a mut self) -> Self::Iter {
        self.iter_mut()
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
    type Storage = [T];
    fn storage(&self) -> &Self::Storage {
        self
    }
}

impl<'a, T> Storage for &'a mut [T] {
    type Storage = [T];
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

impl<'a, T: 'a> CloneWithStorage<Vec<T>> for &'a [T] {
    type CloneType = Vec<T>;
    /// This function simply ignores self and returns storage since self is already
    /// a storage type.
    fn clone_with_storage(&self, storage: Vec<T>) -> Self::CloneType {
        assert_eq!(self.len(), storage.len());
        storage
    }
}

impl<'a, T: 'a> CloneWithStorage<&'a [T]> for &'a [T] {
    type CloneType = &'a [T];
    /// This function simply ignores self and returns storage since self is already
    /// a storage type.
    fn clone_with_storage(&self, storage: &'a [T]) -> Self::CloneType {
        assert_eq!(self.len(), storage.len());
        storage
    }
}

impl<'a, T: 'a> CloneWithStorage<&'a mut [T]> for &'a mut [T] {
    type CloneType = &'a mut [T];
    /// This function simply ignores self and returns storage since self is already
    /// a storage type.
    fn clone_with_storage(&self, storage: &'a mut [T]) -> Self::CloneType {
        assert_eq!(self.len(), storage.len());
        storage
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

impl<T> SwapChunks for &mut [T] {
    /// Swap non-overlapping chunks beginning at the given indices.
    fn swap_chunks(&mut self, i: usize, j: usize, chunk_size: usize) {
        assert!(i + chunk_size <= j || j + chunk_size <= i);

        let (lower, upper) = if i < j { (i, j) } else { (j, i) };
        let (l, r) = self.split_at_mut(upper);
        l[lower..lower + chunk_size].swap_with_slice(&mut r[..chunk_size]);
    }
}

impl<T: PartialOrd + Clone> Sort for [T] {
    /// Sort the given indices into this collection with respect to values provided by this collection.
    /// Invalid values like `NaN` in floats will be pushed to the end.
    fn sort_indices(&self, indices: &mut [usize]) {
        indices.sort_by(|&a, &b| {
            self[a]
                .partial_cmp(&self[b])
                .unwrap_or(std::cmp::Ordering::Less)
        });
    }
}

impl<T> PermuteInPlace for &mut [T] {
    /// Permute this slice according to the given permutation.
    /// The given permutation must have length equal to this slice.
    /// The slice `seen` is provided to keep track of which elements have already been seen.
    /// `seen` is assumed to be initialized to `false` and have length equal or
    /// larger than this slice.
    fn permute_in_place(&mut self, permutation: &[usize], seen: &mut [bool]) {
        let data = std::mem::replace(self, &mut []);
        UniChunked { chunks: 1, data }.permute_in_place(permutation, seen);
    }
}
