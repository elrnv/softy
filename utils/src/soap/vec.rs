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

impl<T> SplitOff for Vec<T> {
    fn split_off(&mut self, mid: usize) -> Self {
        Vec::split_off(self, mid)
    }
}

impl<T, N> SplitPrefix<N> for Vec<T>
where
    N: Unsigned + Array<T>,
    <N as Array<T>>::Array: Default,
{
    type Prefix = N::Array;

    fn split_prefix(mut self) -> Option<(Self::Prefix, Self)> {
        if self.len() < N::to_usize() {
            return None;
        }
        // Note: This is inefficient ( as is the implementation for `remove_prefix` ).
        // As such it shouldn't be used when iterating over `Subset`s of
        // `Vec<T>` or `Subset`s of any other chunked collection that uses
        // `Vec<T>` for storage. We should be able to specialize the
        // implementation of subsets of `Vec<T>` types for better performance.
        self.rotate_left(N::to_usize());
        let at = self.len() - N::to_usize();
        let mut out: N::Array = Default::default();
        unsafe {
            self.set_len(at);
            std::ptr::copy_nonoverlapping(
                self.as_ptr().add(at),
                &mut out as *mut N::Array as *mut T,
                N::to_usize(),
            );
        }
        Some((out, self))
    }
}

impl<T, N> IntoStaticChunkIterator<N> for Vec<T>
where
    N: Unsigned + Array<T>,
{
    type Item = N::Array;
    type IterType = std::vec::IntoIter<N::Array>;

    fn into_static_chunk_iter(self) -> Self::IterType {
        assert_eq!(self.len() % N::to_usize(), 0);
        ReinterpretAsGrouped::<N>::reinterpret_as_grouped(self).into_iter()
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

impl<T> Storage for Vec<T> {
    type Storage = Vec<T>;
    /// `Vec` is a type of storage, simply return an immutable reference to self.
    fn storage(&self) -> &Self::Storage {
        self
    }
}

impl<T> StorageMut for Vec<T> {
    /// `Vec` is a type of storage, simply return a mutable reference to self.
    fn storage_mut(&mut self) -> &mut Self::Storage {
        self
    }
}

impl<T> CloneWithStorage<Vec<T>> for Vec<T> {
    type CloneType = Vec<T>;
    /// This function simply ignores self and returns storage since self is already
    /// a storage type.
    fn clone_with_storage(&self, storage: Vec<T>) -> Self::CloneType {
        assert_eq!(self.len(), storage.len());
        storage
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

/// Since `Vec` already owns its data, this is simply a noop.
impl<T> ToOwnedData for Vec<T> {
    type OwnedData = Self;
    fn to_owned_data(self) -> Self::OwnedData {
        self
    }
}

impl<'a, T, N> ReinterpretAsGrouped<N> for Vec<T>
where
    N: Array<T>,
{
    type Output = Vec<N::Array>;
    #[inline]
    fn reinterpret_as_grouped(self) -> Self::Output {
        unsafe { reinterpret::reinterpret_vec(self) }
    }
}

impl<'a, T, N> ReinterpretAsGrouped<N> for &'a Vec<T>
where
    N: Array<T>,
    <N as Array<T>>::Array: 'a,
{
    type Output = &'a [N::Array];
    #[inline]
    fn reinterpret_as_grouped(self) -> Self::Output {
        unsafe { reinterpret::reinterpret_slice(self.as_slice()) }
    }
}

impl<'a, T, N> ReinterpretAsGrouped<N> for &'a mut Vec<T>
where
    N: Array<T>,
    <N as Array<T>>::Array: 'a,
{
    type Output = &'a mut [N::Array];
    #[inline]
    fn reinterpret_as_grouped(self) -> Self::Output {
        unsafe { reinterpret::reinterpret_mut_slice(self.as_mut_slice()) }
    }
}

impl<T> Dummy for Vec<T> {
    unsafe fn dummy() -> Self {
        Vec::new()
    }
}

impl<T> Truncate for Vec<T> {
    fn truncate(&mut self, new_len: usize) {
        Vec::truncate(self, new_len);
    }
}

impl<T: Clone> ExtendFromSlice for Vec<T> {
    type Item = T;
    fn extend_from_slice(&mut self, other: &[Self::Item]) {
        Vec::extend_from_slice(self, other);
    }
}

/*
 * These are base cases for `ConvertStorage`. We apply the conversion at this point since `Vec` is
 * a storage type. The following are some common conversion behaviours.
 */

/// Convert a `Vec` of one type into a `Vec` of another type given that the element types can be
/// converted.
///
/// # Example
///
/// ```
/// use utils::soap::*;
/// let sentences = vec!["First", "sentence", "about", "nothing", ".", "Second", "sentence", "."];
/// let chunked = Chunked::from_sizes(vec![5,3], sentences);
/// let owned_sentences: Chunked<Vec<String>> = chunked.storage_into();
/// assert_eq!(Some(&["Second".to_string(), "sentence".to_string(), ".".to_string()][..]), owned_sentences.view().get(1));
/// ```
impl<T, S: Into<T>> StorageInto<Vec<T>> for Vec<S> {
    type Output = Vec<T>;
    fn storage_into(self) -> Self::Output {
        self.into_iter().map(|x| x.into()).collect()
    }
}

/*
 * End of ConvertStorage impls
 */

impl<T> PermuteInPlace for Vec<T> {
    /// Permute this collection according to the given permutation.
    /// The given permutation must have length equal to this collection.
    /// The slice `seen` is provided to keep track of which elements have already been seen.
    /// `seen` is assumed to be initialized to `false` and have length equal or
    /// larger than this collection.
    fn permute_in_place(&mut self, permutation: &[usize], seen: &mut [bool]) {
        self.as_mut_slice().permute_in_place(permutation, seen);
    }
}
