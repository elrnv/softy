use super::*;

impl<'a, T, N> GetIndex<'a, &'a [T]> for StaticRange<N>
where
    N: num::Unsigned,
    T: Grouped<N>,
    <T as Grouped<N>>::Type: 'a,
{
    type Output = &'a T::Type;
    fn get(self, set: &&'a [T]) -> Option<Self::Output> {
        if self.end() <= set.len() {
            let slice = *set;
            Some(unsafe { &*(slice.as_ptr().add(self.start()) as *const T::Type) })
        } else {
            None
        }
    }
}

impl<'a, T, N> GetIndex<'a, &'a mut [T]> for StaticRange<N>
where
    N: num::Unsigned,
    T: Grouped<N>,
    <T as Grouped<N>>::Type: 'a,
{
    type Output = &'a T::Type;
    fn get(self, set: &&'a mut [T]) -> Option<Self::Output> {
        if self.end() <= set.len() {
            Some(unsafe { &*((*set).as_ptr().add(self.start()) as *const T::Type) })
        } else {
            None
        }
    }
}
impl<'a, T, N> GetMutIndex<'a, &'a mut [T]> for StaticRange<N>
where
    N: num::Unsigned,
    T: Grouped<N>,
    <T as Grouped<N>>::Type: 'a,
{
    type Output = &'a mut T::Type;
    fn get_mut(self, set: &mut &'a mut [T]) -> Option<Self::Output> {
        if self.end() <= set.len() {
            Some(unsafe { &mut *((*set).as_mut_ptr().add(self.start()) as *mut T::Type) })
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

impl<'a, T, I> GetIndex<'a, &'a mut [T]> for I
where
    I: std::slice::SliceIndex<[T]>,
    <[T] as std::ops::Index<I>>::Output: 'a,
{
    type Output = &'a <[T] as std::ops::Index<I>>::Output;
    fn get(self, set: &&'a mut [T]) -> Option<Self::Output> {
        let slice = unsafe { std::slice::from_raw_parts(set.as_ptr(), set.len()) };
        Some(std::ops::Index::<I>::index(slice, self))
    }
}

impl<'a, T, I> GetMutIndex<'a, &'a mut [T]> for I
where
    I: std::slice::SliceIndex<[T]>,
    <[T] as std::ops::Index<I>>::Output: 'a,
{
    type Output = &'a mut <[T] as std::ops::Index<I>>::Output;
    fn get_mut(self, set: &mut &'a mut [T]) -> Option<Self::Output> {
        let slice = unsafe { std::slice::from_raw_parts_mut(set.as_mut_ptr(), set.len()) };
        Some(std::ops::IndexMut::<I>::index_mut(slice, self))
    }
}

impl<'a, T: 'a, I> Get<'a, I> for &'a [T]
where
    I: GetIndex<'a, &'a [T]>,
{
    type Output = I::Output;
    /// Index into a standard slice `[T]` using the `Get` trait.
    ///
    /// # Example
    ///
    /// ```rust
    /// assert_eq!(utils::soap::Get::get(&&[1,2,3,4,5][..], 2), Some(&3));
    /// ```
    fn get(&self, idx: I) -> Option<Self::Output> {
        GetIndex::get(idx, self)
    }
}

impl<'a, T: 'a, I> Get<'a, I> for &'a mut [T]
where
    I: GetIndex<'a, &'a mut [T]>,
{
    type Output = I::Output;
    /// Immutable index into a standard mutable slice `[T]` using the `Get` trait.
    fn get(&self, idx: I) -> Option<Self::Output> {
        GetIndex::get(idx, self)
    }
}

impl<'a, T: 'a, I> GetMut<'a, I> for &'a mut [T]
where
    I: GetMutIndex<'a, &'a mut [T]>,
{
    type Output = I::Output;
    /// Mutably index into a standard slice `[T]` using the `GetMut` trait.
    /// This function is not intended to be used directly, but it allows getters
    /// for chunked collections to work.
    ///
    /// # Example
    ///
    /// ```rust
    /// let mut v = vec![1,2,3,4,5];
    /// *utils::soap::GetMut::get_mut(&mut v.as_mut_slice(), 2).unwrap() = 100;
    /// assert_eq!(v, vec![1,2,100,4,5]);
    /// ```
    fn get_mut(&mut self, idx: I) -> Option<Self::Output> {
        GetMutIndex::get_mut(idx, self)
    }
}

impl<T> Set for [T] {
    type Elem = T;
    fn len(&self) -> usize {
        <[T]>::len(self)
    }
}

impl<'a, T, N> SplitPrefix<N> for &'a [T]
where
    T: Grouped<N>,
    <T as Grouped<N>>::Type: 'a,
    N: num::Unsigned,
{
    type Prefix = &'a T::Type;

    fn split_prefix(self) -> Option<(Self::Prefix, Self)> {
        if self.len() < N::value() {
            return None;
        }

        let (prefix, rest) = unsafe {
            let prefix = self.as_ptr() as *const T::Type;
            (&*prefix, &self[N::value()..])
        };
        Some((prefix, rest))
    }
}

impl<'a, T, N> SplitPrefix<N> for &'a mut [T]
where
    T: Grouped<N>,
    <T as Grouped<N>>::Type: 'a,
    N: num::Unsigned,
{
    type Prefix = &'a mut T::Type;

    fn split_prefix(self) -> Option<(Self::Prefix, Self)> {
        if self.len() < N::value() {
            return None;
        }

        let (prefix, rest) = unsafe {
            let prefix = self.as_mut_ptr() as *mut T::Type;
            (&mut *prefix, &mut self[N::value()..])
        };
        Some((prefix, rest))
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
    fn dummy() -> Self {
        &[]
    }
}

impl<T> Dummy for &mut [T] {
    fn dummy() -> Self {
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
