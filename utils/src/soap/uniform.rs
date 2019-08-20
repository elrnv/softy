use super::*;

/// `UniChunked` Assigns a stride `N` to the specified collection.
///
/// # Example
///
/// ```rust
/// use utils::soap::*;
/// let s = Chunked2::from_flat(vec![1,2,3,4,5,6]);
/// let mut iter = s.iter();
/// assert_eq!(Some(&[1,2]), iter.next());
/// assert_eq!(Some(&[3,4]), iter.next());
/// assert_eq!(Some(&[5,6]), iter.next());
/// assert_eq!(None, iter.next());
/// ```
#[derive(Copy, Clone, PartialEq, Debug)]
pub struct UniChunked<S, N> {
    pub(crate) data: S,
    pub(crate) chunks: N,
}

impl<T, N: Default + Array<T>> UniChunked<Vec<T>, U<N>> {
    /// Create a `UniChunked` collection from a `Vec` of arrays.
    ///
    /// # Example
    ///
    /// ```
    /// use utils::soap::*;
    /// let v = vec![[1,2,3],[4,5,6],[7,8,9]];
    /// let c = Chunked3::from_array_vec(v);
    /// assert_eq!(c.data(), &vec![1,2,3,4,5,6,7,8,9]);
    /// ```
    pub fn from_array_vec(data: Vec<N::Array>) -> UniChunked<Vec<T>, U<N>> {
        UniChunked {
            chunks: Default::default(),
            data: reinterpret::reinterpret_vec(data),
        }
    }
}

impl<'a, T, N: Default + Array<T>> UniChunked<&'a [T], U<N>> {
    /// Create a `UniChunked` collection from a slice of arrays.
    ///
    /// # Example
    ///
    /// ```
    /// use utils::soap::*;
    /// let v = vec![[1,2,3],[4,5,6],[7,8,9]];
    /// let c = Chunked3::from_array_vec(v.clone());
    /// assert_eq!(c.data(), &[1,2,3,4,5,6,7,8,9]);
    /// ```
    pub fn from_array_slice(data: &[N::Array]) -> UniChunked<&[T], U<N>> {
        UniChunked {
            chunks: Default::default(),
            data: reinterpret::reinterpret_slice(data),
        }
    }
}

impl<'a, T, N: Default + Array<T>> UniChunked<&'a mut [T], U<N>> {
    /// Create a `UniChunked` collection from a mutable slice of arrays.
    ///
    /// # Example
    ///
    /// ```
    /// use utils::soap::*;
    /// let mut v = vec![[1,2,3],[4,5,6],[7,8,9]];
    /// let c = Chunked3::from_array_slice_mut(v.as_mut_slice());
    /// assert_eq!(c.data(), &mut [1,2,3,4,5,6,7,8,9]);
    /// ```
    pub fn from_array_slice_mut(data: &'a mut [N::Array]) -> UniChunked<&'a mut [T], U<N>> {
        UniChunked {
            chunks: Default::default(),
            data: reinterpret::reinterpret_mut_slice(data),
        }
    }
}

impl<'a, S: Set + ReinterpretAsGrouped<N>, N: Array<<S as Set>::Elem>> UniChunked<S, U<N>> {
    /// Convert this `UniChunked` collection into arrays.
    ///
    /// # Example
    ///
    /// ```
    /// use utils::soap::*;
    /// let mut v = vec![[1,2,3],[4,5,6],[7,8,9]];
    ///
    /// // Convert to and from a `Vec` of arrays.
    /// let v_new = Chunked3::from_array_vec(v.clone()).into_arrays();
    /// assert_eq!(v.clone(), v_new);
    ///
    /// // Convert to and from an immutable slice of arrays.
    /// let v_new = Chunked3::from_array_slice(v.as_slice()).into_arrays();
    /// assert_eq!(v.as_slice(), v_new);
    ///
    /// // Convert to and from a mutable slice of arrays.
    /// let mut v_exp = v.clone();
    /// let v_result = Chunked3::from_array_slice_mut(v.as_mut_slice()).into_arrays();
    /// assert_eq!(v_exp.as_mut_slice(), v_result);
    /// ```
    pub fn into_arrays(self) -> S::Output {
        ReinterpretAsGrouped::<N>::reinterpret_as_grouped(self.into_inner())
    }
}

//impl<'a, S: Set, N: Array<<S as Set>::Elem>, M: Unsigned + Array<<Set> UniChunked<&'a mut N::Array, U<N>>
//    where &'a mut <N as Array<<S as Set>::Elem>>::Array: ReinterpretAsGrouped<N>
//{
//    /// Convert this `UniChunked` collection into arrays.
//    ///
//    /// # Example
//    ///
//    /// ```
//    /// use utils::soap::*;
//    /// let mut v = vec![[1,2,3],[4,5,6],[7,8,9]];
//    /// let mut v_exp = v.clone();
//    /// let v_result = Chunked3::from_array_slice_mut(v.as_mut_slice()).into_arrays();
//    /// assert_eq!(v_exp.as_mut_slice(), v_result);
//    /// ```
//    pub fn into_arrays(self) -> &'a mut [N::Array] {
//        ReinterpretAsGrouped::<N>::reinterpret_as_grouped(self.into_inner())
//    }
//}

// Note: These must be separate in order to avoid conflict with standard library.
impl<T: Clone, N: Array<T> + Unsigned> Into<Vec<N::Array>> for UniChunked<Vec<T>, U<N>> {
    fn into(self) -> Vec<N::Array> {
        self.into_arrays()
    }
}

impl<'a, T: Clone, N: Array<T> + Unsigned> Into<&'a [N::Array]> for UniChunked<&'a [T], U<N>> {
    fn into(self) -> &'a [N::Array] {
        self.into_arrays()
    }
}

impl<'a, T: Clone, N: Array<T> + Unsigned> Into<&'a mut [N::Array]>
    for UniChunked<&'a mut [T], U<N>>
{
    fn into(self) -> &'a mut [N::Array] {
        self.into_arrays()
    }
}

/// Define aliases for common uniform chunked types.
pub type Chunked3<S> = UniChunked<S, U3>;
pub type Chunked2<S> = UniChunked<S, U2>;
pub type Chunked1<S> = UniChunked<S, U1>;
pub type ChunkedN<S> = UniChunked<S, usize>;

impl<S, N> UniChunked<S, N> {
    /// Get a immutable reference to the underlying data.
    ///
    /// # Example
    ///
    /// ```rust
    /// use utils::soap::*;
    /// let v = vec![1,2,3,4,5,6];
    /// let s = Chunked3::from_flat(v.clone());
    /// assert_eq!(&v, s.data());
    /// ```
    pub fn data(&self) -> &S {
        &self.data
    }
    /// Get a mutable reference to the underlying data.
    ///
    /// # Example
    ///
    /// ```rust
    /// use utils::soap::*;
    /// let mut v = vec![1,2,3,4,5,6];
    /// let mut s = Chunked3::from_flat(v.clone());
    /// v[2] = 100;
    /// s.data_mut()[2] = 100;
    /// assert_eq!(&v, s.data());
    /// ```
    pub fn data_mut(&mut self) -> &mut S {
        &mut self.data
    }
}

impl<S: Default, N: Default> UniChunked<S, U<N>> {
    /// Create an empty `UniChunked` collection that groups elements into `N`
    /// chunks.
    ///
    /// # Example
    ///
    /// ```rust
    /// use utils::soap::*;
    /// let mut s = Chunked3::<Vec<usize>>::new();
    /// assert_eq!(s, Chunked3::from_flat(Vec::new()));
    /// s.push([1,2,3]);
    /// let mut iter = s.iter();
    /// assert_eq!(Some(&[1,2,3]), iter.next());
    /// assert_eq!(None, iter.next());
    /// ```
    pub fn new() -> Self {
        UniChunked {
            chunks: Default::default(), // Zero sized type.
            data: S::default(),
        }
    }
}

impl<S: Set, N: Unsigned + Default> UniChunked<S, U<N>> {
    /// Create a `UniChunked` collection that groups the elements of the
    /// original set into uniformly sized groups at compile time.
    ///
    /// # Example
    ///
    /// ```rust
    /// use utils::soap::*;
    /// let s = Chunked3::from_flat(vec![1,2,3,4,5,6]);
    /// let mut iter = s.iter();
    /// assert_eq!(Some(&[1,2,3]), iter.next());
    /// assert_eq!(Some(&[4,5,6]), iter.next());
    /// assert_eq!(None, iter.next());
    /// ```
    pub fn from_flat(data: S) -> Self {
        assert_eq!(data.len() % N::to_usize(), 0);
        UniChunked {
            chunks: Default::default(), // Zero sized type.
            data,
        }
    }
}

impl<S, N> UniChunked<S, N> {
    /// Convert this `UniChunked` collection into its inner representation.
    pub fn into_inner(self) -> S {
        self.data
    }
}

impl<S: Default> ChunkedN<S> {
    /// Create an empty `UniChunked` collection that groups elements into `n`
    /// chunks.
    ///
    /// # Example
    ///
    /// ```rust
    /// use utils::soap::*;
    /// let mut s = ChunkedN::<Vec<_>>::with_stride(3);
    /// s.push(&[1,2,3][..]);
    /// let mut iter = s.iter();
    /// assert_eq!(Some(&[1,2,3][..]), iter.next());
    /// assert_eq!(None, iter.next());
    /// ```
    pub fn with_stride(n: usize) -> Self {
        UniChunked {
            chunks: n,
            data: S::default(),
        }
    }
}

impl<S: Set> ChunkedN<S> {
    /// Create a `UniChunked` collection that groups the elements of the
    /// original set into uniformly sized groups at compile time.
    ///
    /// # Example
    ///
    /// ```rust
    /// use utils::soap::*;
    /// let s = ChunkedN::from_flat_with_stride(vec![1,2,3,4,5,6], 3);
    /// let mut iter = s.iter();
    /// assert_eq!(Some(&[1,2,3][..]), iter.next());
    /// assert_eq!(Some(&[4,5,6][..]), iter.next());
    /// assert_eq!(None, iter.next());
    /// ```
    pub fn from_flat_with_stride(data: S, n: usize) -> Self {
        assert_eq!(data.len() % n, 0);
        UniChunked { chunks: n, data }
    }
}

impl<T, N> UniChunked<Vec<T>, N> {
    /// This function panics if `src` has doesn't have a length equal to `self.len()*N::value()`.
    pub fn copy_from_flat(&mut self, src: &[T])
    where
        T: Copy,
    {
        assert_eq!(src.len(), self.data.len());
        self.data.copy_from_slice(src);
    }
    /// This function panics if `src` has doesn't have a length equal to `self.len()*N::value()`.
    pub fn clone_from_flat(&mut self, src: &[T])
    where
        T: Clone,
    {
        assert_eq!(src.len(), self.data.len());
        self.data.clone_from_slice(src);
    }
}

impl<T, N: Unsigned + Array<T>> UniChunked<Vec<T>, U<N>> {
    /// This function panics if `src` has doesn't have a length equal to `self.len()`.
    pub fn copy_from_grouped(&mut self, src: &[N::Array])
    where
        T: Copy,
    {
        assert_eq!(src.len(), self.len());
        self.data
            .copy_from_slice(reinterpret::reinterpret_slice(src));
    }
    /// This function panics if `src` has doesn't have a length equal to `self.len()`.
    pub fn clone_from_grouped(&mut self, src: &[N::Array])
    where
        T: Clone,
    {
        assert_eq!(src.len(), self.len());
        self.data
            .clone_from_slice(reinterpret::reinterpret_slice(src));
    }
}

impl<T, N: Unsigned + Array<T>> UniChunked<Vec<T>, U<N>> {
    pub fn reserve(&mut self, new_length: usize) {
        self.data.reserve(new_length * N::to_usize());
    }
}

impl<T> ChunkedN<Vec<T>> {
    pub fn reserve(&mut self, new_length: usize) {
        self.data.reserve(new_length * self.chunks);
    }
}

impl<T, N: Unsigned + Array<T>> UniChunked<Vec<T>, U<N>>
where
    <N as Array<T>>::Array: Clone,
{
    pub fn resize(&mut self, new_length: usize, default: N::Array)
    where
        T: PushArrayToVec<N> + Clone,
    {
        self.reserve(new_length);
        for _ in 0..new_length {
            PushArrayToVec::<N>::push_to_vec(default.clone(), &mut self.data);
        }
    }
}

impl<T> ChunkedN<Vec<T>> {
    pub fn resize(&mut self, new_length: usize, default: &[T])
    where
        T: Clone,
    {
        self.reserve(new_length);
        for _ in 0..new_length {
            self.data.extend_from_slice(default);
        }
    }
}

impl<T, N> UniChunked<Vec<T>, U<N>>
where
    N: Array<T>,
    T: Clone,
{
    /// Extend this chunked `Vec` from a slice of arrays.
    ///
    /// # Example
    ///
    /// ```rust
    /// use utils::soap::*;
    /// let mut s = Chunked2::from_flat(vec![0,1,2,3]);
    /// s.extend_from_slice(&[[4,5], [6,7]]);
    /// assert_eq!(s.data(), &vec![0,1,2,3,4,5,6,7]);
    /// ```
    pub fn extend_from_slice(&mut self, slice: &[N::Array]) {
        self.data
            .extend_from_slice(reinterpret::reinterpret_slice(slice));
    }
}

/// An implementation of `Set` for a `UniChunked` collection of any type that
/// can be grouped as `N` sub-elements.
impl<S: Set, N: Unsigned + Array<<S as Set>::Elem>> Set for UniChunked<S, U<N>> {
    type Elem = N::Array;

    /// Compute the length of this `UniChunked` collection as the number of
    /// grouped elements in the set.
    ///
    /// # Example
    ///
    /// ```rust
    /// use utils::soap::*;
    /// let s = UniChunked::<_, U2>::from_flat(vec![0,1,2,3,4,5]);
    /// assert_eq!(s.len(), 3);
    /// let s = UniChunked::<_, U3>::from_flat(vec![0,1,2,3,4,5]);
    /// assert_eq!(s.len(), 2);
    /// ```
    fn len(&self) -> usize {
        self.data.len() / N::to_usize()
    }
}

/// An implementation of `Set` for a `UniChunked` collection of any type that
/// can be grouped as `N` sub-elements.
impl<S: Set> Set for ChunkedN<S> {
    type Elem = Vec<S::Elem>;

    /// Compute the length of this `UniChunked` collection as the number of
    /// grouped elements in the set.
    ///
    /// # Example
    ///
    /// ```rust
    /// use utils::soap::*;
    /// let s = ChunkedN::from_flat_with_stride(vec![0,1,2,3,4,5], 2);
    /// assert_eq!(s.len(), 3);
    /// let s = ChunkedN::from_flat_with_stride(vec![0,1,2,3,4,5], 3);
    /// assert_eq!(s.len(), 2);
    /// ```
    fn len(&self) -> usize {
        self.data.len() / self.chunks
    }
}

impl<S, N> Push<N::Array> for UniChunked<S, U<N>>
where
    N: Unsigned + Array<<S as Set>::Elem>,
    <S as Set>::Elem: PushArrayTo<S, N>,
    S: Set + Push<<S as Set>::Elem>,
{
    /// Push a grouped element onto the `UniChunked` type. The pushed element must
    /// have exactly `N` sub-elements.
    ///
    /// # Example
    ///
    /// ```rust
    /// use utils::soap::*;
    /// let mut s = UniChunked::<_, U3>::from_flat(vec![1,2,3]);
    /// s.push([4,5,6]);
    /// let mut iter = s.iter();
    /// assert_eq!(Some(&[1,2,3]), iter.next());
    /// assert_eq!(Some(&[4,5,6]), iter.next());
    /// assert_eq!(None, iter.next());
    /// ```
    fn push(&mut self, element: N::Array) {
        S::Elem::push_to(element, &mut self.data);
    }
}

impl<S> Push<&[<S as Set>::Elem]> for ChunkedN<S>
where
    S: Set + Push<<S as Set>::Elem>,
    <S as Set>::Elem: Clone,
{
    /// Push a grouped element onto the `UniChunked` type. The pushed element must
    /// have exactly `N` sub-elements.
    ///
    /// # Example
    ///
    /// ```rust
    /// use utils::soap::*;
    /// let mut s = Chunked3::from_flat(vec![1,2,3]);
    /// s.push([4,5,6]);
    /// let mut iter = s.iter();
    /// assert_eq!(Some(&[1,2,3]), iter.next());
    /// assert_eq!(Some(&[4,5,6]), iter.next());
    /// assert_eq!(None, iter.next());
    /// ```
    fn push(&mut self, element: &[<S as Set>::Elem]) {
        for e in element {
            self.data.push(e.clone());
        }
    }
}

impl<S, N> IntoIterator for UniChunked<S, U<N>>
where
    S: Set + IntoStaticChunkIterator<N>,
    N: Unsigned,
{
    type Item = <S as IntoStaticChunkIterator<N>>::Item;
    type IntoIter = <S as IntoStaticChunkIterator<N>>::IterType;

    /// Convert a `UniChunked` collection into an iterator over grouped elements.
    ///
    /// # Example
    ///
    /// ```rust
    /// use utils::soap::*;
    /// let mut s = UniChunked::<_, U3>::from_flat(vec![1,2,3,4,5,6]);
    /// let mut iter = s.into_iter();
    /// assert_eq!(Some([1,2,3]), iter.next());
    /// assert_eq!(Some([4,5,6]), iter.next());
    /// assert_eq!(None, iter.next());
    /// ```
    fn into_iter(self) -> Self::IntoIter {
        self.data.into_static_chunk_iter()
    }
}

impl<S> IntoIterator for ChunkedN<S>
where
    S: Set + IntoChunkIterator,
{
    type Item = <S as IntoChunkIterator>::Item;
    type IntoIter = <S as IntoChunkIterator>::IterType;

    /// Convert a `ChunkedN` collection into an iterator over grouped elements.
    ///
    /// # Example
    ///
    /// ```rust
    /// use utils::soap::*;
    /// let mut s = ChunkedN::from_flat_with_stride(vec![1,2,3,4,5,6], 3);
    /// let mut iter = s.view().into_iter();
    /// assert_eq!(Some(&[1,2,3][..]), iter.next());
    /// assert_eq!(Some(&[4,5,6][..]), iter.next());
    /// assert_eq!(None, iter.next());
    /// ```
    fn into_iter(self) -> Self::IntoIter {
        self.data.into_chunk_iter(self.chunks)
    }
}

impl<S, N> std::iter::FromIterator<N::Array> for UniChunked<S, U<N>>
where
    N: Unsigned + Array<<S as Set>::Elem> + Default,
    <S as Set>::Elem: PushArrayTo<S, N>,
    S: Set + Default + Push<<S as Set>::Elem>,
{
    /// Construct a `UniChunked` collection from an iterator that produces
    /// chunked elements.
    ///
    /// # Example
    ///
    /// ```rust
    /// use utils::soap::*;
    /// let v = vec![[1,2,3],[4,5,6]];
    /// let s: Chunked3::<Vec<usize>> = v.into_iter().collect();
    /// let mut iter = s.iter();
    /// assert_eq!(Some(&[1,2,3]), iter.next());
    /// assert_eq!(Some(&[4,5,6]), iter.next());
    /// assert_eq!(None, iter.next());
    /// ```
    fn from_iter<T>(iter: T) -> Self
    where
        T: IntoIterator<Item = N::Array>,
    {
        let mut s = UniChunked::default();
        for i in iter {
            s.push(i);
        }
        s
    }
}

impl<S: Set + Default, N: Unsigned + Default> Default for UniChunked<S, U<N>> {
    fn default() -> Self {
        Self::from_flat(S::default())
    }
}

pub trait ReinterpretAsGrouped<N> {
    type Output;
    fn reinterpret_as_grouped(self) -> Self::Output;
}

impl<'a, S, N, M> ReinterpretAsGrouped<N> for UniChunked<S, U<M>>
where
    S: ReinterpretAsGrouped<M>,
    <S as ReinterpretAsGrouped<M>>::Output: ReinterpretAsGrouped<N>,
{
    type Output = <<S as ReinterpretAsGrouped<M>>::Output as ReinterpretAsGrouped<N>>::Output;
    #[inline]
    fn reinterpret_as_grouped(self) -> Self::Output {
        self.data.reinterpret_as_grouped().reinterpret_as_grouped()
    }
}

pub trait PushArrayTo<S, N>
where
    S: Set,
    N: Array<<S as Set>::Elem>,
{
    /// This method tells this type how it can be pushed to a set as an array.
    fn push_to(element: N::Array, set: &mut S);
}

pub trait PushArrayToVec<N>
where
    N: Array<Self>,
    Self: Sized,
{
    /// This method tells this type how it can be pushed to a `Vec` as an array.
    fn push_to_vec(element: N::Array, set: &mut Vec<Self>);
}

impl<S: Set + Push<<S as Set>::Elem>, N: Array<<S as Set>::Elem>> PushArrayTo<S, N> for S::Elem
where
    <S as Set>::Elem: Clone,
{
    fn push_to(element: N::Array, set: &mut S) {
        for i in N::iter(&element) {
            set.push(i.clone());
        }
    }
}
impl<T: Clone, N: Array<T>> PushArrayToVec<N> for T {
    fn push_to_vec(element: N::Array, set: &mut Vec<T>) {
        set.extend_from_slice(N::as_slice(&element));
    }
}

//impl<T, N> std::borrow::Borrow<[T]> for UniChunked<&[T], N> {
//    fn borrow(&self) -> &[T] {
//        self.data
//    }
//}
//
//impl<T, N> std::borrow::Borrow<[T]> for UniChunked<&mut [T], N> {
//    fn borrow(&self) -> &[T] {
//        self.data
//    }
//}
//
//impl<T, N> std::borrow::BorrowMut<[T]> for UniChunked<&mut [T], N> {
//    fn borrow_mut(&mut self) -> &mut [T] {
//        self.data
//    }
//}

impl<T, N: Array<T> + Unsigned> std::borrow::Borrow<[N::Array]> for UniChunked<&[T], U<N>> {
    fn borrow(&self) -> &[N::Array] {
        reinterpret::reinterpret_slice(self.data)
    }
}
impl<T, N: Array<T> + Unsigned> std::borrow::Borrow<[N::Array]> for UniChunked<&mut [T], U<N>> {
    fn borrow(&self) -> &[N::Array] {
        reinterpret::reinterpret_slice(self.data)
    }
}
impl<T, N: Array<T> + Unsigned> std::borrow::BorrowMut<[N::Array]> for UniChunked<&mut [T], U<N>> {
    fn borrow_mut(&mut self) -> &mut [N::Array] {
        reinterpret::reinterpret_mut_slice(self.data)
    }
}

impl<S, N> ToOwned for UniChunked<S, N>
where
    S: ToOwned,
    N: Copy,
{
    type Owned = UniChunked<<S as ToOwned>::Owned, N>;

    /// Convert this `UniChunked` collection to an owned one.
    ///
    /// # Example
    ///
    /// ```rust
    /// use utils::soap::*;
    /// let v = vec![1,2,3,4,5,6];
    /// let s_view = UniChunked::<_, U3>::from_flat(v.as_slice());
    /// let s_owned = UniChunked::<_, U3>::from_flat(v.clone());
    /// assert_eq!(utils::soap::ToOwned::to_owned(s_view), s_owned);
    /// ```
    fn to_owned(self) -> Self::Owned {
        UniChunked {
            data: self.data.to_owned(),
            chunks: self.chunks,
        }
    }
}

impl<S, N> ToOwnedData for UniChunked<S, N>
where
    S: ToOwnedData,
    N: Copy,
{
    type OwnedData = UniChunked<S::OwnedData, N>;
    fn to_owned_data(self) -> Self::OwnedData {
        let UniChunked { chunks, data } = self;
        UniChunked {
            chunks,
            data: data.to_owned_data(),
        }
    }
}

/*
 * Indexing
 */

impl<'a, S, N> GetIndex<'a, UniChunked<S, U<N>>> for usize
where
    S: Set + Get<'a, StaticRange<N>>,
    N: Unsigned + Array<<S as Set>::Elem>,
{
    type Output = S::Output;

    /// Get an element of the given `UniChunked` collection.
    fn get(self, chunked: &UniChunked<S, U<N>>) -> Option<Self::Output> {
        if self < chunked.len() {
            chunked.data.get(StaticRange::new(self * N::to_usize()))
        } else {
            None
        }
    }
}

impl<'a, S, N> GetIndex<'a, UniChunked<S, U<N>>> for std::ops::Range<usize>
where
    S: Set + Get<'a, std::ops::Range<usize>>,
    N: Unsigned + Default + Array<<S as Set>::Elem>,
{
    type Output = UniChunked<S::Output, U<N>>;

    /// Get a `[begin..end)` subview of the given `UniChunked` collection.
    fn get(self, chunked: &UniChunked<S, U<N>>) -> Option<Self::Output> {
        if self.start <= self.end && self.end <= chunked.len() {
            chunked
                .data
                .get(N::to_usize() * self.start..N::to_usize() * self.end)
                .map(|data| UniChunked {
                    data,
                    chunks: Default::default(),
                })
        } else {
            None
        }
    }
}

impl<'a, S> GetIndex<'a, ChunkedN<S>> for usize
where
    S: Set + Get<'a, std::ops::Range<usize>>,
{
    type Output = S::Output;

    /// Get an element of the given `ChunkedN` collection.
    fn get(self, chunked: &ChunkedN<S>) -> Option<Self::Output> {
        if self < chunked.len() {
            let stride = chunked.chunks;
            chunked.data.get(std::ops::Range {
                start: self * stride,
                end: (self + 1) * stride,
            })
        } else {
            None
        }
    }
}

impl<'a, S> GetIndex<'a, ChunkedN<S>> for std::ops::Range<usize>
where
    S: Set + Get<'a, std::ops::Range<usize>>,
{
    type Output = ChunkedN<S::Output>;

    /// Get a `[begin..end)` subview of the given `ChunkedN` collection.
    fn get(self, chunked: &ChunkedN<S>) -> Option<Self::Output> {
        if self.start <= self.end && self.end <= chunked.len() {
            let stride = chunked.chunks;
            chunked
                .data
                .get(stride * self.start..stride * self.end)
                .map(|data| UniChunked {
                    data,
                    chunks: stride,
                })
        } else {
            None
        }
    }
}

impl<S, N> IsolateIndex<UniChunked<S, U<N>>> for usize
where
    S: Set + Isolate<StaticRange<N>>,
    N: Unsigned + Array<<S as Set>::Elem>,
{
    type Output = S::Output;

    /// Isolate a chunk of the given `UniChunked` collection.
    fn try_isolate(self, chunked: UniChunked<S, U<N>>) -> Option<Self::Output> {
        if self < chunked.len() {
            chunked
                .data
                .try_isolate(StaticRange::new(self * N::to_usize()))
        } else {
            None
        }
    }
}

impl<S, N> IsolateIndex<UniChunked<S, U<N>>> for std::ops::Range<usize>
where
    S: Set + Isolate<std::ops::Range<usize>>,
    N: Unsigned + Default + Array<<S as Set>::Elem>,
{
    type Output = UniChunked<S::Output, U<N>>;

    /// Isolate a `[begin..end)` range of the given `UniChunked` collection.
    fn try_isolate(self, chunked: UniChunked<S, U<N>>) -> Option<Self::Output> {
        if self.start <= self.end && self.end <= chunked.len() {
            chunked
                .data
                .try_isolate(N::to_usize() * self.start..N::to_usize() * self.end)
                .map(|data| UniChunked {
                    data,
                    chunks: Default::default(),
                })
        } else {
            None
        }
    }
}

impl<S> IsolateIndex<ChunkedN<S>> for usize
where
    S: Set + Isolate<std::ops::Range<usize>>,
{
    type Output = S::Output;

    /// Isolate a chunk of the given `ChunkedN` collection.
    fn try_isolate(self, chunked: ChunkedN<S>) -> Option<Self::Output> {
        if self < chunked.len() {
            let stride = chunked.chunks;
            chunked.data.try_isolate(std::ops::Range {
                start: self * stride,
                end: (self + 1) * stride,
            })
        } else {
            None
        }
    }
}

impl<S> IsolateIndex<ChunkedN<S>> for std::ops::Range<usize>
where
    S: Set + Isolate<std::ops::Range<usize>>,
{
    type Output = ChunkedN<S::Output>;

    /// Isolate a `[begin..end)` range of the given `ChunkedN` collection.
    fn try_isolate(self, chunked: ChunkedN<S>) -> Option<Self::Output> {
        if self.start <= self.end && self.end <= chunked.len() {
            let stride = chunked.chunks;
            chunked
                .data
                .try_isolate(stride * self.start..stride * self.end)
                .map(|data| UniChunked {
                    data,
                    chunks: stride,
                })
        } else {
            None
        }
    }
}

impl<S, N, I> Isolate<I> for UniChunked<S, N>
where
    I: IsolateIndex<Self>,
{
    type Output = I::Output;
    /// Get a mutable subview from this `UniChunked` collection according to the
    /// given range. If the range is a single index, then a single chunk is
    /// returned instead.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use utils::soap::*;
    /// let mut v = vec![1,2,3, 4,5,6, 0,0,0, 10,11,12];
    /// let mut s = Chunked3::from_flat(v.as_mut_slice());
    ///
    /// s.view_mut().try_isolate(2).unwrap().copy_from_slice(&[7,8,9]);
    /// assert_eq!(s.view().get(2), Some(&[7,8,9])); // Single index
    /// assert_eq!(v, vec![1,2,3, 4,5,6, 7,8,9, 10,11,12]);
    /// ```
    fn try_isolate(self, range: I) -> Option<I::Output> {
        range.try_isolate(self)
    }
}

impl<T, N> std::ops::Index<usize> for UniChunked<Vec<T>, U<N>>
where
    N: Unsigned + Array<T>,
{
    type Output = N::Array;

    /// Index the `UniChunked` `Vec` by `usize`. Note that this
    /// works for chunked collections that are themselves not chunked, since the
    /// item at the index of a doubly chunked collection is itself chunked,
    /// which cannot be represented by a single borrow. For more complex
    /// indexing use the `get` method provided by the `Get` trait.
    ///
    /// # Example
    ///
    /// ```rust
    /// use utils::soap::*;
    /// let s = Chunked3::from_flat(vec![1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]);
    /// assert_eq!([7,8,9], s[2]);
    /// ```
    fn index(&self, idx: usize) -> &Self::Output {
        ReinterpretAsGrouped::<N>::reinterpret_as_grouped(&self.data).index(idx)
    }
}

impl<T, N> std::ops::Index<usize> for UniChunked<&[T], U<N>>
where
    N: Unsigned + Array<T>,
{
    type Output = N::Array;

    /// Immutably index the `UniChunked` borrowed slice by `usize`. Note
    /// that this works for chunked collections that are themselves not chunked,
    /// since the item at the index of a doubly chunked collection is itself
    /// chunked, which cannot be represented by a single borrow. For more
    /// complex indexing use the `get` method provided by the `Get` trait.
    ///
    /// # Example
    ///
    /// ```rust
    /// use utils::soap::*;
    /// let s = Chunked3::from_flat(vec![1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]);
    /// assert_eq!([7,8,9], s.view()[2]);
    /// ```
    fn index(&self, idx: usize) -> &Self::Output {
        ReinterpretAsGrouped::<N>::reinterpret_as_grouped(self.data).index(idx)
    }
}

impl<T, N> std::ops::Index<usize> for UniChunked<&mut [T], U<N>>
where
    N: Unsigned + Array<T>,
{
    type Output = N::Array;

    /// Immutably index the `UniChunked` mutably borrowed slice by `usize`. Note
    /// that this works for chunked collections that are themselves not chunked,
    /// since the item at the index of a doubly chunked collection is itself
    /// chunked, which cannot be represented by a single borrow. For more
    /// complex indexing use the `get` method provided by the `Get` trait.
    ///
    /// # Example
    ///
    /// ```rust
    /// use utils::soap::*;
    /// let mut s = Chunked3::from_flat(vec![1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]);
    /// assert_eq!([7,8,9], s.view_mut()[2]);
    /// ```
    fn index(&self, idx: usize) -> &Self::Output {
        ReinterpretAsGrouped::<N>::reinterpret_as_grouped(&*self.data).index(idx)
    }
}

impl<T, N> std::ops::IndexMut<usize> for UniChunked<Vec<T>, U<N>>
where
    N: Unsigned + Array<T>,
{
    /// Mutably index the `UniChunked` `Vec` by `usize`. Note that this
    /// works for chunked collections that are themselves not chunked, since the
    /// item at the index of a doubly chunked collection is itself chunked,
    /// which cannot be represented by a single borrow. For more complex
    /// indexing use the `get` method provided by the `Get` trait.
    ///
    /// # Example
    ///
    /// ```rust
    /// use utils::soap::*;
    /// let mut v = vec![1,2,3,4,5,6,0,0,0,10,11,12];
    /// let mut s = Chunked3::from_flat(v);
    /// s[2] = [7,8,9];
    /// assert_eq!(vec![1,2,3,4,5,6,7,8,9,10,11,12], s.into_flat().to_vec());
    /// ```
    fn index_mut(&mut self, idx: usize) -> &mut Self::Output {
        ReinterpretAsGrouped::<N>::reinterpret_as_grouped(&mut self.data).index_mut(idx)
    }
}

impl<T, N> std::ops::IndexMut<usize> for UniChunked<&mut [T], U<N>>
where
    N: Unsigned + Array<T>,
{
    /// Mutably index the `UniChunked` mutably borrowed slice by `usize`.
    /// Note that this works for chunked collections that are themselves not
    /// chunked, since the item at the index of a doubly chunked collection is
    /// itself chunked, which cannot be represented by a single borrow. For more
    /// complex indexing use the `get` method provided by the `Get` trait.
    ///
    /// # Example
    ///
    /// ```rust
    /// use utils::soap::*;
    /// let mut v = vec![1,2,3,4,5,6,0,0,0,10,11,12];
    /// let mut s = Chunked3::from_flat(v.as_mut_slice());
    /// s[2] = [7,8,9];
    /// assert_eq!(vec![1,2,3,4,5,6,7,8,9,10,11,12], v);
    /// ```
    fn index_mut(&mut self, idx: usize) -> &mut Self::Output {
        ReinterpretAsGrouped::<N>::reinterpret_as_grouped(&mut *self.data).index_mut(idx)
    }
}

/*
 * Iteration
 */

impl<'a, S, N> UniChunked<S, U<N>>
where
    S: View<'a>,
    <S as View<'a>>::Type: IntoStaticChunkIterator<N>,
    N: Unsigned,
{
    /// Produce an iterator over borrowed grouped elements of the `UniChunked`.
    ///
    /// # Examples
    ///
    /// The following is a simple test for iterating over a uniformly organized `Vec`.
    ///
    /// ```rust
    /// use utils::soap::*;
    /// let mut s = UniChunked::<_, U2>::from_flat(vec![1,2,3,4]);
    /// let mut iter = s.iter();
    /// assert_eq!(Some(&[1,2]), iter.next());
    /// assert_eq!(Some(&[3,4]), iter.next());
    /// assert_eq!(None, iter.next());
    /// ```
    ///
    /// A more complex example consists of data organized as a nested `UniChunked`.
    ///
    /// ```rust
    /// use utils::soap::*;
    /// let s0 = UniChunked::<_, U2>::from_flat(vec![1,2, 3,4, 5,6, 7,8, 9,10, 11,12]);
    /// let s1 = UniChunked::<_, U3>::from_flat(s0);
    /// let mut iter1 = s1.iter();
    /// let mut s0 = iter1.next().unwrap();
    /// let mut iter0 = s0.iter();
    /// assert_eq!(Some(&[1,2]), iter0.next());
    /// assert_eq!(Some(&[3,4]), iter0.next());
    /// assert_eq!(Some(&[5,6]), iter0.next());
    /// assert_eq!(None, iter0.next());
    /// let s0 = iter1.next().unwrap();
    /// let mut iter0 = s0.iter();
    /// assert_eq!(Some(&[7,8]), iter0.next());
    /// assert_eq!(Some(&[9,10]), iter0.next());
    /// assert_eq!(Some(&[11,12]), iter0.next());
    /// assert_eq!(None, iter0.next());
    /// ```
    pub fn iter(&'a self) -> <<S as View<'a>>::Type as IntoStaticChunkIterator<N>>::IterType {
        self.data.view().into_static_chunk_iter()
    }
}

impl<'a, S, N> UniChunked<S, U<N>>
where
    S: ViewMut<'a>,
    <S as ViewMut<'a>>::Type: IntoStaticChunkIterator<N>,
    N: Unsigned,
{
    /// Produce an iterator over mutably borrowed grouped elements of `UniChunked`.
    ///
    /// # Examples
    ///
    /// The following example shows a simple modification of a uniformly organized `Vec`.
    ///
    /// ```rust
    /// use utils::soap::*;
    /// let mut s = UniChunked::<_, U2>::from_flat(vec![0,1,2,3]);
    /// for i in s.iter_mut() {
    ///     i[0] += 1;
    ///     i[1] += 1;
    /// }
    /// let mut iter = s.iter();
    /// assert_eq!(Some(&[1,2]), iter.next());
    /// assert_eq!(Some(&[3,4]), iter.next());
    /// assert_eq!(None, iter.next());
    /// ```
    ///
    /// Nested `UniChunked`s can also be modified as follows:
    ///
    /// ```rust
    /// use utils::soap::*;
    /// let mut s0 = UniChunked::<_, U2>::from_flat(vec![1,2, 3,4, 5,6, 7,8, 9,10, 11,12]);
    /// let mut s1 = UniChunked::<_, U3>::from_flat(s0);
    /// for mut i in s1.iter_mut() {
    ///     for j in i.iter_mut() {
    ///         j[0] += 1;
    ///         j[1] += 2;
    ///     }
    /// }
    /// let mut iter1 = s1.iter();
    /// let s0 = iter1.next().unwrap();
    /// let mut iter0 = s0.iter();
    /// assert_eq!(Some(&[2,4]), iter0.next());
    /// assert_eq!(Some(&[4,6]), iter0.next());
    /// assert_eq!(Some(&[6,8]), iter0.next());
    /// assert_eq!(None, iter0.next());
    /// let s0 = iter1.next().unwrap();
    /// let mut iter0 = s0.iter();
    /// assert_eq!(Some(&[8,10]), iter0.next());
    /// assert_eq!(Some(&[10,12]), iter0.next());
    /// assert_eq!(Some(&[12,14]), iter0.next());
    /// assert_eq!(None, iter0.next());
    /// ```
    pub fn iter_mut(
        &'a mut self,
    ) -> <<S as ViewMut<'a>>::Type as IntoStaticChunkIterator<N>>::IterType {
        self.data.view_mut().into_static_chunk_iter()
    }
}

/// A generic version of the `Chunks` iterator used by slices. This is used by
/// uniformly (but not statically) chunked collections.
pub struct Chunks<S> {
    chunk_size: usize,
    data: S,
}

impl<S> Iterator for Chunks<S>
where
    S: SplitAt + Set + Dummy,
{
    type Item = S;

    fn next(&mut self) -> Option<Self::Item> {
        let data_slice = std::mem::replace(&mut self.data, Dummy::dummy());
        if data_slice.is_empty() {
            None
        } else {
            let n = std::cmp::min(data_slice.len(), self.chunk_size);
            let (l, r) = data_slice.split_at(n);
            self.data = r;
            Some(l)
        }
    }
}

impl<'a, S> UniChunked<S, usize>
where
    S: View<'a>,
{
    pub fn iter(&'a self) -> Chunks<S::Type> {
        let UniChunked { chunks, data } = self;
        Chunks {
            chunk_size: *chunks,
            data: data.view(),
        }
    }
}

impl<'a, S> UniChunked<S, usize>
where
    S: ViewMut<'a>,
{
    /// Mutably iterate over `Chunked` data.
    ///
    /// # Example
    ///
    /// ```rust
    /// use utils::soap::*;
    /// let mut s = ChunkedN::from_flat_with_stride(vec![1,2,3,4,5,6], 3);
    /// s.view_mut().isolate(1).copy_from_slice(&[0; 3]);
    /// let mut iter = s.iter();
    /// assert_eq!(Some(&[1,2,3][..]), iter.next());
    /// assert_eq!(Some(&[0,0,0][..]), iter.next());
    /// assert_eq!(None, iter.next());
    /// ```
    pub fn iter_mut(&'a mut self) -> Chunks<S::Type> {
        let UniChunked { chunks, data } = self;
        Chunks {
            chunk_size: *chunks,
            data: data.view_mut(),
        }
    }
}

/*
 * View implementations.
 */

impl<'a, S, N> View<'a> for UniChunked<S, N>
where
    S: Set + View<'a>,
    N: Copy,
    <S as View<'a>>::Type: Set,
{
    type Type = UniChunked<<S as View<'a>>::Type, N>;

    /// Create a `UniChunked` contiguous immutable (shareable) view into the
    /// underlying collection.
    ///
    /// # Example
    ///
    /// ```rust
    /// use utils::soap::*;
    /// let s = UniChunked::<_, U2>::from_flat(vec![0,1,2,3]);
    /// let v1 = s.view(); // s is now inaccessible.
    /// let v2 = v1.clone();
    /// let mut view1_iter = v1.iter();
    /// assert_eq!(Some(&[0,1]), view1_iter.next());
    /// assert_eq!(Some(&[2,3]), view1_iter.next());
    /// assert_eq!(None, view1_iter.next());
    /// for ((a, b), c) in v1.iter().zip(v2.iter()).zip(s.iter()) {
    ///     assert_eq!(a,b);
    ///     assert_eq!(b,c);
    /// }
    /// ```
    fn view(&'a self) -> Self::Type {
        UniChunked {
            data: self.data.view(),
            chunks: self.chunks,
        }
    }
}

impl<'a, S, N> ViewMut<'a> for UniChunked<S, N>
where
    S: Set + ViewMut<'a>,
    N: Copy,
    <S as ViewMut<'a>>::Type: Set,
{
    type Type = UniChunked<<S as ViewMut<'a>>::Type, N>;

    /// Create a `UniChunked` contiguous mutable (unique) view into the
    /// underlying collection.
    ///
    /// # Example
    ///
    /// ```rust
    /// use utils::soap::*;
    /// let mut s = Chunked2::from_flat(vec![0,1,2,3]);
    /// let mut v = s.view_mut();
    /// {
    ///    v.iter_mut().next().unwrap()[0] = 100;
    /// }
    /// let mut view_iter = v.iter();
    /// assert_eq!(Some(&[100,1]), view_iter.next());
    /// assert_eq!(Some(&[2,3]), view_iter.next());
    /// assert_eq!(None, view_iter.next());
    /// ```
    fn view_mut(&'a mut self) -> Self::Type {
        UniChunked {
            data: self.data.view_mut(),
            chunks: self.chunks,
        }
    }
}

impl<S: SplitAt + Set, N: Copy + Unsigned> SplitAt for UniChunked<S, U<N>> {
    /// Split the current set into two distinct sets at the given index `mid`.
    ///
    /// # Example
    ///
    /// ```rust
    /// use utils::soap::*;
    /// let s = Chunked2::from_flat(vec![0,1,2,3]);
    /// let (l, r) = s.split_at(1);
    /// assert_eq!(l, Chunked2::from_flat(vec![0,1]));
    /// assert_eq!(r, Chunked2::from_flat(vec![2,3]));
    /// ```
    fn split_at(self, mid: usize) -> (Self, Self) {
        let UniChunked { data, chunks } = self;
        let (l, r) = data.split_at(mid * N::to_usize());
        (
            UniChunked { data: l, chunks },
            UniChunked { data: r, chunks },
        )
    }
}

impl<S: SplitAt + Set> SplitAt for ChunkedN<S> {
    /// Split the current set into two distinct sets at the given index `mid`.
    ///
    /// # Example
    ///
    /// ```rust
    /// use utils::soap::*;
    /// let s = ChunkedN::from_flat_with_stride(vec![0,1,2,3], 2);
    /// let (l, r) = s.split_at(1);
    /// assert_eq!(l, ChunkedN::from_flat_with_stride(vec![0,1], 2));
    /// assert_eq!(r, ChunkedN::from_flat_with_stride(vec![2,3], 2));
    /// ```
    fn split_at(self, mid: usize) -> (Self, Self) {
        let UniChunked { data, chunks } = self;
        let (l, r) = data.split_at(mid * chunks);
        (
            UniChunked { data: l, chunks },
            UniChunked { data: r, chunks },
        )
    }
}

impl<S: SplitPrefix<N> + Set, N: Copy> SplitFirst for UniChunked<S, U<N>> {
    type First = S::Prefix;
    fn split_first(self) -> Option<(Self::First, Self)> {
        let UniChunked { data, chunks } = self;
        data.split_prefix()
            .map(|(prefix, rest)| (prefix, UniChunked { data: rest, chunks }))
    }
}

impl<
        S: SplitPrefix<<N as std::ops::Mul<M>>::Output> + Set,
        N: Unsigned + std::ops::Mul<M> + Copy,
        M: Unsigned,
    > SplitPrefix<M> for UniChunked<S, U<N>>
{
    type Prefix = UniChunked<S::Prefix, U<N>>;
    fn split_prefix(self) -> Option<(Self::Prefix, Self)> {
        let UniChunked { data, chunks } = self;
        data.split_prefix().map(|(prefix, rest)| {
            (
                UniChunked {
                    data: prefix,
                    chunks,
                },
                UniChunked { data: rest, chunks },
            )
        })
    }
}

impl<S, N, M> IntoStaticChunkIterator<N> for UniChunked<S, M>
where
    Self: Set + SplitPrefix<N> + Dummy,
    N: Unsigned,
{
    type Item = <Self as SplitPrefix<N>>::Prefix;
    type IterType = UniChunkedIter<Self, N>;
    fn into_static_chunk_iter(self) -> Self::IterType {
        self.into_generic_static_chunk_iter()
    }
}

impl<S: Dummy, N: Default> Dummy for UniChunked<S, N> {
    fn dummy() -> Self {
        UniChunked {
            data: Dummy::dummy(),
            chunks: N::default(),
        }
    }
}

impl<S: IntoFlat, N> IntoFlat for UniChunked<S, N> {
    type FlatType = <S as IntoFlat>::FlatType;
    /// Strip away the uniform organization of the underlying data, and return the underlying data.
    fn into_flat(self) -> Self::FlatType {
        self.data.into_flat()
    }
}

/// Required for building `Subset`s of `UniChunked` types.
impl<S: RemovePrefix, N: Unsigned> RemovePrefix for UniChunked<S, U<N>> {
    fn remove_prefix(&mut self, n: usize) {
        self.data.remove_prefix(n * N::to_usize());
    }
}

impl<S: RemovePrefix> RemovePrefix for ChunkedN<S> {
    fn remove_prefix(&mut self, n: usize) {
        self.data.remove_prefix(n * self.chunks);
    }
}

impl<S: Clear, N> Clear for UniChunked<S, N> {
    fn clear(&mut self) {
        self.data.clear();
    }
}
