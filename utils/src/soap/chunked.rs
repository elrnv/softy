use super::*;

/// A partitioning of the collection `S` into distinct chunks.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Chunked<S, O = Vec<usize>> {
    /// This can be either offsets of a uniform chunk size, if
    /// chunk size is specified at compile time.
    pub(crate) chunks: O,
    pub(crate) data: S,
}


impl<S, C> Chunked<S, C> {
    /// Get a immutable reference to the underlying data.
    ///
    /// # Example
    ///
    /// ```rust
    /// use utils::soap::*;
    /// let v = vec![1,2,3,4,5,6];
    /// let s = Chunked::from_offsets(vec![0,3,4,6], v.clone());
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
    /// let mut s = Chunked::from_offsets(vec![0,3,4,6], v.clone());
    /// v[2] = 100;
    /// s.data_mut()[2] = 100;
    /// assert_eq!(&v, s.data());
    /// ```
    pub fn data_mut(&mut self) -> &mut S {
        &mut self.data
    }
}

impl<S: Set, O: Buffer<usize>> Chunked<S, O> {
    /// Construct a `Chunked` collection of elements given a collection of
    /// offsets into `S`. This is the most efficient constructor for creating
    /// variable sized chunks, however it is also the most error prone.
    ///
    /// # Panics
    ///
    /// The absolute value of `offsets` is not significant, however their
    /// relative quantities are. More specifically, if `x` is the first offset,
    /// then the last element of offsets must always be `data.len() + x`.
    /// This also implies that `offsets` cannot be empty. This function panics
    /// if any one of these invariants isn't satisfied.
    ///
    /// # Example
    ///
    /// ```rust
    /// use utils::soap::*;
    /// let s = Chunked::from_offsets(vec![0,3,4,6], vec![1,2,3,4,5,6]);
    /// let mut iter = s.iter();
    /// assert_eq!(vec![1,2,3], iter.next().unwrap().to_vec());
    /// assert_eq!(vec![4], iter.next().unwrap().to_vec());
    /// assert_eq!(vec![5,6], iter.next().unwrap().to_vec());
    /// assert_eq!(None, iter.next());
    /// ```
    pub fn from_offsets(offsets: O, data: S) -> Self {
        assert!(!offsets.is_empty());
        assert_eq!(*offsets.last().unwrap(), data.len() + *offsets.first().unwrap());
        Chunked { chunks: offsets, data }
    }

    pub fn offsets(&self) -> &O {
        &self.chunks
    }
    pub fn offsets_mut(&mut self) -> &mut O {
        &mut self.chunks
    }

    /// Convert this `Chunked` into its inner representation, which consists of a
    /// collection of offsets (first output) along with the underlying data
    ///
    /// # Example
    ///
    /// ```rust
    /// use utils::soap::*;
    /// let data = vec![1,2,3,4,5,6];
    /// let offsets = vec![0,3,4,6];
    /// let s = Chunked::from_offsets(offsets.clone(), data.clone());
    /// assert_eq!(s.into_inner(), (offsets, data));
    /// ```
    /// storage type (second output).
    pub fn into_inner(self) -> (O, S) {
        let Chunked {
            chunks,
            data
        } = self;
        (chunks, data)
    }
}


impl<S> Chunked<S>
where
    S: Set + Push<<S as Set>::Elem> + Default,
    <S as Set>::Elem: Sized,
{
    /// Construct a `Chunked` `Vec` from a nested `Vec`.
    ///
    /// # Example
    ///
    /// ```rust
    /// use utils::soap::*;
    /// let s = Chunked::<Vec<_>>::from_nested_vec(vec![vec![1,2,3],vec![4],vec![5,6]]);
    /// let mut iter = s.iter();
    /// assert_eq!(vec![1,2,3], iter.next().unwrap().to_vec());
    /// assert_eq!(vec![4], iter.next().unwrap().to_vec());
    /// assert_eq!(vec![5,6], iter.next().unwrap().to_vec());
    /// assert_eq!(None, iter.next());
    /// ```
    pub fn from_nested_vec(nested_data: Vec<Vec<<S as Set>::Elem>>) -> Self {
        nested_data.into_iter().collect()
    }
}

impl<S, O> Set for Chunked<S, O>
where S: Set, O: Set
{
    type Elem = Vec<S::Elem>;

    /// Get the number of elements in a `Chunked`.
    ///
    /// # Example
    ///
    /// ```rust
    /// use utils::soap::*;
    /// let s = Chunked::from_offsets(vec![0,3,4,6], vec![1,2,3,4,5,6]);
    /// assert_eq!(3, s.len());
    /// ```
    fn len(&self) -> usize {
        self.chunks.len() - 1
    }
}

impl<'a, S: Set + Push<<S as Set>::Elem>> Push<<Self as Set>::Elem> for Chunked<S> {
    /// Push an element onto this `Chunked`.
    ///
    /// # Example
    ///
    /// ```rust
    /// use utils::soap::*;
    /// let mut s = Chunked::<Vec<usize>>::from_offsets(vec![0,1,4], vec![0,1,2,3]);
    /// s.push(vec![4,5]);
    /// let v1 = s.view();
    /// let mut view1_iter = v1.into_iter();
    /// assert_eq!(Some(&[0][..]), view1_iter.next());
    /// assert_eq!(Some(&[1,2,3][..]), view1_iter.next());
    /// assert_eq!(Some(&[4,5][..]), view1_iter.next());
    /// assert_eq!(None, view1_iter.next());
    /// ```
    fn push(&mut self, element: <Self as Set>::Elem) {
        for elem in element.into_iter() {
            self.data.push(elem);
        }
        self.chunks.push(self.data.len());
    }
}

impl<S, O> ToOwned for Chunked<S, O>
where S: ToOwned,
      O: ToOwned,
{
    type Owned = Chunked<S::Owned, O::Owned>;

    fn to_owned(self) -> Self::Owned {
        Chunked {
            chunks: self.chunks.to_owned(),
            data: self.data.to_owned(),
        }
    }
}

// NOTE: There is currently no way to split ownership of a Vec without
// allocating. For this reason we opt to use a slice and defer allocation to
// a later step when the results may be collected into another Vec. This saves
// an extra allocation. We could make this more righteous with a custom
// allocator.
impl<'a, S> std::iter::FromIterator<&'a mut [<S as Set>::Elem]> for Chunked<S>
where
    S: Set
        + ExtendFromSlice<Item = <S as Set>::Elem>
        + Default
        + std::iter::FromIterator<&'a mut [<S as Set>::Elem]>,
    <S as Set>::Elem: Sized + 'a,
{
    fn from_iter<T>(iter: T) -> Self
    where
        T: IntoIterator<Item = &'a mut [<S as Set>::Elem]>,
    {
        let mut s = Chunked::default();
        for i in iter {
            s.push_slice(i);
        }
        s
    }
}

// For convenience we also implement a `FromIterator` trait for building from
// nested `Vec`s, however as mentioned in the note above, this is typically
// inefficient because it relies on intermediate allocations. This is acceptable
// during initialization, for instance.
impl<S> std::iter::FromIterator<Vec<<S as Set>::Elem>> for Chunked<S>
where
    S: Set + Default + Push<<S as Set>::Elem>,
    <S as Set>::Elem: Sized,
{
    /// Construct a `Chunked` from an iterator over `Vec` types.
    ///
    /// # Example
    ///
    /// ```rust
    /// use utils::soap::*;
    /// use std::iter::FromIterator;
    /// let s = Chunked::<Vec<_>>::from_iter(vec![vec![1,2,3],vec![4],vec![5,6]].into_iter());
    /// dbg!(&s);
    /// let mut iter = s.iter();
    /// assert_eq!(vec![1,2,3], iter.next().unwrap().to_vec());
    /// assert_eq!(vec![4], iter.next().unwrap().to_vec());
    /// assert_eq!(vec![5,6], iter.next().unwrap().to_vec());
    /// assert_eq!(None, iter.next());
    /// ```
    fn from_iter<T>(iter: T) -> Self
    where
        T: IntoIterator<Item = Vec<<S as Set>::Elem>>,
    {
        let mut s = Chunked::default();
        for i in iter {
            s.push(i);
        }
        s
    }
}

/*
 * Indexing
 */

impl<'o, 'i: 'o, S, O> GetIndex<'i, 'o, Chunked<S, O>> for usize
where S: Set + Get<'i, 'o, std::ops::Range<usize>>,
      O: Set + Get<'i, 'o, usize, Output=&'o usize>,
{
    type Output = S::Output;

    /// Get a n element of the given `Chunked` collection.
    fn get(self, chunked: &'i Chunked<S, O>) -> Option<Self::Output> {
        if self <= chunked.len() {
            let begin = *chunked.chunks.get(self);
            let end = *chunked.chunks.get(self + 1);
            Some(chunked.data.get(begin..end))
        } else {
            None
        }
    }
}

impl<'o, 'i: 'o, S, O> GetIndex<'i, 'o, Chunked<S, O>> for std::ops::Range<usize>
where S: Set + Get<'i, 'o, std::ops::Range<usize>>,
      O: Set + Get<'i, 'o, std::ops::Range<usize>, Output=&'o [usize]>,
{
    type Output = Chunked<S::Output, &'o [usize]>;

    /// Get a `[begin..end)` subview of the given `Chunked` collection.
    fn get(mut self, chunked: &'i Chunked<S, O>) -> Option<Self::Output> {
        if self.start <= self.end && self.end <= chunked.len() {
            self.end += 1;
            let chunks = chunked.chunks.get(self);
            let data = chunked.data.get(*chunks.first().unwrap()..*chunks.last().unwrap());
            Some(Chunked {
                chunks,
                data,
            })
        } else {
            None
        }
    }
}

impl<'o, 'i: 'o, S, O> GetIndex<'i, 'o, Chunked<S, O>> for std::ops::RangeFrom<usize>
where S: Set + Get<'i, 'o, std::ops::Range<usize>>,
      O: Set + Get<'i, 'o, std::ops::Range<usize>, Output=&'o [usize]>,
{
    type Output = Chunked<S::Output, &'o [usize]>;

    /// Get a `[begin..)` subview of the given `Chunked` collection.
    fn get(self, chunked: &'i Chunked<S, O>) -> Option<Self::Output> {
        (self.start..chunked.len()).get(chunked)
    }
}

impl<'o, 'i: 'o, S, O> GetIndex<'i, 'o, Chunked<S, O>> for std::ops::RangeTo<usize>
where S: Set + Get<'i, 'o, std::ops::Range<usize>>,
      O: Set + Get<'i, 'o, std::ops::Range<usize>, Output=&'o [usize]>,
{
    type Output = Chunked<S::Output, &'o [usize]>;

    /// Get a `[..end)` subview of the given `Chunked` collection.
    fn get(self, chunked: &'i Chunked<S, O>) -> Option<Self::Output> {
        (0..self.end).get(chunked)
    }
}

impl<'o, 'i: 'o, S, O> GetIndex<'i, 'o, Chunked<S, O>> for std::ops::RangeFull
where S: Set + Get<'i, 'o, std::ops::Range<usize>>,
      O: Set + Get<'i, 'o, std::ops::Range<usize>, Output=&'o [usize]>,
{
    type Output = Chunked<S::Output, &'o [usize]>;

    /// Get a view of the given `Chunked` collection. This is synonymous with
    /// `chunked.view()`.
    fn get(self, chunked: &'i Chunked<S, O>) -> Option<Self::Output> {
        (0..chunked.len()).get(chunked)
    }
}

impl<'o, 'i: 'o, S, O> GetIndex<'i, 'o, Chunked<S, O>> for std::ops::RangeInclusive<usize>
where S: Set + Get<'i, 'o, std::ops::Range<usize>>,
      O: Set + Get<'i, 'o, std::ops::Range<usize>, Output=&'o [usize]>,
{
    type Output = Chunked<S::Output, &'o [usize]>;

    /// Get a `[begin..end]` (including the element at `end`) subview of the
    /// given `Chunked` collection.
    fn get(self, chunked: &'i Chunked<S, O>) -> Option<Self::Output> {
        if *self.end() == usize::max_value() { None }
        else {(*self.start()..*self.end() + 1).get(chunked) }
    }
}

impl<'o, 'i: 'o, S, O> GetIndex<'i, 'o, Chunked<S, O>> for std::ops::RangeToInclusive<usize>
where S: Set + Get<'i, 'o, std::ops::Range<usize>>,
      O: Set + Get<'i, 'o, std::ops::Range<usize>, Output=&'o [usize]>,
{
    type Output = Chunked<S::Output, &'o [usize]>;

    /// Get a `[..end]` (including the element at `end`) subview of the given
    /// `Chunked` collection.
    fn get(self, chunked: &'i Chunked<S, O>) -> Option<Self::Output> {
        (0..=self.end).get(chunked)
    }
}

impl<'o, 'i: 'o, S, O, I> Get<'i, 'o, I> for Chunked<S, O>
where I: GetIndex<'i, 'o, Self>
{
    type Output = I::Output;
    /// Get a subview from this `Chunked` collection according to the given
    /// range. If the range is a single index, then a single chunk is returned
    /// instead.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use utils::soap::*;
    /// let v = vec![1,2,3,4,5,6,7,8,9,10,11];
    /// let s = Chunked::from_offsets(vec![0,3,4,6,9,11], v.clone());
    ///
    /// assert_eq!(s.get(2), &s[2]); // Single index
    ///
    /// let r = s.get(1..3);         // Range
    /// let mut iter = r.iter();
    /// assert_eq!(Some(&[4][..]), iter.next());
    /// assert_eq!(Some(&[5,6][..]), iter.next());
    /// assert_eq!(None, iter.next());
    ///
    /// let r = s.get(3..);         // RangeFrom
    /// let mut iter = r.iter();
    /// assert_eq!(Some(&[7,8,9][..]), iter.next());
    /// assert_eq!(Some(&[10,11][..]), iter.next());
    /// assert_eq!(None, iter.next());
    ///
    /// let r = s.get(..2);         // RangeTo
    /// let mut iter = r.iter();
    /// assert_eq!(Some(&[1,2,3][..]), iter.next());
    /// assert_eq!(Some(&[4][..]), iter.next());
    /// assert_eq!(None, iter.next());
    ///
    /// assert_eq!(s.view(), s.get(..)); // RangeFull
    /// assert_eq!(s.view(), s.view().get(..));
    ///
    /// let r = s.get(1..=2);         // RangeInclusive
    /// let mut iter = r.iter();
    /// assert_eq!(Some(&[4][..]), iter.next());
    /// assert_eq!(Some(&[5,6][..]), iter.next());
    /// assert_eq!(None, iter.next());
    ///
    /// let r = s.get(..=1);         // RangeToInclusive
    /// let mut iter = r.iter();
    /// assert_eq!(Some(&[1,2,3][..]), iter.next());
    /// assert_eq!(Some(&[4][..]), iter.next());
    /// assert_eq!(None, iter.next());
    /// ```
    fn get(&'i self, range: I) -> I::Output {
        range.get(self).expect("Index out of bounds")
    }
}

impl<'o, 'i: 'o, S, O> GetMutIndex<'i, 'o, Chunked<S, O>> for usize
where S: Set + GetMut<'i, 'o, std::ops::Range<usize>>,
      O: Set + Get<'i, 'o, usize, Output=&'o usize>,
{
    type Output = S::Output;

    /// Get a mutable reference to a chunk of the given `Chunked` collection.
    fn get_mut(self, chunked: &'i mut Chunked<S, O>) -> Option<Self::Output> {
        if self <= chunked.len() {
            let begin = *chunked.chunks.get(self);
            let end = *chunked.chunks.get(self + 1);
            Some(chunked.data.get_mut(begin..end))
        } else {
            None
        }
    }
}

impl<'o, 'i: 'o, S, O> GetMutIndex<'i, 'o, Chunked<S, O>> for std::ops::Range<usize>
where S: Set + GetMut<'i, 'o, std::ops::Range<usize>>,
      O: Set + Get<'i, 'o, std::ops::Range<usize>, Output=&'o [usize]>,
{
    type Output = Chunked<S::Output, &'o [usize]>;

    /// Get a mutable `[begin..end)` subview of the given `Chunked` collection.
    fn get_mut(mut self, chunked: &'i mut Chunked<S, O>) -> Option<Self::Output> {
        if self.start <= self.end && self.end <= chunked.len() {
            self.end += 1;
            let chunks = chunked.chunks.get(self);
            let data = chunked.data.get_mut(*chunks.first().unwrap()..*chunks.last().unwrap());
            Some(Chunked {
                chunks,
                data,
            })
        } else {
            None
        }
    }
}

impl<'o, 'i: 'o, S, O> GetMutIndex<'i, 'o, Chunked<S, O>> for std::ops::RangeFrom<usize>
where S: Set + GetMut<'i, 'o, std::ops::Range<usize>>,
      O: Set + Get<'i, 'o, std::ops::Range<usize>, Output=&'o [usize]>,
{
    type Output = Chunked<S::Output, &'o [usize]>;

    /// Get a `[begin..)` subview of the given `Chunked` collection.
    fn get_mut(self, chunked: &'i mut Chunked<S, O>) -> Option<Self::Output> {
        (self.start..chunked.len()).get_mut(chunked)
    }
}

impl<'o, 'i: 'o, S, O> GetMutIndex<'i, 'o, Chunked<S, O>> for std::ops::RangeTo<usize>
where S: Set + GetMut<'i, 'o, std::ops::Range<usize>>,
      O: Set + Get<'i, 'o, std::ops::Range<usize>, Output=&'o [usize]>,
{
    type Output = Chunked<S::Output, &'o [usize]>;

    /// Get a `[..end)` subview of the given `Chunked` collection.
    fn get_mut(self, chunked: &'i mut Chunked<S, O>) -> Option<Self::Output> {
        (0..self.end).get_mut(chunked)
    }
}

impl<'o, 'i: 'o, S, O> GetMutIndex<'i, 'o, Chunked<S, O>> for std::ops::RangeFull
where S: Set + GetMut<'i, 'o, std::ops::Range<usize>>,
      O: Set + Get<'i, 'o, std::ops::Range<usize>, Output=&'o [usize]>,
{
    type Output = Chunked<S::Output, &'o [usize]>;

    /// Get a view of the given `Chunked` collection. This is synonymous with
    /// `chunked.view()`.
    fn get_mut(self, chunked: &'i mut Chunked<S, O>) -> Option<Self::Output> {
        (0..chunked.len()).get_mut(chunked)
    }
}

impl<'o, 'i: 'o, S, O> GetMutIndex<'i, 'o, Chunked<S, O>> for std::ops::RangeInclusive<usize>
where S: Set + GetMut<'i, 'o, std::ops::Range<usize>>,
      O: Set + Get<'i, 'o, std::ops::Range<usize>, Output=&'o [usize]>,
{
    type Output = Chunked<S::Output, &'o [usize]>;

    /// Get a `[begin..end]` (including the element at `end`) subview of the
    /// given `Chunked` collection.
    fn get_mut(self, chunked: &'i mut Chunked<S, O>) -> Option<Self::Output> {
        if *self.end() == usize::max_value() { None }
        else {(*self.start()..*self.end() + 1).get_mut(chunked) }
    }
}

impl<'o, 'i: 'o, S, O> GetMutIndex<'i, 'o, Chunked<S, O>> for std::ops::RangeToInclusive<usize>
where S: Set + GetMut<'i, 'o, std::ops::Range<usize>>,
      O: Set + Get<'i, 'o, std::ops::Range<usize>, Output=&'o [usize]>,
{
    type Output = Chunked<S::Output, &'o [usize]>;

    /// Get a `[..end]` (including the element at `end`) subview of the given
    /// `Chunked` collection.
    fn get_mut(self, chunked: &'i mut Chunked<S, O>) -> Option<Self::Output> {
        (0..=self.end).get_mut(chunked)
    }
}

impl<'o, 'i: 'o, S, O, I> GetMut<'i, 'o, I> for Chunked<S, O>
where I: GetMutIndex<'i, 'o, Self>
{
    type Output = I::Output;
    /// Get a mutable subview from this `Chunked` collection according to the
    /// given range. If the range is a single index, then a single mutable chunk
    /// reference is returned instead.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use utils::soap::*;
    /// let mut v = vec![1,2,3,4,0,0,7,8,9,10,11];
    /// let mut s = Chunked::from_offsets(vec![0,3,4,6,9,11], v.clone());
    ///
    /// s.get_mut(2).copy_from_slice(&[5,6]);        // Single index
    /// assert_eq!(s.data(), &vec![1,2,3,4,5,6,7,8,9,10,11]);
    /// ```
    fn get_mut(&'i mut self, range: I) -> I::Output {
        range.get_mut(self).expect("Index out of bounds")
    }
}

impl<T, O> std::ops::Index<usize> for Chunked<Vec<T>, O>
where O: std::ops::Index<usize, Output=usize>
{
    type Output = <[T] as std::ops::Index<std::ops::Range<usize>>>::Output;
    /// Get reference to a chunk at the given index. Note that this works for
    /// `Chunked` collections that are themselves NOT `Chunked`, since a chunk
    /// of a doubly `Chunked` collection is itself `Chunked`, which cannot be
    /// represented by a single borrow. For more complex indexing use the `get`
    /// method provided by the `Get` trait.
    ///
    /// # Example
    ///
    /// ```rust
    /// use utils::soap::*;
    /// let v = vec![1,2,3,4,5,6,7,8,9,10,11];
    /// let s = Chunked::from_offsets(vec![0,3,4,6,9,11], v.clone());
    /// assert_eq!(&[5,6], &s[2]);
    /// ```
    fn index(&self, idx: usize) -> &Self::Output {
        &self.data[self.chunks[idx]..self.chunks[idx+1]]
    }
}

impl<T, O> std::ops::Index<usize> for Chunked<&[T], O>
where O: std::ops::Index<usize, Output=usize>
{
    type Output = <[T] as std::ops::Index<std::ops::Range<usize>>>::Output;

    /// Immutably index the `Chunked` borrowed slice by `usize`. Note
    /// that this works for chunked collections that are themselves not chunked,
    /// since the item at the index of a doubly chunked collection is itself
    /// chunked, which cannot be represented by a single borrow. For more
    /// complex indexing use the `get` method provided by the `Get` trait.
    ///
    /// # Example
    ///
    /// ```rust
    /// use utils::soap::*;
    /// let v = vec![1,2,3,4,5,6,7,8,9,10,11];
    /// let s = Chunked::from_offsets(vec![0,3,4,6,9,11], v.as_slice());
    /// assert_eq!(&[5,6], &s[2]);
    /// ```
    fn index(&self, idx: usize) -> &Self::Output {
        &self.data[self.chunks[idx]..self.chunks[idx+1]]
    }
}

impl<T, O> std::ops::Index<usize> for Chunked<&mut [T], O>
where O: std::ops::Index<usize, Output=usize>
{
    type Output = <[T] as std::ops::Index<std::ops::Range<usize>>>::Output;

    /// Immutably index the `Chunked` mutably borrowed slice by `usize`. Note
    /// that this works for chunked collections that are themselves not chunked,
    /// since the item at the index of a doubly chunked collection is itself
    /// chunked, which cannot be represented by a single borrow. For more
    /// complex indexing use the `get` method provided by the `Get` trait.
    ///
    /// # Example
    ///
    /// ```rust
    /// use utils::soap::*;
    /// let mut v = vec![1,2,3,4,5,6,7,8,9,10,11];
    /// let s = Chunked::from_offsets(vec![0,3,4,6,9,11], v.as_mut_slice());
    /// assert_eq!(&[5,6], &s[2]);
    /// ```
    fn index(&self, idx: usize) -> &Self::Output {
        &self.data[self.chunks[idx]..self.chunks[idx+1]]
    }
}

impl<T, O> std::ops::IndexMut<usize> for Chunked<Vec<T>, O>
where O: std::ops::Index<usize, Output=usize>
{
    /// Mutably index the `Chunked` `Vec` by `usize`. Note that this
    /// works for chunked collections that are themselves not chunked, since the
    /// item at the index of a doubly chunked collection is itself chunked,
    /// which cannot be represented by a single borrow. For more complex
    /// indexing use the `get` method provided by the `Get` trait.
    ///
    /// # Example
    ///
    /// ```rust
    /// use utils::soap::*;
    /// let mut v = vec![1,2,3,4,0,0,7,8,9,10,11];
    /// let mut s = Chunked::from_offsets(vec![0,3,4,6,9,11], v.clone());
    /// s[2].copy_from_slice(&[5,6]);
    /// assert_eq!(vec![1,2,3,4,5,6,7,8,9,10,11], s.into_flat());
    /// ```
    fn index_mut(&mut self, idx: usize) -> &mut Self::Output {
        &mut self.data[self.chunks[idx]..self.chunks[idx+1]]
    }
}

impl<T, O> std::ops::IndexMut<usize> for Chunked<&mut [T], O>
where O: std::ops::Index<usize, Output=usize>
{
    /// Mutably index the `Chunked` mutably borrowed slice by `usize`.
    /// Note that this works for chunked collections that are themselves not
    /// chunked, since the item at the index of a doubly chunked collection is
    /// itself chunked, which cannot be represented by a single borrow. For more
    /// complex indexing use the `get` method provided by the `Get` trait.
    ///
    /// # Example
    ///
    /// ```rust
    /// use utils::soap::*;
    /// let mut v = vec![1,2,3,4,0,0,7,8,9,10,11];
    /// let mut s = Chunked::from_offsets(vec![0,3,4,6,9,11], v.as_mut_slice());
    /// s[2].copy_from_slice(&[5,6]);
    /// assert_eq!(vec![1,2,3,4,5,6,7,8,9,10,11], v);
    /// ```
    fn index_mut(&mut self, idx: usize) -> &mut Self::Output {
        &mut self.data[self.chunks[idx]..self.chunks[idx+1]]
    }
}

impl<S> Chunked<S>
where
    S: Set + ExtendFromSlice<Item = <S as Set>::Elem>,
    <S as Set>::Elem: Sized
{
    /// Push a slice of elements onto this `Chunked`.
    ///
    /// # Example
    ///
    /// ```rust
    /// use utils::soap::*;
    /// let mut s = Chunked::from_offsets(vec![0,3,5], vec![1,2,3,4,5]);
    /// assert_eq!(2, s.len());
    /// s.push_slice(&[1,2]);
    /// assert_eq!(3, s.len());
    /// ```
    pub fn push_slice(&mut self, element: &[<S as Set>::Elem]) {
        self.data.extend_from_slice(element);
        self.chunks.push(self.data.len());
    }
}

impl<'a, T> IntoIterator for Chunked<&'a [T], &'a [usize]> {
    type Item = &'a [T];
    type IntoIter = VarIter<'a, &'a [T]>;

    fn into_iter(self) -> Self::IntoIter {
        VarIter {
            offsets: self.chunks,
            data: self.data,
        }
    }
}

impl<'a, S, O> Chunked<S, O>
where S: View<'a>,
      O: std::borrow::Borrow<[usize]>,
{
    /// Produce an iterator over elements (borrowed slices) of a `Chunked`.
    ///
    /// # Examples
    ///
    /// The following simple example demonstrates how to iterate over a `Chunked`
    /// of integers stored in a flat `Vec`.
    ///
    /// ```rust
    /// use utils::soap::*;
    /// let s = Chunked::from_offsets(vec![0,3,4,6], vec![1,2,3,4,5,6]);
    /// let mut iter = s.iter();
    /// let mut e0_iter = iter.next().unwrap().iter();
    /// assert_eq!(Some(&1), e0_iter.next());
    /// assert_eq!(Some(&2), e0_iter.next());
    /// assert_eq!(Some(&3), e0_iter.next());
    /// assert_eq!(None, e0_iter.next());
    /// assert_eq!(Some(&[4][..]), iter.next());
    /// assert_eq!(Some(&[5,6][..]), iter.next());
    /// assert_eq!(None, iter.next());
    /// ```
    ///
    /// Nested `Chunked`s can also be used to create more complex data organization:
    ///
    /// ```rust
    /// use utils::soap::*;
    /// let s0 = Chunked::from_offsets(vec![0,3,4,6,9,11], vec![1,2,3,4,5,6,7,8,9,10,11]);
    /// let s1 = Chunked::from_offsets(vec![0,1,4,5], s0);
    /// let mut iter1 = s1.iter();
    /// let v0 = iter1.next().unwrap();
    /// let mut iter0 = v0.iter();
    /// assert_eq!(Some(&[1,2,3][..]), iter0.next());
    /// assert_eq!(None, iter0.next());
    /// let v0 = iter1.next().unwrap();
    /// let mut iter0 = v0.iter();
    /// assert_eq!(Some(&[4][..]), iter0.next());
    /// assert_eq!(Some(&[5,6][..]), iter0.next());
    /// assert_eq!(Some(&[7,8,9][..]), iter0.next());
    /// assert_eq!(None, iter0.next());
    /// let v0 = iter1.next().unwrap();
    /// let mut iter0 = v0.iter();
    /// assert_eq!(Some(&[10,11][..]), iter0.next());
    /// assert_eq!(None, iter0.next());
    /// ```
    pub fn iter(&'a self) -> VarIter<'a, <S as View<'a>>::Type> {
        VarIter {
            offsets: self.chunks.borrow(),
            data: self.data.view(),
        }
    }
}


impl<'a, S, O> Chunked<S, O>
where
    S: ViewMut<'a>,
    O: std::borrow::Borrow<[usize]>,
{
    /// Produce a mutable iterator over elements (borrowed slices) of a
    /// `Chunked`.
    ///
    /// # Example
    ///
    /// ```rust
    /// use utils::soap::*;
    /// let mut s = Chunked::from_offsets(vec![0,3,4,6], vec![1,2,3,4,5,6]);
    /// for i in s.view_mut().iter_mut() {
    ///     for j in i.iter_mut() {
    ///         *j += 1;
    ///     }
    /// }
    /// let mut iter = s.iter();
    /// assert_eq!(vec![2,3,4], iter.next().unwrap().to_vec());
    /// assert_eq!(vec![5], iter.next().unwrap().to_vec());
    /// assert_eq!(vec![6,7], iter.next().unwrap().to_vec());
    /// assert_eq!(None, iter.next());
    /// ```
    ///
    /// Nested `Chunked`s can also be used to create more complex data organization:
    ///
    /// ```rust
    /// use utils::soap::*;
    /// let mut s0 = Chunked::from_offsets(vec![0,3,4,6,9,11], vec![0,1,2,3,4,5,6,7,8,9,10]);
    /// let mut s1 = Chunked::from_offsets(vec![0,1,4,5], s0);
    /// for mut v0 in s1.view_mut().iter_mut() {
    ///     for i in v0.iter_mut() {
    ///         for j in i.iter_mut() {
    ///             *j += 1;
    ///         }
    ///     }
    /// }
    /// let v1 = s1.view();
    /// let mut iter1 = v1.iter();
    /// let v0 = iter1.next().unwrap();
    /// let mut iter0 = v0.iter();
    /// assert_eq!(Some(&[1,2,3][..]), iter0.next());
    /// assert_eq!(None, iter0.next());
    /// let v0 = iter1.next().unwrap();
    /// let mut iter0 = v0.iter();
    /// assert_eq!(Some(&[4][..]), iter0.next());
    /// assert_eq!(Some(&[5,6][..]), iter0.next());
    /// assert_eq!(Some(&[7,8,9][..]), iter0.next());
    /// assert_eq!(None, iter0.next());
    /// let v0 = iter1.next().unwrap();
    /// let mut iter0 = v0.iter();
    /// assert_eq!(Some(&[10,11][..]), iter0.next());
    /// assert_eq!(None, iter0.next());
    /// ```
    pub fn iter_mut(&'a mut self) -> VarIterMut<'a, <S as ViewMut<'a>>::Type> {
        VarIterMut {
            offsets: self.chunks.borrow(),
            data: self.data.view_mut(),
        }
    }
}

impl<V: SplitAt + Set> SplitAt for Chunked<V, &[usize]> {
    fn split_at(self, mid: usize) -> (Self, Self) {
        let (offsets_l, offsets_r, off) = split_offsets_at(self.chunks, mid);
        let (data_l, data_r) = self.data.split_at(off);
        ( Chunked { chunks: offsets_l, data: data_l },
          Chunked { chunks: offsets_r, data: data_r } )
    }
}

/// A special iterator capable of iterating over a `Chunked`.
pub struct VarIter<'a, S> {
    offsets: &'a [usize],
    data: S,
}

/// Splits a slice of offsets at the given index into two slices such that each
/// slice is a valid slice of offsets. This means that the element at index
/// `mid` is shared between the two output slices. In addition, return the
/// offset of the middle element: this is the value `offsets[mid] - offsets[0]`.
///
/// # WARNING
/// Calling this function with an empty `offsets` slice or with `mid >=
/// offsets.len()` will cause Undefined Behaviour.
fn split_offsets_at(offsets: &[usize], mid: usize) -> (&[usize], &[usize], usize) {
    debug_assert!(!offsets.is_empty());
    debug_assert!(mid < offsets.len());
    let l = &offsets[..mid+1];
    let r = &offsets[mid..];
    // Skip bounds checking here since this function is not exposed to the user.
    let off = unsafe { *r.get_unchecked(0) - *l.get_unchecked(0) };
    (l, r, off)
}

/// Test for the `split_offset_at` helper function.
#[test]
fn split_offset_at_test() {
    let offsets = vec![0,1,2,3,4,5];
    let (l, r, off) = split_offsets_at(offsets.as_slice(), 3);
    assert_eq!(l, &[0,1,2,3]);
    assert_eq!(r, &[3,4,5]);
    assert_eq!(off, 3);
}

/// Pops an offset from the given slice of offsets and produces an increment for
/// advancing the data pointer. This is a helper function for implementing
/// iterators over `Chunked` types.
/// This function panics if offsets is empty.
fn pop_offset(offsets: &mut &[usize]) -> Option<usize> {
    debug_assert!(!offsets.is_empty(), "Chunked is corrupted and cannot be iterated.");
    offsets.split_first().and_then(|(head, tail)| {
        if tail.is_empty() {
            return None;
        }
        *offsets = tail;
        Some(unsafe { *tail.get_unchecked(0) } - *head)
    })
}

impl<'a, V> Iterator for VarIter<'a, V>
where V: SplitAt + Set + Dummy,
{
    type Item = V;

    fn next(&mut self) -> Option<Self::Item> {
        let data_slice = std::mem::replace(&mut self.data, Dummy::dummy());
        pop_offset(&mut self.offsets).map(|n| {
            let (l, r) = data_slice.split_at(n);
            self.data = r;
            l
        })
    }
}

/// Mutable variant of `VarIter`.
pub struct VarIterMut<'a, S> {
    offsets: &'a [usize],
    data: S,
}

impl<'a, V: 'a> Iterator for VarIterMut<'a, V>
where V: SplitAt + Set + Dummy,
{
    type Item = V;

    fn next(&mut self) -> Option<Self::Item> {
        // Get a unique mutable reference for the data.
        let data_slice = std::mem::replace(&mut self.data, Dummy::dummy());

        pop_offset(&mut self.offsets).map(move |n| {
            let (l, r) = data_slice.split_at(n);
            self.data = r;
            l
        })
    }
}

/*
 * `IntoIterator` implementation for `Chunked`. Note that this type of
 * iterator allocates a new `Vec` at each iteration. This is an expensive
 * operation and is here for compatibility with the rest of Rust's ecosystem.
 * However, this iterator should be used sparingly.
 */

/// IntoIter for `Chunked`.
pub struct VarIntoIter<S> {
    offsets: std::iter::Peekable<std::vec::IntoIter<usize>>,
    data: S,
}

impl<S> Iterator for VarIntoIter<S>
    where S: SplitOff + Set
{
    type Item = S;

    fn next(&mut self) -> Option<Self::Item> {
        let begin = self
            .offsets
            .next()
            .expect("Chunked is corrupted and cannot be iterated.");
        if self.offsets.len() <= 1 {
            return None; // Ignore the last offset
        }
        let end = *self.offsets.peek().unwrap();
        let n = end - begin;
        let mut l = self.data.split_off(n);
        std::mem::swap(&mut l, &mut self.data);
        Some(l) // These are the elements [0..n).
    }
}

impl<S: SplitOff + Set> SplitOff for Chunked<S> {
    fn split_off(&mut self, mid: usize) -> Self {
        // Note: Allocations in this function heavily outweigh any cost in bounds checking.
        assert!(!self.chunks.is_empty());
        assert!(mid < self.chunks.len());
        let off = self.chunks[mid] - self.chunks[0];
        let offsets_l = self.chunks[..mid+1].to_vec();
        let offsets_r = self.chunks[mid..].to_vec();
        self.chunks = offsets_l;
        let data_r = self.data.split_off(off);
        Chunked::from_offsets(offsets_r, data_r)
    }
}

impl<T> SplitOff for Vec<T> {
    fn split_off(&mut self, mid: usize) -> Self {
        Vec::split_off(self, mid)
    }
}


impl<S> IntoIterator for Chunked<S>
where
    S: SplitOff + Set,
{
    type Item = S;
    type IntoIter = VarIntoIter<S>;

    fn into_iter(self) -> Self::IntoIter {
        let Chunked {
            chunks,
            data,
        } = self;
        VarIntoIter {
            offsets: chunks.into_iter().peekable(),
            data: data,
        }
    }
}

impl<'a, S, O> View<'a> for Chunked<S, O>
where
    S: Set + View<'a>,
    O: std::borrow::Borrow<[usize]>,
    <S as View<'a>>::Type: Set,
{
    type Type = Chunked<S::Type, &'a [usize]>;

    /// Create a contiguous immutable (shareable) view into this set.
    ///
    /// # Example
    ///
    /// ```rust
    /// use utils::soap::*;
    /// let s = Chunked::<Vec<usize>>::from_offsets(vec![0,1,4,6], vec![0,1,2,3,4,5]);
    /// let v1 = s.view();
    /// let v2 = v1.clone();
    /// let mut view1_iter = v1.into_iter();
    /// assert_eq!(Some(&[0][..]), view1_iter.next());
    /// assert_eq!(Some(&[1,2,3][..]), view1_iter.next());
    /// assert_eq!(Some(&[4,5][..]), view1_iter.next());
    /// assert_eq!(None, view1_iter.next());
    /// for (a,b) in v1.into_iter().zip(v2.into_iter()) {
    ///     assert_eq!(a,b);
    /// }
    /// ```
    fn view(&'a self) -> Self::Type {
        Chunked::from_offsets(self.chunks.borrow(), self.data.view())
    }
}

impl<'a, S, O> ViewMut<'a> for Chunked<S, O>
where
    S: Set + ViewMut<'a>,
    <S as ViewMut<'a>>::Type: Set,
    O: std::borrow::Borrow<[usize]>,
{
    type Type = Chunked<S::Type, &'a [usize]>;

    /// Create a contiguous mutable (unique) view into this set.
    ///
    /// # Example
    ///
    /// ```rust
    /// use utils::soap::*;
    /// let mut s = Chunked::<Vec<usize>>::from_offsets(vec![0,1,4,6], vec![0,1,2,3,4,5]);
    /// let mut v1 = s.view_mut();
    /// v1.iter_mut().next().unwrap()[0] = 100;
    /// let mut view1_iter = v1.iter();
    /// assert_eq!(Some(&[100][..]), view1_iter.next());
    /// assert_eq!(Some(&[1,2,3][..]), view1_iter.next());
    /// assert_eq!(Some(&[4,5][..]), view1_iter.next());
    /// assert_eq!(None, view1_iter.next());
    /// ```
    fn view_mut(&'a mut self) -> Self::Type {
        Chunked::from_offsets(self.chunks.borrow(), self.data.view_mut())
    }
}


impl<S: IntoFlat> IntoFlat for Chunked<S> {
    type FlatType = S::FlatType;
    /// Strip all organizational information from this set, returning the
    /// underlying storage type.
    ///
    /// # Example
    ///
    /// ```rust
    /// use utils::soap::*;
    /// let v = vec![1,2,3,4,5,6,7,8,9,10,11];
    /// let s0 = Chunked::from_offsets(vec![0,3,4,6,9,11], v.clone());
    /// let s1 = Chunked::from_offsets(vec![0,1,4,5], s0.clone());
    /// assert_eq!(s1.into_flat(), v);
    /// assert_eq!(s0.into_flat(), v);
    /// ```
    fn into_flat(self) -> Self::FlatType {
        self.data.into_flat()
    }
}


/*
 * Utility traits intended to expose the necessary behaviour to implement `Chunked` types.
 */

pub trait ExtendFromSlice {
    type Item;
    fn extend_from_slice(&mut self, other: &[Self::Item]);
}

/*
 * Implement helper traits for supported `Set` types
 */

impl<T: Clone> ExtendFromSlice for Vec<T> {
    type Item = T;
    fn extend_from_slice(&mut self, other: &[Self::Item]) {
        Vec::extend_from_slice(self, other);
    }
}

/// A helper trait used to abstract over owned `Vec`s and slices.
pub trait Buffer<T>: Set {
    fn first(&self) -> Option<&T>;
    fn last(&self) -> Option<&T>;
}

impl<T> Buffer<T> for Vec<T> {
    fn first(&self) -> Option<&T> {
        <[T]>::first(self)
    }
    fn last(&self) -> Option<&T> {
        <[T]>::last(self)
    }
}

impl<T> Buffer<T> for [T] {
    fn first(&self) -> Option<&T> {
        <[T]>::first(self)
    }
    fn last(&self) -> Option<&T> {
        <[T]>::last(self)
    }
}

impl<'a, T, S: Buffer<T> + Set + ?Sized> Buffer<T> for &'a S {
    fn first(&self) -> Option<&T> {
        <S as Buffer<T>>::first(*self)
    }
    fn last(&self) -> Option<&T> {
        <S as Buffer<T>>::last(*self)
    }
}

impl<'a, T, S: Buffer<T> + Set + ?Sized> Buffer<T> for &'a mut S {
    fn first(&self) -> Option<&T> {
        <S as Buffer<T>>::first(*self)
    }
    fn last(&self) -> Option<&T> {
        <S as Buffer<T>>::last(*self)
    }
}

impl<S: Default> Default for Chunked<S> {
    /// Construct an empty `Chunked`.
    fn default() -> Self {
        Chunked {
            data: Default::default(),
            chunks: vec![0],
        }
    }
}

impl<S: Dummy, O: Dummy> Dummy for Chunked<S, O> {
    fn dummy() -> Self {
        Chunked { data: Dummy::dummy(), chunks: Dummy::dummy() }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_length_chunk() {
        let empty: Vec<usize> = vec![];
        // In the beginning
        let s = Chunked::from_offsets(vec![0,0,3,4,6], vec![1,2,3,4,5,6]);
        let mut iter = s.iter();
        assert_eq!(empty.clone(), iter.next().unwrap().to_vec());
        assert_eq!(vec![1,2,3], iter.next().unwrap().to_vec());
        assert_eq!(vec![4], iter.next().unwrap().to_vec());
        assert_eq!(vec![5,6], iter.next().unwrap().to_vec());
        assert_eq!(None, iter.next());

        // In the middle
        let s = Chunked::from_offsets(vec![0,3,3,4,6], vec![1,2,3,4,5,6]);
        let mut iter = s.iter();
        assert_eq!(vec![1,2,3], iter.next().unwrap().to_vec());
        assert_eq!(empty.clone(), iter.next().unwrap().to_vec());
        assert_eq!(vec![4], iter.next().unwrap().to_vec());
        assert_eq!(vec![5,6], iter.next().unwrap().to_vec());
        assert_eq!(None, iter.next());

        // At the end
        let s = Chunked::from_offsets(vec![0,3,4,6,6], vec![1,2,3,4,5,6]);
        let mut iter = s.iter();
        assert_eq!(vec![1,2,3], iter.next().unwrap().to_vec());
        assert_eq!(vec![4], iter.next().unwrap().to_vec());
        assert_eq!(vec![5,6], iter.next().unwrap().to_vec());
        assert_eq!(empty.clone(), iter.next().unwrap().to_vec());
        assert_eq!(None, iter.next());
    }
}
