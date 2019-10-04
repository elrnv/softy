use super::*;
use std::convert::{AsMut, AsRef};

/// A Set that is a non-contiguous, unordered and possibly duplicated selection
/// of some larger collection. `S` can be any borrowed collection type that
/// implements [`Set`]. Note that it doesn't make much sense to have a `Select`
/// type own the data that it selects from, although it's possible to create
/// one.
///
/// # Simple Usage Examples
///
/// The following example shows how to `Select` from a range.
///
/// ```
/// use utils::soap::*;
/// let selection = Select::new(vec![0,2,4,0,1], 5..10);
/// let mut iter = selection.iter();
/// assert_eq!(Some((0, 5)), iter.next());
/// assert_eq!(Some((2, 7)), iter.next());
/// assert_eq!(Some((4, 9)), iter.next());
/// assert_eq!(Some((0, 5)), iter.next());
/// assert_eq!(Some((1, 6)), iter.next());
/// assert_eq!(None, iter.next());
/// ```
///
/// The next example shows how to `Select` from a [`UniChunked`] view.
///
/// ```
/// use utils::soap::*;
/// let mut v = Chunked3::from_flat((1..=15).collect::<Vec<_>>());
/// let mut selection = Select::new(vec![1,0,4,4,1], v.view_mut());
/// *selection.view_mut().isolate(0).1 = [0; 3];
/// {
///     let selection_view = selection.view();
///     let mut iter = selection_view.iter();
///     assert_eq!(Some((1, &[0,0,0])), iter.next());
///     assert_eq!(Some((0, &[1,2,3])), iter.next());
///     assert_eq!(Some((4, &[13,14,15])), iter.next());
///     assert_eq!(Some((4, &[13,14,15])), iter.next());
///     assert_eq!(Some((1, &[0,0,0])), iter.next());
///     assert_eq!(None, iter.next());
/// }
/// ```
///
/// # Mutable `Select`ions
///
/// A `Select`ion of a mutable borrow cannot be [`SplitAt`], which means it
/// cannot be [`Chunked`]. This is because a split selection must have a copy of
/// the mutable borrow since an index from any half of the split can access any
/// part of the data. This of course breaks Rust's aliasing rules. It is
/// possible, however to bypass this restriction by using interior mutability.
///
///
/// # Common Uses
///
/// Selections are a useful way to annotate arrays of indices into some other
/// array or even a range. It is not uncommon to use a `Vec<usize>` to represent
/// indices into another collection. Using `Select` instead lets the user be
/// explicit about where these indices are pointing without having to annotate
/// the indices themselves.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Select<S, I = Vec<usize>> {
    pub indices: I,
    pub target: S,
}

/// A borrowed selection.
pub type SelectView<'a, S> = Select<S, &'a [usize]>;

impl<S: Set, I: AsRef<[usize]>> Select<S, I> {
    /// Create a selection of elements from the original set from the given
    /// indices.
    ///
    /// # Example
    ///
    /// ```
    /// use utils::soap::*;
    /// let v = vec!['a', 'b', 'c'];
    /// let selection = Select::new(vec![1,2,1], v.as_slice());
    /// assert_eq!('b', selection[0]);
    /// assert_eq!('c', selection[1]);
    /// assert_eq!('b', selection[2]);
    /// ```
    pub fn new(indices: I, target: S) -> Self {
        Self::validate(Select { indices, target })
    }

    /// Panics if this selection has out of bounds indices.
    #[inline]
    fn validate(self) -> Self {
        if !self.indices.as_ref().iter().all(|&i| i < self.target.len()) {
            panic!("Select index out of bounds.");
        }
        self
    }
}

impl<'a, S, I> Select<S, I>
where
    S: Set + IntoOwned + Get<'a, usize>,
    <S as IntoOwned>::Owned: std::iter::FromIterator<<S as Set>::Elem>,
    <S as Get<'a, usize>>::Output: IntoOwned<Owned = <S as Set>::Elem>,
    I: AsRef<[usize]>,
{
    /// Collapse the target values pointed to by `indices` into the structure
    /// given by the indices. In other words, replace `indices` with the target
    /// data they point to producing a new collection. This function allocates.
    ///
    /// # Examples
    ///
    /// In the following simple example, we convert a selection of characters
    /// from a standard `Vec` into a standard owned `Vec` of characters.
    ///
    /// ```
    /// use utils::soap::*;
    /// let v = vec!['a', 'b', 'c'];
    /// let selection = Select::new(vec![1,2,1], v.as_slice());
    /// assert_eq!(vec!['b', 'c', 'b'], selection.collapse());
    /// ```
    ///
    /// A more complex example below shows how selections of chunked types can
    /// be collapsed as well.
    ///
    /// ```
    /// use utils::soap::*;
    /// // Start with a vector of words stored as Strings.
    /// let v = vec!["World", "Coffee", "Cat", " ", "Hello", "Refrigerator", "!"];
    ///
    /// // Convert the strings to bytes.
    /// let bytes: Vec<Vec<u8>> = v
    ///     .into_iter()
    ///     .map(|word| word.to_string().into_bytes())
    ///     .collect();
    ///
    /// // Chunk the nested vector at word boundaries.
    /// let words = Chunked::<Vec<u8>>::from_nested_vec(bytes);
    ///
    /// // Select some of the words from the collection.
    /// let selection = Select::new(vec![4, 3, 0, 6, 3, 4, 6], words);
    ///
    /// // Collapse the selected words into an owned collection.
    /// let collapsed = selection.view().collapse();
    ///
    /// assert_eq!(
    ///     "Hello World! Hello!",
    ///     String::from_utf8(collapsed.data().clone()).unwrap().as_str()
    /// );
    /// ```
    pub fn collapse(self) -> S::Owned {
        self.indices
            .as_ref()
            .iter()
            .map(|&i| self.target.at(i).into_owned())
            .collect()
    }
}

// Note to self:
// To enable a collection to be chunked, we need to implement:
// Set, View, SplitAt
// For mutability we also need ViewMut,
// For UniChunked we need:
// Set, Vew, ReinterpretSet (this needs to be refined)

// Required for `Chunked` and `UniChunked` selections.
impl<S: Set, I: AsRef<[usize]>> Set for Select<S, I> {
    type Elem = S::Elem;
    /// Get the number of selected elements.
    ///
    /// # Example
    ///
    /// ```
    /// use utils::soap::*;
    /// let v = vec![1,2,3,4,5];
    /// let selection = Select::new(vec![4,0,1,4], v.as_slice());
    /// assert_eq!(4, selection.len());
    /// ```
    fn len(&self) -> usize {
        self.indices.as_ref().len()
    }
}

// Required for `Chunked` and `UniChunked` selections.
impl<'a, S, I> View<'a> for Select<S, I>
where
    S: View<'a>,
    I: AsRef<[usize]>,
    //<S as View<'a>>::Type: Set,
    //Select<<S as View<'a>>::Type, &'a [usize]>: IntoIterator,
{
    type Type = Select<S::Type, &'a [usize]>;
    fn view(&'a self) -> Self::Type {
        Select {
            indices: self.indices.as_ref(),
            target: self.target.view(),
        }
    }
}

impl<'a, S, I> ViewMut<'a> for Select<S, I>
where
    S: Set + ViewMut<'a>,
    I: AsRef<[usize]>,
    //Select<<S as ViewMut<'a>>::Type, &'a [usize]>: IntoIterator,
{
    type Type = Select<S::Type, &'a [usize]>;
    /// Create a mutable view of this selection.
    ///
    /// # Example
    ///
    /// ```
    /// use utils::soap::*;
    /// let mut v = vec!['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'];
    /// let mut selection = Select::new(vec![1,2,4,1], v.as_mut_slice());
    ///
    /// {
    ///     let view = selection.view();
    ///     let mut iter = view.iter();
    ///     assert_eq!(Some((1, &'b')), iter.next());
    ///     assert_eq!(Some((2, &'c')), iter.next());
    ///     assert_eq!(Some((4, &'e')), iter.next());
    ///     assert_eq!(Some((1, &'b')), iter.next());
    ///     assert_eq!(None, iter.next());
    /// }
    ///
    /// // Change all referenced elements to 'a'.
    /// let mut view = selection.view_mut();
    /// for &i in view.indices.iter() {
    ///     view.target[i] = 'a';
    /// }
    ///
    /// let view = selection.view();
    /// let mut iter = view.iter();
    /// assert_eq!(Some((1, &'a')), iter.next());
    /// assert_eq!(Some((2, &'a')), iter.next());
    /// assert_eq!(Some((4, &'a')), iter.next());
    /// assert_eq!(Some((1, &'a')), iter.next());
    /// assert_eq!(None, iter.next());
    /// ```
    fn view_mut(&'a mut self) -> Self::Type {
        Select {
            indices: self.indices.as_ref(),
            target: self.target.view_mut(),
        }
    }
}

// This impl enables `Chunked` `Select`ions
impl<V, I> SplitAt for Select<V, I>
where
    V: Set + Clone,
    I: SplitAt,
{
    /// Split this selection into two at the given index `mid`.
    ///
    /// # Example
    ///
    /// ```
    /// use utils::soap::*;
    /// let v = vec![1,2,3,4,5];
    /// let indices = vec![3,2,0,4,2];
    /// let selection = Select::new(indices.as_slice(), v.as_slice());
    /// let (l, r) = selection.split_at(2);
    /// let mut iter_l = l.iter();
    /// assert_eq!(Some((3, &4)), iter_l.next());
    /// assert_eq!(Some((2, &3)), iter_l.next());
    /// assert_eq!(None, iter_l.next());
    /// let mut iter_r = r.iter();
    /// assert_eq!(Some((0, &1)), iter_r.next());
    /// assert_eq!(Some((4, &5)), iter_r.next());
    /// assert_eq!(Some((2, &3)), iter_r.next()); // Note that 3 is shared between l and r
    /// assert_eq!(None, iter_r.next());
    /// ```
    fn split_at(self, mid: usize) -> (Self, Self) {
        let Select { target, indices } = self;
        let (indices_l, indices_r) = indices.split_at(mid);
        (
            Select {
                indices: indices_l,
                target: target.clone(),
            },
            Select {
                indices: indices_r,
                target: target,
            },
        )
    }
}

impl<S, I: RemovePrefix> RemovePrefix for Select<S, I> {
    fn remove_prefix(&mut self, n: usize) {
        self.indices.remove_prefix(n);
    }
}

impl<'a, S, I> Select<S, I>
where
    S: Set + Get<'a, usize, Output = &'a <S as Set>::Elem> + View<'a>,
    I: AsRef<[usize]>,
    <S as View<'a>>::Type: IntoIterator<Item = S::Output>,
    <S as Set>::Elem: 'a,
{
    /// The typical way to use this function is to clone from a `SelectView`
    /// into a mutable `S` type. This function disregards indies, and simply
    /// clones the underlying target data.
    ///
    /// # Panics
    ///
    /// This function panics if `other` has a length unequal to `self.len()`.
    ///
    /// # Example
    ///
    /// ```
    /// use utils::soap::*;
    /// let v = vec![1,2,3,4,5];
    /// let indices = vec![3,3,4,0];
    /// let selection = Select::new(indices.as_slice(), v.as_slice());
    /// let mut owned = vec![0; 5];
    /// selection.clone_values_into(&mut owned[..4]); // Need 4 elements to avoid panics.
    /// let mut iter_owned = owned.iter();
    /// assert_eq!(owned, vec![4,4,5,1,0]);
    /// ```
    pub fn clone_values_into<V>(&'a self, other: &'a mut V)
    where
        V: ViewMut<'a> + ?Sized,
        <V as ViewMut<'a>>::Type: Set + IntoIterator<Item = &'a mut S::Elem>,
        <S as Set>::Elem: Clone,
    {
        let other_view = other.view_mut();
        assert_eq!(other_view.len(), self.len());
        for (theirs, mine) in other_view.into_iter().zip(self.iter()) {
            theirs.clone_from(&mine.1);
        }
    }
}

/*
 * Get API provides a way to access the index and its associated value for each
 * of the selected elements.
 */

impl<'a, S, I> GetIndex<'a, Select<S, I>> for usize
where
    I: AsRef<[usize]>,
    S: Get<'a, usize>,
{
    type Output = (usize, <S as Get<'a, usize>>::Output);

    fn get(self, selection: &Select<S, I>) -> Option<Self::Output> {
        selection
            .indices
            .as_ref()
            .get(self)
            .and_then(|&idx| selection.target.get(idx).map(|val| (idx, val)))
    }
}

impl<S, I> IsolateIndex<Select<S, I>> for usize
where
    I: Isolate<usize>,
    <I as Isolate<usize>>::Output: std::borrow::Borrow<usize>,
    S: Isolate<usize>,
{
    type Output = (I::Output, S::Output);

    fn try_isolate(self, selection: Select<S, I>) -> Option<Self::Output> {
        use std::borrow::Borrow;
        let Select { indices, target } = selection;
        indices
            .try_isolate(self)
            .and_then(move |idx| target.try_isolate(*idx.borrow()).map(|val| (idx, val)))
    }
}

/// Isolating a range from a selection will preserve the original target data set.
impl<S, I> IsolateIndex<Select<S, I>> for std::ops::Range<usize>
where
    I: Isolate<std::ops::Range<usize>>,
{
    type Output = Select<S, I::Output>;

    fn try_isolate(self, selection: Select<S, I>) -> Option<Self::Output> {
        let Select { indices, target } = selection;
        indices
            .try_isolate(self)
            .map(move |indices| Select { indices, target })
    }
}

impl_isolate_index_for_static_range!(impl<S, I> for Select<S, I>);

//impl<S, I, Idx> Isolate<Idx> for Select<S, I>
//where
//    Idx: IsolateIndex<Self>,
//{
//    type Output = Idx::Output;
//
//    /// Isolate an element or a range in this selection.
//    ///
//    /// # Panics
//    ///
//    /// This function panics if the index is out of bounds.
//    fn try_isolate(self, range: Idx) -> Option<Self::Output> {
//        range.try_isolate(self)
//    }
//}

/*
 * Indexing operators for convenience. Users familiar with indexing by `usize`
 * may find these implementations convenient. However, these do not have the
 * same function as `Get` provides, because they necessarily return a borrow of
 * some inner value. Since `Select`ions store target data and indices
 * separately, only one of them can be returned as a reference. As such, the
 * indexing operators only provide the select values without their respective
 * indices. To get indices and values, the `Get` and `GetMut` traits should be
 * used instead.
 */

impl<'a, S, I> std::ops::Index<usize> for Select<S, I>
where
    S: std::ops::Index<usize> + Set + ValueType,
    I: AsRef<[usize]>,
{
    type Output = S::Output;

    /// Immutably index the selection.
    ///
    /// # Panics
    ///
    /// This function panics if the index is out of bounds or if the selection is empty.
    ///
    /// # Example
    ///
    /// ```
    /// use utils::soap::*;
    /// let selection = Select::new(vec![0,2,0,4], Chunked2::from_flat(1..=12));
    /// assert_eq!((0, 1..3), selection.at(0));
    /// assert_eq!((2, 5..7), selection.at(1));
    /// assert_eq!((0, 1..3), selection.at(2));
    /// assert_eq!((4, 9..11), selection.at(3));
    /// ```
    fn index(&self, idx: usize) -> &Self::Output {
        self.target.index(self.indices.as_ref()[idx])
    }
}

impl<'a, S, I> std::ops::IndexMut<usize> for Select<S, I>
where
    S: std::ops::IndexMut<usize> + Set + ValueType,
    I: AsRef<[usize]>,
{
    /// Mutably index the selection.
    ///
    /// # Panics
    ///
    /// This function panics if the index is out of bounds or if the selection is empty.
    ///
    /// # Example
    ///
    /// ```
    /// use utils::soap::*;
    /// let mut v = vec![1,2,3,4,5];
    /// let mut selection = Select::new(vec![0,2,0,4], v.as_mut_slice());
    /// assert_eq!(selection[0], 1);
    /// assert_eq!(selection[1], 3);
    /// assert_eq!(selection[2], 1);
    /// assert_eq!(selection[3], 5);
    /// selection[2] = 100;
    /// assert_eq!(selection[0], 100);
    /// assert_eq!(selection[1], 3);
    /// assert_eq!(selection[2], 100);
    /// assert_eq!(selection[3], 5);
    /// ```
    fn index_mut(&mut self, idx: usize) -> &mut Self::Output {
        self.target.index_mut(self.indices.as_ref()[idx])
    }
}

impl<'a, T, I> std::ops::Index<usize> for Select<&'a [T], I>
where
    I: AsRef<[usize]>,
{
    type Output = T;
    /// Immutably index the selection of elements from a borrowed slice.
    ///
    /// # Panics
    ///
    /// This function panics if the index is out of bounds or if the selection is empty.
    ///
    /// # Example
    ///
    /// ```
    /// use utils::soap::*;
    /// let v = vec![1,2,3,4,5];
    /// let selection = Select::new(vec![0,2,0,4], v.as_slice());
    /// assert_eq!(3, selection[1]);
    /// assert_eq!(1, selection[2]);
    /// ```
    fn index(&self, idx: usize) -> &Self::Output {
        self.target.index(self.indices.as_ref()[idx])
    }
}

impl<'a, T, I> std::ops::Index<usize> for Select<&'a mut [T], I>
where
    I: AsRef<[usize]>,
{
    type Output = T;
    /// Immutably index a selection of elements from a mutably borrowed slice.
    ///
    /// # Panics
    ///
    /// This function panics if the index is out of bounds or if the selection is empty.
    ///
    /// # Example
    ///
    /// ```
    /// use utils::soap::*;
    /// let mut v = vec![1,2,3,4,5];
    /// let mut subset = Subset::from_indices(vec![3,2,0,4], v.as_mut_slice());
    /// assert_eq!(3, subset[1]);
    /// ```
    fn index(&self, idx: usize) -> &Self::Output {
        self.target.index(self.indices.as_ref()[idx])
    }
}

impl<'a, T, I> std::ops::IndexMut<usize> for Select<&'a mut [T], I>
where
    I: AsRef<[usize]>,
{
    /// Mutably index a selection of elements from a mutably borrowed slice.
    ///
    /// # Panics
    ///
    /// This function panics if the index is out of bounds or if the selection is empty.
    ///
    /// # Example
    ///
    /// ```
    /// use utils::soap::*;
    /// let mut v = vec![1,2,3,4,5];
    /// let mut selection = Select::new(vec![4,0,2,4], v.as_mut_slice());
    /// assert_eq!(selection[0], 5);
    /// selection[0] = 100;
    /// assert_eq!(selection[0], 100);
    /// assert_eq!(selection[1], 1);
    /// assert_eq!(selection[2], 3);
    /// assert_eq!(selection[3], 100);
    /// ```
    fn index_mut(&mut self, idx: usize) -> &mut Self::Output {
        self.target.index_mut(self.indices.as_ref()[idx])
    }
}

/*
 * Iteration
 */

impl<'a, S, I> Select<S, I>
where
    S: Set + Get<'a, usize> + View<'a>,
    I: AsRef<[usize]>,
    <S as View<'a>>::Type: IntoIterator<Item = S::Output>,
{
    pub fn iter(&'a self) -> impl Iterator<Item = (usize, <S as Get<'a, usize>>::Output)> + Clone {
        self.indices
            .as_ref()
            .iter()
            .cloned()
            .filter_map(move |idx| self.target.get(idx).map(|val| (idx, val)))
    }
}

impl<S, I> Select<S, I>
where
    I: AsRef<[usize]>,
{
    pub fn index_iter(&self) -> std::slice::Iter<'_, usize> {
        self.indices.as_ref().iter()
    }
}

impl<S, I> Select<S, I>
where
    I: AsMut<[usize]>,
{
    pub fn index_iter_mut(&mut self) -> std::slice::IterMut<'_, usize> {
        self.indices.as_mut().iter_mut()
    }
}

impl<S: Dummy, I: Dummy> Dummy for Select<S, I> {
    unsafe fn dummy() -> Self {
        Select {
            indices: Dummy::dummy(),
            target: Dummy::dummy(),
        }
    }
}

impl<S, I: Truncate> Truncate for Select<S, I> {
    fn truncate(&mut self, new_len: usize) {
        // The target data remains untouched.
        self.indices.truncate(new_len);
    }
}

// Clear selection
impl<S, I: Clear> Clear for Select<S, I> {
    fn clear(&mut self) {
        self.indices.clear();
    }
}

/*
 * Conversions
 */

/// Pass through the conversion for structure type `Select`.
impl<S: StorageInto<T>, I, T> StorageInto<T> for Select<S, I> {
    type Output = Select<S::Output, I>;
    fn storage_into(self) -> Self::Output {
        Select {
            target: self.target.storage_into(),
            indices: self.indices,
        }
    }
}

/*
 * Target data Access
 */

impl<S: Storage, I> Storage for Select<S, I> {
    type Storage = S::Storage;
    /// Return an immutable reference to the underlying storage type.
    ///
    /// # Example
    ///
    /// ```rust
    /// use utils::soap::*;
    /// let v = vec![1,2,3,4,5,6,7,8,9,10,11,12];
    /// let s0 = Chunked3::from_flat(v.clone());
    /// let s1 = Select::new(vec![1, 1, 0, 2], s0.clone());
    /// assert_eq!(s1.storage(), &v);
    /// ```
    fn storage(&self) -> &Self::Storage {
        self.target.storage()
    }
}

impl<S: StorageMut, I> StorageMut for Select<S, I> {
    /// Return a mutable reference to the underlying storage type.
    ///
    /// # Example
    ///
    /// ```rust
    /// use utils::soap::*;
    /// let mut v = vec![1,2,3,4,5,6,7,8,9,10,11,12];
    /// let mut s0 = Chunked3::from_flat(v.clone());
    /// let mut s1 = Select::new(vec![1, 1, 0, 2], s0.clone());
    /// assert_eq!(s1.storage_mut(), &mut v);
    /// ```
    fn storage_mut(&mut self) -> &mut Self::Storage {
        self.target.storage_mut()
    }
}

/*
 * Selections of Unichunked types
 */

impl<S: ChunkSize, I> ChunkSize for Select<S, I> {
    fn chunk_size(&self) -> usize {
        self.target.chunk_size()
    }
}

/*
 * Convert views to owned types
 */

impl<S: IntoOwned, I: IntoOwned> IntoOwned for Select<S, I> {
    type Owned = Select<S::Owned, I::Owned>;

    fn into_owned(self) -> Self::Owned {
        Select {
            indices: self.indices.into_owned(),
            target: self.target.into_owned(),
        }
    }
}

impl<S, I> IntoOwnedData for Select<S, I>
where
    S: IntoOwnedData,
{
    type OwnedData = Select<S::OwnedData, I>;
    fn into_owned_data(self) -> Self::OwnedData {
        Select {
            indices: self.indices,
            target: self.target.into_owned_data(),
        }
    }
}

/*
 * Impls for uniformly chunked sparse types
 */

impl<S, I, M> UniChunkable<M> for Select<S, I> {
    type Chunk = Select<S, I>;
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Test into_owned for selections.
    #[test]
    fn into_owned() {
        let indices = vec![1, 2, 3];
        let target = 0..4;
        let select = Select {
            indices: indices.as_slice(),
            target: target.clone(),
        };

        assert_eq!(select.into_owned(), Select { indices, target });
    }
}
