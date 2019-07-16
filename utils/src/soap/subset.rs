use super::*;

/// A Set that is a non-contiguous subset of some larger set (which could have any type).
/// `B` can be any borrowed collection type.
///
/// # Example
///
/// The following example shows how to create a `Subset` from a standard `Vec`.
///
/// ```rust
/// use utils::soap::*;
/// let v = vec![1,2,3,4,5];
/// let subset = Subset::from_indices(vec![0,2,4], v.as_slice());
/// let mut subset_iter = subset.iter();
/// assert_eq!(Some(&1), subset_iter.next());
/// assert_eq!(Some(&3), subset_iter.next());
/// assert_eq!(Some(&5), subset_iter.next());
/// assert_eq!(None, subset_iter.next());
/// ```
/////
///// The next example shows how to create a `Subset` from a [`UniSet`].
/////
///// ```rust
///// use utils::soap::*;
///// let mut v = UniSet::<_, num::U3>::from_flat(vec![1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]);
///// let mut subset = Subset::from_indices(vec![0,2,4], v.view_mut());
///// {
/////     let mut subset_iter = subset.iter();
/////     assert_eq!(Some(&[1,2,3]), subset_iter.next());
/////     assert_eq!(Some(&[7,8,9]), subset_iter.next());
/////     assert_eq!(Some(&[13,14,15]), subset_iter.next());
/////     assert_eq!(None, subset_iter.next());
///// }
///// subset[1] = [0; 3];
///// assert_eq!([0,0,0], subset[1]);
///// ```
#[derive(Clone, Debug)]
pub struct Subset<S> {
    indices: Vec<usize>,
    set: S,
}

impl<'a, S: Set> Subset<S> {
    /// Create a subset of elements from the original set given at the specified indices.
    ///
    /// # Example
    ///
    /// ```rust
    /// use utils::soap::*;
    /// let v = vec![1,2,3];
    /// let subset = Subset::from_indices(vec![0,2], v.as_slice());
    /// assert_eq!(1, subset[0]);
    /// assert_eq!(3, subset[1]);
    /// ```
    pub fn from_indices(mut indices: Vec<usize>, set: S) -> Self {
        // Ensure that indices are sorted and there are no duplicates.
        // Failure to enforce this invariant can cause race conditions.

        indices.sort_unstable();
        indices.dedup();

        Self::validate(Subset { indices, set })
    }

    /// Create a subset with all elements from the original set.
    ///
    /// # Example
    ///
    /// ```rust
    /// use utils::soap::*;
    /// let subset = Subset::all(vec![1,2,3]);
    /// let mut subset_iter = subset.iter();
    /// assert_eq!(Some(&1), subset_iter.next());
    /// assert_eq!(Some(&2), subset_iter.next());
    /// assert_eq!(Some(&3), subset_iter.next());
    /// assert_eq!(None, subset_iter.next());
    /// ```
    pub fn all(set: S) -> Self {
        Self::validate(Subset {
            indices: (0..set.len()).collect(),
            set,
        })
    }

    /// Return the superset of this `Subset`. This is just the set it was created with.
    pub fn into_super(self) -> S {
        self.set
    }

    /// Panics if this subset is invald.
    #[inline]
    fn validate(self) -> Self {
        for &i in self.indices.iter() {
            assert!(i < self.set.len(), "Subset index out of bounds.");
        }
        self
    }
}

/*
 * Indexing operators for convenience. Users familiar with indexing by `usize`
 * may find these implementations convenient.
 */

impl<'a, S: std::ops::Index<usize> + ?Sized> std::ops::Index<usize> for Subset<&'a S> {
    type Output = <S as std::ops::Index<usize>>::Output;
    /// Immutably index the subset.
    ///
    /// # Example
    ///
    /// ```rust
    /// use utils::soap::*;
    /// let mut v = vec![1,2,3,4,5];
    /// let mut subset = Subset::from_indices(vec![0,2,4], &v);
    /// assert_eq!(3, subset[1]);
    /// ```
    fn index(&self, idx: usize) -> &Self::Output {
        self.set.index(self.indices[idx])
    }
}

impl<'a, S: std::ops::Index<usize> + ?Sized> std::ops::Index<usize> for Subset<&'a mut S> {
    type Output = <S as std::ops::Index<usize>>::Output;
    /// Immutably index the subset.
    ///
    /// # Example
    ///
    /// ```rust
    /// use utils::soap::*;
    /// let mut v = vec![1,2,3,4,5];
    /// let mut subset = Subset::from_indices(vec![0,2,4], v.as_mut_slice());
    /// assert_eq!(3, subset[1]);
    /// ```
    fn index(&self, idx: usize) -> &Self::Output {
        self.set.index(self.indices[idx])
    }
}

impl<'a, S: std::ops::IndexMut<usize> + ?Sized> std::ops::IndexMut<usize> for Subset<&'a mut S> {
    /// Mutably index the subset.
    ///
    /// # Example
    ///
    /// ```rust
    /// use utils::soap::*;
    /// let mut v = vec![1,2,3,4,5];
    /// let mut subset = Subset::from_indices(vec![0,2,4], v.as_mut_slice());
    /// assert_eq!(subset[1], 3);
    /// subset[1] = 100;
    /// assert_eq!(subset[0], 1);
    /// assert_eq!(subset[1], 100);
    /// assert_eq!(subset[2], 5);
    /// ```
    fn index_mut(&mut self, idx: usize) -> &mut Self::Output {
        self.set.index_mut(self.indices[idx])
    }
}

//impl<S, N> std::ops::Index<usize> for Subset<UniSet<S, N>> {
//    type Output = <UniSet<S, N> as std::ops::Index<usize>>::Output;
//    /// Immutably index the subset.
//    ///
//    /// # Example
//    ///
//    /// ```rust
//    /// use utils::soap::*;
//    /// let mut v = UniSet::<_, num::U3>::from_flat(vec![1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]);
//    /// let subset = Subset::from_indices(vec![0,2,4], v.view());
//    /// assert_eq!([7,8,9], subset[1]);
//    /// ```
//    fn index(&self, idx: usize) -> &Self::Output {
//        self.set.index(self.indices[idx])
//    }
//}

/*
 * Iteration
 */

impl<S> Subset<S> {
    pub fn iter<'a, T: 'a>(&'a self) -> impl Iterator<Item = &'a T>
    where
        S: std::borrow::Borrow<[T]>,
    {
        let ptr = self.set.borrow().as_ptr();
        self.indices.iter().map(move |&i| unsafe { &*ptr.add(i) })
    }
}

impl<S> Subset<S> {
    /// Mutably iterate over a subset.
    ///
    /// # Example
    ///
    /// ```rust
    /// use utils::soap::*;
    /// let mut v = vec![1,2,3,4,5];
    /// let mut subset = Subset::from_indices(vec![0,2,4], v.as_mut_slice());
    /// for i in subset.iter_mut() {
    ///     *i += 1;
    /// }
    /// assert_eq!(v, vec![2,2,4,4,6]);
    /// ```
    pub fn iter_mut<'a, T: 'a>(&'a mut self) -> impl Iterator<Item = &'a mut T>
    where
        S: std::borrow::BorrowMut<[T]>,
    {
        let ptr = self.set.borrow_mut().as_mut_ptr();
        self.indices
            .iter()
            .map(move |&i| unsafe { &mut *ptr.add(i) })
    }
}
