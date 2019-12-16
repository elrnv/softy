mod lazy;

use super::*;
pub use lazy::*;
use num_traits::{Float, One, Zero};
use std::num::FpCategory;
use std::ops::{
    Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Rem, RemAssign, Sub, SubAssign,
};
use unroll::unroll_for_loops;

/// A generic type that accepts algebraic expressions.
///
/// The type parameter `I` determines the indexing structure of the tensor.
/// For instance for unique types `I0` and `I1`, the type `I == (I0, I1)` represents a matrix with
/// outer index `I0` and inner index `I1`. This means that a transpose can be implemented simply by
/// swapping positions of `I0` and `I1`, which means a matrix with `I == (I1, I0)` has structure
/// that is transpose of the matix with `I = (I0, I1)`.
#[derive(Copy, Clone, Debug, PartialOrd)]
#[repr(transparent)]
pub struct Tensor<T: ?Sized> {
    pub data: T,
}

impl<U: ?Sized, T: ?Sized + PartialEq<U>> PartialEq<Tensor<U>> for Tensor<T> {
    #[inline]
    fn eq(&self, other: &Tensor<U>) -> bool {
        self.data.eq(&other.data)
    }
}

// The following type aliases are useful for disambiguating between the `new` constructors without
// having to spell out the entire type.
pub type UTensor<S, N> = Tensor<UniChunked<S, N>>;
pub type UTensorN<S> = Tensor<ChunkedN<S>>;
pub type UTensor1<S> = Tensor<Chunked1<S>>;
pub type UTensor2<S> = Tensor<Chunked2<S>>;
pub type UTensor3<S> = Tensor<Chunked3<S>>;
pub type UTensor4<S> = Tensor<Chunked4<S>>;
pub type UTensor5<S> = Tensor<Chunked5<S>>;
pub type UTensor6<S> = Tensor<Chunked6<S>>;
pub type UTensor7<S> = Tensor<Chunked7<S>>;
pub type UTensor8<S> = Tensor<Chunked8<S>>;
pub type UTensor9<S> = Tensor<Chunked9<S>>;
pub type JaggedTensor<S, I> = Tensor<Chunked<S, I>>;
pub type SparseTensor<S, T, I> = Tensor<Sparse<S, T, I>>;
pub type SubsetTensor<S, I> = Tensor<Subset<S, I>>;
pub type SelectTensor<S, I> = Tensor<Select<S, I>>;
pub type Vector<T> = Tensor<Vec<T>>;

impl<S, N> Tensor<UniChunked<S, N>> {
    #[inline]
    pub fn new<D>(data: UniChunked<D, N>) -> Self
    where
        D: IntoTensor<Tensor = S>,
        S: IntoData<Data = D>,
    {
        data.into_tensor()
    }
}

impl<S, O> Tensor<Chunked<S, O>> {
    #[inline]
    pub fn new<D>(data: Chunked<D, O>) -> Self
    where
        D: IntoTensor<Tensor = S>,
        S: IntoData<Data = D>,
    {
        data.into_tensor()
    }
}

impl<S, T, I> Tensor<Sparse<S, T, I>> {
    #[inline]
    pub fn new<D>(data: Sparse<D, T, I>) -> Self
    where
        D: IntoTensor<Tensor = S>,
        S: IntoData<Data = D>,
    {
        data.into_tensor()
    }
}

impl<S, I> Tensor<Subset<S, I>> {
    #[inline]
    pub fn new<D>(data: Subset<D, I>) -> Self
    where
        D: IntoTensor<Tensor = S>,
        S: IntoData<Data = D>,
    {
        data.into_tensor()
    }
}

impl<S, I> Tensor<Select<S, I>> {
    #[inline]
    pub fn new<D>(data: Select<D, I>) -> Self
    where
        D: IntoTensor<Tensor = S>,
        S: IntoData<Data = D>,
    {
        data.into_tensor()
    }
}

impl<T> Tensor<Vec<T>> {
    #[inline]
    pub fn new(data: Vec<T>) -> Self {
        Tensor { data }
    }
}

/// Synonymous with `AsRef<Tensor<_>>`.
pub trait AsTensor {
    type Inner: ?Sized;
    fn as_tensor(&self) -> &Tensor<Self::Inner>;
}

pub trait AsMutTensor: AsTensor {
    fn as_mut_tensor(&mut self) -> &mut Tensor<Self::Inner>;
}

macro_rules! impl_as_tensor {
    //(impl<$($type_vars:tt)*> for $type:ty { type Inner = $inner:ty }) => {
    //    impl_as_tensor!(impl<$($type_vars),*> for $type where { type Inner = $inner });
    //};
    (impl ($($type_vars:tt)*) for $type:ty { type Inner = $inner:ty }) => {
        impl <$($type_vars)*> AsTensor for $type {
            type Inner = $inner;
            #[inline]
            fn as_tensor(&self) -> &Tensor<Self::Inner> {
                unsafe { Tensor::as_ref(self) }
            }
        }

        impl <$($type_vars)*> AsMutTensor for $type {
            #[inline]
            fn as_mut_tensor(&mut self) -> &mut Tensor<Self::Inner> {
                unsafe { Tensor::as_mut(self) }
            }
        }
    }
}
impl_as_tensor!(impl (T) for [T] { type Inner = [T] });
impl_as_tensor!(impl (S: IntoTensor, N) for UniChunked<S, N> {
    type Inner = UniChunked<S::Tensor, N>
});
impl_as_tensor!(impl (S: IntoTensor, I) for Chunked<S, I> {
    type Inner = Chunked<S::Tensor, I>
});
impl_as_tensor!(impl (S: IntoTensor, T, I) for Sparse<S, T, I> {
    type Inner = Sparse<S::Tensor, T, I>
});
impl_as_tensor!(impl (S: IntoTensor, I) for Subset<S, I> {
    type Inner = Subset<S::Tensor, I>
});
impl_as_tensor!(impl (S: IntoTensor, I) for Select<S, I> {
    type Inner = Select<S::Tensor, I>
});

/// Trait intended to strip away all Tensor wrappers in a given reference type.
pub trait AsData {
    type Data: ?Sized;
    fn as_data(&self) -> &Self::Data;
}

/// Trait intended to strip away all Tensor wrappers in a given mutable reference type.
pub trait AsMutData: AsData {
    fn as_mut_data(&mut self) -> &mut Self::Data;
}

impl<T: AsData + ?Sized> AsData for Tensor<T> {
    type Data = T::Data;
    #[inline]
    fn as_data(&self) -> &Self::Data {
        self.data.as_data()
    }
}

impl<T: AsMutData + ?Sized> AsMutData for Tensor<T> {
    #[inline]
    fn as_mut_data(&mut self) -> &mut Self::Data {
        self.data.as_mut_data()
    }
}

impl<S: IntoData, N> AsData for UniChunked<S, N> {
    type Data = UniChunked<S::Data, N>;
    #[inline]
    fn as_data(&self) -> &Self::Data {
        unsafe { &*(self as *const UniChunked<S, N> as *const UniChunked<S::Data, N>) }
    }
}

impl<S: IntoData, N> AsMutData for UniChunked<S, N> {
    #[inline]
    fn as_mut_data(&mut self) -> &mut Self::Data {
        unsafe { &mut *(self as *mut UniChunked<S, N> as *mut UniChunked<S::Data, N>) }
    }
}

impl<S: IntoData, I> AsData for Chunked<S, I> {
    type Data = Chunked<S::Data, I>;
    #[inline]
    fn as_data(&self) -> &Self::Data {
        unsafe { &*(self as *const Chunked<S, I> as *const Chunked<S::Data, I>) }
    }
}

impl<S: IntoData, I> AsMutData for Chunked<S, I> {
    #[inline]
    fn as_mut_data(&mut self) -> &mut Self::Data {
        unsafe { &mut *(self as *mut Chunked<S, I> as *mut Chunked<S::Data, I>) }
    }
}

impl<S: IntoData, T, I> AsData for Sparse<S, T, I> {
    type Data = Sparse<S::Data, T, I>;
    #[inline]
    fn as_data(&self) -> &Self::Data {
        unsafe { &*(self as *const Sparse<S, T, I> as *const Sparse<S::Data, T, I>) }
    }
}

impl<S: IntoData, T, I> AsMutData for Sparse<S, T, I> {
    #[inline]
    fn as_mut_data(&mut self) -> &mut Self::Data {
        unsafe { &mut *(self as *mut Sparse<S, T, I> as *mut Sparse<S::Data, T, I>) }
    }
}

impl<S: IntoData, I> AsData for Subset<S, I> {
    type Data = Subset<S::Data, I>;
    #[inline]
    fn as_data(&self) -> &Self::Data {
        unsafe { &*(self as *const Subset<S, I> as *const Subset<S::Data, I>) }
    }
}

impl<S: IntoData, I> AsMutData for Subset<S, I> {
    #[inline]
    fn as_mut_data(&mut self) -> &mut Self::Data {
        unsafe { &mut *(self as *mut Subset<S, I> as *mut Subset<S::Data, I>) }
    }
}

impl<S: IntoData, I> AsData for Select<S, I> {
    type Data = Select<S::Data, I>;
    #[inline]
    fn as_data(&self) -> &Self::Data {
        unsafe { &*(self as *const Select<S, I> as *const Select<S::Data, I>) }
    }
}

impl<S: IntoData, I> AsMutData for Select<S, I> {
    #[inline]
    fn as_mut_data(&mut self) -> &mut Self::Data {
        unsafe { &mut *(self as *mut Select<S, I> as *mut Select<S::Data, I>) }
    }
}

impl<T> AsData for [T] {
    type Data = [T];
    #[inline]
    fn as_data(&self) -> &Self::Data {
        self
    }
}

impl<T> AsMutData for [T] {
    #[inline]
    fn as_mut_data(&mut self) -> &mut Self::Data {
        self
    }
}

impl<T> AsData for Vec<T> {
    type Data = Vec<T>;
    #[inline]
    fn as_data(&self) -> &Self::Data {
        self
    }
}

impl<T> AsMutData for Vec<T> {
    #[inline]
    fn as_mut_data(&mut self) -> &mut Self::Data {
        self
    }
}

pub trait IntoData {
    type Data;
    fn into_data(self) -> Self::Data;
}

impl<S: NonTensor> IntoData for S {
    type Data = Self;
    fn into_data(self) -> Self::Data {
        self
    }
}

impl<S: IntoData, N> IntoData for Tensor<UniChunked<S, N>> {
    type Data = UniChunked<S::Data, N>;
    /// Converts this tensor into its raw data representation.
    #[inline]
    fn into_data(self) -> Self::Data {
        UniChunked {
            chunk_size: self.data.chunk_size,
            data: self.data.data.into_data(),
        }
    }
}

impl<S: IntoData, O> IntoData for Tensor<Chunked<S, O>> {
    type Data = Chunked<S::Data, O>;
    /// Converts this tensor into its raw data representation.
    #[inline]
    fn into_data(self) -> Self::Data {
        Chunked {
            chunks: self.data.chunks,
            data: self.data.data.into_data(),
        }
    }
}

impl<S: IntoData, T, I> IntoData for Tensor<Sparse<S, T, I>> {
    type Data = Sparse<S::Data, T, I>;
    /// Converts this tensor into its raw data representation.
    #[inline]
    fn into_data(self) -> Self::Data {
        Sparse {
            selection: self.data.selection,
            source: self.data.source.into_data(),
        }
    }
}

impl<S: IntoData, I> IntoData for Tensor<Subset<S, I>> {
    type Data = Subset<S::Data, I>;
    /// Converts this tensor into its raw data representation.
    #[inline]
    fn into_data(self) -> Self::Data {
        Subset {
            indices: self.data.indices,
            data: self.data.data.into_data(),
        }
    }
}

impl<S: IntoData, I> IntoData for Tensor<Select<S, I>> {
    type Data = Select<S::Data, I>;
    /// Converts this tensor into its raw data representation.
    #[inline]
    fn into_data(self) -> Self::Data {
        Select {
            indices: self.data.indices,
            target: self.data.target.into_data(),
        }
    }
}

impl<T: Flat> IntoData for Tensor<T> {
    type Data = T;
    /// Converts this tensor into its raw data representation.
    #[inline]
    fn into_data(self) -> Self::Data {
        self.data
    }
}

impl<'a, T: ?Sized> IntoData for &'a Tensor<T> {
    type Data = &'a T;
    /// Converts this tensor into its raw data representation.
    #[inline]
    fn into_data(self) -> Self::Data {
        &self.data
    }
}

impl<'a, T: ?Sized> IntoData for &'a mut Tensor<T> {
    type Data = &'a mut T;
    /// Converts this tensor into its raw data representation.
    #[inline]
    fn into_data(self) -> Self::Data {
        &mut self.data
    }
}

pub trait IntoTensor
where
    Self: Sized,
{
    type Tensor; //: IntoData<Data = Self>;
    fn into_tensor(self) -> Self::Tensor;
}

impl<S: IntoTensor, N> IntoTensor for UniChunked<S, N> {
    type Tensor = Tensor<UniChunked<S::Tensor, N>>;
    /// Converts this collection into its tensor representation.
    #[inline]
    fn into_tensor(self) -> Self::Tensor {
        Tensor {
            data: UniChunked {
                chunk_size: self.chunk_size,
                data: self.data.into_tensor(),
            },
        }
    }
}

impl<S: IntoTensor, O> IntoTensor for Chunked<S, O> {
    type Tensor = Tensor<Chunked<S::Tensor, O>>;
    #[inline]
    fn into_tensor(self) -> Self::Tensor {
        Tensor {
            data: Chunked {
                chunks: self.chunks,
                data: self.data.into_tensor(),
            },
        }
    }
}

impl<S: IntoTensor, T, I> IntoTensor for Sparse<S, T, I> {
    type Tensor = Tensor<Sparse<S::Tensor, T, I>>;
    #[inline]
    fn into_tensor(self) -> Self::Tensor {
        Tensor {
            data: Sparse {
                selection: self.selection,
                source: self.source.into_tensor(),
            },
        }
    }
}

impl<S: IntoTensor, I> IntoTensor for Subset<S, I> {
    type Tensor = Tensor<Subset<S::Tensor, I>>;
    #[inline]
    fn into_tensor(self) -> Self::Tensor {
        Tensor {
            data: Subset {
                indices: self.indices,
                data: self.data.into_tensor(),
            },
        }
    }
}

impl<S: IntoTensor, I> IntoTensor for Select<S, I> {
    type Tensor = Tensor<Select<S::Tensor, I>>;
    #[inline]
    fn into_tensor(self) -> Self::Tensor {
        Tensor {
            data: Select {
                indices: self.indices,
                target: self.target.into_tensor(),
            },
        }
    }
}

impl<T> IntoTensor for Vec<T> {
    type Tensor = Tensor<Vec<T>>;
    #[inline]
    fn into_tensor(self) -> Self::Tensor {
        Tensor { data: self }
    }
}

impl<'a, T: ?Sized + NonTensor> IntoTensor for &'a T {
    type Tensor = &'a Tensor<T>;
    #[inline]
    fn into_tensor(self) -> Self::Tensor {
        unsafe { Tensor::as_ref(self) }
    }
}

impl<'a, T: ?Sized + NonTensor> IntoTensor for &'a mut T {
    type Tensor = &'a mut Tensor<T>;
    #[inline]
    fn into_tensor(self) -> Self::Tensor {
        unsafe { Tensor::as_mut(self) }
    }
}

impl<S: IntoTensor, N> From<UniChunked<S, N>> for Tensor<UniChunked<S::Tensor, N>> {
    fn from(data: UniChunked<S, N>) -> Self {
        data.into_tensor()
    }
}
impl<S: IntoTensor, O> From<Chunked<S, O>> for Tensor<Chunked<S::Tensor, O>> {
    fn from(data: Chunked<S, O>) -> Self {
        data.into_tensor()
    }
}
impl<S: IntoTensor, T, I> From<Sparse<S, T, I>> for Tensor<Sparse<S::Tensor, T, I>> {
    fn from(data: Sparse<S, T, I>) -> Self {
        data.into_tensor()
    }
}
impl<S: IntoTensor, I> From<Subset<S, I>> for Tensor<Subset<S::Tensor, I>> {
    fn from(data: Subset<S, I>) -> Self {
        data.into_tensor()
    }
}
impl<S: IntoTensor, I> From<Select<S, I>> for Tensor<Select<S::Tensor, I>> {
    fn from(data: Select<S, I>) -> Self {
        data.into_tensor()
    }
}
impl<T> From<Vec<T>> for Tensor<Vec<T>> {
    fn from(data: Vec<T>) -> Self {
        data.into_tensor()
    }
}
impl<'a, T: 'a> From<&'a [T]> for &'a Tensor<[T]> {
    fn from(data: &'a [T]) -> Self {
        data.into_tensor()
    }
}
impl<'a, T: 'a> From<&'a mut [T]> for &'a mut Tensor<[T]> {
    fn from(data: &'a mut [T]) -> Self {
        data.into_tensor()
    }
}

impl<S, I> From<Tensor<S>> for Tensor<Subset<Tensor<S>, I>> {
    fn from(tensor: Tensor<S>) -> Tensor<Subset<Tensor<S>, I>> {
        Tensor {
            data: Subset::all(tensor),
        }
    }
}

impl<S: IntoData, N> From<Tensor<UniChunked<S, N>>> for UniChunked<S::Data, N> {
    fn from(data: Tensor<UniChunked<S, N>>) -> Self {
        data.into_data()
    }
}
impl<S: IntoData, O> From<Tensor<Chunked<S, O>>> for Chunked<S::Data, O> {
    fn from(data: Tensor<Chunked<S, O>>) -> Self {
        data.into_data()
    }
}
impl<S: IntoData, T, I> From<Tensor<Sparse<S, T, I>>> for Sparse<S::Data, T, I> {
    fn from(data: Tensor<Sparse<S, T, I>>) -> Self {
        data.into_data()
    }
}
impl<S: IntoData, I> From<Tensor<Subset<S, I>>> for Subset<S::Data, I> {
    fn from(data: Tensor<Subset<S, I>>) -> Self {
        data.into_data()
    }
}
impl<S: IntoData, I> From<Tensor<Select<S, I>>> for Select<S::Data, I> {
    fn from(data: Tensor<Select<S, I>>) -> Self {
        data.into_data()
    }
}

impl<T> IntoExpr for Tensor<T> {
    type Expr = Self;
    #[inline]
    fn into_expr(self) -> Self::Expr {
        self
    }
}

//impl<T> AsRef<T> for Tensor<T> {
//    fn as_ref(&self) -> &T {
//        &self.data
//    }
//}
//
//impl<T> AsMut<T> for Tensor<T> {
//    fn as_mut(&mut self) -> &mut T {
//        &mut self.data
//    }
//}

//impl<'a, T: ?Sized> From<&'a mut T> for &'a mut Tensor<T> {
//    fn from(t: &'a mut T) -> &'a mut Tensor<T> {
//        t.as_mut_tensor()
//    }
//}
//
//impl<'a, T: ?Sized> From<&'a T> for &'a Tensor<T> {
//    fn from(t: &'a T) -> &'a Tensor<T> {
//        t.as_tensor()
//    }
//}

impl<T: Default> Default for Tensor<T> {
    fn default() -> Self {
        Tensor {
            data: Default::default(),
        }
    }
}

impl<T: ?Sized> Tensor<T> {
    /// Create a reference to the given type as a `Tensor`.
    ///
    /// It is only valid to convert `T` to another composite type `U` that is the same as `T` but
    /// is allowed to contain Tensor wrappers.
    pub(crate) unsafe fn as_ref<U: ?Sized>(c: &T) -> &Tensor<U> {
        debug_assert_eq!(std::mem::size_of::<&Tensor<U>>(), std::mem::size_of::<&T>(),);
        std::mem::transmute_copy(&c)
    }

    /// Same as `as_ref` but creates a mutable reference to the given type as a `Tensor`.
    ///
    /// It is only valid to convert `T` to another composite type `U` that is the same as `T` but
    /// is allowed to contain Tensor wrappers.
    pub(crate) unsafe fn as_mut<U: ?Sized>(c: &mut T) -> &mut Tensor<U> {
        debug_assert_eq!(
            std::mem::size_of::<&mut Tensor<U>>(),
            std::mem::size_of::<&mut T>(),
        );
        std::mem::transmute_copy(&c)
    }
}

impl<T> Tensor<T> {
    pub(crate) unsafe fn reinterpret<U>(c: T) -> Tensor<U> {
        debug_assert_eq!(std::mem::size_of::<Tensor<U>>(), std::mem::size_of::<T>(),);
        std::mem::transmute_copy(&c)
    }
}

impl<S: ?Sized> Tensor<S> {
    /// Negate all elements in this tensor. This works on any tensor whose
    /// underlying elements are copyable negateable types.
    ///
    /// # Example
    ///
    /// ```
    /// use utils::soap::*;
    /// let mut v = vec![1.0, -2.0, 3.0];
    /// v.as_mut_tensor().negate();
    /// assert_eq!(v, vec![-1.0, 2.0, -3.0]);
    /// ```
    pub fn negate<'a>(&'a mut self)
    where
        S: Set + StorageMut,
        <S as Storage>::Storage: ViewMutIterator<'a, Item = &'a mut S::Atom>,
        <S as Set>::Atom: std::ops::Neg<Output = S::Atom> + Copy,
    {
        for v in self.storage_mut().view_mut_iter() {
            *v = -*v;
        }
    }
}

/*
 * Tensor as a Set
 * Forward soap impls to T in Tensor<T>.
 */

impl<T: Set + ?Sized> Set for Tensor<T> {
    type Elem = T::Elem;
    type Atom = T::Atom;
    fn len(&self) -> usize {
        self.data.len()
    }
}

impl<'a, T: StorageView<'a> + ?Sized> StorageView<'a> for Tensor<T> {
    type StorageView = T::StorageView;
    fn storage_view(&'a self) -> T::StorageView {
        self.data.storage_view()
    }
}

impl<T: Storage + ?Sized> Storage for Tensor<T> {
    type Storage = T::Storage;
    fn storage(&self) -> &T::Storage {
        self.data.storage()
    }
}

impl<T: StorageMut + ?Sized> StorageMut for Tensor<T> {
    fn storage_mut(&mut self) -> &mut T::Storage {
        self.data.storage_mut()
    }
}

impl<T, S: Push<T>> Push<T> for Tensor<S> {
    fn push(&mut self, element: T) {
        self.data.push(element);
    }
}

impl<S: ExtendFromSlice> ExtendFromSlice for Tensor<S> {
    type Item = S::Item;
    fn extend_from_slice(&mut self, other: &[Self::Item]) {
        self.data.extend_from_slice(other);
    }
}

impl<S: Truncate> Truncate for Tensor<S> {
    fn truncate(&mut self, len: usize) {
        self.data.truncate(len);
    }
}
impl<S: Clear> Clear for Tensor<S> {
    fn clear(&mut self) {
        self.data.clear();
    }
}
impl<S: IntoFlat> IntoFlat for Tensor<S> {
    type FlatType = Tensor<S::FlatType>;
    fn into_flat(self) -> Self::FlatType {
        Tensor {
            data: self.data.into_flat(),
        }
    }
}
impl<T, S: StorageInto<T>> StorageInto<T> for Tensor<S> {
    type Output = Tensor<S::Output>;
    fn storage_into(self) -> Self::Output {
        Tensor {
            data: self.data.storage_into(),
        }
    }
}
impl<S, T: CloneWithStorage<S>> CloneWithStorage<S> for Tensor<T> {
    type CloneType = Tensor<T::CloneType>;
    fn clone_with_storage(&self, storage: S) -> Self::CloneType {
        Tensor {
            data: self.data.clone_with_storage(storage),
        }
    }
}
impl<S: IntoOwned> IntoOwned for Tensor<S> {
    type Owned = Tensor<S::Owned>;
    fn into_owned(self) -> Self::Owned {
        Tensor {
            data: self.data.into_owned(),
        }
    }
}

impl<S: IntoOwnedData> IntoOwnedData for Tensor<S> {
    type OwnedData = Tensor<S::OwnedData>;
    fn into_owned_data(self) -> Self::OwnedData {
        Tensor {
            data: self.data.into_owned_data(),
        }
    }
}

impl<'a, S> GetIndex<'a, Tensor<S>> for usize
where
    usize: GetIndex<'a, S>,
{
    type Output = Tensor<<usize as GetIndex<'a, S>>::Output>;
    fn get(self, tensor: &Tensor<S>) -> Option<Self::Output> {
        tensor.data.get(self).map(|data| Tensor { data })
    }
}

impl<'a, S> GetIndex<'a, Tensor<S>> for std::ops::Range<usize>
where
    std::ops::Range<usize>: GetIndex<'a, S>,
{
    type Output = Tensor<<std::ops::Range<usize> as GetIndex<'a, S>>::Output>;
    fn get(self, tensor: &Tensor<S>) -> Option<Self::Output> {
        tensor.data.get(self).map(|data| Tensor { data })
    }
}

impl<S> IsolateIndex<Tensor<S>> for usize
where
    usize: IsolateIndex<S>,
{
    type Output = Tensor<<usize as IsolateIndex<S>>::Output>;
    fn try_isolate(self, tensor: Tensor<S>) -> Option<Self::Output> {
        tensor.data.try_isolate(self).map(|data| Tensor { data })
    }
}

impl<S> IsolateIndex<Tensor<S>> for std::ops::Range<usize>
where
    std::ops::Range<usize>: IsolateIndex<S>,
{
    type Output = Tensor<<std::ops::Range<usize> as IsolateIndex<S>>::Output>;
    fn try_isolate(self, tensor: Tensor<S>) -> Option<Self::Output> {
        tensor.data.try_isolate(self).map(|data| Tensor { data })
    }
}

impl<S: SplitAt> SplitAt for Tensor<S> {
    fn split_at(self, mid: usize) -> (Self, Self) {
        let (a, b) = self.data.split_at(mid);
        (Tensor { data: a }, Tensor { data: b })
    }
}

impl<S: SplitOff> SplitOff for Tensor<S> {
    fn split_off(&mut self, mid: usize) -> Self {
        Tensor {
            data: self.data.split_off(mid),
        }
    }
}

impl<N, S: SplitPrefix<N>> SplitPrefix<N> for Tensor<S> {
    type Prefix = Tensor<S::Prefix>;

    fn split_prefix(self) -> Option<(Self::Prefix, Self)> {
        self.data
            .split_prefix()
            .map(|(prefix, rest)| (Tensor { data: prefix }, Tensor { data: rest }))
    }
}

impl<S: SplitFirst> SplitFirst for Tensor<S> {
    type First = Tensor<S::First>;

    fn split_first(self) -> Option<(Self::First, Self)> {
        self.data
            .split_first()
            .map(|(first, rest)| (Tensor { data: first }, Tensor { data: rest }))
    }
}

impl<S: Dummy> Dummy for Tensor<S> {
    unsafe fn dummy() -> Self {
        Tensor {
            data: Dummy::dummy(),
        }
    }
}

impl<S: RemovePrefix> RemovePrefix for Tensor<S> {
    fn remove_prefix(&mut self, n: usize) {
        self.data.remove_prefix(n);
    }
}

impl<S: IntoParChunkIterator> IntoParChunkIterator for Tensor<S> {
    type Item = S::Item;
    type IterType = S::IterType;

    fn into_par_chunk_iter(self, chunk_size: usize) -> Self::IterType {
        self.data.into_par_chunk_iter(chunk_size)
    }
}

impl<S: UniChunkable<N> + ?Sized, N> UniChunkable<N> for Tensor<S> {
    type Chunk = Tensor<S::Chunk>;
}

impl<'a, S: UniChunkable<N> + ?Sized, N> UniChunkable<N> for &'a Tensor<S>
where
    S::Chunk: 'a,
{
    type Chunk = &'a Tensor<S::Chunk>;
}

impl<'a, S: UniChunkable<N> + ?Sized, N> UniChunkable<N> for &'a mut Tensor<S>
where
    S::Chunk: 'a,
{
    type Chunk = &'a mut Tensor<S::Chunk>;
}

impl<N: Unsigned, S> IntoStaticChunkIterator<N> for Tensor<S>
where
    S: Sized + Set + Dummy + IntoStaticChunkIterator<N>,
{
    type Item = S::Item;
    type IterType = S::IterType;

    fn into_static_chunk_iter(self) -> Self::IterType {
        self.data.into_static_chunk_iter()
    }
}

impl<S: Reserve> Reserve for Tensor<S> {
    fn reserve(&mut self, n: usize) {
        self.data.reserve(n);
    }
    fn reserve_with_storage(&mut self, n: usize, storage_n: usize) {
        self.data.reserve_with_storage(n, storage_n);
    }
}

impl<S: SwapChunks> SwapChunks for Tensor<S> {
    fn swap_chunks(&mut self, begin_a: usize, begin_b: usize, chunk_size: usize) {
        self.data.swap_chunks(begin_a, begin_b, chunk_size);
    }
}

impl<S: Sort> Sort for Tensor<S> {
    fn sort_indices(&self, indices: &mut [usize]) {
        self.data.sort_indices(indices);
    }
}

impl<S: PermuteInPlace> PermuteInPlace for Tensor<S> {
    fn permute_in_place(&mut self, indices: &[usize], seen: &mut [bool]) {
        self.data.permute_in_place(indices, seen);
    }
}

impl<S: ChunkSize> ChunkSize for Tensor<S> {
    fn chunk_size(&self) -> usize {
        self.data.chunk_size()
    }
}

impl<T: ?Sized, S: CloneIntoOther<T>> CloneIntoOther<T> for Tensor<S> {
    fn clone_into_other(&self, other: &mut T) {
        self.data.clone_into_other(other);
    }
}

impl<'a, S: AtomIterator<'a>> AtomIterator<'a> for Tensor<S> {
    type Item = S::Item;
    type Iter = S::Iter;
    fn atom_iter(&'a self) -> Self::Iter {
        S::atom_iter(&self.data)
    }
}

impl<'a, S: AtomMutIterator<'a>> AtomMutIterator<'a> for Tensor<S> {
    type Item = S::Item;
    type Iter = S::Iter;
    fn atom_mut_iter(&'a mut self) -> Self::Iter {
        S::atom_mut_iter(&mut self.data)
    }
}

// Marker traits
impl<S: LocalGeneric> LocalGeneric for Tensor<S> {}
impl<S: ValueType> ValueType for Tensor<S> {}
impl<S: Viewed + ?Sized> Viewed for Tensor<S> {}
impl<S: DynamicCollection + ?Sized> DynamicCollection for Tensor<S> {}
impl<S: Dense + ?Sized> Dense for Tensor<S> {}
impl<S: Flat> Flat for Tensor<S> {}

/*
 * View Impls
 */

impl<'a, T: 'a> View<'a> for Tensor<Vec<T>> {
    type Type = &'a Tensor<[T]>;
    fn view(&'a self) -> Self::Type {
        self.data.view().as_tensor()
    }
}

impl<'a, T: 'a> ViewMut<'a> for Tensor<Vec<T>> {
    type Type = &'a mut Tensor<[T]>;
    fn view_mut(&'a mut self) -> Self::Type {
        self.data.view_mut().as_mut_tensor()
    }
}

impl<'a, T: 'a> View<'a> for Tensor<[T]> {
    type Type = &'a Tensor<[T]>;
    fn view(&'a self) -> Self::Type {
        self
    }
}

impl<'a, T: 'a> ViewMut<'a> for Tensor<[T]> {
    type Type = &'a mut Tensor<[T]>;
    fn view_mut(&'a mut self) -> Self::Type {
        self
    }
}

impl<'a, T: LocalGeneric + View<'a>> View<'a> for Tensor<T> {
    type Type = Tensor<T::Type>;
    fn view(&'a self) -> Self::Type {
        Tensor {
            data: self.data.view(),
        }
    }
}

impl<'a, T: LocalGeneric + ViewMut<'a>> ViewMut<'a> for Tensor<T> {
    type Type = Tensor<T::Type>;
    fn view_mut(&'a mut self) -> Self::Type {
        Tensor {
            data: self.data.view_mut(),
        }
    }
}

// Std impls
impl<S: IntoIterator> IntoIterator for Tensor<S> {
    type Item = S::Item;
    type IntoIter = S::IntoIter;
    fn into_iter(self) -> Self::IntoIter {
        self.data.into_iter()
    }
}

/// Plain old data trait. Types that implement this trait contain no references and can be copied
/// with `memcpy`.
pub trait Pod: 'static + Copy + Sized + Send + Sync {}
impl<T> Pod for T where T: 'static + Copy + Sized + Send + Sync {}

impl<T: Scalar> AsTensor for T {
    type Inner = T;
    #[inline]
    fn as_tensor(&self) -> &Tensor<Self::Inner> {
        unsafe { Tensor::as_ref(self) }
    }
}

impl<T: Scalar> AsMutTensor for T {
    #[inline]
    fn as_mut_tensor(&mut self) -> &mut Tensor<Self::Inner> {
        unsafe { Tensor::as_mut(self) }
    }
}

impl<T: Scalar> IntoTensor for Tensor<T> {
    type Tensor = Self;
    fn into_tensor(self) -> Self::Tensor {
        self
    }
}

/*
 * Scalar trait
 */

pub trait Scalar:
    Pod
    + Flat
    + std::fmt::Debug
    + PartialOrd
    + num_traits::NumCast
    + num_traits::NumAssign
    + num_traits::FromPrimitive
    + std::iter::Sum
    + Dummy
    + IntoTensor<Tensor = Tensor<Self>>
{
}

impl<T: Scalar + num_traits::Signed> num_traits::Signed for Tensor<T> {
    fn abs(&self) -> Self {
        Tensor {
            data: self.data.abs(),
        }
    }
    fn abs_sub(&self, other: &Self) -> Self {
        Tensor {
            data: self.data.abs_sub(&other.data),
        }
    }
    fn signum(&self) -> Self {
        Tensor {
            data: self.data.signum(),
        }
    }
    fn is_positive(&self) -> bool {
        self.data.is_positive()
    }
    fn is_negative(&self) -> bool {
        self.data.is_negative()
    }
}

impl<T: Scalar> num_traits::ToPrimitive for Tensor<T> {
    fn to_i64(&self) -> Option<i64> {
        self.data.to_i64()
    }
    fn to_u64(&self) -> Option<u64> {
        self.data.to_u64()
    }
    fn to_isize(&self) -> Option<isize> {
        self.data.to_isize()
    }
    fn to_i8(&self) -> Option<i8> {
        self.data.to_i8()
    }
    fn to_i16(&self) -> Option<i16> {
        self.data.to_i16()
    }
    fn to_i32(&self) -> Option<i32> {
        self.data.to_i32()
    }
    fn to_i128(&self) -> Option<i128> {
        self.data.to_i128()
    }
    fn to_usize(&self) -> Option<usize> {
        self.data.to_usize()
    }
    fn to_u8(&self) -> Option<u8> {
        self.data.to_u8()
    }
    fn to_u16(&self) -> Option<u16> {
        self.data.to_u16()
    }
    fn to_u32(&self) -> Option<u32> {
        self.data.to_u32()
    }
    fn to_u128(&self) -> Option<u128> {
        self.data.to_u128()
    }
    fn to_f32(&self) -> Option<f32> {
        self.data.to_f32()
    }
    fn to_f64(&self) -> Option<f64> {
        self.data.to_f64()
    }
}
impl<S: Scalar> num_traits::NumCast for Tensor<S> {
    fn from<T: num_traits::ToPrimitive>(n: T) -> Option<Self> {
        S::from(n).map(|x| x.into_tensor())
    }
}
impl<T: Scalar> num_traits::Num for Tensor<T> {
    type FromStrRadixErr = T::FromStrRadixErr;
    fn from_str_radix(str: &str, radix: u32) -> Result<Self, Self::FromStrRadixErr> {
        T::from_str_radix(str, radix).map(|x| x.into_tensor())
    }
}
impl<T: Scalar + Float> num_traits::Float for Tensor<T> {
    fn nan() -> Self {
        Tensor { data: T::nan() }
    }
    fn infinity() -> Self {
        T::infinity().into_tensor()
    }
    fn neg_infinity() -> Self {
        T::neg_infinity().into_tensor()
    }
    fn neg_zero() -> Self {
        T::neg_zero().into_tensor()
    }
    fn min_value() -> Self {
        T::min_value().into_tensor()
    }
    fn min_positive_value() -> Self {
        T::min_positive_value().into_tensor()
    }
    fn max_value() -> Self {
        T::max_value().into_tensor()
    }
    fn is_nan(self) -> bool {
        self.data.is_nan()
    }
    fn is_infinite(self) -> bool {
        self.data.is_infinite()
    }
    fn is_finite(self) -> bool {
        self.data.is_finite()
    }
    fn is_normal(self) -> bool {
        self.data.is_normal()
    }
    fn classify(self) -> FpCategory {
        self.data.classify()
    }
    fn floor(self) -> Self {
        self.data.floor().into_tensor()
    }
    fn ceil(self) -> Self {
        self.data.ceil().into_tensor()
    }
    fn round(self) -> Self {
        self.data.round().into_tensor()
    }
    fn trunc(self) -> Self {
        self.data.trunc().into_tensor()
    }
    fn fract(self) -> Self {
        self.data.fract().into_tensor()
    }
    fn abs(self) -> Self {
        self.data.abs().into_tensor()
    }
    fn signum(self) -> Self {
        self.data.signum().into_tensor()
    }
    fn is_sign_positive(self) -> bool {
        self.data.is_sign_positive()
    }
    fn is_sign_negative(self) -> bool {
        self.data.is_sign_negative()
    }
    fn mul_add(self, a: Self, b: Self) -> Self {
        self.data.mul_add(a.data, b.data).into_tensor()
    }
    fn recip(self) -> Self {
        self.data.recip().into_tensor()
    }
    fn powi(self, n: i32) -> Self {
        self.data.powi(n).into_tensor()
    }
    fn powf(self, n: Self) -> Self {
        self.data.powf(n.data).into_tensor()
    }
    fn sqrt(self) -> Self {
        self.data.sqrt().into_tensor()
    }
    fn exp(self) -> Self {
        self.data.exp().into_tensor()
    }
    fn exp2(self) -> Self {
        self.data.exp2().into_tensor()
    }
    fn ln(self) -> Self {
        self.data.ln().into_tensor()
    }
    fn log(self, base: Self) -> Self {
        self.data.log(base.data).into_tensor()
    }
    fn log2(self) -> Self {
        self.data.log2().into_tensor()
    }
    fn log10(self) -> Self {
        self.data.log10().into_tensor()
    }
    fn max(self, other: Self) -> Self {
        self.data.max(other.data).into_tensor()
    }
    fn min(self, other: Self) -> Self {
        self.data.min(other.data).into_tensor()
    }
    fn abs_sub(self, other: Self) -> Self {
        self.data.abs_sub(other.data).into_tensor()
    }
    fn cbrt(self) -> Self {
        self.data.cbrt().into_tensor()
    }
    fn hypot(self, other: Self) -> Self {
        self.data.hypot(other.data).into_tensor()
    }
    fn sin(self) -> Self {
        self.data.sin().into_tensor()
    }
    fn cos(self) -> Self {
        self.data.cos().into_tensor()
    }
    fn tan(self) -> Self {
        self.data.tan().into_tensor()
    }
    fn asin(self) -> Self {
        self.data.asin().into_tensor()
    }
    fn acos(self) -> Self {
        self.data.acos().into_tensor()
    }
    fn atan(self) -> Self {
        self.data.atan().into_tensor()
    }
    fn atan2(self, other: Self) -> Self {
        self.data.atan2(other.data).into_tensor()
    }
    fn sin_cos(self) -> (Self, Self) {
        let (s, c) = self.data.sin_cos();
        (s.into_tensor(), c.into_tensor())
    }
    fn exp_m1(self) -> Self {
        self.data.exp_m1().into_tensor()
    }
    fn ln_1p(self) -> Self {
        self.data.ln_1p().into_tensor()
    }
    fn sinh(self) -> Self {
        self.data.sinh().into_tensor()
    }
    fn cosh(self) -> Self {
        self.data.cosh().into_tensor()
    }
    fn tanh(self) -> Self {
        self.data.tanh().into_tensor()
    }
    fn asinh(self) -> Self {
        self.data.asinh().into_tensor()
    }
    fn acosh(self) -> Self {
        self.data.acosh().into_tensor()
    }
    fn atanh(self) -> Self {
        self.data.atanh().into_tensor()
    }
    fn integer_decode(self) -> (u64, i16, i8) {
        self.data.integer_decode()
    }
}

#[cfg(feature = "approx")]
impl<U: Scalar, T: Scalar + approx::AbsDiffEq<U>> approx::AbsDiffEq<Tensor<U>> for Tensor<T>
where
    T::Epsilon: Copy,
{
    type Epsilon = T::Epsilon;
    #[inline]
    fn default_epsilon() -> Self::Epsilon {
        T::default_epsilon()
    }
    #[inline]
    fn abs_diff_eq(&self, other: &Tensor<U>, epsilon: Self::Epsilon) -> bool {
        self.data.abs_diff_eq(&other.data, epsilon)
    }
}
#[cfg(feature = "approx")]
impl<U: Scalar, T: Scalar + approx::RelativeEq<U>> approx::RelativeEq<Tensor<U>> for Tensor<T>
where
    T::Epsilon: Copy,
{
    #[inline]
    fn default_max_relative() -> Self::Epsilon {
        T::default_max_relative()
    }
    #[inline]
    fn relative_eq(
        &self,
        other: &Tensor<U>,
        epsilon: Self::Epsilon,
        max_relative: Self::Epsilon,
    ) -> bool {
        self.data.relative_eq(&other.data, epsilon, max_relative)
    }
}
#[cfg(feature = "approx")]
impl<U: Scalar, T: Scalar + approx::UlpsEq<U>> approx::UlpsEq<Tensor<U>> for Tensor<T>
where
    T::Epsilon: Copy,
{
    #[inline]
    fn default_max_ulps() -> u32 {
        T::default_max_ulps()
    }
    #[inline]
    fn ulps_eq(&self, other: &Tensor<U>, epsilon: Self::Epsilon, max_ulps: u32) -> bool {
        self.data.ulps_eq(&other.data, epsilon, max_ulps)
    }
}

macro_rules! impl_scalar {
    ($($type:ty),*) => {
        $(
            impl NonTensor for $type { }
            impl Scalar for $type { }
            impl Flat for $type { }
            impl Dummy for $type {
                #[inline]
                unsafe fn dummy() -> Self {
                    Self::default()
                }
            }

            impl<'a> Expr<'a> for $type {
                type Output = Tensor<$type>;
                #[inline]
                fn expr(&'a self) -> Self::Output {
                    Tensor { data: *self }
                }
            }
            impl<'a> ExprMut<'a> for $type {
                type Output = &'a mut Tensor<$type>;
                #[inline]
                fn expr_mut(&'a mut self) -> Self::Output {
                    self.into_tensor()
                }
            }

            impl IntoExpr for $type {
                type Expr = Tensor<$type>;
                #[inline]
                fn into_expr(self) -> Self::Expr {
                    Tensor { data: self }
                }
            }

            impl IntoExpr for &$type {
                type Expr = Tensor<$type>;
                #[inline]
                fn into_expr(self) -> Self::Expr {
                    Tensor { data: *self }
                }
            }

            impl<'a> IntoExpr for &'a mut $type {
                type Expr = &'a mut Tensor<$type>;
                #[inline]
                fn into_expr(self) -> Self::Expr {
                    self.into_tensor()
                }
            }

            impl IntoTensor for $type {
                type Tensor = Tensor<$type>;
                #[inline]
                fn into_tensor(self) -> Self::Tensor {
                    Tensor { data: self }
                }
            }
            impl Expression for $type {}

            impl ExprSize for $type {
                #[inline]
                fn expr_size(&self) -> usize {
                    1
                }
            }
            impl TotalExprSize for $type {
                #[inline]
                fn total_size_hint(&self, _cwise_reduce: u32) -> Option<usize> {
                    Some(1)
                }
            }

            impl AsSlice<$type> for $type {
                #[inline]
                fn as_slice(&self) -> &[$type] {
                    unsafe {
                        std::slice::from_raw_parts(self as *const _, 1)
                    }
                }
            }

            impl CwiseMulOp for $type {
                type Output = Tensor<Self>;
                #[inline]
                fn cwise_mul(self, rhs: Self) -> Self::Output {
                    Tensor { data: self * rhs }
                }
            }
            impl AddAssign<Tensor<$type>> for Tensor<[$type]> {
                #[inline]
                fn add_assign(&mut self, rhs: Tensor<$type>) {
                    self.data[0] += rhs.data
                }
            }
            impl SubAssign<Tensor<$type>> for Tensor<[$type]> {
                #[inline]
                fn sub_assign(&mut self, rhs: Tensor<$type>) {
                    self.data[0] -= rhs.data
                }
            }

        )*
    }
}

impl_scalar!(f64, f32, usize, u64, u32, u16, u8, i64, i32, i16, i8);

#[cfg(feature = "autodiff")]
mod autodiff_impls {
    use super::*;
    use autodiff::F;
    impl NonTensor for F {}
    impl Scalar for F {}
    impl Flat for F {}
    impl Dummy for F {
        unsafe fn dummy() -> Self {
            Self::default()
        }
    }
    impl IntoTensor for F {
        type Tensor = Tensor<F>;
        #[inline]
        fn into_tensor(self) -> Self::Tensor {
            Tensor { data: self }
        }
    }

    impl IntoExpr for F {
        type Expr = Tensor<F>;
        fn into_expr(self) -> Self::Expr {
            Tensor { data: self }
        }
    }

    impl IntoExpr for &F {
        type Expr = Tensor<F>;
        fn into_expr(self) -> Self::Expr {
            Tensor { data: *self }
        }
    }
    impl<'a> IntoExpr for &'a mut F {
        type Expr = &'a mut Tensor<F>;
        fn into_expr(self) -> Self::Expr {
            self.as_mut_tensor()
        }
    }
    impl DotOp for F {
        type Output = Tensor<Self>;
        fn dot_op(self, rhs: Self) -> Self::Output {
            Tensor { data: self * rhs }
        }
    }
}

pub trait Real: Scalar + Float {}
impl<T> Real for T where T: Scalar + Float {}

/*
 * Implement 0-tensor algebra.
 */

impl<T: Scalar> Mul for Tensor<T> {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        (self.data * rhs.data).into_tensor()
    }
}

impl<T: Scalar> CwiseMulOp for Tensor<T> {
    type Output = Self;
    #[inline]
    fn cwise_mul(mut self, rhs: Self) -> Self {
        self *= rhs;
        self
    }
}

impl<T: Scalar + MulAssign<S>, S: Scalar> MulAssign<Tensor<S>> for Tensor<T> {
    #[inline]
    fn mul_assign(&mut self, rhs: Tensor<S>) {
        self.data *= rhs.data;
    }
}

impl<T: Div<Output = T>> Div for Tensor<T> {
    type Output = Self;
    #[inline]
    fn div(self, rhs: Self) -> Self {
        Tensor {
            data: self.data / rhs.data,
        }
    }
}

impl<T: DivAssign> DivAssign for Tensor<T> {
    #[inline]
    fn div_assign(&mut self, rhs: Self) {
        self.data /= rhs.data;
    }
}

impl<T: Rem<Output = T>> Rem for Tensor<T> {
    type Output = Self;
    #[inline]
    fn rem(self, rhs: Self) -> Self {
        Tensor {
            data: self.data % rhs.data,
        }
    }
}

impl<T: RemAssign> RemAssign for Tensor<T> {
    #[inline]
    fn rem_assign(&mut self, rhs: Self) {
        self.data %= rhs.data;
    }
}

impl<T: Scalar> Add for Tensor<T> {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        Tensor {
            data: self.data + rhs.data,
        }
    }
}

impl<T: Scalar> AddAssign for Tensor<T> {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        self.data += rhs.data
    }
}

impl<T: Scalar> Sub for Tensor<T> {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Tensor {
            data: self.data - rhs.data,
        }
    }
}

impl<T: Scalar> SubAssign for Tensor<T> {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        self.data -= rhs.data
    }
}

impl<T: Neg<Output = T> + Scalar> Neg for Tensor<T> {
    type Output = Self;
    #[inline]
    fn neg(mut self) -> Self::Output {
        self.data = -self.data;
        self
    }
}

impl<T: Scalar> Zero for Tensor<T> {
    #[inline]
    fn zero() -> Self {
        Tensor { data: Zero::zero() }
    }
    #[inline]
    fn is_zero(&self) -> bool {
        self.data == Zero::zero()
    }
}

impl<T: Scalar> One for Tensor<T> {
    #[inline]
    fn one() -> Self {
        Tensor { data: One::one() }
    }
}

impl<T> std::iter::Sum for Tensor<T>
where
    Self: Add + num_traits::Zero,
{
    #[inline]
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(num_traits::Zero::zero(), |acc, x| acc + x)
    }
}

/*
 * Implement 1-tensor algebra.
 */

/*
 * Tensor addition and subtraction.
 */

macro_rules! impl_array_tensors {
    ($n:expr) => {
        impl<T> LocalGeneric for &Tensor<[T; $n]> {}

        impl<T: Copy> Neg for Tensor<[T; $n]>
        where
            T: Neg<Output = T>,
        {
            type Output = Self;
            #[inline]
            #[unroll_for_loops]
            fn neg(mut self) -> Self::Output {
                for i in 0..$n {
                    self.data[i] = -self.data[i];
                }
                self
            }
        }

        impl<T: Copy> Neg for &Tensor<[T; $n]>
        where
            T: Neg<Output = T>,
        {
            type Output = Tensor<[T; $n]>;
            #[inline]
            fn neg(self) -> Self::Output {
                Neg::neg(*self)
            }
        }
    };
}

impl_array_tensors!(1);
impl_array_tensors!(2);
impl_array_tensors!(3);
impl_array_tensors!(4);

macro_rules! impl_slice_add {
    ($other:ty) => {
        fn add(self, other: $other) -> Self::Output {
            assert_eq!(other.data.len(), self.data.len());
                Tensor { data: self.data
                    .iter()
                    .zip(other.data.iter())
                    .map(|(&a, &b)| a + b)
                    .collect::<Vec<_>>() }
        }
    }
}

//impl<T: Add<Output = T> + Copy> Add for Tensor<&[T]> {
//    type Output = Tensor<Vec<T>>;
//
//    /// Add two tensor slices together into a resulting tensor `Vec`.
//    ///
//    /// # Example
//    ///
//    /// ```
//    /// use utils::soap::*;
//    /// let a = vec![1,2,3,4];
//    /// let b = vec![5,6,7,8];
//    /// assert_eq!(
//    ///     Tensor::new(vec![6,8,10,12]),
//    ///     Tensor::new(a.view()) + Tensor::new(b.view())
//    /// );
//    /// ```
//    impl_slice_add!(Self);
//}

impl<T: Add<Output = T> + Copy> Add for &Tensor<[T]> {
    type Output = Tensor<Vec<T>>;

    /// Add two tensor slices together into a resulting tensor `Vec`.
    ///
    /// # Example
    ///
    /// ```
    /// use utils::soap::*;
    /// let a = vec![1,2,3,4];
    /// let b = vec![5,6,7,8];
    /// assert_eq!(
    ///     [6,8,10,12].as_tensor(),
    ///     a.view().as_tensor() + b.view().as_tensor())
    /// );
    /// ```
    impl_slice_add!(Self);
}

//impl<T: Add<Output = T> + Copy> Add<Tensor<&[T]>> for &Tensor<[T]> {
//    type Output = Tensor<Vec<T>>;
//
//    /// Add two tensor slices together into a resulting tensor `Vec`.
//    ///
//    /// # Example
//    ///
//    /// ```
//    /// use utils::soap::*;
//    /// let a = vec![1,2,3,4];
//    /// let b = vec![5,6,7,8];
//    /// assert_eq!(
//    ///     Tensor::new(vec![6,8,10,12]),
//    ///     a.view().as_tensor() + Tensor::new(b.view()))
//    /// );
//    /// ```
//    impl_slice_add!(Tensor<&[T]>);
//}

impl<T: AddAssign + Copy> Add<Tensor<Vec<T>>> for &Tensor<[T]> {
    type Output = Tensor<Vec<T>>;

    /// Add two tensors together into a resulting tensor `Vec`.
    ///
    /// # Example
    ///
    /// ```
    /// use utils::soap::*;
    /// let a = vec![1,2,3,4];
    /// let b = vec![5,6,7,8];
    /// assert_eq!(
    ///     Vector::new(vec![6,8,10,12]),
    ///     a.view().as_tensor() + Vector::new(b)
    /// );
    /// ```
    fn add(self, mut other: Tensor<Vec<T>>) -> Self::Output {
        other.add_assign(self);
        other
    }
}

//impl<T: Add<Output = T> + Copy> Add<&Tensor<[T]>> for Tensor<&[T]> {
//    type Output = Tensor<Vec<T>>;
//
//    /// Add two tensor slices together into a resulting tensor `Vec`.
//    ///
//    /// # Example
//    ///
//    /// ```
//    /// use utils::soap::*;
//    /// let a = vec![1,2,3,4];
//    /// let b = vec![5,6,7,8];
//    /// assert_eq!(
//    ///     [6,8,10,12].as_tensor(),
//    ///     a.view().into_tensor() + b.view().as_tensor())
//    /// );
//    /// ```
//    impl_slice_add!(&Tensor<[T]>);
//}

//impl<T: AddAssign + Copy> Add<Tensor<Vec<T>>> for Tensor<&[T]> {
//    type Output = Tensor<Vec<T>>;
//
//    /// Add two tensor slices together into a resulting tensor `Vec`.
//    ///
//    /// # Example
//    ///
//    /// ```
//    /// use utils::soap::*;
//    /// let a = vec![1,2,3,4];
//    /// let b = vec![5,6,7,8];
//    /// assert_eq!(
//    ///     Vector::new(vec![6,8,10,12]),
//    ///     a.view().into_tensor() + Vector::new(b)
//    /// );
//    /// ```
//    fn add(self, mut other: Tensor<Vec<T>>) -> Self::Output {
//        other.add_assign(self);
//        other
//    }
//}

impl<T: AddAssign + Copy> Add<&Tensor<[T]>> for Tensor<Vec<T>> {
    type Output = Self;

    /// Add a tensor slice to a tensor `Vec` into a resulting tensor `Vec`.
    ///
    /// # Example
    ///
    /// ```
    /// use utils::soap::*;
    /// let a = vec![1,2,3,4];
    /// let b = vec![5,6,7,8];
    /// assert_eq!(
    ///     Vector::new(vec![6,8,10,12]),
    ///     Vector::new(a) + b.view().as_tensor()
    /// );
    /// ```
    fn add(mut self, other: &Tensor<[T]>) -> Self::Output {
        self.add_assign(other);
        self
    }
}

//impl<T: AddAssign + Copy> Add<Tensor<&[T]>> for Tensor<Vec<T>> {
//    type Output = Self;
//
//    /// Add a tensor slice to a tensor `Vec` into a resulting tensor `Vec`.
//    ///
//    /// # Example
//    ///
//    /// ```
//    /// use utils::soap::*;
//    /// let a = vec![1,2,3,4];
//    /// let b = vec![5,6,7,8];
//    /// assert_eq!(
//    ///     Tensor::new(vec![6,8,10,12]),
//    ///     Tensor::new(a) + Tensor::new(b.view())
//    /// );
//    /// ```
//    fn add(mut self, other: Tensor<&[T]>) -> Self::Output {
//        self.add_assign(other);
//        self
//    }
//}

macro_rules! impl_slice_add_assign {
    ($self:ident, $other:ident) => {{
        assert_eq!($other.data.len(), $self.data.len());
        for (a, &b) in $self.data.iter_mut().zip($other.data.view_iter()) {
            *a += b;
        }
    }};
}

//impl<T, S> AddAssign<S> for Tensor<Vec<T>>
//where
//    S: ValueType + IntoData,
//    S::Data: Set + for<'a> ViewIterator<'a, Item = &'a T>,
//    T: AddAssign + Copy,
//{
//    /// Add a generic tensor to this tensor `Vec`.
//    ///
//    /// # Example
//    ///
//    /// ```
//    /// use utils::soap::*;
//    /// let a = Subset::from_unique_ordered_indices(vec![1,3,4,5], vec![1,2,3,4,5,6,7]);
//    /// let b = vec![5,6,7,8];
//    /// let mut tensor = Vector::new(b.clone());
//    /// tensor += SubsetTensor::new(a.view());
//    /// assert_eq!(vec![7,10,12,14], tensor.data);
//    /// ```
//    fn add_assign(&mut self, other: S) {
//        let other_data = other.into_data();
//        assert_eq!(other_data.len(), self.data.len());
//        for (a, &b) in self.data.iter_mut().zip(other_data.view_iter()) {
//            *a += b;
//        }
//    }
//}

//impl<T, S> AddAssign<&Tensor<S>> for Tensor<Vec<T>>
//where
//    S: Set + LocalGeneric + for<'a> ViewIterator<'a, Item = &'a T>,
//    T: AddAssign + Copy,
//{
//    /// Add a generic tensor to this tensor `Vec`.
//    ///
//    /// # Example
//    ///
//    /// ```
//    /// use utils::soap::*;
//    /// let a = Subset::from_unique_ordered_indices(vec![1,3,4,5], vec![1,2,3,4,5,6,7]);
//    /// let b = vec![5,6,7,8];
//    /// let mut tensor = b.clone().into_tensor();
//    /// tensor += a.as_tensor();
//    /// assert_eq!(vec![7,10,12,14], tensor.data);
//    /// ```
//    fn add_assign(&mut self, other: &Tensor<S>) {
//        impl_slice_add_assign!(self, other);
//    }
//}

//impl<T, S> AddAssign<S> for Tensor<[T]>
//where
//    S: ValueType + IntoData,
//    S::Data: Set + for<'a> ViewIterator<'a, Item = &'a T>,
//    T: AddAssign + Copy,
//{
//    /// Add a generic tensor to this slice tensor.
//    ///
//    /// # Example
//    ///
//    /// ```
//    /// use utils::soap::*;
//    /// let a = Subset::from_unique_ordered_indices(vec![1,3,4,5], vec![1,2,3,4,5,6,7]);
//    /// let mut b = vec![5,6,7,8];
//    /// *b.as_mut_tensor() += a.view().into_tensor();
//    /// assert_eq!(vec![7,10,12,14], b);
//    /// ```
//    fn add_assign(&mut self, other: Tensor<S>) {
//        let other_data = other.into_data();
//        assert_eq!(other_data.len(), self.data.len());
//        for (a, &b) in self.data.iter_mut().zip(other_data.view_iter()) {
//            *a += b;
//        }
//    }
//}

//impl<T, S> AddAssign<&Tensor<S>> for Tensor<[T]>
//where
//    S: Set + LocalGeneric + for<'a> ViewIterator<'a, Item = &'a T>,
//    T: AddAssign + Copy,
//{
//    /// Add a generic tensor to this slice tensor.
//    ///
//    /// # Example
//    ///
//    /// ```
//    /// use utils::soap::*;
//    /// let a = Subset::from_unique_ordered_indices(vec![1,3,4,5], vec![1,2,3,4,5,6,7]);
//    /// let mut b = vec![5,6,7,8];
//    /// *b.as_mut_tensor() += a.as_tensor();
//    /// assert_eq!(vec![7,10,12,14], b);
//    /// ```
//    fn add_assign(&mut self, other: &Tensor<S>) {
//        impl_slice_add_assign!(self, other);
//    }
//}

impl<T: AddAssign + Copy> AddAssign<&Tensor<[T]>> for Tensor<Vec<T>> {
    /// Add a tensor slice to this tensor.
    ///
    /// # Example
    ///
    /// ```
    /// use utils::soap::*;
    /// let a = vec![1,2,3,4];
    /// let b = vec![5,6,7,8];
    /// let mut tensor = Vector::new(a.clone());
    /// tensor += b.view().into_tensor();
    /// assert_eq!(vec![6,8,10,12], tensor.data);
    /// ```
    fn add_assign(&mut self, other: &Tensor<[T]>) {
        impl_slice_add_assign!(self, other);
    }
}

//impl<T: AddAssign + Copy> AddAssign<Tensor<&[T]>> for Tensor<Vec<T>> {
//    /// Add a tensor slice to this tensor.
//    ///
//    /// # Example
//    ///
//    /// ```rust
//    /// use utils::soap::*;
//    /// let a = vec![1,2,3,4];
//    /// let b = vec![5,6,7,8];
//    /// let mut tensor = Vector::new(a.clone());
//    /// tensor += b.view().as_tensor();
//    /// assert_eq!(vec![6,8,10,12], tensor.data);
//    /// ```
//    fn add_assign(&mut self, other: Tensor<&[T]>) {
//        impl_slice_add_assign!(self, other);
//    }
//}

//impl<T: AddAssign + Copy> AddAssign<Tensor<Vec<T>>> for Tensor<Vec<T>> {
//    /// Add a tensor `Vec` to this tensor.
//    ///
//    /// # Example
//    ///
//    /// ```rust
//    /// use utils::soap::*;
//    /// let a = vec![1,2,3,4];
//    /// let b = vec![5,6,7,8];
//    /// let mut tensor = Vector::new(a.clone());
//    /// tensor += Vector::new(b);
//    /// assert_eq!(vec![6,8,10,12], tensor.data);
//    /// ```
//    fn add_assign(&mut self, other: Tensor<Vec<T>>) {
//        impl_slice_add_assign!(self, other);
//    }
//}

//impl<T: AddAssign + Copy> AddAssign<Tensor<Vec<T>>> for Tensor<&mut [T]> {
//    /// Add a tensor `Vec` to this tensor.
//    ///
//    /// # Example
//    ///
//    /// ```
//    /// use utils::soap::*;
//    /// let mut a = vec![1,2,3,4];
//    /// let b = vec![5,6,7,8];
//    /// let mut view = a.view_mut().into_tensor();
//    /// view += Vector::new(b);
//    /// assert_eq!(vec![6,8,10,12], a);
//    /// ```
//    fn add_assign(&mut self, other: Tensor<Vec<T>>) {
//        impl_slice_add_assign!(self, other);
//    }
//}

//impl<T: AddAssign + Copy> AddAssign<Tensor<&[T]>> for Tensor<&mut [T]> {
//    /// Add a tensor slice to this tensor.
//    ///
//    /// # Example
//    ///
//    /// ```
//    /// use utils::soap::*;
//    /// let mut a = vec![1,2,3,4];
//    /// let b = vec![5,6,7,8];
//    /// let mut view = a.view_mut().into_tensor();
//    /// view += Vector::new(b.view());
//    /// assert_eq!(vec![6,8,10,12], a);
//    /// ```
//    fn add_assign(&mut self, other: Tensor<&[T]>) {
//        impl_slice_add_assign!(self, other);
//    }
//}

//impl<T: AddAssign + Copy> AddAssign<&Tensor<[T]>> for Tensor<&mut [T]> {
//    /// Add a tensor slice to this tensor.
//    ///
//    /// # Example
//    ///
//    /// ```
//    /// use utils::soap::*;
//    /// let mut a = vec![1,2,3,4];
//    /// let b = vec![5,6,7,8];
//    /// let mut view = a.view_mut().into_tensor();
//    /// view += b.view().as_tensor();
//    /// assert_eq!(vec![6,8,10,12], a);
//    /// ```
//    fn add_assign(&mut self, other: &Tensor<[T]>) {
//        impl_slice_add_assign!(self, other);
//    }
//}

impl<T: AddAssign + Copy> AddAssign<Tensor<Vec<T>>> for Tensor<[T]> {
    /// Add a tensor `Vec` to this tensor.
    ///
    /// # Example
    ///
    /// ```
    /// use utils::soap::*;
    /// let mut a = vec![1,2,3,4];
    /// let b = vec![5,6,7,8];
    /// *a.as_mut_slice().as_mut_tensor() += Vector::new(b);
    /// assert_eq!(vec![6,8,10,12], a);
    /// ```
    fn add_assign(&mut self, other: Tensor<Vec<T>>) {
        impl_slice_add_assign!(self, other);
    }
}

impl<T: AddAssign + Copy> AddAssign<&Tensor<[T]>> for Tensor<[T]> {
    /// Add a tensor slice to this tensor.
    ///
    /// # Example
    ///
    /// ```
    /// use utils::soap::*;
    /// let mut a = vec![1,2,3,4];
    /// let b = vec![5,6,7,8];
    /// *a.as_mut_slice().as_mut_tensor() += b.as_slice().as_tensor();
    /// assert_eq!(vec![6,8,10,12], a);
    /// ```
    fn add_assign(&mut self, other: &Tensor<[T]>) {
        impl_slice_add_assign!(self, other);
    }
}

//impl<T: AddAssign + Copy> AddAssign<Tensor<&[T]>> for Tensor<[T]> {
//    /// Add a tensor slice to this tensor.
//    ///
//    /// # Example
//    ///
//    /// ```
//    /// use utils::soap::*;
//    /// let mut a = vec![1,2,3,4];
//    /// let b = vec![5,6,7,8];
//    /// *a.as_mut_slice().as_mut_tensor() += b.as_slice().into_tensor();
//    /// assert_eq!(vec![6,8,10,12], a);
//    /// ```
//    fn add_assign(&mut self, other: Tensor<&[T]>) {
//        impl_slice_add_assign!(self, other);
//    }
//}

macro_rules! impl_slice_sub {
    ($other:ty) => {
        fn sub(self, other: $other) -> Self::Output {
            assert_eq!(other.data.len(), self.data.len());
            Tensor { data:
                self.data
                    .iter()
                    .zip(other.data.iter())
                    .map(|(&a, &b)| a - b)
                    .collect::<Vec<_>>()}
        }
    }
}

//impl<T: Sub<Output = T> + Copy> Sub for Tensor<&[T]> {
//    type Output = Tensor<Vec<T>>;
//
//    /// Subtract one slice tensor from another.
//    ///
//    /// # Example
//    ///
//    /// ```
//    /// use utils::soap::*;
//    /// let a = vec![5,6,7,8];
//    /// let b = vec![1,2,3,4];
//    /// assert_eq!(
//    ///     Vector::new(vec![4,4,4,4]),
//    ///     a.view().into_tensor() - b.view().into_tensor()
//    /// );
//    /// ```
//    impl_slice_sub!(Self);
//}

impl<T: Sub<Output = T> + Copy> Sub for &Tensor<[T]> {
    type Output = Tensor<Vec<T>>;

    /// Subtract one slice tensor from another.
    ///
    /// # Example
    ///
    /// ```
    /// use utils::soap::*;
    /// let a = vec![5,6,7,8];
    /// let b = vec![1,2,3,4];
    /// assert_eq!(
    ///     [4,4,4,4].as_tensor(),
    ///     a.view().as_tensor() - b.view().as_tensor())
    /// );
    /// ```
    impl_slice_sub!(Self);
}

//impl<T: Sub<Output = T> + Copy> Sub<Tensor<&[T]>> for &Tensor<[T]> {
//    type Output = Tensor<Vec<T>>;
//
//    /// Subtract one slice tensor from another.
//    ///
//    /// # Example
//    ///
//    /// ```
//    /// use utils::soap::*;
//    /// let a = vec![5,6,7,8];
//    /// let b = vec![1,2,3,4];
//    /// assert_eq!(
//    ///     vec![4,4,4,4].into_tensor(),
//    ///     a.view().as_tensor() - b.view().into_tensor())
//    /// );
//    /// ```
//    impl_slice_sub!(Tensor<&[T]>);
//}

impl<T: Sub<Output = T> + Copy> Sub<Tensor<Vec<T>>> for &Tensor<[T]> {
    type Output = Tensor<Vec<T>>;

    /// Subtract a `Vec` tensor from a slice tensor.
    ///
    /// # Example
    ///
    /// ```
    /// use utils::soap::*;
    /// let a = vec![5,6,7,8];
    /// let b = vec![1,2,3,4];
    /// assert_eq!(
    ///     Vector::new(vec![4,4,4,4]),
    ///     a.view().as_tensor() - Vector::new(b)
    /// );
    /// ```
    fn sub(self, mut other: Tensor<Vec<T>>) -> Self::Output {
        assert_eq!(other.data.len(), self.data.len());
        for (&a, b) in self.data.iter().zip(other.data.iter_mut()) {
            *b = a - *b;
        }
        other
    }
}

//impl<T: Sub<Output = T> + Copy> Sub<&Tensor<[T]>> for Tensor<&[T]> {
//    type Output = Tensor<Vec<T>>;
//
//    /// Subtract one slice tensor from another.
//    ///
//    /// # Example
//    ///
//    /// ```
//    /// use utils::soap::*;
//    /// let a = vec![5,6,7,8];
//    /// let b = vec![1,2,3,4];
//    /// assert_eq!(
//    ///     [4,4,4,4].as_tensor(),
//    ///     a.view().into_tensor() - b.view().as_tensor())
//    /// );
//    /// ```
//    impl_slice_sub!(&Tensor<[T]>);
//}

//impl<T: Sub<Output = T> + Copy> Sub<Tensor<Vec<T>>> for Tensor<&[T]> {
//    type Output = Tensor<Vec<T>>;
//
//    /// Subtract a `Vec` tensor from a slice tensor.
//    ///
//    /// # Example
//    ///
//    /// ```
//    /// use utils::soap::*;
//    /// let a = vec![5,6,7,8];
//    /// let b = vec![1,2,3,4];
//    /// assert_eq!(
//    ///     Vector::new(vec![4,4,4,4]),
//    ///     a.view().into_tensor() - Vector::new(b)
//    /// );
//    /// ```
//    fn sub(self, mut other: Tensor<Vec<T>>) -> Self::Output {
//        assert_eq!(other.data.len(), self.data.len());
//        for (&a, b) in self.data.iter().zip(other.data.iter_mut()) {
//            *b = a - *b;
//        }
//        other
//    }
//}

impl<T: Scalar> Sub<&Tensor<[T]>> for Tensor<Vec<T>> {
    type Output = Self;

    /// Subtract a tensor slice from another.
    ///
    /// # Example
    ///
    /// ```
    /// use utils::soap::*;
    /// let a = vec![5,6,7,8];
    /// let b = vec![1,2,3,4];
    /// assert_eq!(
    ///     Vector::new(vec![4,4,4,4]),
    ///     Vector::new(a) - b.view().as_tensor()
    /// );
    /// ```
    fn sub(mut self, other: &Tensor<[T]>) -> Self::Output {
        self.sub_assign(other);
        self
    }
}

//impl<T: Scalar> Sub<Tensor<&[T]>> for Tensor<Vec<T>> {
//    type Output = Self;
//
//    /// Subtract a slice tensor from a `Vec` tensor.
//    ///
//    /// # Example
//    ///
//    /// ```
//    /// use utils::soap::*;
//    /// let a = vec![5,6,7,8];
//    /// let b = vec![1,2,3,4];
//    /// assert_eq!(
//    ///     Vector::new(vec![4,4,4,4]),
//    ///     Vector::new(a) - b.view().into_tensor()
//    /// );
//    /// ```
//    fn sub(mut self, other: Tensor<&[T]>) -> Self::Output {
//        self.sub_assign(other);
//        self
//    }
//}

macro_rules! impl_sub_assign {
    ($self:ident, $other:ident) => {
        assert_eq!($other.data.len(), $self.data.len());
        for (a, &b) in $self.data.iter_mut().zip($other.data.view_iter()) {
            *a -= b;
        }
    };
}

//impl<T, S> SubAssign<S> for Tensor<Vec<T>>
//where
//    S: Set + LocalGeneric + for<'a> ViewIterator<'a, Item = &'a T>,
//    T: SubAssign + Copy,
//{
//    /// Subtract a generic tensor from this `Vec` tensor.
//    ///
//    /// # Example
//    ///
//    /// ```
//    /// use utils::soap::*;
//    /// let a = Subset::from_unique_ordered_indices(vec![1,3,4,5], vec![1,2,3,4,5,6,7]);
//    /// let mut tensor = Vector::new(vec![5,6,7,8]);
//    /// tensor -= SubsetTensor::new(a);
//    /// assert_eq!(vec![3,2,2,2], tensor.data);
//    /// ```
//    fn sub_assign(&mut self, other: Tensor<S>) {
//        impl_sub_assign!(self, other);
//    }
//}
//
//impl<T, S> SubAssign<&Tensor<S>> for Tensor<Vec<T>>
//where
//    S: Set + LocalGeneric + for<'a> ViewIterator<'a, Item = &'a T>,
//    T: SubAssign + Copy,
//{
//    /// Subtract a generic tensor reference from this `Vec` tensor.
//    ///
//    /// # Example
//    ///
//    /// ```
//    /// use utils::soap::*;
//    /// let a = Subset::from_unique_ordered_indices(vec![1,3,4,5], vec![1,2,3,4,5,6,7]);
//    /// let mut tensor = Vector::new(vec![5,6,7,8]);
//    /// tensor -= a.as_tensor();
//    /// assert_eq!(vec![3,2,2,2], tensor.data);
//    /// ```
//    fn sub_assign(&mut self, other: &Tensor<S>) {
//        impl_sub_assign!(self, other);
//    }
//}
//
//impl<T, S> SubAssign<Tensor<S>> for Tensor<[T]>
//where
//    S: Set + LocalGeneric + for<'a> ViewIterator<'a, Item = &'a T>,
//    T: SubAssign + Copy,
//{
//    /// Subtract a generic tensor from this slice tensor.
//    ///
//    /// # Example
//    ///
//    /// ```
//    /// use utils::soap::*;
//    /// let a = Subset::from_unique_ordered_indices(vec![1,3,4,5], vec![1,2,3,4,5,6,7]);
//    /// let mut b = vec![5,6,7,8];
//    /// *b.as_mut_tensor() -= SubsetTensor::new(a);
//    /// assert_eq!(vec![3,2,2,2], b);
//    /// ```
//    fn sub_assign(&mut self, other: Tensor<S>) {
//        impl_sub_assign!(self, other);
//    }
//}
//
//impl<T, S> SubAssign<&Tensor<S>> for Tensor<[T]>
//where
//    S: Set + LocalGeneric + for<'a> ViewIterator<'a, Item = &'a T>,
//    T: SubAssign + Copy,
//{
//    /// Subtract a generic tensor reference from this slice tensor.
//    ///
//    /// # Example
//    ///
//    /// ```
//    /// use utils::soap::*;
//    /// let a = Subset::from_unique_ordered_indices(vec![1,3,4,5], vec![1,2,3,4,5,6,7]);
//    /// let mut b = vec![5,6,7,8];
//    /// *b.as_mut_tensor() -= a.into_tensor();
//    /// assert_eq!(vec![3,2,2,2], b);
//    /// ```
//    fn sub_assign(&mut self, other: &Tensor<S>) {
//        impl_sub_assign!(self, other);
//    }
//}

//impl<T: SubAssign + Copy> SubAssign<Tensor<&[T]>> for Tensor<Vec<T>> {
//    /// Subtract a tensor slice from this `Vec` tensor.
//    ///
//    /// # Example
//    ///
//    /// ```
//    /// use utils::soap::*;
//    /// let a = vec![1,2,3,4];
//    /// let mut tensor = Vector::new(vec![5,6,7,8]);
//    /// tensor -= a.view().into_tensor();
//    /// assert_eq!(vec![4,4,4,4], tensor.data);
//    /// ```
//    fn sub_assign(&mut self, other: Tensor<&[T]>) {
//        impl_sub_assign!(self, other);
//    }
//}

impl<T: SubAssign + Copy> SubAssign<&Tensor<[T]>> for Tensor<Vec<T>> {
    /// Subtract a tensor slice from this `Vec` tensor.
    ///
    /// # Example
    ///
    /// ```
    /// use utils::soap::*;
    /// let a = vec![1,2,3,4];
    /// let mut tensor = Vector::new(vec![5,6,7,8]);
    /// tensor -= a.view().as_tensor();
    /// assert_eq!(vec![4,4,4,4], tensor.data);
    /// ```
    fn sub_assign(&mut self, other: &Tensor<[T]>) {
        impl_sub_assign!(self, other);
    }
}

//impl<T: SubAssign + Copy> SubAssign<Tensor<Vec<T>>> for Tensor<&mut [T]> {
//    /// Subtract a `Vec` tensor from this tensor.
//    ///
//    /// # Example
//    ///
//    /// ```
//    /// use utils::soap::*;
//    /// let a = vec![1,2,3,4];
//    /// let mut b = vec![5,6,7,8];
//    /// let mut view = b.view_mut().into_tensor();
//    /// view -= Vector::new(a);
//    /// assert_eq!(vec![4,4,4,4], b);
//    /// ```
//    fn sub_assign(&mut self, other: Tensor<Vec<T>>) {
//        impl_sub_assign!(self, other);
//    }
//}

//impl<T: SubAssign + Copy> SubAssign<Tensor<&[T]>> for Tensor<&mut [T]> {
//    /// Subtract a tensor slice from this tensor.
//    ///
//    /// # Example
//    ///
//    /// ```
//    /// use utils::soap::*;
//    /// let a = vec![1,2,3,4];
//    /// let mut b = vec![5,6,7,8];
//    /// let mut view = b.view_mut().into_tensor();
//    /// view -= a.view().into_tensor();
//    /// assert_eq!(vec![4,4,4,4], b);
//    /// ```
//    fn sub_assign(&mut self, other: Tensor<&[T]>) {
//        impl_sub_assign!(self, other);
//    }
//}
//
//impl<T: SubAssign + Copy> SubAssign<&Tensor<[T]>> for Tensor<&mut [T]> {
//    /// Subtract a tensor slice from this tensor.
//    ///
//    /// # Example
//    ///
//    /// ```
//    /// use utils::soap::*;
//    /// let a = vec![1,2,3,4];
//    /// let mut b = vec![5,6,7,8];
//    /// let mut view = b.view_mut().into_tensor();
//    /// view -= a.view().as_tensor();
//    /// assert_eq!(vec![4,4,4,4], b);
//    /// ```
//    fn sub_assign(&mut self, other: &Tensor<[T]>) {
//        impl_sub_assign!(self, other);
//    }
//}

impl<T: SubAssign + Copy> SubAssign<Tensor<Vec<T>>> for Tensor<[T]> {
    /// Subtract a tensor slice from this tensor.
    ///
    /// # Example
    ///
    /// ```
    /// use utils::soap::*;
    /// let a = vec![1,2,3,4];
    /// let mut b = vec![5,6,7,8];
    /// *b.view_mut().as_mut_tensor() -= Vector::new(a);
    /// assert_eq!(vec![4,4,4,4], b);
    /// ```
    fn sub_assign(&mut self, other: Tensor<Vec<T>>) {
        impl_sub_assign!(self, other);
    }
}

impl<T: SubAssign + Copy> SubAssign<&Tensor<[T]>> for Tensor<[T]> {
    /// Subtract a tensor slice from this tensor.
    ///
    /// # Example
    ///
    /// ```
    /// use utils::soap::*;
    /// let a = vec![1,2,3,4];
    /// let mut b = vec![5,6,7,8];
    /// *b.view_mut().as_mut_tensor() -= a.view().as_tensor();
    /// assert_eq!(vec![4,4,4,4], b);
    /// ```
    fn sub_assign(&mut self, other: &Tensor<[T]>) {
        impl_sub_assign!(self, other);
    }
}

//impl<T: SubAssign + Copy> SubAssign<Tensor<&[T]>> for Tensor<[T]> {
//    /// Subtract a tensor slice from this tensor.
//    ///
//    /// # Example
//    ///
//    /// ```
//    /// use utils::soap::*;
//    /// let a = vec![1,2,3,4];
//    /// let mut b = vec![5,6,7,8];
//    /// *b.as_mut_tensor() -= a.view().into_tensor();
//    /// assert_eq!(vec![4,4,4,4], b);
//    /// ```
//    fn sub_assign(&mut self, other: Tensor<&[T]>) {
//        impl_sub_assign!(self, other);
//    }
//}

/*
 * Scalar multiplication
 */

impl<T: Scalar> Mul<T> for &Tensor<[T]> {
    type Output = Tensor<Vec<T>>;

    /// Multiply a tensor slice by a scalar producing a new `Vec` tensor.
    ///
    /// # Example
    ///
    /// ```
    /// use utils::soap::*;
    /// let a = vec![1,2,3,4];
    /// assert_eq!(Vector::new(vec![3,6,9,12]), a.view().into_tensor() * 3);
    /// ```
    fn mul(self, other: T) -> Self::Output {
        Tensor {
            data: self.data.iter().map(|&a| a * other).collect::<Vec<_>>(),
        }
    }
}

//impl<T: Scalar> Mul<T> for Tensor<&[T]> {
//    type Output = Tensor<Vec<T>>;
//
//    /// Multiply a tensor slice by a scalar producing a new `Vec` tensor.
//    ///
//    /// # Example
//    ///
//    /// ```
//    /// use utils::soap::*;
//    /// let a = vec![1,2,3,4];
//    /// assert_eq!(Vector::new(vec![3,6,9,12]), a.view().into_tensor() * 3);
//    /// ```
//    fn mul(self, other: T) -> Self::Output {
//        Tensor { data: self.data.iter().map(|&a| a * other).collect::<Vec<_>>() }
//    }
//}

//impl<T: Scalar> MulAssign<T> for Tensor<&mut [T]> {
//    /// Multiply this tensor slice by a scalar.
//    ///
//    /// # Example
//    ///
//    /// ```
//    /// use utils::soap::*;
//    /// let mut a = vec![1,2,3,4];
//    /// let mut view = a.view_mut().into_tensor();
//    /// view *= 3;
//    /// assert_eq!(vec![3,6,9,12], a);
//    /// ```
//    fn mul_assign(&mut self, other: T) {
//        for a in self.data.iter_mut() {
//            *a *= other;
//        }
//    }
//}

impl<T: Scalar> MulAssign<T> for Tensor<[T]> {
    /// Multiply this tensor slice by a scalar.
    ///
    /// # Example
    ///
    /// ```
    /// use utils::soap::*;
    /// let mut a = vec![1,2,3,4];
    /// *a.view_mut().as_mut_tensor() *= 3;
    /// assert_eq!(vec![3,6,9,12], a);
    /// ```
    fn mul_assign(&mut self, other: T) {
        for a in self.data.iter_mut() {
            *a *= other;
        }
    }
}

/*
 * Scalar division
 */

impl<T: Scalar> Div<T> for &Tensor<[T]> {
    type Output = Tensor<Vec<T>>;

    /// Divide a tensor slice by a scalar producing a new `Vec` tensor.
    ///
    /// # Example
    ///
    /// ```
    /// use utils::soap::*;
    /// let a = vec![3,6,9,12];
    /// assert_eq!(Vector::new(vec![1,2,3,4]), a.view().into_tensor() / 3);
    /// ```
    fn div(self, other: T) -> Self::Output {
        Tensor {
            data: self.data.iter().map(|&a| a / other).collect::<Vec<_>>(),
        }
    }
}

//impl<T: Scalar> DivAssign<T> for Tensor<&mut [T]> {
//    /// Divide this tensor slice by a scalar.
//    ///
//    /// # Example
//    ///
//    /// ```
//    /// use utils::soap::*;
//    /// let mut a = vec![3,6,9,12];
//    /// let mut view = a.view_mut().into_tensor();
//    /// view /= 3;
//    /// assert_eq!(vec![1,2,3,4], a);
//    /// ```
//    fn div_assign(&mut self, other: T) {
//        for a in self.data.iter_mut() {
//            *a /= other;
//        }
//    }
//}

impl<T: Scalar> DivAssign<T> for Tensor<[T]> {
    /// Divide this tensor slice by a scalar.
    ///
    /// # Example
    ///
    /// ```
    /// use utils::soap::*;
    /// let mut a = vec![3,6,9,12];
    /// *a.as_mut_slice().as_mut_tensor() /= 3;
    /// assert_eq!(vec![1,2,3,4], a);
    /// ```
    fn div_assign(&mut self, other: T) {
        for a in self.data.iter_mut() {
            *a /= other;
        }
    }
}

/*
 * All additions and subtractions on 1-tensors represented by chunked vectors can be performed at the lowest level (flat)
 */

//impl<T: ?Sized, U, V: ?Sized> AddAssign<Tensor<U>> for Tensor<V>
//where
//    V: LocalGeneric + Set + for<'b> ViewMutIterator<'b, Item = &'b mut T>,
//    U: LocalGeneric + Set + for<'c> ViewIterator<'c, Item = &'c T>,
//    Tensor<T>: for<'a> AddAssign<&'a Tensor<T>>,
//{
//    fn add_assign(&mut self, other: Tensor<U>) {
//        for (out, b) in self.data.view_mut_iter().zip(other.data.view_iter()) {
//            let out_tensor = Tensor::as_mut(out);
//            *out_tensor += Tensor::as_ref(b);
//        }
//    }
//}
//
//impl<T: ?Sized, U, V: ?Sized> SubAssign<Tensor<U>> for Tensor<V>
//where
//    V: LocalGeneric + Set + for<'b> ViewMutIterator<'b, Item = &'b mut T>,
//    U: LocalGeneric + Set + for<'c> ViewIterator<'c, Item = &'c T>,
//    Tensor<T>: for<'a> SubAssign<&'a Tensor<T>>,
//{
//    fn sub_assign(&mut self, other: Tensor<U>) {
//        for (out, b) in self.data.view_mut_iter().zip(other.data.view_iter()) {
//            let out_tensor = Tensor::as_mut(out);
//            *out_tensor -= Tensor::as_ref(b);
//        }
//    }
//}

macro_rules! impl_chunked_tensor_arithmetic {
    ($chunked:ident, $chunks:ident) => {
        //impl<S, O> Add for Tensor<$chunked<S, O>>
        //where
        //    $chunked<S, O>: Set,
        //    S: IntoOwnedData,
        //    S: Add<Output = Tensor<S::OwnedData>>,
        //{
        //    type Output = Tensor<$chunked<S::OwnedData, O>>;
        //    fn add(self, other: Self) -> Self::Output {
        //        assert_eq!(self.data.len(), other.data.len());
        //        let $chunked { $chunks, data } = self.data;

        //        Tensor::new($chunked {
        //            $chunks,
        //            data: (Tensor::new(data) + Tensor::new(other.data.data)).data,
        //        })
        //    }
        //}

        //impl<S, O> Sub for Tensor<$chunked<S, O>>
        //where
        //    $chunked<S, O>: Set,
        //    S: IntoOwnedData,
        //    Tensor<S>: Sub<Output = Tensor<S::OwnedData>>,
        //{
        //    type Output = Tensor<$chunked<S::OwnedData, O>>;

        //    /// Subtract a tensor of chunked from another.
        //    fn sub(self, other: Self) -> Self::Output {
        //        assert_eq!(self.data.len(), other.data.len());
        //        let $chunked { $chunks, data } = self.data;
        //        Tensor::new($chunked {
        //            $chunks,
        //            data: (Tensor::new(data) - Tensor::new(other.data.data)).data,
        //        })
        //    }
        //}

        /*
         * Scalar multiplication
         */

        //impl<S, O, T> Mul<T> for Tensor<$chunked<S, O>>
        //where
        //    T: Scalar,
        //    $chunked<S, O>: Set,
        //    S: IntoOwnedData,
        //    Tensor<S>: Mul<T, Output = Tensor<S::OwnedData>>,
        //{
        //    type Output = Tensor<$chunked<S::OwnedData, O>>;

        //    fn mul(self, other: T) -> Self::Output {
        //        let $chunked { $chunks, data } = self.data;
        //        Tensor::new($chunked {
        //            $chunks,
        //            data: (Tensor::new(data) * other).data,
        //        })
        //    }
        //}
        /*
         * Scalar division
         */

        impl<S, O, T, D> Div<T> for Tensor<$chunked<S, O>>
        where
            S: IntoOwnedData<OwnedData = D>,
            S: Div<T, Output = D>,
        {
            type Output = Tensor<$chunked<D, O>>;

            fn div(self, other: T) -> Self::Output {
                let $chunked { $chunks, data } = self.data;
                Tensor {
                    data: $chunked {
                        $chunks,
                        data: data / other,
                    },
                }
            }
        }

        /*
         * Mul/Div assign variants of the above operators.
         */

        impl<S, O, T> MulAssign<T> for Tensor<$chunked<S, O>>
        where
            S: MulAssign<T>,
        {
            fn mul_assign(&mut self, other: T) {
                self.data.data *= other;
            }
        }

        impl<S, O, T> DivAssign<T> for Tensor<$chunked<S, O>>
        where
            S: DivAssign<T>,
        {
            fn div_assign(&mut self, other: T) {
                self.data.data /= other;
            }
        }
    };
}

impl_chunked_tensor_arithmetic!(Chunked, chunks);
impl_chunked_tensor_arithmetic!(UniChunked, chunk_size);

/*
 * Tensor norms
 */

pub enum LpNorm {
    P(i32),
    Inf,
}

// TODO: Split this trait into one that works for integers.
pub trait Norm<T> {
    fn lp_norm(&self, norm: LpNorm) -> T
    where
        T: Float;
    fn norm_squared(&self) -> T;
    fn norm(&self) -> T
    where
        T: Float,
    {
        self.norm_squared().sqrt()
    }
}

impl<S: ?Sized, T> Norm<T> for Tensor<S>
where
    T: Scalar,
    S: for<'a> AtomIterator<'a, Item = &'a T>,
    T: num_traits::FromPrimitive,
{
    fn lp_norm(&self, norm: LpNorm) -> T
    where
        T: Float,
    {
        match norm {
            LpNorm::P(p) => self
                .data
                .atom_iter()
                .map(|&x| x.abs().powi(p))
                .sum::<T>()
                .powf(T::one() / T::from_i32(p).expect("Failed to convert integer to flaot type.")),
            LpNorm::Inf => self
                .data
                .atom_iter()
                .map(|&x| x.abs())
                .max_by(|x, y| {
                    x.partial_cmp(y)
                        .expect("Detected NaN when computing Inf-norm.")
                })
                .unwrap_or(T::zero()),
        }
    }
    fn norm_squared(&self) -> T {
        self.data.atom_iter().map(|&x| x * x).sum::<T>()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    //#[test]
    //fn tensor_chunked() {
    //    let offsets = [0, 3, 4];
    //    let mut a = Chunked::from_offsets(&offsets[..], vec![1, 2, 3, 4]);
    //    let b = Chunked::from_offsets(&offsets[..], vec![5, 6, 7, 8]);

    //    // Add
    //    let res = Chunked::from_offsets(&offsets[..], vec![6, 8, 10, 12]);
    //    assert_eq!(
    //        Tensor::new(res.clone()),
    //        Tensor::new(a.view()) + Tensor::new(b.view())
    //    );

    //    // AddAssign
    //    let mut tensor_a = Tensor::new(a.view_mut());
    //    tensor_a += Tensor::new(b.view());
    //    assert_eq!(res.view(), a.view());

    //    // MulAssign
    //    let res = Chunked::from_offsets(&offsets[..], vec![600, 800, 1000, 1200]);
    //    let mut tensor_a = Tensor::new(a.view_mut());
    //    tensor_a *= 100;
    //    assert_eq!(res.view(), a.view());

    //    // SubAssign
    //    let res = Chunked::from_offsets(&offsets[..], vec![595, 794, 993, 1192]);
    //    let mut tensor_a = Tensor::new(a.view_mut());
    //    SubAssign::sub_assign(&mut tensor_a, Tensor::new(b.view()));
    //    assert_eq!(res.view(), a.view());
    //}

    //#[test]
    //fn tensor_uni_chunked() {
    //    let a = Chunked2::from_flat(vec![1, 2, 3, 4]);
    //    let b = Chunked2::from_flat(vec![5, 6, 7, 8]);
    //    assert_eq!(
    //        Tensor::new(Chunked2::from_flat(vec![6, 8, 10, 12])),
    //        Tensor::new(a.view()) + Tensor::new(b.view())
    //    );
    //}

    // #[test]
    // fn tensor_subset_sub_assign() {
    //     let a = Subset::from_unique_ordered_indices(
    //         vec![1, 3],
    //         Chunked2::from_flat(vec![1, 2, 3, 4, 5, 6, 7, 8]),
    //     );
    //     let mut b = Chunked2::from_flat(vec![9, 10, 13, 14]);
    //     let mut b_tensor = Tensor::new(b.view_mut());
    //     let a_tensor = Tensor::new(a.view());
    //     SubAssign::sub_assign(&mut b_tensor, a_tensor);
    //     assert_eq!(b.view().at(0), &[6, 6]);
    //     assert_eq!(b.view().at(1), &[6, 6]);
    // }

    // #[test]
    // fn tensor_subset_add_assign() {
    //     let a = Subset::from_unique_ordered_indices(
    //         vec![1, 3],
    //         Chunked2::from_flat(vec![1, 2, 3, 4, 5, 6, 7, 8]),
    //     );
    //     let mut b = Chunked2::from_flat(vec![9, 10, 13, 14]);
    //     *b.as_mut_tensor() += Tensor::new(a.view());
    //     assert_eq!(b.view().at(0), &[12, 14]);
    //     assert_eq!(b.view().at(1), &[20, 22]);
    // }

    // This test demonstrates the different ways to use tensors for assignment ops like AddAssign.
    #[test]
    fn tensor_assign_ops() {
        let mut v0 = vec![1, 2, 3, 4];
        let v1 = vec![2, 3, 4, 5];

        // With RHS being a tensor reference:
        let rhs = v1.as_slice().as_tensor();

        // As transient mutable tensor reference.
        *v0.view_mut().as_mut_tensor() += rhs;
        assert_eq!(v0, vec![3, 5, 7, 9]);

        // As a persistent owned tensor object.
        let t0 = v0.as_mut_slice().into_tensor();
        *t0 += rhs;
        assert_eq!(&t0.data, &[5, 8, 11, 14]);

        // With RHS being a persistent owned tensor object.
        let t1 = v1.as_slice().into_tensor();
        *t0 += t1;
        assert_eq!(&t0.data, &[7, 11, 15, 19]);

        *v0.view_mut().as_mut_tensor() += t1;
        assert_eq!(v0, vec![9, 14, 19, 24]);
    }

    #[test]
    fn tensor_add() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        let res = Vector::new(vec![6, 8, 10, 12]);
        assert_eq!(res, a.view().into_tensor() + b.view().into_tensor());
        assert_eq!(res, a.view().as_tensor() + b.view().into_tensor());
        assert_eq!(res, a.view().into_tensor() + b.view().as_tensor());
        assert_eq!(res, a.view().as_tensor() + b.view().as_tensor());
    }

    #[test]
    fn tensor_norm() {
        let a = vec![1, 2, 3, 4];
        assert_eq!(Norm::norm_squared(a.as_tensor()), 30);
        assert_eq!(Vector::new(a).norm_squared(), 30);

        let f = vec![1.0, 2.0, 3.0, 1.0, 4.0, 2.0, 1.0];
        assert_eq!(f.as_tensor().norm(), 6.0);
        assert_eq!(Vector::new(f.clone()).norm(), 6.0);

        assert_eq!(f.as_tensor().lp_norm(LpNorm::P(2)), 6.0);
        assert_eq!(f.as_tensor().lp_norm(LpNorm::P(1)), 14.0);
        assert_eq!(f.as_tensor().lp_norm(LpNorm::Inf), 4.0);
    }
}
