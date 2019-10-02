/// This suite of tests checks various ways to index collections using the `Get` trait.
use utils::soap::*;

/// Get a subview from this `UniChunked` collection according to the given
/// range. If the range is a single index, then a single chunk is returned
/// instead.
#[test]
fn get_unichunked() {
    let v = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
    let s = Chunked3::from_flat(v.view());

    assert_eq!(s.get(2), Some(&[7, 8, 9])); // Single index
    assert_eq!(s.get(2), Some(&s[2]));

    let r = s.get(1..3).unwrap(); // Range
    let mut iter = r.iter();
    assert_eq!(Some(&[4, 5, 6]), iter.next());
    assert_eq!(Some(&[7, 8, 9]), iter.next());
    assert_eq!(None, iter.next());

    let r = s.get(2..).unwrap(); // RangeFrom
    let mut iter = r.iter();
    assert_eq!(Some(&[7, 8, 9]), iter.next());
    assert_eq!(Some(&[10, 11, 12]), iter.next());
    assert_eq!(None, iter.next());

    let r = s.get(..2).unwrap(); // RangeTo
    let mut iter = r.iter();
    assert_eq!(Some(&[1, 2, 3]), iter.next());
    assert_eq!(Some(&[4, 5, 6]), iter.next());
    assert_eq!(None, iter.next());

    assert_eq!(s.view(), s.get(..).unwrap()); // RangeFull
    assert_eq!(s.view(), s.view().get(..).unwrap());

    let r = s.get(1..=2).unwrap(); // RangeInclusive
    let mut iter = r.iter();
    assert_eq!(Some(&[4, 5, 6]), iter.next());
    assert_eq!(Some(&[7, 8, 9]), iter.next());
    assert_eq!(None, iter.next());

    let r = s.get(..=1).unwrap(); // RangeToInclusive
    let mut iter = r.iter();
    assert_eq!(Some(&[1, 2, 3]), iter.next());
    assert_eq!(Some(&[4, 5, 6]), iter.next());
    assert_eq!(None, iter.next());
}

#[test]
fn isolate_unichunked() {
    let v = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
    let s = Chunked3::from_flat(v.view());

    assert_eq!(s.view().try_isolate(2), Some(&[7, 8, 9])); // Single index
    assert_eq!(s.view().try_isolate(2), Some(&s[2]));

    let r = s.view().isolate(1..3); // Range
    let mut iter = r.iter();
    assert_eq!(Some(&[4, 5, 6]), iter.next());
    assert_eq!(Some(&[7, 8, 9]), iter.next());
    assert_eq!(None, iter.next());

    let r = s.view().isolate(2..); // RangeFrom
    let mut iter = r.iter();
    assert_eq!(Some(&[7, 8, 9]), iter.next());
    assert_eq!(Some(&[10, 11, 12]), iter.next());
    assert_eq!(None, iter.next());

    let r = s.view().isolate(..2); // RangeTo
    let mut iter = r.iter();
    assert_eq!(Some(&[1, 2, 3]), iter.next());
    assert_eq!(Some(&[4, 5, 6]), iter.next());
    assert_eq!(None, iter.next());

    assert_eq!(s.view(), s.view().isolate(..)); // RangeFull

    let r = s.view().isolate(1..=2); // RangeInclusive
    let mut iter = r.iter();
    assert_eq!(Some(&[4, 5, 6]), iter.next());
    assert_eq!(Some(&[7, 8, 9]), iter.next());
    assert_eq!(None, iter.next());

    let r = s.view().isolate(..=1); // RangeToInclusive
    let mut iter = r.iter();
    assert_eq!(Some(&[1, 2, 3]), iter.next());
    assert_eq!(Some(&[4, 5, 6]), iter.next());
    assert_eq!(None, iter.next());
}

#[test]
fn isolate_nested_unichunked() {
    let v = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
    let s = Chunked3::from_flat(Chunked2::from_flat(v.view()));

    assert_eq!(s.view().isolate(0).isolate(0), &[1, 2]);
    assert_eq!(s.view().isolate(0).isolate(1), &[3, 4]);
    assert_eq!(s.view().isolate(0).isolate(2), &[5, 6]);

    assert_eq!(s.view().isolate(1).isolate(0), &[7, 8]);
    assert_eq!(s.view().isolate(1).isolate(1), &[9, 10]);
    assert_eq!(s.view().isolate(1).isolate(2), &[11, 12]);
}

/// Get a subview from this `Chunked` collection according to the given
/// range. If the range is a single index, then a single chunk is returned
/// instead.
#[test]
fn get_chunked() {
    let v = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11];
    let s = Chunked::from_offsets(vec![0, 3, 4, 6, 9, 11], v.clone());
    let s = s.view();

    assert_eq!(s.get(2), Some(&s[2])); // Single index

    let r = s.get(1..3).unwrap(); // Range
    let mut iter = r.iter();
    assert_eq!(Some(&[4][..]), iter.next());
    assert_eq!(Some(&[5, 6][..]), iter.next());
    assert_eq!(None, iter.next());

    let r = s.get(3..).unwrap(); // RangeFrom
    let mut iter = r.iter();
    assert_eq!(Some(&[7, 8, 9][..]), iter.next());
    assert_eq!(Some(&[10, 11][..]), iter.next());
    assert_eq!(None, iter.next());

    let r = s.get(..2).unwrap(); // RangeTo
    let mut iter = r.iter();
    assert_eq!(Some(&[1, 2, 3][..]), iter.next());
    assert_eq!(Some(&[4][..]), iter.next());
    assert_eq!(None, iter.next());

    assert_eq!(s.view(), s.get(..).unwrap()); // RangeFull
    assert_eq!(s.view(), s.view().get(..).unwrap());

    let r = s.get(1..=2).unwrap(); // RangeInclusive
    let mut iter = r.iter();
    assert_eq!(Some(&[4][..]), iter.next());
    assert_eq!(Some(&[5, 6][..]), iter.next());
    assert_eq!(None, iter.next());

    let r = s.get(..=1).unwrap(); // RangeToInclusive
    let mut iter = r.iter();
    assert_eq!(Some(&[1, 2, 3][..]), iter.next());
    assert_eq!(Some(&[4][..]), iter.next());
    assert_eq!(None, iter.next());
}

#[test]
fn get_selected() {
    let v = vec![1, 2, 3, 4, 5];
    let selection = Select::new(vec![0, 0, 4], v.as_slice());
    assert_eq!((0, &1), selection.get(1).unwrap());
}

#[test]
fn get_subset() {
    let mut v = vec![1, 2, 3, 4, 5];
    let subset = Subset::from_indices(vec![0, 2, 4], v.as_mut_slice());
    assert_eq!(&3, subset.view().get(1).unwrap());
}
