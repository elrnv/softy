/// This suite of tests checks that subsets and chunked sets can be composed.
use utils::soap::*;

#[test]
fn chunked_index() {
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
fn chunked_isolate_mut() {
    let mut v = vec![1, 2, 3, 4, 0, 0, 7, 8, 9, 10, 11];
    let mut s = Chunked::from_offsets(vec![0, 3, 4, 6, 9, 11], v.view_mut());

    s.view_mut()
        .try_isolate(2)
        .unwrap()
        .copy_from_slice(&[5, 6]); // Single index
    assert_eq!(
        *s.data(),
        vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11].as_slice()
    );
}

#[test]
fn chunked_subset() {
    let v: Vec<usize> = (1..12).collect();
    let subset = Subset::from_indices(vec![1, 3, 5, 7, 9, 10], v);
    let chunked = Chunked::from_offsets(vec![0, 2, 6], subset);
    let mut iter = chunked.iter();
    let sub = iter.next().unwrap();
    let mut sub_iter = sub.iter();
    assert_eq!(Some(&2), sub_iter.next());
    assert_eq!(Some(&4), sub_iter.next());
    assert_eq!(None, sub_iter.next());
    let sub = iter.next().unwrap();
    let mut sub_iter = sub.iter();
    assert_eq!(Some(&6), sub_iter.next());
    assert_eq!(Some(&8), sub_iter.next());
    assert_eq!(Some(&10), sub_iter.next());
    assert_eq!(Some(&11), sub_iter.next());
    assert_eq!(None, sub_iter.next());
}

#[test]
fn subset_chunked() {
    let v: Vec<usize> = (1..12).collect();
    let chunked = Chunked::from_offsets(vec![0, 2, 4, 7, 10, 11], v);
    let subset = Subset::from_indices(vec![1, 3, 4], chunked);
    let subset = subset.view();
    let mut iter = subset.iter();
    assert_eq!(Some(&[3, 4][..]), iter.next());
    assert_eq!(Some(&[8, 9, 10][..]), iter.next());
    assert_eq!(Some(&[11][..]), iter.next());
    assert_eq!(None, iter.next());
}

#[test]
fn subset_unichunked_clone_into() {
    let v: Vec<usize> = (1..=15).collect();
    let uni = Chunked3::from_flat(v);
    let subset = Subset::from_indices(vec![1, 3, 4], uni);
    let subset = subset.view();
    let mut other = Chunked3::from_flat(vec![0; 9]);
    subset.clone_into_other(&mut other);
    assert_eq!(other.into_inner(), vec![4, 5, 6, 10, 11, 12, 13, 14, 15]);
}

#[test]
fn subset_unichunked_const() {
    let v: Vec<usize> = (1..=15).collect();
    let uni = Chunked3::from_flat(v.clone());
    let subset = Subset::from_indices(vec![1, 3, 4], uni.view());
    let mut iter = subset.iter();
    assert_eq!(Some(&[4, 5, 6]), iter.next());
    assert_eq!(Some(&[10, 11, 12]), iter.next());
    assert_eq!(Some(&[13, 14, 15]), iter.next());
    assert_eq!(None, iter.next());

    // Verify that this works for views as well.
    let uni = Chunked3::from_flat(v.as_slice());
    let subset = Subset::from_unique_ordered_indices(&[1, 3, 4][..], uni);
    let mut iter = subset.iter();
    assert_eq!(Some(&[4, 5, 6]), iter.next());
    assert_eq!(Some(&[10, 11, 12]), iter.next());
    assert_eq!(Some(&[13, 14, 15]), iter.next());
    assert_eq!(None, iter.next());
}

#[test]
fn subset_unichunked_mut() {
    use utils::soap::*;
    let v: Vec<usize> = (1..=15).collect();
    let mut uni = Chunked3::from_flat(v.clone());
    let mut subset = Subset::from_indices(vec![0, 2, 4], uni.view_mut());
    {
        let subset_view = subset.view();
        let mut subset_iter = subset_view.iter();
        assert_eq!(Some(&[1, 2, 3]), subset_iter.next());
        assert_eq!(Some(&[7, 8, 9]), subset_iter.next());
        assert_eq!(Some(&[13, 14, 15]), subset_iter.next());
        assert_eq!(None, subset_iter.next());
    }
    *subset.view_mut().isolate(1) = [0; 3];
    assert_eq!(&[0, 0, 0], subset.view().at(1));

    // Verify that this works for views as well.
    let mut subset = Subset::from_unique_ordered_indices(&[0, 2, 4][..], uni.view_mut());
    {
        let subset_view = subset.view();
        let mut subset_iter = subset_view.iter();
        assert_eq!(Some(&[1, 2, 3]), subset_iter.next());
        assert_eq!(Some(&[0; 3]), subset_iter.next());
        assert_eq!(Some(&[13, 14, 15]), subset_iter.next());
        assert_eq!(None, subset_iter.next());
    }

    // Try with iter_mut this time setting all elements to 1.
    for elem in subset.iter_mut() {
        *elem = [1; 3];
    }

    for i in 0..3 {
        assert_eq!(&[1, 1, 1], subset.view().at(i));
    }
}

#[test]
fn subset_unichunked_index() {
    let v: Vec<usize> = (1..=15).collect();
    let uni = Chunked3::from_flat(v);
    let subset = Subset::from_indices(vec![1, 3, 4], uni.view());
    assert_eq!(Some(&[4, 5, 6]), subset.get(0));
    assert_eq!(Some(&[10, 11, 12]), subset.get(1));
    assert_eq!(Some(&[13, 14, 15]), subset.get(2));
    assert_eq!(&[4, 5, 6], subset.at(0));
    assert_eq!(&[10, 11, 12], subset.at(1));
    assert_eq!(&[13, 14, 15], subset.at(2));
    assert_eq!([4, 5, 6], subset[0]);
    assert_eq!([10, 11, 12], subset[1]);
    assert_eq!([13, 14, 15], subset[2]);

    let v: Vec<usize> = (1..=15).collect();
    let mut uni = Chunked3::from_flat(v);
    let indices = vec![1, 3, 4];
    let mut subset = Subset::from_unique_ordered_indices(indices.as_slice(), uni.view_mut());
    subset[1] = [0, 0, 0];
    let subset_view = subset.view();
    assert_eq!(Some(&[4, 5, 6]), subset_view.get(0));
    assert_eq!(Some(&[0, 0, 0]), subset_view.get(1));
    assert_eq!(Some(&[13, 14, 15]), subset_view.get(2));
    assert_eq!(&[4, 5, 6], subset_view.at(0));
    assert_eq!(&[0, 0, 0], subset_view.at(1));
    assert_eq!(&[13, 14, 15], subset_view.at(2));
    assert_eq!([4, 5, 6], subset[0]);
    assert_eq!([0, 0, 0], subset[1]);
    assert_eq!([13, 14, 15], subset[2]);
}

fn get<'a>(x: ChunkedView<'a, &'a [usize]>) -> &'a [usize] {
    x.at(1)
}

#[test]
fn chunked_return_get_result_from_fn() {
    let v: Vec<_> = (1..=10).collect();
    let s = Chunked::from_sizes(vec![1, 4, 5], v);
    assert_eq!(get(s.view()), &[2, 3, 4, 5]);
}

// Test that we can create and return a subset of a substructure of a nested chunked set
fn subset<'a, 'b>(
    indices: &'a [usize],
    x: ChunkedView<'b, ChunkedView<'b, Chunked3<&'b [usize]>>>,
) -> SubsetView<'a, Chunked3<&'b [usize]>> {
    Subset::from_unique_ordered_indices(indices, x.at(1).at(1))
}

#[test]
fn subset_nested_chunked() {
    let v: Vec<_> = (1..=45).collect();
    let u = Chunked3::from_flat(v);
    let c1 = Chunked::from_sizes(vec![1, 4, 5, 5], u);
    let c2 = Chunked::from_sizes(vec![1, 2, 1], c1);
    let subset = subset(&[0, 1], c2.view());
    assert_eq!(subset[0], [16, 17, 18]);
    assert_eq!(subset[1], [19, 20, 21]);
}
