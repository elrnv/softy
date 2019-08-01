/// This suite of tests checks that subsets and chunked sets can be composed.
use utils::soap::*;

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
    let mut other = Chunked3::from_flat(vec![0; 9]);
    subset.clone_into_other(&mut other);
    assert_eq!(other.into_inner(), vec![4, 5, 6, 10, 11, 12, 13, 14, 15]);
}

#[test]
fn subset_unichunked_const() {
    let v: Vec<usize> = (1..=15).collect();
    let uni = Chunked3::from_flat(v.clone());
    let subset = Subset::from_indices(vec![1, 3, 4], uni);
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
        let mut subset_iter = subset.iter();
        assert_eq!(Some(&[1, 2, 3]), subset_iter.next());
        assert_eq!(Some(&[7, 8, 9]), subset_iter.next());
        assert_eq!(Some(&[13, 14, 15]), subset_iter.next());
        assert_eq!(None, subset_iter.next());
    }
    *subset.at_mut(1) = [0; 3];
    assert_eq!(&[0, 0, 0], subset.at(1));

    // Verify that this works for views as well.
    let mut subset = Subset::from_unique_ordered_indices(&[0, 2, 4][..], uni.view_mut());
    {
        let mut subset_iter = subset.iter();
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
        assert_eq!(&[1, 1, 1], subset.at(i));
    }
}

#[test]
fn subset_unichunked_index() {
    let v: Vec<usize> = (1..=15).collect();
    let uni = Chunked3::from_flat(v);
    let subset = Subset::from_indices(vec![1, 3, 4], uni);
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
    let uni = Chunked3::from_flat(v);
    let indices = vec![1, 3, 4];
    let mut subset = Subset::from_unique_ordered_indices(indices.as_slice(), uni);
    subset[1] = [0, 0, 0];
    assert_eq!(Some(&[4, 5, 6]), subset.get(0));
    assert_eq!(Some(&[0, 0, 0]), subset.get(1));
    assert_eq!(Some(&[13, 14, 15]), subset.get(2));
    assert_eq!(&[4, 5, 6], subset.at(0));
    assert_eq!(&[0, 0, 0], subset.at(1));
    assert_eq!(&[13, 14, 15], subset.at(2));
    assert_eq!([4, 5, 6], subset[0]);
    assert_eq!([0, 0, 0], subset[1]);
    assert_eq!([13, 14, 15], subset[2]);
}

// Test that we can create a subset of a substructure of a nested chunked set
fn subset<'a, 'b>(
    indices: &'a [usize],
    x: ChunkedView<'b, ChunkedView<'b, Chunked3<&'b [usize]>>>,
) -> SubsetView<'a, Chunked3<&'b [usize]>> {
    Subset::from_unique_ordered_indices(indices, x.at(1).at(1))
}

#[test]
fn subset_nested_chunked() {}
