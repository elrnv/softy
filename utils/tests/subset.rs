/// This suite of tests checks behaviour for standard subsets.
use utils::soap::*;

#[test]
fn subset_index() {
    let v: Vec<usize> = (1..=15).collect();
    let subset = Subset::from_indices(vec![1, 3, 4], v);
    assert_eq!(2, subset[0]);
    assert_eq!(4, subset[1]);
    assert_eq!(5, subset[2]);

    let subset = subset.view();
    assert_eq!(Some(&2), subset.get(0));
    assert_eq!(Some(&4), subset.get(1));
    assert_eq!(Some(&5), subset.get(2));
    assert_eq!(&2, subset.at(0));
    assert_eq!(&4, subset.at(1));
    assert_eq!(&5, subset.at(2));
}

#[test]
fn subset_of_subsets_of_chunked_iter() {
    let set = Chunked2::from_flat(vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
    let subset = Subset::from_unique_ordered_indices(vec![1, 3, 5], set);
    let subsubset = Subset::from_unique_ordered_indices(vec![0, 2], subset);
    let mut iter = subsubset.iter();
    assert_eq!(Some(&[3, 4]), iter.next());
    assert_eq!(Some(&[11, 12]), iter.next());
    assert_eq!(None, iter.next());
}
