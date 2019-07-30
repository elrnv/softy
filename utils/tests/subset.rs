/// This suite of tests checks behaviour for standard subsets.
use utils::soap::*;

#[test]
fn subset_index() {
    let v: Vec<usize> = (1..=15).collect();
    let subset = Subset::from_indices(vec![1, 3, 4], v);
    assert_eq!(2, subset[0]);
    assert_eq!(4, subset[1]);
    assert_eq!(5, subset[2]);
    assert_eq!(Some(&2), subset.get(0));
    assert_eq!(Some(&4), subset.get(1));
    assert_eq!(Some(&5), subset.get(2));
    assert_eq!(&2, subset.at(0));
    assert_eq!(&4, subset.at(1));
    assert_eq!(&5, subset.at(2));
}
