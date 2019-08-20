/// This suite of tests checks splitting behaviour. This crate contains a number of traits responsible for splitting up
use utils::soap::*;

#[test]
fn split_prefix() {
    let s = Chunked2::from_flat(vec![0, 1, 2, 3, 4, 5]);
    let (prefix, rest) = SplitPrefix::<consts::U1>::split_prefix(s).unwrap();
    assert_eq!(prefix, Chunked2::from_flat([0, 1]));
    assert_eq!(rest, Chunked2::from_flat(vec![2, 3, 4, 5]));
}

#[test]
fn split_first() {
    let s = Chunked2::from_flat(vec![0, 1, 2, 3, 4, 5]);
    let (first, rest) = SplitFirst::split_first(s).unwrap();
    assert_eq!(first, [0, 1]);
    assert_eq!(rest, Chunked2::from_flat(vec![2, 3, 4, 5]));
}
