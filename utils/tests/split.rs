/// This suite of tests checks splitting behaviour. This crate contains a number of traits responsible for splitting up
use utils::soap::*;

#[test]
fn split_prefix() {
    let s = UniChunked::<_, num::U2>::from_flat(vec![0, 1, 2, 3, 4, 5]);
    let (first, rest) = SplitPrefix::<num::U1>::split_prefix(s).unwrap();
    assert_eq!(first, [0, 1]);
    assert_eq!(rest, UniChunked::<_, num::U2>::from_flat(vec![2, 3, 4, 5]));
}
