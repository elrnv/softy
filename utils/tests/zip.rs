use utils::zip;

#[test]
fn zip() {
    let va = vec![1, 2, 3, 4];
    let vb = vec![4, 2, 1, 0];
    let vc = vec![(1, 4), (2, 2), (3, 1), (4, 0)];
    for (a, b, c) in zip!(va.into_iter(), vb.into_iter(), vc.into_iter()) {
        assert_eq!((a, b), c)
    }
}
