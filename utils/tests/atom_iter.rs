/// This suite of tests checks behaviour for standard subsets.
use utils::soap::*;

#[test]
fn vec() {
    let v: Vec<usize> = (1..=15).collect();
    for (a, b) in v.atom_iter().zip(v.iter()) {
        assert_eq!(a,b);
    }
}

#[test]
fn unichunked() {
    let v: Vec<usize> = (1..=15).collect();
    let set = Chunked3::from_flat(v.clone());
    for (a, b) in set.atom_iter().zip(v.iter()) {
        assert_eq!(a, b);
    }
    for (a, b) in set.view().atom_iter().zip(v.iter()) {
        assert_eq!(a, b);
    }
}
