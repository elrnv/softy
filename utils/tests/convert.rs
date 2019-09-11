/// This suite of tests checks splitting behaviour. This crate contains a number of traits responsible for splitting up
use utils::soap::*;

#[test]
fn unichunked_to_arrays() {
    let mut array = [1, 2, 3, 4, 5, 6, 7, 8, 9];
    // Convert to an array of arrays.
    let v_result = Chunked3::from_flat(&array).into_arrays();
    assert_eq!(&[[1, 2, 3], [4, 5, 6], [7, 8, 9]], v_result);

    // Same as above but for mutable array borrows.
    let v_result = Chunked3::from_flat(&mut array).into_arrays();
    assert_eq!(&mut [[1, 2, 3], [4, 5, 6], [7, 8, 9]], v_result);
}
