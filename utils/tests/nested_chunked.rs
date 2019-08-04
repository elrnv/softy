/// This suite of tests checks that chunked collections compose work as expected.
use utils::soap::*;

#[test]
fn chunked_unichunked_iter() {
    let v: Vec<usize> = (1..13).collect();
    let uni = Chunked3::from_flat(v);
    let chunked = Chunked::from_offsets(vec![0, 1, 4], uni);

    // Simple iteration
    let mut chunked_iter = chunked.iter();
    let uni = chunked_iter.next().unwrap();
    let mut uni_iter = uni.iter();
    assert_eq!(Some(&[1, 2, 3]), uni_iter.next());
    assert_eq!(None, uni_iter.next());
    let uni = chunked_iter.next().unwrap();
    let mut uni_iter = uni.iter();
    assert_eq!(Some(&[4, 5, 6]), uni_iter.next());
    assert_eq!(Some(&[7, 8, 9]), uni_iter.next());
    assert_eq!(Some(&[10, 11, 12]), uni_iter.next());
    assert_eq!(None, uni_iter.next());
}

#[test]
fn chunked_chunked_mut() {
    let v: Vec<usize> = (1..=10).collect();
    let chunked = Chunked::from_offsets(vec![0, 3, 5, 8, 10], v);
    let mut chunked = Chunked::from_offsets(vec![0, 1, 4], chunked);

    chunked.view_mut().at_mut(1).at_mut(1)[1] = 100;
    assert_eq!(chunked.view().at(1).at(1)[1], 100);

    // Simple iteration
    let mut iter1 = chunked.iter();
    let chunked0 = iter1.next().unwrap();
    let mut iter0 = chunked0.iter();
    assert_eq!(Some(&[1, 2, 3][..]), iter0.next());
    assert_eq!(None, iter0.next());
    let chunked0 = iter1.next().unwrap();
    let mut iter0 = chunked0.iter();
    assert_eq!(Some(&[4, 5][..]), iter0.next());
    assert_eq!(Some(&[6, 100, 8][..]), iter0.next());
    assert_eq!(Some(&[9, 10][..]), iter0.next());
    assert_eq!(None, iter0.next());
}

#[test]
fn chunked_chunked_chunked_mut() {
    let v: Vec<usize> = (1..=11).collect();
    let chunked = Chunked::from_offsets(vec![1, 4, 6, 8, 10, 11, 12, 12], v);
    let chunked = Chunked::from_offsets(vec![1, 3, 5, 6, 6, 8], chunked);
    let mut chunked = Chunked::from_offsets(vec![1, 3, 6], chunked);

    chunked.view_mut().at_mut(0).at_mut(1).at_mut(1)[1] = 100;
    assert_eq!(chunked.view().at(0).at(1).at(1)[1], 100);

    // Simple iteration
    let mut iter2 = chunked.iter();

    let chunked1 = iter2.next().unwrap();
    let mut iter1 = chunked1.iter();

    let chunked0 = iter1.next().unwrap();
    let mut iter0 = chunked0.iter();
    assert_eq!(Some(&[1, 2, 3][..]), iter0.next());
    assert_eq!(Some(&[4, 5][..]), iter0.next());
    assert_eq!(None, iter0.next());
    let chunked0 = iter1.next().unwrap();
    let mut iter0 = chunked0.iter();
    assert_eq!(Some(&[6, 7][..]), iter0.next());
    assert_eq!(Some(&[8, 100][..]), iter0.next());
    assert_eq!(None, iter0.next());
    assert_eq!(None, iter1.next());

    let chunked1 = iter2.next().unwrap();
    let mut iter1 = chunked1.iter();

    let chunked0 = iter1.next().unwrap();
    let mut iter0 = chunked0.iter();
    assert_eq!(Some(&[10][..]), iter0.next());
    assert_eq!(None, iter0.next());
    let chunked0 = iter1.next().unwrap();
    let mut iter0 = chunked0.iter();
    assert_eq!(None, iter0.next());
    let chunked0 = iter1.next().unwrap();
    let mut iter0 = chunked0.iter();
    assert_eq!(Some(&[11][..]), iter0.next());
    assert_eq!(Some(&[][..]), iter0.next());
    assert_eq!(None, iter0.next());
    assert_eq!(None, iter1.next());
    assert_eq!(None, iter2.next());
}

#[test]
fn chunked_unichunked_push() {
    let mut chunked = Chunked::<Chunked2<Vec<usize>>>::new();

    for i in 0..4 {
        let data: Vec<usize> = (i..i + 4).collect();
        let uni = Chunked2::from_flat(data);
        chunked.push(uni.into());
    }

    assert_eq!(
        chunked.into_flat(),
        vec![0, 1, 2, 3, 1, 2, 3, 4, 2, 3, 4, 5, 3, 4, 5, 6]
    );
}

//#[test]
//fn unichunked_chunked() {
//    let v: Vec<usize> = (1..13).collect();
//    let chunked = Chunked::from_offsets(vec![0, 1, 3, 6, 8, 10, 13], v);
//    let uni = Chunked3::from_flat(chunked);
//    let mut uni_iter = uni.iter();
//    let chunked = uni_iter.next().unwrap();
//    let mut chunked_iter = uni.iter();
//    assert_eq!(Some(&[1][..]), chunked_iter.next());
//    assert_eq!(Some(&[1, 2][..]), chunked_iter.next());
//    assert_eq!(Some(&[3, 4, 5][..]), chunked_iter.next());
//    assert_eq!(None, chunked_iter.next());
//    let chunked = uni_iter.next().unwrap();
//    let mut chunked_iter = uni.iter();
//    assert_eq!(Some(&[6, 7][..]), chunked_iter.next());
//    assert_eq!(Some(&[8, 9][..]), chunked_iter.next());
//    assert_eq!(Some(&[10, 11, 12][..]), chunked_iter.next());
//    assert_eq!(None, chunked_iter.next());
//}
