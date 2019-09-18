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

    chunked.view_mut().isolate(1).isolate(1)[1] = 100;
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

    chunked.view_mut().isolate(0).isolate(1).isolate(1)[1] = 100;
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
        chunked.push(uni);
    }

    assert_eq!(
        chunked.into_flat(),
        vec![0, 1, 2, 3, 1, 2, 3, 4, 2, 3, 4, 5, 3, 4, 5, 6]
    );
}

#[test]
fn unichunked_unichunked() {
    let v: Vec<usize> = (1..=12).collect();
    let uni0 = Chunked2::from_flat(v);
    let uni1 = Chunked3::from_flat(uni0);
    let mut uni1_iter = uni1.iter();
    let uni0 = uni1_iter.next().unwrap();
    let mut uni0_iter = uni0.iter();
    assert_eq!(Some(&[1, 2]), uni0_iter.next());
    assert_eq!(Some(&[3, 4]), uni0_iter.next());
    assert_eq!(Some(&[5, 6]), uni0_iter.next());
    assert_eq!(None, uni0_iter.next());
    let uni0 = uni1_iter.next().unwrap();
    let mut uni0_iter = uni0.iter();
    assert_eq!(Some(&[7, 8]), uni0_iter.next());
    assert_eq!(Some(&[9, 10]), uni0_iter.next());
    assert_eq!(Some(&[11, 12]), uni0_iter.next());
    assert_eq!(None, uni0_iter.next());
}

#[test]
fn unichunked_unichunked_as_ref() {
    let v: Vec<usize> = (1..=12).collect();
    let uni0 = Chunked2::from_flat(v);
    let uni1 = Chunked3::from_flat(uni0);
    assert_eq!(
        &[[[1, 2], [3, 4], [5, 6]], [[7, 8], [9, 10], [11, 12]]],
        uni1.as_ref()
    );
}

#[test]
fn unichunked_chunked() {
    let v: Vec<usize> = (1..=12).collect();
    let chunked = Chunked::from_sizes(vec![2, 1, 2, 3, 3, 1], v);
    let uni = Chunked3::from_flat(chunked);
    let mut uni_iter = uni.iter();
    let chunked = uni_iter.next().unwrap();
    let mut chunked_iter = chunked.iter();
    assert_eq!(Some(&[1, 2][..]), chunked_iter.next());
    assert_eq!(Some(&[3][..]), chunked_iter.next());
    assert_eq!(Some(&[4, 5][..]), chunked_iter.next());
    assert_eq!(None, chunked_iter.next());
    let chunked = uni_iter.next().unwrap();
    let mut chunked_iter = chunked.iter();
    assert_eq!(Some(&[6, 7, 8][..]), chunked_iter.next());
    assert_eq!(Some(&[9, 10, 11][..]), chunked_iter.next());
    assert_eq!(Some(&[12][..]), chunked_iter.next());
    assert_eq!(None, chunked_iter.next());
}

#[test]
fn sparse_unichunked() {
    let v: Vec<usize> = (1..=6).collect();
    let uni = Chunked2::from_flat(v);
    let sparse = Sparse::from_dim(vec![0, 2, 3], 4, uni);
    let mut sparse_iter = sparse.iter();

    assert_eq!(Some((0, &[1, 2], 0)), sparse_iter.next());
    assert_eq!(Some((2, &[3, 4], 2)), sparse_iter.next());
    assert_eq!(Some((3, &[5, 6], 3)), sparse_iter.next());
    assert_eq!(None, sparse_iter.next());
}

#[test]
fn sparse_unichunked_mut() {
    let v: Vec<usize> = (1..=6).collect();
    let uni = Chunked2::from_flat(v);
    let mut sparse = Sparse::from_dim(vec![0, 2, 3], 4, uni);
    for (_, v3) in sparse.iter_mut() {
        for elem in v3.iter_mut() {
            *elem *= 10;
        }
    }

    let mut sparse_iter = sparse.iter();
    assert_eq!(Some((0, &[10, 20], 0)), sparse_iter.next());
    assert_eq!(Some((2, &[30, 40], 2)), sparse_iter.next());
    assert_eq!(Some((3, &[50, 60], 3)), sparse_iter.next());
    assert_eq!(None, sparse_iter.next());
}

#[test]
fn sparse_unichunked_unichunked() {
    let v: Vec<usize> = (1..=12).collect();
    let uni0 = Chunked2::from_flat(v);
    let uni1 = Chunked3::from_flat(uni0);
    let sparse = Sparse::from_dim(vec![0, 2], 3, uni1);
    let mut sparse_iter = sparse.iter();

    let (_, uni0, _) = sparse_iter.next().unwrap();
    let mut uni0_iter = uni0.iter();
    assert_eq!(Some(&[1, 2]), uni0_iter.next());
    assert_eq!(Some(&[3, 4]), uni0_iter.next());
    assert_eq!(Some(&[5, 6]), uni0_iter.next());
    assert_eq!(None, uni0_iter.next());
    let (_, uni0, _) = sparse_iter.next().unwrap();
    let mut uni0_iter = uni0.iter();
    assert_eq!(Some(&[7, 8]), uni0_iter.next());
    assert_eq!(Some(&[9, 10]), uni0_iter.next());
    assert_eq!(Some(&[11, 12]), uni0_iter.next());
    assert_eq!(None, uni0_iter.next());

    assert_eq!(None, sparse_iter.next());
}

#[test]
fn sparse_unichunked_unichunked_mut() {
    let v: Vec<usize> = (1..=12).collect();
    let uni0 = Chunked2::from_flat(v);
    let uni1 = Chunked3::from_flat(uni0);
    let mut sparse = Sparse::from_dim(vec![0, 2], 3, uni1);
    for (_, mut v3) in sparse.iter_mut() {
        for v2 in v3.iter_mut() {
            for elem in v2.iter_mut() {
                *elem *= 10;
            }
        }
    }

    let mut sparse_iter = sparse.iter();

    let (_, uni0, _) = sparse_iter.next().unwrap();
    let mut uni0_iter = uni0.iter();
    assert_eq!(Some(&[10, 20]), uni0_iter.next());
    assert_eq!(Some(&[30, 40]), uni0_iter.next());
    assert_eq!(Some(&[50, 60]), uni0_iter.next());
    assert_eq!(None, uni0_iter.next());
    let (_, uni0, _) = sparse_iter.next().unwrap();
    let mut uni0_iter = uni0.iter();
    assert_eq!(Some(&[70, 80]), uni0_iter.next());
    assert_eq!(Some(&[90, 100]), uni0_iter.next());
    assert_eq!(Some(&[110, 120]), uni0_iter.next());
    assert_eq!(None, uni0_iter.next());

    assert_eq!(None, sparse_iter.next());
}

#[test]
fn chunked_sparse_unichunked() {
    let v: Vec<usize> = (1..=12).collect();
    let uni = Chunked2::from_flat(v);
    let sparse = Sparse::from_dim(vec![0, 2, 5, 1, 2, 5], 6, uni);
    let chunked = Chunked::from_sizes(vec![2, 0, 1, 3], sparse);
    let mut chunked_iter = chunked.iter();

    let sparse = chunked_iter.next().unwrap();
    let mut sparse_iter = sparse.iter();
    assert_eq!(Some((0, &[1, 2], 0)), sparse_iter.next());
    assert_eq!(Some((2, &[3, 4], 2)), sparse_iter.next());
    assert_eq!(None, sparse_iter.next());

    let sparse = chunked_iter.next().unwrap();
    let mut sparse_iter = sparse.iter();
    assert_eq!(None, sparse_iter.next());

    let sparse = chunked_iter.next().unwrap();
    let mut sparse_iter = sparse.iter();
    assert_eq!(Some((5, &[5, 6], 5)), sparse_iter.next());
    assert_eq!(None, sparse_iter.next());

    let sparse = chunked_iter.next().unwrap();
    let mut sparse_iter = sparse.iter();
    assert_eq!(Some((1, &[7, 8], 1)), sparse_iter.next());
    assert_eq!(Some((2, &[9, 10], 2)), sparse_iter.next());
    assert_eq!(Some((5, &[11, 12], 5)), sparse_iter.next());
    assert_eq!(None, sparse_iter.next());

    assert_eq!(None, chunked_iter.next());
}

#[test]
fn unichunked_sparse_unichunked_mut() {
    let v: Vec<usize> = (1..=12).collect();
    let uni = Chunked2::from_flat(v);
    let sparse = Sparse::from_dim(vec![0, 2, 5, 1, 2, 5], 6, uni);
    let mut chunked = ChunkedN::from_flat_with_stride(sparse, 3);
    for mut sparse in chunked.iter_mut() {
        for (_, v3) in sparse.iter_mut() {
            for elem in v3.iter_mut() {
                *elem *= 10;
            }
        }
    }

    let mut chunked_iter = chunked.iter();

    let sparse = chunked_iter.next().unwrap();
    let mut sparse_iter = sparse.iter();
    assert_eq!(Some((0, &[10, 20], 0)), sparse_iter.next());
    assert_eq!(Some((2, &[30, 40], 2)), sparse_iter.next());
    assert_eq!(Some((5, &[50, 60], 5)), sparse_iter.next());
    assert_eq!(None, sparse_iter.next());

    let sparse = chunked_iter.next().unwrap();
    let mut sparse_iter = sparse.iter();
    assert_eq!(Some((1, &[70, 80], 1)), sparse_iter.next());
    assert_eq!(Some((2, &[90, 100], 2)), sparse_iter.next());
    assert_eq!(Some((5, &[110, 120], 5)), sparse_iter.next());
    assert_eq!(None, sparse_iter.next());

    assert_eq!(None, chunked_iter.next());
}

#[test]
fn chunked_sparse_unichunked_mut() {
    let v: Vec<usize> = (1..=12).collect();
    let uni = Chunked2::from_flat(v);
    let sparse = Sparse::from_dim(vec![0, 2, 5, 1, 2, 5], 6, uni);
    let mut chunked = Chunked::from_sizes(vec![2, 0, 1, 3], sparse);
    for mut sparse in chunked.iter_mut() {
        for (_, v2) in sparse.iter_mut() {
            for elem in v2.iter_mut() {
                *elem *= 10;
            }
        }
    }
    let mut chunked_iter = chunked.iter();

    let sparse = chunked_iter.next().unwrap();
    let mut sparse_iter = sparse.iter();
    assert_eq!(Some((0, &[10, 20], 0)), sparse_iter.next());
    assert_eq!(Some((2, &[30, 40], 2)), sparse_iter.next());
    assert_eq!(None, sparse_iter.next());

    let sparse = chunked_iter.next().unwrap();
    let mut sparse_iter = sparse.iter();
    assert_eq!(None, sparse_iter.next());

    let sparse = chunked_iter.next().unwrap();
    let mut sparse_iter = sparse.iter();
    assert_eq!(Some((5, &[50, 60], 5)), sparse_iter.next());
    assert_eq!(None, sparse_iter.next());

    let sparse = chunked_iter.next().unwrap();
    let mut sparse_iter = sparse.iter();
    assert_eq!(Some((1, &[70, 80], 1)), sparse_iter.next());
    assert_eq!(Some((2, &[90, 100], 2)), sparse_iter.next());
    assert_eq!(Some((5, &[110, 120], 5)), sparse_iter.next());
    assert_eq!(None, sparse_iter.next());

    assert_eq!(None, chunked_iter.next());
}

#[test]
fn chunked_sparse_unichunked_unichunked_view() {
    let v: Vec<usize> = (1..=24).collect();
    let uni0 = Chunked2::from_flat(v);
    let uni1 = Chunked2::from_flat(uni0);
    let sparse = Sparse::from_dim(vec![0, 2, 5, 1, 2, 5], 6, uni1);
    let chunked = Chunked::from_sizes(vec![2, 0, 1, 3], sparse);
    let chunked_view = chunked.view();
    let mut chunked_iter = chunked_view.iter();

    let sparse = chunked_iter.next().unwrap();
    let mut sparse_iter = sparse.iter();
    let (idx, uni0, target) = sparse_iter.next().unwrap();
    assert_eq!((idx, target), (0, 0));
    let mut uni0_iter = uni0.iter();
    assert_eq!(Some(&[1, 2]), uni0_iter.next());
    assert_eq!(Some(&[3, 4]), uni0_iter.next());
    assert_eq!(None, uni0_iter.next());

    let (idx, uni0, target) = sparse_iter.next().unwrap();
    assert_eq!((idx, target), (2, 2));
    let mut uni0_iter = uni0.iter();
    assert_eq!(Some(&[5, 6]), uni0_iter.next());
    assert_eq!(Some(&[7, 8]), uni0_iter.next());
    assert_eq!(None, uni0_iter.next());
    assert_eq!(None, sparse_iter.next());

    let sparse = chunked_iter.next().unwrap();
    let mut sparse_iter = sparse.iter();
    assert_eq!(None, sparse_iter.next());

    let sparse = chunked_iter.next().unwrap();
    let mut sparse_iter = sparse.iter();
    let (idx, uni0, target) = sparse_iter.next().unwrap();
    assert_eq!((idx, target), (5, 5));
    let mut uni0_iter = uni0.iter();
    assert_eq!(Some(&[9, 10]), uni0_iter.next());
    assert_eq!(Some(&[11, 12]), uni0_iter.next());
    assert_eq!(None, uni0_iter.next());
    assert_eq!(None, sparse_iter.next());

    let sparse = chunked_iter.next().unwrap();
    let mut sparse_iter = sparse.iter();
    let (idx, uni0, target) = sparse_iter.next().unwrap();
    assert_eq!((idx, target), (1, 1));
    let mut uni0_iter = uni0.iter();
    assert_eq!(Some(&[13, 14]), uni0_iter.next());
    assert_eq!(Some(&[15, 16]), uni0_iter.next());
    assert_eq!(None, uni0_iter.next());

    let (idx, uni0, target) = sparse_iter.next().unwrap();
    assert_eq!((idx, target), (2, 2));
    let mut uni0_iter = uni0.iter();
    assert_eq!(Some(&[17, 18]), uni0_iter.next());
    assert_eq!(Some(&[19, 20]), uni0_iter.next());
    assert_eq!(None, uni0_iter.next());

    let (idx, uni0, target) = sparse_iter.next().unwrap();
    assert_eq!((idx, target), (5, 5));
    let mut uni0_iter = uni0.iter();
    assert_eq!(Some(&[21, 22]), uni0_iter.next());
    assert_eq!(Some(&[23, 24]), uni0_iter.next());
    assert_eq!(None, uni0_iter.next());
    assert_eq!(None, sparse_iter.next());
    assert_eq!(None, chunked_iter.next());
}

#[test]
fn sparse_chunked() {
    let v: Vec<usize> = (1..=6).collect();
    let chunked = Chunked::from_sizes(vec![2, 0, 1, 3], v);
    let selection = Select::new(vec![0, 3, 6, 10], 10..30);
    let sparse = Sparse::new(selection, chunked);
    let mut sparse_iter = sparse.iter();

    assert_eq!(Some((0, &[1, 2][..], 10)), sparse_iter.next());
    assert_eq!(Some((3, &[][..], 13)), sparse_iter.next());
    assert_eq!(Some((6, &[3][..], 16)), sparse_iter.next());
    assert_eq!(Some((10, &[4, 5, 6][..], 20)), sparse_iter.next());
    assert_eq!(None, sparse_iter.next());
}

#[test]
fn sparse_chunked_sparse_unichunked() {
    let v: Vec<usize> = (1..=12).collect();
    let uni = Chunked2::from_flat(v);
    let sparse0 = Sparse::from_dim(vec![0, 2, 5, 1, 2, 5], 6, uni);
    let chunked = Chunked::from_sizes(vec![2, 0, 1, 3], sparse0);
    let sparse1 = Sparse::from_dim(vec![0, 3, 6, 10], 11, chunked);
    let mut sparse1_iter = sparse1.iter();

    let (idx, sparse0, target) = sparse1_iter.next().unwrap();
    assert_eq!((idx, target), (0, 0));
    let mut sparse0_iter = sparse0.iter();
    assert_eq!(Some((0, &[1, 2], 0)), sparse0_iter.next());
    assert_eq!(Some((2, &[3, 4], 2)), sparse0_iter.next());
    assert_eq!(None, sparse0_iter.next());

    let (idx, sparse0, target) = sparse1_iter.next().unwrap();
    assert_eq!((idx, target), (3, 3));
    let mut sparse0_iter = sparse0.iter();
    assert_eq!(None, sparse0_iter.next());

    let (idx, sparse0, target) = sparse1_iter.next().unwrap();
    assert_eq!((idx, target), (6, 6));
    let mut sparse0_iter = sparse0.iter();
    assert_eq!(Some((5, &[5, 6], 5)), sparse0_iter.next());
    assert_eq!(None, sparse0_iter.next());

    let (idx, sparse0, target) = sparse1_iter.next().unwrap();
    assert_eq!((idx, target), (10, 10));
    let mut sparse0_iter = sparse0.iter();
    assert_eq!(Some((1, &[7, 8], 1)), sparse0_iter.next());
    assert_eq!(Some((2, &[9, 10], 2)), sparse0_iter.next());
    assert_eq!(Some((5, &[11, 12], 5)), sparse0_iter.next());
    assert_eq!(None, sparse0_iter.next());
    assert_eq!(None, sparse1_iter.next());
}

#[test]
fn sparse_chunked_sparse_unichunked_mut() {
    let v: Vec<usize> = (1..=12).collect();
    let uni = Chunked2::from_flat(v);
    let sparse0 = Sparse::from_dim(vec![0, 2, 5, 1, 2, 5], 6, uni);
    let chunked = Chunked::from_sizes(vec![2, 0, 1, 3], sparse0);
    let mut sparse1 = Sparse::from_dim(vec![0, 3, 6, 10], 11, chunked);
    for (_, mut sparse0) in sparse1.iter_mut() {
        for (_, v2) in sparse0.iter_mut() {
            for elem in v2.iter_mut() {
                *elem *= 10;
            }
        }
    }

    let mut sparse1_iter = sparse1.iter();

    let (idx, sparse0, target) = sparse1_iter.next().unwrap();
    assert_eq!((idx, target), (0, 0));
    let mut sparse0_iter = sparse0.iter();
    assert_eq!(Some((0, &[10, 20], 0)), sparse0_iter.next());
    assert_eq!(Some((2, &[30, 40], 2)), sparse0_iter.next());
    assert_eq!(None, sparse0_iter.next());

    let (idx, sparse0, target) = sparse1_iter.next().unwrap();
    assert_eq!((idx, target), (3, 3));
    let mut sparse0_iter = sparse0.iter();
    assert_eq!(None, sparse0_iter.next());

    let (idx, sparse0, target) = sparse1_iter.next().unwrap();
    assert_eq!((idx, target), (6, 6));
    let mut sparse0_iter = sparse0.iter();
    assert_eq!(Some((5, &[50, 60], 5)), sparse0_iter.next());
    assert_eq!(None, sparse0_iter.next());

    let (idx, sparse0, target) = sparse1_iter.next().unwrap();
    assert_eq!((idx, target), (10, 10));
    let mut sparse0_iter = sparse0.iter();
    assert_eq!(Some((1, &[70, 80], 1)), sparse0_iter.next());
    assert_eq!(Some((2, &[90, 100], 2)), sparse0_iter.next());
    assert_eq!(Some((5, &[110, 120], 5)), sparse0_iter.next());
    assert_eq!(None, sparse0_iter.next());
    assert_eq!(None, sparse1_iter.next());
}

#[test]
fn sparse_unichunked_sparse_unichunked_mut() {
    let v: Vec<usize> = (1..=12).collect();
    let uni = Chunked2::from_flat(v);
    let sparse0 = Sparse::from_dim(vec![0, 2, 5, 1, 2, 5], 6, uni);
    let mut chunked = ChunkedN::from_flat_with_stride(sparse0, 3);
    let mut sparse1 = Sparse::from_dim(vec![0, 10], 11, chunked.view_mut());
    for (_, mut sparse0) in sparse1.iter_mut() {
        for (_, v2) in sparse0.iter_mut() {
            for elem in v2.iter_mut() {
                *elem *= 10;
            }
        }
    }

    let mut sparse1_iter = sparse1.iter();

    let (idx, sparse0, target) = sparse1_iter.next().unwrap();
    assert_eq!((idx, target), (0, 0));
    let mut sparse0_iter = sparse0.iter();
    assert_eq!(Some((0, &[10, 20], 0)), sparse0_iter.next());
    assert_eq!(Some((2, &[30, 40], 2)), sparse0_iter.next());
    assert_eq!(Some((5, &[50, 60], 5)), sparse0_iter.next());
    assert_eq!(None, sparse0_iter.next());

    let (idx, sparse0, target) = sparse1_iter.next().unwrap();
    assert_eq!((idx, target), (10, 10));
    let mut sparse0_iter = sparse0.iter();
    assert_eq!(Some((1, &[70, 80], 1)), sparse0_iter.next());
    assert_eq!(Some((2, &[90, 100], 2)), sparse0_iter.next());
    assert_eq!(Some((5, &[110, 120], 5)), sparse0_iter.next());
    assert_eq!(None, sparse0_iter.next());
    assert_eq!(None, sparse1_iter.next());
}

#[test]
fn chunked_chars() {
    let v = vec!["World", "Coffee", "Cat", " ", "Hello", "Refrigerator", "!"];
    let bytes: Vec<Vec<u8>> = v
        .into_iter()
        .map(|word| word.to_string().into_bytes())
        .collect();
    let words = Chunked::<Vec<u8>>::from_nested_vec(bytes);
    let selection = Select::new(vec![4, 3, 0, 6, 3, 4, 6], words);
    let collapsed = selection.view().collapse();
    assert_eq!(
        "Hello World! Hello!",
        String::from_utf8(collapsed.data().clone())
            .unwrap()
            .as_str()
    );
}
