use utils::soap::*;

//const A: Matrix2<f64> = Matrix2([[1.0, 3.0], [2.0, 4.0]]);
//const B: Matrix3<f64> = Matrix3([[1.0, 3.0, 4.0], [2.0, 2.0, 2.0], [4.0, 3.0, 2.0]]);
//const C: Matrix3<f64> = Matrix3([[2.0, 4.0, 3.0], [1.0, 2.0, 5.0], [3.0, 1.5, 2.0]]);

//#[test]
//fn fold_test() {
//    assert_eq!(A.sum(), 10.0);
//    assert_eq!(B.sum(), 23.0);
//    assert_eq!(A.fold(0.0, |acc, x| x + acc), 10.0);
//    assert_eq!(B.fold(0.0, |acc, x| x + acc), 23.0);
//}
#[test]
fn dot_product_test() {
    let a = Vector3::flat([1.0, 2.0, 3.0]);
    let b = Vector3::flat([3.0, 2.0, 1.0]);
    assert_eq!(a.dot(b), 10.0);
}
#[test]
fn negation_test() {
    // for vectors
    let a = Vector3::flat([1.0, -2.0, 3.0]);
    assert_eq!(-a, Vector3::flat([-1.0, 2.0, -3.0]));
    // for matrices
    //assert_eq!(-A, Matrix2([[-1.0, -3.0], [-2.0, -4.0]]));
}
#[test]
fn norm_test() {
    let a = Vector3::flat([1.0, 2.0, 3.0]);
    assert_eq!(a.norm_squared(), 14.0);
    assert_eq!(a.norm(), 14.0_f64.sqrt());
}

/*
#[test]
fn add_vec_test() {
    let a = Vector3([0.0, 1.0, -2.0]);
    let b = Vector3([3.0, 1.5, -2.5]);
    assert_eq!(a - b, Vector3([-3.0, -0.5, 0.5]));
    assert_eq!(a + b, Vector3([3.0, 2.5, -4.5]));
}

#[test]
fn determinant_test() {
    assert_eq!(A.determinant(), -2.0);
    assert_eq!(B.determinant(), 2.0);
    assert_eq!(C.determinant(), 31.5);
}

#[test]
fn transpose_test() {
    let expected = Matrix2([[1.0, 2.0], [3.0, 4.0]]);
    assert_eq!(A.transpose(), expected);
}

#[test]
fn inverse_test() {
    // Test non-singular matrices.
    let expected = Matrix2([[-2.0, 1.5], [1.0, -0.5]]);
    assert_eq!(A.inverse(), Some(expected));
    let expected = Matrix3([[-1.0, 3.0, -1.0], [2.0, -7.0, 3.0], [-1.0, 4.5, -2.0]]);
    assert_eq!(B.inverse(), Some(expected));

    // Test the inverse transpose
    let expected = Matrix2([[-2.0, 1.0], [1.5, -0.5]]);
    assert_eq!(A.inverse_transpose(), Some(expected));
    let expected = Matrix3([[-1.0, 2.0, -1.0], [3.0, -7.0, 4.5], [-1.0, 3.0, -2.0]]);
    assert_eq!(B.inverse_transpose(), Some(expected));

    // Test inversion of a singular matrix
    assert_eq!(
        Matrix3([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]).inverse(),
        None
    );
}

#[test]
fn invert_test() {
    let mut a = A.clone();
    assert!(a.invert());
    assert_eq!(A.inverse(), Some(a));
    let mut b = B.clone();
    assert!(b.invert());
    assert_eq!(B.inverse(), Some(b));

    // Test inverting a singular matrix.
    let sing = Matrix3([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]);
    let mut a = sing.clone();
    assert!(!a.invert()); // check that inversion failed.
    assert_eq!(a, sing); // check that it was unchanged
}

#[test]
fn map_and_apply_test() {
    let expected = Matrix3([[2.0, 4.0, 5.0], [3.0, 3.0, 3.0], [5.0, 4.0, 3.0]]);
    assert_eq!(*B.apply(|x| x + 1.0), expected);
    assert_eq!(B.map(|x| x + 1.0), expected);
}
#[test]
fn covec_vec_mult_test() {
    let a = Vector3([0.5, 1.0, -2.0]);
    let b = RowVector3([1.0, -4.0, 2.0]);
    assert_eq!(b * a, -7.5);
}
#[test]
fn vec_covec_mult_test() {
    let a = Vector3([0.5, 1.0, -2.0]);
    let b = RowVector3([1.0, -4.0, 2.0]);
    assert_eq!(
        a * b,
        Matrix3([[0.5, 1.0, -2.0], [-2.0, -4.0, 8.0], [1.0, 2.0, -4.0]])
    );
}
#[test]
fn mtx_vec_mult_test() {
    let a = Vector3([0.0, 1.0, -2.0]);
    assert_eq!(B * a, Vector3([-6.0, -4.0, -2.0]));
}
#[test]
fn covec_mtx_mult_test() {
    let b = RowVector3([1.0, -4.0, 2.0]);
    assert_eq!(b * B, RowVector3([-3.0, -2.0, -4.0]));
}
#[test]
fn matrix_mult_test() {
    let expected = Matrix3([[22.0, 23.0, 22.0], [25.0, 22.0, 18.0], [14.0, 18.0, 19.0]]);
    assert_eq!(B * C, expected);
}
#[test]
fn matrix_vectorize_test() {
    let expected = Vector4([1.0, 3.0, 2.0, 4.0]);
    assert_eq!(A.vec(), expected);
    assert_eq!(A.vec_ref(), &expected);
}
#[test]
fn matrix_scalar_mult_test() {
    let exp_a = Matrix2([[2.0, 6.0], [4.0, 8.0]]);
    let exp_b = Matrix3([[0.5, 1.5, 2.0], [1.0, 1.0, 1.0], [2.0, 1.5, 1.0]]);
    assert_eq!(A * 2.0, exp_a);
    assert_eq!(2.0 * A, exp_a);
    assert_eq!(B * 0.5, exp_b);
    assert_eq!(0.5 * B, exp_b);
}
#[test]
fn matrix_trace_test() {
    assert_eq!(A.trace(), 5.0);
    assert_eq!(B.trace(), 5.0);
    assert_eq!(C.trace(), 6.0);
}
#[test]
fn cast_test() {
    let a = Vector3([0.0f32, 1.0, -2.0]);
    let b = Vector3([0.0f64, 1.0, -2.0]);
    assert_eq!(a.cast::<f64>().unwrap(), b);
}

#[test]
fn cross_product_test() {
    let a = Vector3([4.5, 7.5, -2.0]);
    let b = Vector3([3.1, -1.2, 4.9]);
            let a_cross_b = Vector3([34.35, -28.25, -28.65]);
    assert_eq!(a.cross(b), a_cross_b);
    assert_eq!(a.skew()*b, a_cross_b);
}

#[test]
fn sum_test() {
    let v = vec![
        Vector3([4.5, 7.5, -2.0]),
        Vector3([3.1, -1.2, 4.9]),
        Vector3([-1.3, -2.4, 1.3])];
    let expected = v[0] + v[1] + v[2];

    let sum_ref: Vector3<f64> = v.iter().sum();
    let sum: Vector3<f64> = v.into_iter().sum();
    assert_eq!(sum_ref, expected);
    assert_eq!(sum, expected);
}

#[test]
fn diag_test() {
    let v = Vector3([4.5, 7.5, -2.0]);
    let d = Matrix3::diag(v.into());
    assert_eq!(d[0][0], v[0]);
    assert_eq!(d[0][1], 0.0);
    assert_eq!(d[0][2], 0.0);
    assert_eq!(d[1][0], 0.0);
    assert_eq!(d[1][1], v[1]);
    assert_eq!(d[1][2], 0.0);
    assert_eq!(d[2][0], 0.0);
    assert_eq!(d[2][1], 0.0);
    assert_eq!(d[2][2], v[2]);

    let id = Matrix3::identity();
    assert_eq!(id, Matrix3([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]));
}

#[cfg(feature = "serde")]
#[test]
fn serde_test() {
    let v = Vector3([4.5, 7.5, -2.0]);
    let v_str = serde_json::to_string(&v).expect("Failed to serialize a Vector3.");
    let v_new: Vector3<f64> = serde_json::from_str(&v_str).expect("Failed to deserialize a Vector3.");
    for i in 0..3 {
        assert_eq!(v[i], v_new[i]);
    }

    let m_str = serde_json::to_string(&B).expect("Failed to serialize a Matrix3.");
    let m_new: Matrix3<f64> = serde_json::from_str(&m_str).expect("Failed to deserialize a Matrix3.");
    for i in 0..3 {
        for j in 0..3 {
            assert_eq!(B[i][j], m_new[i][j]);
        }
    }
}
*/
