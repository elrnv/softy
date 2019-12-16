use utils::soap::*;

fn a2() -> Matrix2<f64> {
    Matrix2::new([[1.0, 3.0], [2.0, 4.0]])
}
fn a3() -> Matrix3<f64> {
    Matrix3::new([[1.0, 3.0, 4.0], [2.0, 2.0, 2.0], [4.0, 3.0, 2.0]])
}
fn b3() -> Matrix3<f64> {
    Matrix3::new([[2.0, 4.0, 3.0], [1.0, 2.0, 5.0], [3.0, 1.5, 2.0]])
}
fn a32() -> Matrix3x2<f64> {
    Matrix3x2::new([[2.0, 4.0], [8.0, 4.0], [12.0, 1.6]])
}

#[test]
fn fold() {
    assert_eq!(a2().sum_inner(), 10.0);
    assert_eq!(a3().sum_inner(), 23.0);
    assert_eq!(a2().fold_inner(0.0, |acc, x| x + acc), 10.0);
    assert_eq!(a3().fold_inner(0.0, |acc, x| x + acc), 23.0);
}
#[test]
fn dot_product() {
    let a = Vector3::new([1.0, 2.0, 3.0]);
    let b = Vector3::new([3.0, 2.0, 1.0]);
    assert_eq!(a.dot(b), 10.0);
}
#[test]
fn negation() {
    // for vectors
    let a = Vector3::new([1.0, -2.0, 3.0]);
    assert_eq!(-a, Vector3::new([-1.0, 2.0, -3.0]));
    // for matrices
    assert_eq!(-a2(), Matrix2::new([[-1.0, -3.0], [-2.0, -4.0]]));
}
#[test]
fn norm() {
    let a = Vector3::new([1.0, 2.0, 3.0]);
    assert_eq!(a.norm_squared(), 14.0);
    assert_eq!(a.norm(), 14.0_f64.sqrt());
}

#[test]
fn frob_norm() {
    assert_eq!(a2().frob_norm_squared(), 30.0);
    assert_eq!(a2().frob_norm(), 30.0_f64.sqrt());
}

#[test]
fn add_vec() {
    let a = Vector3::new([0.0, 1.0, -2.0]);
    let b = Vector3::new([3.0, 1.5, -2.5]);
    assert_eq!(a - b, Vector3::new([-3.0, -0.5, 0.5]));
    assert_eq!(a + b, Vector3::new([3.0, 2.5, -4.5]));
}

#[test]
fn determinant() {
    assert_eq!(a2().determinant(), -2.0);
    assert_eq!(a3().determinant(), 2.0);
    assert_eq!(b3().determinant(), 31.5);
}

#[test]
fn transpose() {
    let expected = Matrix2::new([[1.0, 2.0], [3.0, 4.0]]);
    assert_eq!(a2().transpose(), expected);
}
#[test]
fn mtx_vec_mult() {
    let a = Vector3::new([0.0, 1.0, -2.0]);
    assert_eq!(a3() * a, Vector3::new([-5.0, -2.0, -1.0]));
}
#[test]
fn rowvec_mtx_mult() {
    let b = RowVector3::new([[1.0, -4.0, 2.0]]);
    assert_eq!(b * a3(), RowVector3::new([[1.0, 1.0, 0.0]]));
}
#[test]
fn matrix_mult() {
    let expected = Matrix3::new([[17.0, 16.0, 26.0], [12.0, 15.0, 20.0], [17.0, 25.0, 31.0]]);
    assert_eq!(a3() * b3(), expected);
}
#[test]
fn matrix_scalar_mult() {
    let exp_a = Matrix2::new([[2.0, 6.0], [4.0, 8.0]]);
    let exp_b = Matrix3::new([[0.5, 1.5, 2.0], [1.0, 1.0, 1.0], [2.0, 1.5, 1.0]]);
    assert_eq!(a2() * 2.0, exp_a);
    assert_eq!(2.0.into_tensor() * a2(), exp_a);
    assert_eq!(a3() * 0.5, exp_b);
    assert_eq!(0.5.into_tensor() * a3(), exp_b);
}

#[test]
fn div_assign() {
    let mut d = a32().clone();
    d /= 2.0;
    assert_eq!(d, Matrix3x2::new([[1.0, 2.0], [4.0, 2.0], [6.0, 0.8]]));
}

#[test]
fn cross_product() {
    let a = Vector3::new([4.5, 7.5, -2.0]);
    let b = Vector3::new([3.1, -1.2, 4.9]);
    let a_cross_b = Vector3::new([34.35, -28.25, -28.65]);
    assert_eq!(a.cross(b), a_cross_b);
    assert_eq!(a.skew() * b, a_cross_b);
}

#[test]
fn inverse() {
    // Test non-singular matrices.
    let expected = Matrix2::new([[-2.0, 1.5], [1.0, -0.5]]);
    assert_eq!(a2().inverse(), Some(expected));
    let expected = Matrix3::new([[-1.0, 3.0, -1.0], [2.0, -7.0, 3.0], [-1.0, 4.5, -2.0]]);
    assert_eq!(a3().inverse(), Some(expected));

    // Test the inverse transpose
    let expected = Matrix2::new([[-2.0, 1.0], [1.5, -0.5]]);
    assert_eq!(a2().inverse_transpose(), Some(expected));
    let expected = Matrix3::new([[-1.0, 2.0, -1.0], [3.0, -7.0, 4.5], [-1.0, 3.0, -2.0]]);
    assert_eq!(a3().inverse_transpose(), Some(expected));

    // Test inversion of a singular matrix
    assert_eq!(
        Matrix3::new([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]).inverse(),
        None
    );
}

#[test]
fn invert() {
    let mut a = a2().clone();
    assert!(a.invert());
    assert_eq!(a, Matrix2::new([[-2.0, 1.5], [1.0, -0.5]]));
    assert_eq!(a2().inverse(), Some(a));
    let mut b = a3().clone();
    assert!(b.invert());
    assert_eq!(a3().inverse(), Some(b));

    // Test inverting a singular matrix.
    let sing = Matrix3::new([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]);
    let mut a = sing.clone();
    assert!(!a.invert()); // check that inversion failed.
    assert_eq!(a, sing); // check that it was unchanged
}

#[test]
fn rowvec_vec_mult() {
    let a = Vector3::new([0.5, 1.0, -2.0]);
    let b = RowVector3::new([[1.0, -4.0, 2.0]]);
    assert_eq!(b * a, Vector1::new([-7.5]));
}

#[test]
fn map_inner() {
    let expected = Matrix3::new([[2.0, 4.0, 5.0], [3.0, 3.0, 3.0], [5.0, 4.0, 3.0]]);
    assert_eq!(a3().map_inner(|x| x + 1.0.into_tensor()), expected);
}

#[test]
fn vec_rowvec_mult_test() {
    let a = Vector3::new([0.5, 1.0, -2.0]);
    let b = RowVector3::new([[1.0, -4.0, 2.0]]);
    assert_eq!(
        a * b,
        Matrix3::new([[0.5, -2.0, 1.0], [1.0, -4.0, 2.0], [-2.0, 8.0, -4.0]])
    );
}

#[test]
fn matrix_vectorize_test() {
    let expected = Vector4::new([1.0, 3.0, 2.0, 4.0]);
    assert_eq!(a2().vec(), expected);
}
/*
#[test]
fn matrix_trace_test() {
    assert_eq!(A.trace(), 5.0);
    assert_eq!(B.trace(), 5.0);
    assert_eq!(C.trace(), 6.0);
}
#[test]
fn cast_test() {
    let a = Vector3::new([0.0f32, 1.0, -2.0]);
    let b = Vector3::new([0.0f64, 1.0, -2.0]);
    assert_eq!(a.cast::<f64>().unwrap(), b);
}

#[test]
fn sum_test() {
    let v = vec![
        Vector3::new([4.5, 7.5, -2.0]),
        Vector3::new([3.1, -1.2, 4.9]),
        Vector3::new([-1.3, -2.4, 1.3]),
    ];
    let expected = v[0] + v[1] + v[2];

    let sum_ref: Vector3<f64> = v.iter().sum();
    let sum: Vector3<f64> = v.into_iter().sum();
    assert_eq!(sum_ref, expected);
    assert_eq!(sum, expected);
}

#[test]
fn diag_test() {
    let v = Vector3::new([4.5, 7.5, -2.0]);
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
    assert_eq!(
        id,
        Matrix3::new([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    );
}

#[cfg(feature = "serde")]
#[test]
fn serde_test() {
    let v = Vector3::new([4.5, 7.5, -2.0]);
    let v_str = serde_json::to_string(&v).expect("Failed to serialize a Vector3.");
    let v_new: Vector3<f64> =
        serde_json::from_str(&v_str).expect("Failed to deserialize a Vector3.");
    for i in 0..3 {
        assert_eq!(v[i], v_new[i]);
    }

    let m_str = serde_json::to_string(&B).expect("Failed to serialize a Matrix3.");
    let m_new: Matrix3<f64> =
        serde_json::from_str(&m_str).expect("Failed to deserialize a Matrix3.");
    for i in 0..3 {
        for j in 0..3 {
            assert_eq!(B[i][j], m_new[i][j]);
        }
    }
}
*/
