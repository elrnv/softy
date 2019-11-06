use geo::{
    mesh::VertexPositions,
};

/// Rotate a mesh around a given vector using the right hand rule by `theta` radians.
/// This function modifies the vertex positions of the given vertex mesh.
pub fn rotate<T, V3, M>(mesh: &mut M, u: V3, theta: T)
where
    V3: Copy + Into<[T; 3]>,
    M: VertexPositions<Element = V3>,
    [T; 3]: Into<V3>,
{
    // Normalize axis rotation vector
    let mut u = u.into();
    let norm_u = u.norm_squared();
    if norm_u > T::zero() {
        u /= norm_u;
    }

    // Compute rotation matrix
    // R = cos(theta) * I + sin(theta)*[u]_X + (1 - cos(theta))(uu^T)
    let id = Matrix3::identity();
    let u_skew = u.skew();
    let cos_theta = theta.cos();

    let rotation_matrix =
        id * cos_theta + u_skew * theta.sin() + u * (u.transpose() * (T::one() - cos_theta));

    // Trnasform mesh.
    for p in mesh.vertex_position_iter_mut() {
        let pos: Vector3<T> = (*p).into();
        *p = (rotation_matrix * pos).into();
    }
}

/// Translate a mesh by a given translation vector.
/// This function modifies the vertex positions of the given vertex mesh.
pub fn translate<T, V3, M>(mesh: &mut M, t: V3)
where
    T: Real,
    V3: Copy + Into<Vector3<T>>,
    M: VertexPositions<Element = V3>,
    Vector3<T>: Into<V3>,
{
    for p in mesh.vertex_position_iter_mut() {
        let pos = Into::<Vector3<T>>::into(*p);
        *p = (pos + t.into()).into();
    }
}

/// Scale a mesh by a given scale vector.
/// The mesh is scaled in each of the 3 orthogonal axis directions.
pub fn scale<T, V3, M>(mesh: &mut M, s: V3)
where
    T: Real,
    V3: Copy + Into<Vector3<T>>,
    M: VertexPositions<Element = V3>,
    Vector3<T>: Into<V3>,
{
    let scale = s.into();
    for p in mesh.vertex_position_iter_mut() {
        let mut pos = Into::<Vector3<T>>::into(*p);
        pos[0] *= scale[0];
        pos[1] *= scale[1];
        pos[2] *= scale[2];
        *p = pos.into();
    }
}

/// Scale a mesh uniformly in all directions by the given factor.
pub fn uniform_scale<T, V3, M>(mesh: &mut M, s: T)
where
    T: Real,
    V3: Copy + Into<Vector3<T>>,
    M: VertexPositions<Element = V3>,
    Vector3<T>: Into<V3>,
{
    for p in mesh.vertex_position_iter_mut() {
        let pos = Into::<Vector3<T>>::into(*p);
        *p = (pos * s).into();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::*;
    use approx::*;
    use geo::ops::*;

    #[test]
    fn rotate_grid() {
        let mut grid = make_grid(Grid {
            rows: 1,
            cols: 1,
            orientation: AxisPlaneOrientation::ZX,
        });
        rotate(&mut grid, [0.0, 1.0, 0.0], std::f64::consts::PI * 0.25);
        let bbox = grid.bounding_box();

        let bound = 2.0_f64.sqrt();
        let minb = bbox.min_corner().into_inner();
        let maxb = bbox.max_corner().into_inner();

        assert_relative_eq!(minb[0], -bound);
        assert_eq!(minb[1], 0.0);
        assert_relative_eq!(minb[2], -bound);
        assert_relative_eq!(maxb[0], bound);
        assert_eq!(maxb[1], 0.0);
        assert_relative_eq!(maxb[2], bound);
    }
}
