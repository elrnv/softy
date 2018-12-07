use geometry::{
    mesh::VertexPositions,
    math::Vector3,
    Real,
};

/// Translate a mesh by a given translation vector.
/// This function modifies the vertex positions of the given vertex mesh.
pub fn translate<T, V3, M>(mesh: &mut M, t: V3)
    where T: Real,
          V3: Copy + Into<Vector3<T>>,
          M: VertexPositions<Element=V3>,
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
    where T: Real,
          V3: Copy + Into<Vector3<T>>,
          M: VertexPositions<Element=V3>,
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
    where T: Real,
          V3: Copy + Into<Vector3<T>>,
          M: VertexPositions<Element=V3>,
          Vector3<T>: Into<V3>,
{
    for p in mesh.vertex_position_iter_mut() {
        let pos = Into::<Vector3<T>>::into(*p);
        *p = (pos * s).into();
    }
}
