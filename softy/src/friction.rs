mod solver;
mod polar_solver;

pub use solver::*;
use reinterpret::*;
use na::{Vector2, Vector3, Matrix3};

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct FrictionParams {
    pub dynamic_friction: f64,
    pub inner_iterations: usize,
    pub tolerance: f64,
    pub print_level: u8,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Friction {
    pub use_cylindrical_coords: bool,
    pub params: FrictionParams,
    pub force: Vec<[f64; 3]>,
    contact_normals: Vec<Vector3<f64>>,
    contact_tangents: Vec<Vector3<f64>>,
}

impl Friction {
    pub fn new(params: FrictionParams, use_cylindrical_coords: bool) -> Friction {
        Friction {
            use_cylindrical_coords,
            params,
            force: Vec::new(),
            contact_normals: Vec::new(),
            contact_tangents: Vec::new(),
        }
    }

    /// Transform a vector at the given contact point index to contact coordinates. The index
    /// determines which local contact coordinates to use.
    pub fn to_contact_coordinates<V3>(&self, v: V3, contact_index: usize) -> Vector3<f64>
    where
        V3: Into<Vector3<f64>>,
    {
        let n = self.contact_normals[contact_index];
        let t = self.contact_tangents[contact_index];
        let b = n.cross(&t);
        //b = b/b.norm(); // b may need to be renormalized here.
        let vc = Matrix3::from_columns(&[n, t, b]).transpose() * v.into();
        if !self.use_cylindrical_coords {
            return vc;
        }

        // Cylindrical coordinates have the form: [Normal, Radius, Angle]
        Vector3::new(vc[0], Vector2::new(vc[1], vc[2]).norm(), f64::atan2(vc[2], vc[1]))
    }

    /// Transform a vector at the given contact point index to physical coordinates. The index
    /// determines which local contact coordinates to use.
    pub fn to_physical_coordinates<V3>(&self, v: V3, contact_index: usize) -> Vector3<f64>
    where
        V3: Into<Vector3<f64>>,
    {
        let n = self.contact_normals[contact_index];
        let t = self.contact_tangents[contact_index];
        let b = n.cross(&t);
        //b = b/b.norm(); // b may need to be renormalized here.
        let v = v.into();

        let v = if self.use_cylindrical_coords {
            // Cylindrical coordinates have the form: [Normal, Radius, Angle]
            let r = v[1];
            let theta = v[2];
            Vector3::new(v[0], r*f64::cos(theta), r*f64::sin(theta))
        } else {
            v
        };

        Matrix3::from_columns(&[n, t, b]) * v
    }

    /// Transform a given stacked vector of vectors in physical space to values in normal direction
    /// to each contact point and stacked 2D vectors in the tangent space of the contact point.
    pub fn to_contact_space(&self, physical: Vec<[f64; 3]>) -> Vec<[f64; 3]> {
        physical
            .iter()
            .enumerate()
            .map(|(i, &v)| self.to_contact_coordinates(v, i).into())
            .collect()
    }

    /// Transform a given stacked vector of vectors in contact space to vectors in physical space.
    pub fn to_physical_space(&self, contact: Vec<[f64; 3]>) -> Vec<[f64; 3]> {
        contact
            .iter()
            .enumerate()
            .map(|(i, &v)| self.to_physical_coordinates(v, i).into())
            .collect()
    }

    /// Update the basis for the contact space at each contact point given the specified set of
    /// normals. The tangent space is chosen arbitrarily
    pub fn update_contact_basis_from_normals(&mut self, normals: Vec<[f64; 3]>) {
        self.contact_tangents
            .resize(normals.len(), Vector3::zeros());
        self.contact_normals = reinterpret_vec(normals);

        for (&n, t) in self
            .contact_normals
            .iter()
            .zip(self.contact_tangents.iter_mut())
        {
            // Find the axis that is most aligned with the normal, then use the next axis for the
            // tangent.
            let tangent_axis = (n.iamax() + 1) % 3;
            t[tangent_axis] = 1.0;

            // Project out the normal component.
            *t -= n * n[tangent_axis];

            // Normalize in-place.
            t.normalize_mut();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::*;

    // Verify that converting to contact space and back to physical space produces the same
    // vectors.
    #[test]
    fn contact_physical_space_conversion_test() -> Result<(), crate::Error> {
        use geo::mesh::{topology::*, TriMesh, VertexPositions};

        let run = |trimesh: TriMesh<f64>| -> Result<(), crate::Error> {
            let mut normals = vec![geo::math::Vector3::zeros(); trimesh.num_vertices()];
            geo::algo::compute_vertex_area_weighted_normals(
                trimesh.vertex_positions(),
                reinterpret_slice(trimesh.indices.as_slice()),
                &mut normals,
            );

            for n in normals.iter_mut() {
                *n /= n.norm();
            }

            let params = FrictionParams {
                dynamic_friction: 0.5,
                inner_iterations: 30,
                tolerance: 1e-5,
                print_level: 0,
            };

            let mut friction = Friction::new(params, false);
            friction.update_contact_basis_from_normals(reinterpret_vec(normals));

            let vecs = utils::random_vectors(trimesh.num_vertices());
            let contact_vecs =
                friction.to_contact_space(reinterpret_vec(vecs.clone()));
            let physical_vecs = friction.to_physical_space(contact_vecs);

            for (a, b) in vecs.into_iter().zip(physical_vecs.into_iter()) {
                for i in 0..3 {
                    assert_relative_eq!(a[i], b[i]);
                }
            }
            Ok(())
        };

        let trimesh = utils::make_sample_octahedron();
        run(trimesh)?;
        let trimesh = utils::make_regular_icosahedron();
        run(trimesh)
    }
}
