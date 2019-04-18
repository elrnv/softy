use na::{Matrix3, Vector3};

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum ContactType {
    Implicit,
    Point,
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct SmoothContactParams {
    pub kernel: implicits::KernelType,
    pub contact_type: ContactType,
    pub friction_params: Option<FrictionParams>,
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct FrictionParams {
    pub dynamic_friction: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Friction {
    pub params: FrictionParams,
    pub force: Vec<[f64;3]>,
    contact_normals: Vec<Vector3<f64>>,
    contact_tangents: Vec<Vector3<f64>>,
}

impl Friction {
    pub fn new(params: FrictionParams) -> Friction {
        Friction {
            params,
            force: Vec::new(),
            contact_normals: Vec::new(),
            contact_tangents: Vec::new(),
        }
    }

    /// Transform a vector at the given contact point index to contact coordinates. The index
    /// determines which local contact coordinates to use.
    pub fn to_contact_coordinates<V3>(&self, v: V3, contact_index: usize) -> Vector3<f64>
        where V3: Into<Vector3<f64>>
    {
        let n = self.contact_normals[contact_index];
        let t = self.contact_tangents[contact_index];
        let b = n.cross(&t);
        //b = b/b.norm(); // b may need to be renormalized here.
        Matrix3::from_columns(&[n, t, b]).transpose() * v.into()
    }

    /// Transform a vector at the given contact point index to physical coordinates. The index
    /// determines which local contact coordinates to use.
    pub fn to_physical_coordinates<V3>(&self, v: V3, contact_index: usize) -> Vector3<f64> 
        where V3: Into<Vector3<f64>>
    {
        let n = self.contact_normals[contact_index];
        let t = self.contact_tangents[contact_index];
        let b = n.cross(&t);
        //b = b/b.norm(); // b may need to be renormalized here.
        Matrix3::from_columns(&[n, t, b]) * v.into()
    }

    /// Transform a given stacked vector of vectors in physical space to values in normal direction
    /// to each contact point and stacked 2D vectors in the tangent space of the contact point.
    pub fn to_contact_space(&self, physical: Vec<[f64; 3]>) -> Vec<[f64; 3]> {
        physical.iter().enumerate().map(|(i, &v)| self.to_contact_coordinates(v, i).into()).collect()
    }

    /// Transform a given stacked vector of vectors in contact space to vectors in physical space.
    pub fn to_physical_space(&self, contact: Vec<[f64; 3]>) -> Vec<[f64; 3]> {
        contact.iter().enumerate().map(|(i, &v)| self.to_physical_coordinates(v, i).into()).collect()
    }

    /// Update the basis for the contact space at each contact point given the specified set of
    /// normals. The tangent space is chosen arbitrarily
    pub fn update_contact_basis_from_normals(&mut self, normals: Vec<[f64; 3]>) {
        self.contact_tangents.resize(normals.len(), Vector3::zeros());
        self.contact_normals = reinterpret::reinterpret_vec(normals);

        for (&n, t) in self.contact_normals.iter().zip(self.contact_tangents.iter_mut()) {
            // Find the axis that is most aligned with the normal, then use the next axis for the
            // tangent.
            let tangent_axis = (n.imax() + 1) % 3;
            t[tangent_axis] = 1.0;

            // Project out the normal component.
            *t -= n * n[tangent_axis];

            // Normalize in-place.
            t.normalize_mut();
        }
    }
}
