use super::*;
use crate::constraint::*;
use crate::contact::*;
use crate::friction::*;
use crate::matrix::*;
use crate::Error;
use crate::Index;
use crate::TetMesh;
use crate::TriMesh;
use geo::math::Vector3;
use geo::mesh::topology::*;
use geo::mesh::{Attrib, VertexPositions};
use implicits::*;
use reinterpret::*;
use std::{cell::RefCell, rc::Rc};
use utils::zip;

/// Enforce a contact constraint on a mesh against an animated implicit surface. This constraint prevents
/// vertices of the simulation mesh from penetrating through the implicit surface.
#[derive(Clone, Debug)]
pub struct SPImplicitContactConstraint {
    /// Implicit surface that represents the collision object.
    pub implicit_surface: RefCell<ImplicitSurface>,
    pub simulation_mesh: Rc<RefCell<TetMesh>>,
    pub collision_object: Rc<RefCell<TriMesh>>,
    /// Mapping from points on the surface of the simulation mesh to the actual
    /// vertices on the tetrahedron mesh.
    pub sim_verts: Vec<usize>,
    /// A buffer of vertex positions on the simulation mesh. This is used to avoid reallocating
    /// contiguous space for these positions every time the constraint is evaluated.
    pub query_points: RefCell<Vec<[f64; 3]>>,
    /// Friction impulses applied during contact.
    pub frictional_contact: Option<FrictionalContact>,
    /// A mass for each vertex in the object mesh.
    pub vertex_masses: Vec<f64>,

    /// Internal constraint function buffer used to store temporary constraint computations.
    constraint_buffer: RefCell<Vec<f64>>,

    /// Worspace vector to keep track of active constraint indices.
    active_constraint_indices: RefCell<Vec<usize>>,
}

impl SPImplicitContactConstraint {
    /// Build an implicit surface from the given trimesh, and constrain the tetmesh vertices to lie
    /// strictly outside of it.
    pub fn new(
        object: &TriMesh,
        collider: &TriMesh,
        kernel: KernelType,
        friction_params: Option<FrictionParams>,
        density: f64,
    ) -> Result<Self, Error> {
        let trimesh = trimesh_rc.borrow();

        let mut surface_builder = ImplicitSurfaceBuilder::new();
        surface_builder
            .trimesh(&trimesh)
            .kernel(kernel)
            .sample_type(SampleType::Face)
            .background_field(BackgroundFieldParams {
                field_type: BackgroundFieldType::DistanceBased,
                weighted: false,
            });

        if let Some(surface) = surface_builder.build() {
            let tetmesh = tetmesh_rc.borrow();
            let mut surf_mesh = tetmesh.surface_trimesh_with_mapping(Some("i"), None, None, None);
            let sim_verts = surf_mesh
                .remove_attrib::<VertexIndex>("i")
                .expect("Failed to map indices.")
                .into_buffer()
                .into_vec::<usize>()
                .expect("Incorrect index type: not usize");

            let query_points = surf_mesh.vertex_positions();

            let vertex_masses = object
                .attrib_as_slice::<MassType, VertexIndex>(MASS_ATTRIB)
                .to_vec();

            let constraint = SPImplicitContactConstraint {
                implicit_surface: RefCell::new(surface),
                simulation_mesh: Rc::clone(tetmesh_rc),
                collision_object: Rc::clone(trimesh_rc),
                sim_verts,
                query_points: RefCell::new(query_points.to_vec()),
                frictional_contact: friction_params.and_then(|fparams| {
                    if fparams.dynamic_friction > 0.0 {
                        Some(FrictionalContact::new(fparams))
                    } else {
                        None
                    }
                }),
                vertex_masses,
                constraint_buffer: RefCell::new(vec![0.0; query_points.len()]),
                active_constraint_indices: RefCell::new(Vec::new()),
            };

            constraint
                .implicit_surface
                .borrow()
                .cache_neighbours(query_points);

            Ok(constraint)
        } else {
            Err(Error::InvalidImplicitSurface)
        }
    }

    /// Given an index into the surface point position coordinates, return the corresponding index
    /// into the original `TetMesh`.
    //pub fn tetmesh_coordinate_index(&self, idx: usize) -> usize {
    //    3 * self.sim_verts[idx / 3] + idx % 3
    //}

    pub fn update_query_points_with_mesh_pos(&self, x: &[f64]) {
        let pos = reinterpret_slice(x);
        self.update_query_points(self.sim_verts.iter().map(|&i| pos[i]));
    }

    pub fn update_query_points(&self, q_iter: impl Iterator<Item = [f64; 3]>) {
        let mut query_points = self.query_points.borrow_mut();
        query_points.clear();
        query_points.extend(q_iter);
    }
}

impl ContactConstraint for SPImplicitContactConstraint {
    fn frictional_contact(&self) -> Option<&FrictionalContact> {
        self.frictional_contact.as_ref()
    }
    fn frictional_contact_mut(&mut self) -> Option<&mut FrictionalContact> {
        self.frictional_contact.as_mut()
    }
    fn vertex_index_mapping(&self) -> Option<&[usize]> {
        Some(&self.sim_verts)
    }
    fn active_surface_vertex_indices(&self) -> ARef<'_, [usize]> {
        {
            let mut active_constraint_indices = self.active_constraint_indices.borrow_mut();
            active_constraint_indices.clear();
            if let Ok(mut indices) = self.active_constraint_indices() {
                active_constraint_indices.append(&mut indices);
            }
        }
        ARef::Cell(std::cell::Ref::map(
            self.active_constraint_indices.borrow(),
            |v| v.as_slice(),
        ))
    }

    fn contact_jacobian_af(&self) -> af::Array<f64> {
        // The contact jacobian for implicit collisions is just a selection matrix of vertices that
        // are in contact, since contacts are colocated with vertex positions.

        let surf_indices = self
            .active_constraint_indices()
            .expect("Failed to retrieve constraint indices.");

        let nnz = surf_indices.len();
        let values = vec![1.0; nnz];
        let rows: Vec<_> = (0i32..nnz as i32).collect();
        let cols: Vec<_> = surf_indices.iter().map(|&i| i as i32).collect();

        // Build ArrayFire matrix
        let nnz = nnz as u64;
        let num_rows = nnz as u64;
        let num_cols = self.sim_verts.len() as u64;

        let values = af::Array::new(&values, af::Dim4::new(&[nnz, 1, 1, 1]));
        let row_indices = af::Array::new(&rows, af::Dim4::new(&[nnz, 1, 1, 1]));
        let col_indices = af::Array::new(&cols, af::Dim4::new(&[nnz, 1, 1, 1]));

        af::sparse(
            num_rows,
            num_cols,
            &values,
            &row_indices,
            &col_indices,
            af::SparseFormat::COO,
        )
    }

    fn contact_jacobian_sprs(&self) -> sprs::CsMat<f64> {
        // The contact jacobian for implicit collisions is just a selection matrix of vertices that
        // are in contact, since contacts are colocated with vertex positions.

        let surf_indices = self
            .active_constraint_indices()
            .expect("Failed to retrieve constraint indices.");

        let nnz = surf_indices.len();
        let values = vec![1.0; nnz];
        let rows: Vec<_> = (0..nnz).collect();
        let cols: Vec<_> = surf_indices;

        let num_rows = nnz;
        let num_cols = self.sim_verts.len();

        sprs::TriMat::from_triplets((num_rows, num_cols), rows, cols, values).to_csr()
    }

    fn update_frictional_contact_impulse(
        &mut self,
        _contact_impulse: &[f64],
        x: &[[f64; 3]],
        v: &[[f64; 3]],
        _constraint_values: &[f64],
        mut friction_steps: u32,
    ) -> u32 {
        if self.frictional_contact.is_none() {
            return 0;
        }

        let normals = self
            .contact_normals(reinterpret::reinterpret_slice(x))
            .expect("Failed to compute contact normals.");
        let surf_indices = self
            .active_constraint_indices()
            .expect("Failed to retrieve constraint indices.");

        {
            // NOTE: here we rely on query points being updated in the contact_normals fn.
            let SPImplicitContactConstraint {
                constraint_buffer,
                implicit_surface,
                ..
            } = self;
            let mut potential_values = constraint_buffer.borrow_mut();
            implicit_surface
                .borrow()
                .potential(&self.query_points.borrow(), &mut potential_values)
                .unwrap();
        }

        // Define a subset of surface indices deemed to be in contact.
        let potential = self.constraint_buffer.borrow();

        // A set of masses on active contact vertices.
        let contact_masses: Vec<_> = surf_indices
            .iter()
            .map(|&surf_idx| self.vertex_masses[self.sim_verts[surf_idx]])
            .collect();

        let SPImplicitContactConstraint {
            ref mut frictional_contact,
            ref mut sim_verts,
            ..
        } = *self;

        let frictional_contact = frictional_contact.as_mut().unwrap(); // Must be checked above.
        frictional_contact
            .contact_basis
            .update_from_normals(normals);
        frictional_contact.impulse.clear();

        eprintln!("#### Solving Friction");
        let (velocity_n, velocity_t): (Vec<_>, Vec<_>) = surf_indices
            .iter()
            .enumerate()
            .map(|(contact_idx, &surf_idx)| {
                let vtx_idx = sim_verts[surf_idx];
                let v = frictional_contact
                    .contact_basis
                    .to_contact_coordinates(v[vtx_idx], contact_idx);
                if potential[surf_idx] <= 0.0 {
                    (v[0], [v[1], v[2]])
                } else {
                    // Dont constrain points that are not actually in contact.
                    (0.0, [0.0, 0.0])
                }
            })
            .unzip();

        let mut r_n = Vec::new();
        let mut r_t = Vec::new();
        while friction_steps > 0 {
            friction_steps -= 1;
            let mut solver = ContactSolver::without_contact_jacobian(
                &velocity_n,
                &r_t,
                &frictional_contact.contact_basis,
                &contact_masses,
            );
            r_n.clear();
            r_n.append(&mut solver.step());

            let mut solver = FrictionSolver::without_contact_jacobian(
                &velocity_t,
                &r_n,
                &frictional_contact.contact_basis,
                &contact_masses,
                frictional_contact.params,
            );
            r_t.clear();
            r_t.append(&mut solver.step());
        }

        assert_eq!(r_n.len(), r_t.len());
        frictional_contact
            .impulse
            .resize(surf_indices.len(), [0.0; 3]);
        for (i, (&r_n, &r_t, r)) in zip!(
            r_n.iter(),
            r_t.iter(),
            frictional_contact.impulse.iter_mut()
        )
        .enumerate()
        {
            *r = frictional_contact
                .contact_basis
                .from_contact_coordinates([r_n, r_t[0], r_t[1]], i)
                .into();
        }
        friction_steps
    }

    fn add_mass_weighted_frictional_contact_impulse(&self, x: &mut [f64]) {
        if let Some(ref frictional_contact) = self.frictional_contact {
            if frictional_contact.impulse.is_empty() {
                return;
            }

            let indices = self
                .active_constraint_indices()
                .expect("Failed to retrieve constraint indices.");

            assert_eq!(indices.len(), frictional_contact.impulse.len());

            for (&i, r) in indices.iter().zip(frictional_contact.impulse.iter()) {
                let surf_vert_idx = self.sim_verts[i];
                let m = self.vertex_masses[surf_vert_idx];
                for (j, impulse) in r.iter().enumerate().take(3) {
                    let idx = 3 * surf_vert_idx + j;
                    x[idx] += impulse / m;
                }
            }
        }
    }

    /// Add the frictional impulse to the given gradient vector.
    fn add_friction_impulse(&self, grad: &mut [f64], multiplier: f64) {
        let grad: &mut [Vector3<f64>] = reinterpret_mut_slice(grad);
        if let Some(ref frictional_contact) = self.frictional_contact() {
            if frictional_contact.impulse.is_empty() {
                return;
            }

            let indices = self.active_surface_vertex_indices();
            if indices.is_empty() {
                return;
            }

            assert_eq!(indices.len(), frictional_contact.impulse.len());
            for (&i, &r) in indices.iter().zip(frictional_contact.impulse.iter()) {
                let vertex_idx = self.vertex_index_mapping().map_or(i, |m| m[i]);
                grad[vertex_idx] += Vector3(r.into()) * multiplier;
            }
        }
    }

    fn remap_frictional_contact(&mut self, old_set: &[usize], new_set: &[usize]) {
        // Remap friction forces the same way we remap constraint multipliers for the contact
        // solve.
        if let Some(ref mut frictional_contact) = self.frictional_contact {
            let new_friction_impulses = crate::constraints::remap_values(
                frictional_contact.impulse.iter().cloned(),
                [0.0; 3],
                old_set.iter().cloned(),
                new_set.iter().cloned(),
            );
            std::mem::replace(&mut frictional_contact.impulse, new_friction_impulses);

            frictional_contact.contact_basis.remap(old_set, new_set);
        }
    }

    /// For visualization purposes.
    fn compute_contact_impulse(
        &self,
        x: &[f64],
        contact_impulse: &[f64],
        impulse: &mut [[f64; 3]],
    ) {
        let normals = self
            .contact_normals(x)
            .expect("Failed to retrieve contact normals.");
        let indices = self
            .active_constraint_indices()
            .expect("Failed to retrieve constraint indices.");

        let contact_impulse = if contact_impulse.is_empty() {
            vec![0.0; normals.len()]
        } else {
            contact_impulse.to_vec()
        };

        assert_eq!(contact_impulse.len(), normals.len());
        assert_eq!(indices.len(), normals.len());
        for (i, (n, &f)) in indices
            .into_iter()
            .zip(normals.into_iter().zip(contact_impulse.iter()))
        {
            impulse[self.sim_verts[i]] = (Vector3(n) * f).into();
        }
    }

    fn contact_normals(&self, x: &[f64]) -> Result<Vec<[f64; 3]>, Error> {
        // Contacts occur at vertex positions of the deforming volume mesh.
        let surf = self.implicit_surface.borrow();
        self.update_query_points_with_mesh_pos(x);
        let query_points = self.query_points.borrow();

        let mut normal_coords = vec![0.0; surf.num_query_jacobian_entries()?];
        surf.query_jacobian_values(&query_points, &mut normal_coords)?;
        let mut normals: Vec<Vector3<f64>> = reinterpret::reinterpret_vec(normal_coords);

        // Normalize normals.
        // Contact normals point away from the surface being collided against.
        // In this case the gradient coincides with this direction.
        for n in normals.iter_mut() {
            let len = n.norm();
            if len > 0.0 {
                *n /= len;
            }
        }

        Ok(reinterpret::reinterpret_vec(normals))
    }

    fn contact_radius(&self) -> f64 {
        self.implicit_surface.borrow().radius()
    }

    fn update_radius_multiplier(&mut self, rad_mult: f64) {
        self.implicit_surface
            .borrow_mut()
            .update_radius_multiplier(rad_mult);
    }

    fn update_max_step(&mut self, step: f64) {
        self.implicit_surface.borrow_mut().update_max_step(step);
    }

    fn active_constraint_indices(&self) -> Result<Vec<usize>, Error> {
        self.implicit_surface
            .borrow()
            .nonempty_neighbourhood_indices()
            .map_err(|_| Error::InvalidImplicitSurface)
    }

    fn update_cache(&mut self, x: Option<&[f64]>) -> bool {
        // If positions are given, then use them, otherwise, update cache using the vertex
        // positions in the simulation mesh.
        if let Some(x) = x {
            self.update_query_points_with_mesh_pos(x);
        } else {
            let sim_mesh = self.simulation_mesh.borrow();
            self.update_query_points_with_mesh_pos(reinterpret_slice(sim_mesh.vertex_positions()));
        }

        let collision_mesh = self.collision_object.borrow();
        let mut surf = self.implicit_surface.borrow_mut();

        let num_vertices_updated = surf.update(collision_mesh.vertex_position_iter().cloned());
        assert_eq!(num_vertices_updated, surf.surface_vertex_positions().len());

        surf.invalidate_query_neighbourhood();
        surf.cache_neighbours(&self.query_points.borrow())
    }

    /// Get a list of indices of the surface vertices on the simulation mesh which are neighbouring
    /// the implicit surface. These indices are with respect to the `sim_verts` vector,
    /// not the actual simulation mesh vertices.
    fn cached_neighbourhood_indices(&self) -> Vec<Index> {
        let surf = self.implicit_surface.borrow();

        let mut cached_neighbourhood_indices = if let Ok(n) = surf.num_cached_query_points() {
            vec![Index::INVALID; n]
        } else {
            return Vec::new();
        };

        let cached_neighbourhood_sizes = match surf.cached_neighbourhood_sizes() {
            Ok(c) => c,
            Err(_) => return cached_neighbourhood_indices,
        };

        for (i, (idx, _)) in cached_neighbourhood_indices
            .iter_mut()
            .zip(cached_neighbourhood_sizes.iter())
            .filter(|&(_, &s)| s != 0)
            .enumerate()
        {
            *idx = Index::new(i);
        }

        cached_neighbourhood_indices
    }
}

impl Constraint<f64> for SPImplicitContactConstraint {
    #[inline]
    fn constraint_size(&self) -> usize {
        0
    }

    #[inline]
    fn constraint_bounds(&self) -> (Vec<f64>, Vec<f64>) {
        let m = self.constraint_size();
        (vec![0.0; m], vec![2e10; m])
    }

    #[inline]
    fn constraint(&self, _x0: &[f64], _x1: &[f64], _value: &mut [f64]) {}
}

impl ConstraintJacobian<f64> for SPImplicitContactConstraint {
    #[inline]
    fn constraint_jacobian_size(&self) -> usize {
        0
    }

    fn constraint_jacobian_indices_iter<'a>(
        &'a self,
    ) -> Result<Box<dyn Iterator<Item = MatrixElementIndex> + 'a>, Error> {
        Ok(Box::new(std::iter::empty()))
    }

    fn constraint_jacobian_values(
        &self,
        _x0: &[f64],
        _x1: &[f64],
        _values: &mut [f64],
    ) -> Result<(), Error> {
        Ok(())
    }
}

impl ConstraintHessian<f64> for SPImplicitContactConstraint {
    #[inline]
    fn constraint_hessian_size(&self) -> usize {
        0
    }

    fn constraint_hessian_indices_iter<'a>(
        &'a self,
    ) -> Result<Box<dyn Iterator<Item = MatrixElementIndex> + 'a>, Error> {
        Ok(Box::new(std::iter::empty()))
    }

    fn constraint_hessian_values(
        &self,
        _x0: &[f64],
        _x1: &[f64],
        _lambda: &[f64],
        _scale: f64,
        _values: &mut [f64],
    ) -> Result<(), Error> {
        Ok(())
    }
}
