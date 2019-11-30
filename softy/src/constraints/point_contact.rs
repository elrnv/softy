use super::*;
use crate::constraint::*;
use crate::contact::*;
use crate::fem::problem::Var;
use crate::friction::*;
use crate::matrix::*;
use crate::Error;
use crate::Index;
use crate::TriMesh;
use geo::bbox::BBox;
use geo::mesh::topology::*;
use geo::mesh::{Attrib, VertexPositions};
use geo::ops::*;
use implicits::*;
use log::{debug, error};
use num_traits::Zero;
#[cfg(feature = "af")]
use reinterpret::*;
use std::cell::RefCell;
use utils::soap::*;
use utils::soap::{Matrix3, Vector2, Vector3};
use utils::zip;

/// Enforce a contact constraint on a mesh against animated vertices. This constraint prevents
/// vertices from occupying the same space as a smooth representation of the simulation mesh.
#[derive(Clone, Debug)]
pub struct PointContactConstraint {
    /// Implicit surface that represents the deforming object.
    pub implicit_surface: QueryTopo,
    /// Points where collision and contact occurs. I.e. all surface vertex positions on the
    /// collider mesh.
    pub contact_points: RefCell<Chunked3<Vec<f64>>>,

    /// Friction impulses applied during contact.
    pub frictional_contact: Option<FrictionalContact>,
    /// A mass inverse for each vertex in the object mesh.
    /// If the object is fixed, masses are effectively zero and this field
    /// will be `None`.
    pub object_mass_inv: Option<Chunked3<Vec<f64>>>,
    /// A mass inverse for each vertex in the collider mesh.
    /// If the collider is fixed, masses are effectively zero and this field
    /// will be `None`.
    pub collider_mass_inv: Option<Chunked3<Vec<f64>>>,

    /// A flag indicating if the object is fixed. Otherwise it's considered
    /// to be deforming, and thus appropriate derivatives are computed.
    object_is_fixed: bool,
    /// A flag indicating if the collider is fixed. Otherwise it's considered
    /// to be deforming, and thus appropriate derivatives are computed.
    collider_is_fixed: bool,

    /// The maximum distance between two points of the given geometry.
    ///
    /// This value is used to produce relative thresholds.
    problem_diameter: f64,

    /// Internal constraint function buffer used to store temporary constraint computations.
    constraint_buffer: RefCell<Vec<f64>>,

    /// Vertex to vertex topology of the collider mesh.
    collider_vertex_topo: Chunked<Vec<usize>>,
}

impl PointContactConstraint {
    #[allow(dead_code)]
    pub fn new(
        // Main object experiencing contact against its implicit surface representation.
        object: Var<&TriMesh>,
        // Collision object consisting of points pushing against the solid object.
        collider: Var<&TriMesh>,
        kernel: KernelType,
        friction_params: Option<FrictionParams>,
    ) -> Result<Self, Error> {
        let mut surface_builder = ImplicitSurfaceBuilder::new();
        let object_is_fixed = object.is_fixed();
        let collider_is_fixed = collider.is_fixed();
        let object = object.untag();
        let collider = collider.untag();
        surface_builder
            .trimesh(object)
            .kernel(kernel)
            .sample_type(SampleType::Face)
            .background_field(BackgroundFieldParams {
                field_type: BackgroundFieldType::DistanceBased,
                weighted: false,
            });

        if let Some(surface) = surface_builder.build_mls() {
            // Sanity check that the surface is built correctly.
            assert_eq!(
                surface.surface_vertex_positions().len(),
                object.num_vertices()
            );

            let query_points = collider.vertex_positions();

            let object_mass_inv = Self::mass_inv_attribute(&object)?;

            let collider_mass_inv = Self::mass_inv_attribute(&collider)?;

            let mut bbox = BBox::empty();
            bbox.absorb(object.bounding_box());
            bbox.absorb(collider.bounding_box());

            let constraint = PointContactConstraint {
                implicit_surface: surface.query_topo(query_points),
                contact_points: RefCell::new(Chunked3::from_array_vec(query_points.to_vec())),
                frictional_contact: friction_params.and_then(|fparams| {
                    if fparams.dynamic_friction > 0.0 {
                        Some(FrictionalContact::new(fparams))
                    } else {
                        None
                    }
                }),
                object_mass_inv,
                collider_mass_inv,
                object_is_fixed,
                collider_is_fixed,
                constraint_buffer: RefCell::new(vec![0.0; query_points.len()]),
                problem_diameter: bbox.diameter(),
                collider_vertex_topo: Self::build_vertex_topo(collider),
            };

            Ok(constraint)
        } else {
            Err(Error::InvalidImplicitSurface)
        }
    }

    fn mass_inv_attribute(mesh: &TriMesh) -> Result<Option<Chunked3<Vec<f64>>>, Error> {
        match mesh.attrib_as_slice::<MassType, VertexIndex>(MASS_ATTRIB) {
            Ok(attrib) => {
                if !attrib.iter().all(|&x| x > 0.0) {
                    Err(Error::InvalidParameter {
                        name: "Zero mass".to_string(),
                    })
                } else {
                    Ok(Some(attrib.iter().map(|&x| [1.0 / x; 3]).collect()))
                }
            }
            _ => Ok(None),
        }
    }

    /// Compute vertex to vertex topology of the entire collider mesh. This makes it easy to query
    /// neighbours when we are computing laplacians.
    fn build_vertex_topo(mesh: &TriMesh) -> Chunked<Vec<usize>> {
        let mut topo = vec![Vec::with_capacity(5); mesh.num_vertices()];

        // Accumulate all the neighbours
        mesh.face_iter().for_each(|&[v0, v1, v2]| {
            topo[v0].push(v1);
            topo[v0].push(v2);
            topo[v1].push(v0);
            topo[v1].push(v2);
            topo[v2].push(v0);
            topo[v2].push(v1);
        });

        // Sort and dedup neighbourhoods
        for nbrhood in topo.iter_mut() {
            nbrhood.sort();
            nbrhood.dedup();
        }

        // Flatten data layout to make access faster.
        Chunked::from_nested_vec(topo)
    }

    /// Build a matrix that smoothes values at contact points with their neighbours by the given
    /// weight. For `weight = 0.0`, no smoothing is performed, and this matrix is the identity.
    fn build_contact_laplacian(
        &self,
        weight: f64,
        active_contact_indices: Option<&[usize]>,
    ) -> DSMatrix {
        let surf = &self.implicit_surface;
        if let Some(active_contact_indices) = active_contact_indices {
            let size = active_contact_indices.len();
            let mut neighbourhood_indices = vec![Index::INVALID; surf.num_query_points()];
            for (i, &aci) in active_contact_indices.iter().enumerate() {
                neighbourhood_indices[aci] = Index::new(i);
            }

            let triplets =
                active_contact_indices
                    .iter()
                    .enumerate()
                    .flat_map(|(valid_idx, &active_idx)| {
                        let nbrhood = &self.collider_vertex_topo[active_idx];
                        let n = nbrhood
                            .iter()
                            .filter(|&nbr_idx| neighbourhood_indices[*nbr_idx].is_valid())
                            .count();
                        std::iter::repeat((valid_idx, weight / n as f64))
                            .zip(nbrhood.iter())
                            .filter_map(|((valid_idx, normalized_weight), nbr_idx)| {
                                neighbourhood_indices[*nbr_idx]
                                    .into_option()
                                    .map(|valid_nbr| (valid_idx, valid_nbr, normalized_weight))
                            })
                            .chain(std::iter::once((valid_idx, valid_idx, 1.0 - weight)))
                    });
            // Don't need to sort or compress.
            DSMatrix::from_sorted_triplets_iter_uncompressed(triplets, size, size)
        } else {
            let neighbourhood_indices = nonempty_neighbourhood_indices(&surf);
            let size = surf.num_neighbourhoods();
            let triplets = self
                .collider_vertex_topo
                .iter()
                .zip(neighbourhood_indices.iter())
                .filter_map(|(nbrhood, idx)| idx.into_option().map(|i| (nbrhood, i)))
                .flat_map(|(nbrhood, valid_idx)| {
                    let n = nbrhood
                        .iter()
                        .filter(|&nbr_idx| neighbourhood_indices[*nbr_idx].is_valid())
                        .count();
                    std::iter::repeat((valid_idx, weight / n as f64))
                        .zip(nbrhood.iter())
                        .filter_map(|((valid_idx, normalized_weight), nbr_idx)| {
                            neighbourhood_indices[*nbr_idx]
                                .into_option()
                                .map(|valid_nbr| (valid_idx, valid_nbr, normalized_weight))
                        })
                        .chain(std::iter::once((valid_idx, valid_idx, 1.0 - weight)))
                });
            // Don't need to sort or compress.
            DSMatrix::from_sorted_triplets_iter_uncompressed(triplets, size, size)
        }
    }

    /// Update implicit surface using the given position data from mesh vertices.
    /// Return the number of positions that were actually updated.
    pub fn update_surface_with_mesh_pos(&mut self, pos: SubsetView<Chunked3<&[f64]>>) -> usize {
        self.implicit_surface.update_surface(pos.iter().cloned())
    }

    pub fn update_contact_points(&mut self, x: SubsetView<Chunked3<&[f64]>>) {
        let mut contact_points = self.contact_points.borrow_mut();
        x.clone_into_other(&mut *contact_points);
    }

    ///// Update implicit surface using the given position and displacement data.
    //pub fn update_surface_with_displacement(&self, x: &[f64], dx: &[f64]) {
    //    let all_displacements: &[Vector3<f64>] = reinterpret_slice(dx);
    //    let all_positions: &[Vector3<f64>] = reinterpret_slice(x);
    //    let points_iter = self
    //        .sim_verts
    //        .iter()
    //        .map(|&i| (all_positions[i] + all_displacements[i]).into());

    //    self.implicit_surface.borrow_mut().update(points_iter);
    //}

    /* Needed for the Linear constraint
    /// Update implicit surface using the given position data.
    pub fn update_surface(&self, x: &[f64]) {
        let all_positions: &[[f64; 3]] = reinterpret_slice(x);
        let points_iter = self.sim_verts.iter().map(|&i| all_positions[i]);

        self.implicit_surface.borrow_mut().update(points_iter);
    }
    */

    #[allow(dead_code)]
    fn background_points(&self) -> Vec<bool> {
        let neighbourhood_sizes = self.implicit_surface.neighbourhood_sizes();

        let mut background_points = vec![true; neighbourhood_sizes.len()];

        for (_, bg) in neighbourhood_sizes
            .iter()
            .zip(background_points.iter_mut())
            .filter(|&(&c, _)| c != 0)
        {
            *bg = false;
        }

        background_points
    }

    /// This function fills the non-local values of the constraint function with a constant signed
    /// value (equal to the contact radius in magnitude) to help the optimization determine
    /// feasible regions. This is done using a flood fill algorithm as follows.
    /// 1. Identify non-local query poitns with `neighbourhood_sizes`.
    /// 2. Partition the primitives of the kinematic object (from which the points are from) into
    ///    connected components of non-local points. This means that points with a valid local
    ///    potential serve as boundaries.
    /// 3. During the splitting above, record whether a component must be inside or outside
    ///    depending on the sign of its boundary points (local points).
    /// 4. It could happen that a connected component has no local points, in which case we do a
    ///    ray cast in the x direction from any point and intersect it with our mesh to determine
    ///    the winding number. (TODO)
    /// 5. It could also happen that the local points don't separate the primitives into inside
    ///    and outside partitions if the radius is not sufficiently large. This is a problem for
    ///    the future (FIXME)
    #[allow(dead_code)]
    fn fill_background_potential(
        mesh: &TriMesh,
        background_points: &[bool],
        abs_fill_val: f64,
        values: &mut [f64],
    ) {
        debug_assert!(abs_fill_val >= 0.0);

        let mut hedge_dest_indices = vec![Vec::new(); mesh.num_vertices()];
        for f in mesh.face_iter() {
            for vtx in 0..3 {
                // Get an edge with vertices in sorted order.
                let edge = [f[vtx], f[(vtx + 1) % 3]];

                let neighbourhood = &mut hedge_dest_indices[edge[0]];

                if let Err(idx) = neighbourhood.binary_search_by(|x: &usize| x.cmp(&edge[1])) {
                    neighbourhood.insert(idx, edge[1]);
                }
            }
        }
        //println!("edges:");
        //for (i, v) in hedge_dest_indices.iter().enumerate() {
        //    for &vtx in v.iter() {
        //        println!("({}, {})", i, vtx);
        //    }
        //}
        //dbg!(background_points);

        let mut vertex_is_inside = vec![false; mesh.num_vertices()];
        for vidx in (0..mesh.num_vertices()).filter(|&i| !background_points[i]) {
            vertex_is_inside[vidx] = values[vidx] < 0.0;
        }

        let mut seen_vertices = vec![false; mesh.num_vertices()];

        let mut queue = std::collections::VecDeque::new();

        for vidx in (0..mesh.num_vertices()).filter(|&i| !background_points[i]) {
            if seen_vertices[vidx] {
                continue;
            }

            let is_inside = vertex_is_inside[vidx];

            queue.push_back(vidx);

            while let Some(vidx) = queue.pop_front() {
                if background_points[vidx] {
                    if seen_vertices[vidx] {
                        continue;
                    } else {
                        vertex_is_inside[vidx] = is_inside;
                    }
                }

                seen_vertices[vidx] = true;

                queue.extend(
                    hedge_dest_indices[vidx]
                        .iter()
                        .filter(|&&i| background_points[i])
                        .filter(|&&i| !seen_vertices[i]),
                );
            }
        }

        for ((&is_inside, &bg), val) in vertex_is_inside
            .iter()
            .zip(background_points.iter())
            .zip(values.iter_mut())
        {
            if bg {
                if is_inside {
                    *val = -abs_fill_val;
                } else {
                    *val = abs_fill_val;
                }
            }
        }
    }

    /// Prune contacts with zero contact_impulse and contacts without neighbouring samples.
    /// This function outputs the indices of contacts as well as a pruned vector of impulses.
    fn in_contact_indices(
        &self,
        contact_impulse: &[f64],
        potential: &[f64],
    ) -> (Vec<usize>, Vec<usize>, Vec<f64>) {
        let surf = &self.implicit_surface;
        let query_points = self.contact_points.borrow();
        let radius = surf.radius() * 0.999;
        let query_indices = self.active_constraint_indices();
        assert_eq!(query_indices.len(), contact_impulse.len());
        assert_eq!(potential.len(), contact_impulse.len());
        let dist_scale = 1.0 / self.problem_diameter;
        let (active_constraint_subset, contact_impulse): (Vec<_>, Vec<_>) = contact_impulse
            .iter()
            .zip(potential.iter())
            .enumerate()
            .filter_map(|(i, (&cf, dist))| {
                if cf != 0.0
                    && dist * dist_scale < 1e-4
                    && surf.num_neighbours_within_distance(query_points[query_indices[i]], radius)
                        > 0
                {
                    Some((i, cf))
                } else {
                    None
                }
            })
            .unzip();

        let active_contact_indices: Vec<_> = active_constraint_subset
            .iter()
            .map(|&i| query_indices[i])
            .collect();

        (
            active_constraint_subset,
            active_contact_indices,
            contact_impulse,
        )
    }

    fn compute_contact_jacobian(&self, active_contact_indices: &[usize]) -> ContactJacobian {
        let query_points = self.contact_points.borrow();
        let surf = &self.implicit_surface;
        let active_contact_points =
            Subset::from_unique_ordered_indices(active_contact_indices, query_points.view());

        // Compute contact Jacobian
        let jac_triplets =
            build_triplet_contact_jacobian(&surf, active_contact_points, query_points.view());
        let jac: ContactJacobian = jac_triplets.into();
        let jac = jac.pruned(|_, _, block| block.into_inner() != [[0.0; 3]; 3]);
        //jac.write_img("./out/jac.png");
        jac
    }

    fn compute_effective_mass_inv(
        &self,
        active_contact_indices: &[usize],
        jac: ContactJacobianView,
    ) -> EffectiveMassInv {
        // Construct diagonal mass matrices for object and collider.
        let object_zero_mass_inv = Chunked3::from_array_vec(vec![[0.0; 3]; jac.num_cols()]);
        let collider_zero_mass_inv = Chunked3::from_array_vec(vec![[0.0; 3]; jac.num_rows()]);

        let object_mass_inv = DiagonalBlockMatrix::from_uniform(
            self.object_mass_inv
                .as_ref()
                .map(|mass_inv| mass_inv.view())
                .unwrap_or_else(|| object_zero_mass_inv.view()),
        );

        // Collider mass matrix is constructed at active contacts only.
        let collider_mass_inv = DiagonalBlockMatrix::from_subset(
            self.collider_mass_inv
                .as_ref()
                .map(|mass_inv| {
                    Subset::from_unique_ordered_indices(active_contact_indices, mass_inv.view())
                })
                .unwrap_or_else(|| Subset::all(collider_zero_mass_inv.view())),
        );

        let mut jac_mass = Tensor::new(jac.data.clone().into_owned());
        jac_mass *= object_mass_inv.view();

        //jac_mass.write_img("./out/jac_mass.png");

        let effective_mass_inv = jac_mass.view() * jac.view().transpose();
        let effective_mass_inv = effective_mass_inv.view() + collider_mass_inv.view();

        //effective_mass_inv.write_img("./out/effective_mass_inv.png");
        effective_mass_inv
    }

    fn compute_predictor_impulse(
        v: [SubsetView<Chunked3<&[f64]>>; 2],
        active_contact_indices: &[usize],
        jac: ContactJacobianView,
        effective_mass_inv: EffectiveMassInvView,
    ) -> Chunked3<Vec<f64>> {
        let collider_velocity = Subset::from_unique_ordered_indices(active_contact_indices, v[1]);

        let mut object_velocity = jac.view() * Tensor::new(v[0]);
        *&mut object_velocity.expr_mut() -= collider_velocity.expr();

        let sprs_effective_mass_inv: sprs::CsMat<f64> = effective_mass_inv.clone().into();
        //        let ldlt_solver =
        //            sprs_ldl::LdlNumeric::<f64, usize>::new(sprs_effective_mass_inv.view()).unwrap();
        //        let predictor_impulse = Chunked3::from_flat(ldlt_solver.solve(rhs.storage()));

        if !object_velocity.is_empty() {
            sprs::linalg::trisolve::lsolve_csr_dense_rhs(
                sprs_effective_mass_inv.view(),
                object_velocity.storage_mut(),
            )
            .unwrap();
        }

        // The solve turns our relative velocity into a relative impulse.
        object_velocity.data
    }
}

/// Enumerate non-empty neighbourhoods in place.
fn nonempty_neighbourhood_indices(surf: &QueryTopo) -> Vec<Index> {
    neighbourhood_indices_with(surf, |_, s| s != 0)
}

/// Prune neighbourhood indices using a given function that takes the index (query point)
/// and size of the neighbourhood.
/// Only those neighbourhoods for which `f` returns true will be present in the output.
fn neighbourhood_indices_with(surf: &QueryTopo, f: impl Fn(usize, usize) -> bool) -> Vec<Index> {
    let mut neighbourhood_indices = vec![Index::INVALID; surf.num_query_points()];

    let neighbourhood_sizes = surf.neighbourhood_sizes();

    for (i, (_, (idx, _))) in neighbourhood_indices
        .iter_mut()
        .zip(neighbourhood_sizes.iter())
        .enumerate()
        .filter(|&(i, (_, &s))| f(i, s))
        .enumerate()
    {
        *idx = Index::new(i);
    }

    neighbourhood_indices
}

impl ContactConstraint for PointContactConstraint {
    // Get the total number of contacts that could potentially occur.
    fn num_potential_contacts(&self) -> usize {
        self.contact_points.borrow().len()
    }
    fn frictional_contact(&self) -> Option<&FrictionalContact> {
        self.frictional_contact.as_ref()
    }
    fn frictional_contact_mut(&mut self) -> Option<&mut FrictionalContact> {
        self.frictional_contact.as_mut()
    }
    fn active_surface_vertex_indices(&self) -> ARef<'_, [usize]> {
        ARef::Plain(&[])
    }

    #[cfg(feature = "af")]
    fn contact_jacobian_af(&self) -> af::Array<f64> {
        // Compute contact jacobian
        let surf = &self.implicit_surface;
        let query_points = self.contact_points.borrow();

        let mut cj_values = vec![0.0; surf.num_contact_jacobian_entries()];
        surf.contact_jacobian_values(
            query_points.view().into(),
            reinterpret_mut_slice(&mut cj_values),
        );
        let cj_indices_iter = surf.contact_jacobian_indices_iter();

        let nnz = self.constraint_jacobian_size();
        let mut rows = vec![0i32; nnz];
        let mut cols = vec![0i32; nnz];

        for ((row, col), (r, c)) in cj_indices_iter.zip(rows.iter_mut().zip(cols.iter_mut())) {
            *r = row as i32;
            *c = col as i32;
        }

        // Build ArrayFire matrix
        let nnz = nnz as u64;
        let num_rows = 3 * query_points.len() as u64;
        let num_cols = 3 * surf.surface_vertex_positions().len() as u64;

        let values = af::Array::new(&cj_values, af::Dim4::new(&[nnz, 1, 1, 1]));
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

    fn collider_contact_normals(&self, mut out_normals: Chunked3<&mut [f64]>) {
        if self.frictional_contact.is_none() {
            return;
        }

        let normals = self.contact_normals();
        let FrictionalContact {
            collider_impulse, // for active point contacts
            ..
        } = self.frictional_contact.as_ref().unwrap();

        let query_indices = self.implicit_surface.nonempty_neighbourhood_indices();
        assert_eq!(query_indices.len(), normals.len());

        // Only interested in normals at contact points on the collider impulse.
        let remapped_normals: Chunked3<Vec<f64>> = crate::constraints::remap_values_iter(
            normals.into_iter(),
            [0.0; 3], // Default normal (there should not be any).
            query_indices.into_iter(),
            collider_impulse.selection().index_iter().cloned(),
        )
        .collect();

        for (&aci, &nml) in zip!(
            collider_impulse.selection().index_iter(),
            remapped_normals.iter(),
        ) {
            out_normals[aci] = nml;
        }
    }

    fn project_friction_impulses(&mut self, x: [SubsetView<Chunked3<&[f64]>>; 2]) {
        if self.frictional_contact.is_none() {
            return;
        }

        let normals = self.contact_normals();
        let query_indices = self.active_constraint_indices();

        let FrictionalContact {
            contact_basis,
            object_impulse,
            collider_impulse, // for active point contacts
            ..
        } = self.frictional_contact.as_mut().unwrap();

        // Only interested in normals at contact points on the collider impulse.
        let remapped_normals: Chunked3<Vec<f64>> = crate::constraints::remap_values_iter(
            normals.into_iter(),
            [0.0; 3], // Default normal (there should not be many).
            query_indices.into_iter(),
            collider_impulse.selection().index_iter().cloned(),
        )
        .collect();

        if remapped_normals.is_empty() {
            return;
        }

        let mut collider_impulse_clone = collider_impulse.clone();
        debug!(
            "Impulse Before: {:?}",
            collider_impulse.source().view().at(0).1
        );
        for (&n, (_, imp)) in remapped_normals
            .iter()
            .zip(collider_impulse.source_iter_mut())
        {
            let n = Vector3::new(n);
            let nml_component = n.dot(Vector3::new(*imp));
            *imp.as_mut_tensor() -= n * nml_component;
        }
        debug!(
            "Impulse After subtract normal: {:?}",
            collider_impulse.source().view().at(0).1
        );

        assert_eq!(remapped_normals.len(), collider_impulse.len());

        contact_basis.update_from_normals(remapped_normals.into());
        dbg!(contact_basis.normals[0]);
        dbg!(contact_basis.tangents[0]);
        if !contact_basis.is_empty() {
            let dense_collider_impulse: Vec<[f64; 3]> = collider_impulse_clone
                .source_iter()
                .map(|(_, &src)| src)
                .collect();

            debug!("Impulse Before: {:?}", &dense_collider_impulse[0]);
            let dense_collider_impulse: Vec<[f64; 2]> = contact_basis
                .to_tangent_space(&dense_collider_impulse)
                .collect();
            debug!("Impulse Contact Space: {:?}", &dense_collider_impulse[0]);
            let dense_collider_impulse: Chunked3<Vec<f64>> = contact_basis
                .from_tangent_space(&dense_collider_impulse)
                .collect();
            debug!(
                "Impulse After project tangent: {:?}",
                &dense_collider_impulse[0]
            );

            assert_eq!(collider_impulse.len(), dense_collider_impulse.len());
            collider_impulse_clone
                .source_mut()
                .data_mut()
                .1
                .copy_from_slice(dense_collider_impulse.data());
        }

        for (i, ((ai, a, at), (bi, b, bt))) in collider_impulse_clone
            .iter()
            .zip(collider_impulse.iter())
            .enumerate()
        {
            assert_eq!(at, bt);
            assert_eq!(ai, bi);
            dbg!(contact_basis.normals[i]);
            dbg!(contact_basis.tangents[i]);
            approx::assert_relative_eq!(a.0.as_tensor(), b.0.as_tensor());
            approx::assert_relative_eq!(a.1.as_tensor(), b.1.as_tensor());
        }

        // Project object impulse
        let normals = self.implicit_surface.surface_vertex_normals();
        for (&n, (_, imp)) in normals.iter().zip(object_impulse.iter_mut()) {
            let n = Vector3::new(n);
            let nml_component = n.dot(Vector3::new(*imp));
            *imp.as_mut_tensor() -= n * nml_component;
        }
    }

    /// Update the position configuration of contacting objects using the given position data.
    fn update_contact_pos(&mut self, x: [SubsetView<Chunked3<&[f64]>>; 2]) {
        self.update_surface_with_mesh_pos(x[0]);
        self.update_contact_points(x[1]);
    }

    fn update_frictional_contact_impulse(
        &mut self,
        orig_contact_impulse_n: &[f64],
        x: [SubsetView<Chunked3<&[f64]>>; 2],
        v: [SubsetView<Chunked3<&[f64]>>; 2],
        potential_values: &[f64],
        mut friction_steps: u32,
    ) -> u32 {
        if self.frictional_contact.is_none() || friction_steps == 0 {
            return 0;
        }

        self.update_contact_pos(x);

        // Note that there is a distinction between active *contacts* and active *constraints*.
        // Active *constraints* correspond to to those points that are in the MLS neighbourhood of
        // influence to be part of the optimization. Active *contacts* are a subset of those that
        // are considered in contact.
        let (active_constraint_subset, active_contact_indices, orig_contact_impulse_n) =
            self.in_contact_indices(orig_contact_impulse_n, potential_values);

        let normals = self.contact_normals();
        let normals_subset = Subset::from_unique_ordered_indices(active_constraint_subset, normals);
        let mut normals = Chunked3::from_array_vec(vec![[0.0; 3]; normals_subset.len()]);
        normals_subset.clone_into_other(&mut normals);

        self.frictional_contact
            .as_mut()
            .unwrap()
            .contact_basis
            .update_from_normals(normals.into());

        let jac = self.compute_contact_jacobian(&active_contact_indices);
        let effective_mass_inv =
            self.compute_effective_mass_inv(&active_contact_indices, jac.view());

        let FrictionalContact {
            contact_basis,
            params,
            object_impulse,
            collider_impulse, // for active point contacts
        } = self.frictional_contact.as_mut().unwrap();

        // A new set of contacts have been determined. We should remap the previous friction
        // impulses to match new impulses.
        let mut prev_friction_impulse: Chunked3<Vec<f64>> = crate::constraints::remap_values_iter(
            collider_impulse.source_iter().map(|x| *x.1),
            [0.0; 3], // Previous impulse for unmatched contacts.
            collider_impulse.selection().index_iter().cloned(),
            active_contact_indices.iter().cloned(),
        )
        .collect();

        // Initialize the new friction impulse in physical space at active contacts.
        let mut friction_impulse =
            Chunked3::from_array_vec(vec![[0.0; 3]; active_contact_indices.len()]);

        // TODO: REMOVE this
        //let mut prev_friction_impulse = friction_impulse.clone();

        if active_contact_indices.is_empty() {
            // If there are no active contacts, there is nothing to update.
            // Clear object_impulse and collider_impulse before returning.
            collider_impulse.clear();
            object_impulse.iter_mut().for_each(|(x, y)| {
                *x = [0.0; 3];
                *y = [0.0; 3]
            });
            return 0;
        }

        let mut contact_impulse: Chunked3<Vec<f64>> = contact_basis
            .from_normal_space(&orig_contact_impulse_n)
            .collect();
        // Prepare true predictor for the friction solve.
        let predictor_impulse = Self::compute_predictor_impulse(
            v,
            &active_contact_indices,
            jac.view(),
            effective_mass_inv.view(),
        );
        // Project out the normal component.
        let predictor_impulse: Vec<_> = contact_basis
            .to_tangent_space(&predictor_impulse.into_arrays())
            .collect();
        let predictor_impulse: Chunked3<Vec<_>> = contact_basis
            .from_tangent_space(&predictor_impulse)
            .collect();

        // Friction impulse to be subtracted.
        let prev_step_friction_impulse = prev_friction_impulse.clone();

        let predictor_impulse: Chunked3<Vec<f64>> =
            (predictor_impulse.expr() + contact_impulse.expr() + prev_friction_impulse.expr())
                .eval();
        let success = if false {
            // Polar coords
            let predictor_impulse_t: Vec<_> = predictor_impulse
                .iter()
                .enumerate()
                .map(|(aqi, &predictor_imp)| {
                    let r = contact_basis.to_cylindrical_contact_coordinates(predictor_imp, aqi);
                    r.tangent
                })
                .collect();
            if true {
                // switch between implicit solver and explicit solver here.
                match FrictionPolarSolver::new(
                    &predictor_impulse_t,
                    &orig_contact_impulse_n,
                    &contact_basis,
                    effective_mass_inv.view(),
                    *params,
                    jac.view(),
                ) {
                    Ok(mut solver) => {
                        debug!("Solving Friction");
                        if let Ok(FrictionSolveResult { solution: r_t, .. }) = solver.step() {
                            for ((aqi, &r), r_out) in
                                r_t.iter().enumerate().zip(friction_impulse.iter_mut())
                            {
                                let r_polar = Polar2 {
                                    radius: r[0],
                                    angle: r[1],
                                };
                                *r_out = contact_basis
                                    .from_cylindrical_contact_coordinates(r_polar.into(), aqi)
                                    .into();
                            }
                            true
                        } else {
                            error!("Failed friction solve");
                            false
                        }
                    }
                    Err(err) => {
                        error!("Failed to create friction solver: {}", err);
                        false
                    }
                }
            } else {
                for (aqi, (&pred_r_t, &cr, r_out)) in zip!(
                    predictor_impulse_t.iter(),
                    orig_contact_impulse_n.iter(),
                    friction_impulse.iter_mut()
                )
                .enumerate()
                {
                    let r_t = if pred_r_t.radius > 0.0 {
                        Polar2 {
                            radius: params.dynamic_friction * cr.abs(),
                            angle: negate_angle(pred_r_t.angle),
                        }
                    } else {
                        Polar2 {
                            radius: 0.0,
                            angle: 0.0,
                        }
                    };
                    *r_out = contact_basis
                        .from_cylindrical_contact_coordinates(r_t.into(), aqi)
                        .into();
                }
                true
            }
        } else {
            let mut contact_impulse_n = orig_contact_impulse_n.clone();
            let prev_friction_impulse_t: Vec<_> = contact_basis
                .to_tangent_space(&prev_friction_impulse.view().into_arrays())
                .collect();

            let prev_friction_impulse_t = vec![[0.0; 2]; prev_friction_impulse_t.len()];

            // Euclidean coords
            if true {
                // Switch between implicit solver and explicit solver here.
                loop {
                    //println!("predictor: {:?}", predictor_impulse.view());
                    let friction_predictor: Chunked3<Vec<f64>> =
                        (predictor_impulse.expr() - contact_impulse.expr()).eval();
                    //println!("f_predictor: {:?}", friction_predictor.view());
                    match crate::friction::solver::FrictionSolver::new(
                        friction_predictor.view().into(),
                        &prev_friction_impulse_t,
                        &contact_impulse_n,
                        &contact_basis,
                        effective_mass_inv.view(),
                        *params,
                    ) {
                        Ok(mut solver) => {
                            debug!("Solving Friction");

                            if let Ok(FrictionSolveResult { solution: r_t, .. }) = solver.step() {
                                friction_impulse = contact_basis.from_tangent_space(&r_t).collect();
                            } else {
                                error!("Failed friction solve");
                                break false;
                            }
                        }
                        Err(err) => {
                            error!("Failed to create friction solver: {}", err);
                            break false;
                        }
                    }

                    //println!("c_before: {:?}", contact_impulse_n);
                    let contact_predictor: Chunked3<Vec<f64>> =
                        (predictor_impulse.expr() - friction_impulse.expr()).eval();
                    //println!("c_predictor: {:?}", contact_predictor.view());

                    let contact_impulse_n_copy = contact_impulse_n.clone();
                    match crate::friction::contact_solver::ContactSolver::new(
                        contact_predictor.view().into(),
                        &contact_impulse_n_copy,
                        &contact_basis,
                        effective_mass_inv.view(),
                        *params,
                    ) {
                        Ok(mut solver) => {
                            debug!("Solving Contact");

                            if let Ok(r_n) = solver.step() {
                                contact_impulse_n.copy_from_slice(&r_n);
                                contact_impulse.clear();
                                contact_impulse
                                    .extend(contact_basis.from_normal_space(&contact_impulse_n));
                            } else {
                                error!("Failed contact solve");
                                break false;
                            }
                        }
                        Err(err) => {
                            error!("Failed to create contact solver: {}", err);
                            break false;
                        }
                    }

                    //println!("c_after: {:?}", contact_impulse_n);
                    //println!("c_after_full: {:?}", contact_impulse.view());

                    let f_prev = Tensor::new(prev_friction_impulse.view());
                    let f_cur = Tensor::new(friction_impulse.view());
                    //println!("prev friction impulse: {:?}", f_prev.norm());
                    //println!("cur friction impulse: {:?}", f_cur.norm());

                    let f_delta: Chunked3<Vec<f64>> = (f_prev.expr() - f_cur.expr()).eval();
                    let rel_err_numerator: f64 = f_delta
                        .expr()
                        .dot((effective_mass_inv.view() * Tensor::new(f_delta.view())).expr());
                    let rel_err = rel_err_numerator
                        / f_prev
                            .expr()
                            .dot::<f64, _>((effective_mass_inv.view() * f_prev.view()).expr());

                    debug!("Friction relative error: {}", rel_err);
                    if rel_err < 1e-3 {
                        friction_steps = 0;
                        break true;
                    }

                    // Update prev_friction_impulse for computing error subsequent iterations.
                    // Note that this should not and does not affect the "prev_friction_impulse_t"
                    // variable which is used in friciton forwarding and set outside the loop.
                    prev_friction_impulse = friction_impulse.clone();

                    friction_steps -= 1;

                    if friction_steps == 0 {
                        break true;
                    }
                }
            } else {
                let predictor_impulse_t: Vec<_> = contact_basis
                    .to_tangent_space(predictor_impulse.view().into())
                    .collect();
                for (aqi, (&pred_r_t, &cr, r_out)) in zip!(
                    predictor_impulse_t.iter(),
                    orig_contact_impulse_n.iter(),
                    friction_impulse.iter_mut(),
                )
                .enumerate()
                {
                    let pred_r_t = Vector2::new(pred_r_t);
                    let pred_r_norm = pred_r_t.norm();
                    let r_t = if pred_r_norm > 0.0 {
                        pred_r_t * (params.dynamic_friction * cr.abs() / pred_r_norm)
                    } else {
                        Vector2::zero()
                    };
                    *r_out = contact_basis
                        .from_contact_coordinates([0.0, r_t[0], r_t[1]], aqi)
                        .into();
                }
                true
            }
        };

        if !success {
            return 0;
        }

        // Now we apply the contact jacobian to map the frictional impulses at contact points (on
        // collider vertices) to the vertices of the simulation mesh. Given contact jacobian J, and
        // frictional impulses r (in physical space), J^T*r produces frictional impulses on the
        // deforming surface mesh. An additional remapping puts these impulses on the volume mesh
        // vertices, but this is applied when the friction impulses are actually used.
        // Compute transpose product J^T*f
        //let prev_contact_impulse: Chunked3<Vec<f64>> = contact_basis
        //    .from_normal_space(&orig_contact_impulse_n)
        //    .collect();
        //let impulse: Chunked3<Vec<f64>> =
        //    (friction_impulse.expr()).eval();
        // Correct friction_impulse by subtracting previous friction impulse
        let impulse_corrector: Chunked3<Vec<f64>> = (friction_impulse.expr()
            //+ contact_impulse.expr()
            //- prev_contact_impulse.expr()
            - prev_step_friction_impulse.expr())
        .eval();
        let mut object_friction_impulse_tensor =
            jac.view().transpose() * Tensor::new(friction_impulse.view());
        object_friction_impulse_tensor.negate();

        let mut object_impulse_corrector_tensor =
            jac.view().transpose() * Tensor::new(impulse_corrector.view());
        object_impulse_corrector_tensor.negate();

        *object_impulse = Chunked3::from_flat((
            object_impulse_corrector_tensor.data.into_flat(),
            object_friction_impulse_tensor.data.into_flat(),
        ));

        *collider_impulse = Sparse::from_dim(
            active_contact_indices.clone(),
            self.contact_points.borrow().len(),
            Chunked3::from_flat((impulse_corrector.into_flat(), friction_impulse.into_flat())),
        );

        if friction_steps > 0 {
            friction_steps - 1
        } else {
            0
        }
    }

    fn add_mass_weighted_frictional_contact_impulse(
        &self,
        [mut object_vel, collider_vel]: [SubsetView<Chunked3<&mut [f64]>>; 2],
    ) {
        if let Some(ref frictional_contact) = self.frictional_contact {
            if !frictional_contact.object_impulse.is_empty() {
                if let Some(masses) = self.object_mass_inv.as_ref() {
                    let mass_mtx = DiagonalBlockMatrix::new(masses.view());
                    let corrector =
                        Chunked3::from_flat(frictional_contact.object_impulse.view().into_flat().0);
                    let add_vel = mass_mtx.view() * Tensor::new(corrector);
                    *&mut object_vel.expr_mut() += add_vel.expr();
                }
            }

            if frictional_contact.collider_impulse.is_empty() || self.collider_mass_inv.is_none() {
                return;
            }
            let indices = frictional_contact.collider_impulse.indices();

            let collider_mass_inv =
                DiagonalBlockMatrix::from_subset(Subset::from_unique_ordered_indices(
                    indices.as_slice(),
                    self.collider_mass_inv.as_ref().unwrap().view(),
                ));
            let corrector = Chunked3::from_flat(
                frictional_contact
                    .collider_impulse
                    .source()
                    .view()
                    .into_flat()
                    .0,
            );
            let add_vel = collider_mass_inv * Tensor::new(corrector);
            let mut out_vel = Subset::from_unique_ordered_indices(indices.as_slice(), collider_vel);
            *&mut out_vel.expr_mut() += add_vel.expr();
        }
    }

    fn add_friction_corrector_impulse(
        &self,
        mut out: [SubsetView<Chunked3<&mut [f64]>>; 2],
        multiplier: f64,
    ) {
        if let Some(ref frictional_contact) = self.frictional_contact() {
            if !frictional_contact.object_impulse.is_empty() && !out[0].is_empty() {
                for (i, (&cr, _)) in frictional_contact.object_impulse.iter().enumerate() {
                    out[0][i] = (Vector3::new(out[0][i]) + Vector3::new(cr) * multiplier).into();
                }
            }

            if frictional_contact.collider_impulse.is_empty() || out[1].is_empty() {
                return;
            }

            for (contact_idx, (i, (&cr, _))) in frictional_contact
                .collider_impulse
                .indexed_source_iter()
                .enumerate()
            {
                out[1][i] = (Vector3::new(out[1][i]) + Tensor::new(cr) * multiplier).into();
            }
        }
    }

    fn add_friction_impulse(
        &self,
        mut grad: [SubsetView<Chunked3<&mut [f64]>>; 2],
        multiplier: f64,
    ) {
        if let Some(ref frictional_contact) = self.frictional_contact() {
            if !frictional_contact.object_impulse.is_empty() && !grad[0].is_empty() {
                for (i, (_, &r)) in frictional_contact.object_impulse.iter().enumerate() {
                    grad[0][i] = (Vector3::new(grad[0][i]) + Vector3::new(r) * multiplier).into();
                }
            }

            if frictional_contact.collider_impulse.is_empty() || grad[1].is_empty() {
                return;
            }

            for (contact_idx, (i, (_, &r))) in frictional_contact
                .collider_impulse
                .indexed_source_iter()
                .enumerate()
            {
                //// Project out the normal component
                //let r_t = if !frictional_contact.contact_basis.is_empty() {
                //    let f = frictional_contact
                //        .contact_basis
                //        .to_contact_coordinates(r, contact_idx);
                //    Vector3::new(
                //        frictional_contact
                //            .contact_basis
                //            .from_contact_coordinates([0.0, f[1], f[2]], contact_idx)
                //            .into(),
                //    )
                //} else {
                //    Vector3::zero()
                //};

                grad[1][i] = (Vector3::new(grad[1][i]) + Tensor::new(r) * multiplier).into();
            }
        }
    }

    fn frictional_dissipation(&self, v: [SubsetView<Chunked3<&[f64]>>; 2]) -> f64 {
        let mut dissipation = 0.0;
        if let Some(ref frictional_contact) = self.frictional_contact {
            for (i, (_, f)) in frictional_contact.object_impulse.iter().enumerate() {
                for j in 0..3 {
                    dissipation += v[0][i][j] * f[j];
                }
            }

            if frictional_contact.collider_impulse.is_empty() {
                return dissipation;
            }

            for (_contact_idx, (i, (_, &r))) in frictional_contact
                .collider_impulse
                .indexed_source_iter()
                .enumerate()
            {
                if let Some(i) = i.into() {
                    // Project out normal component.
                    //let r_t = if !frictional_contact.contact_basis.is_empty() {
                    //    let f = frictional_contact
                    //        .contact_basis
                    //        .to_contact_coordinates(r, contact_idx);
                    //    Vector3::new(
                    //        frictional_contact
                    //            .contact_basis
                    //            .from_contact_coordinates([0.0, f[1], f[2]], contact_idx)
                    //            .into(),
                    //    )
                    //} else {
                    //    Vector3::zero()
                    //};

                    dissipation += Vector3::new(v[1][i]).dot(Tensor::new(r));
                }
            }
        }

        dissipation
    }

    /// For visualization purposes.
    fn add_contact_impulse(
        &mut self,
        _x: [SubsetView<Chunked3<&[f64]>>; 2],
        contact_impulse: &[f64],
        mut impulse: [Chunked3<&mut [f64]>; 2],
    ) {
        //self.update_surface_with_mesh_pos(x[0]);
        //self.update_contact_points(x[1]);

        let active_constraint_indices = self.active_constraint_indices();
        let normals = self.contact_normals();

        assert_eq!(contact_impulse.len(), normals.len());
        assert_eq!(active_constraint_indices.len(), normals.len());

        for (aci, &nml, &cr) in zip!(
            active_constraint_indices.into_iter(),
            normals.iter(),
            contact_impulse.iter()
        ) {
            impulse[1][aci] = (Vector3::new(nml) * cr).into();
        }

        let query_points = self.contact_points.borrow();
        assert_eq!(impulse[1].len(), query_points.len());

        let surf = &self.implicit_surface;
        let mut cj_matrices = vec![[[0.0; 3]; 3]; surf.num_contact_jacobian_matrices()];

        surf.contact_jacobian_matrices(query_points.view().into(), &mut cj_matrices);

        let cj_indices_iter = surf.contact_jacobian_matrix_indices_iter();

        for ((row, col), jac) in cj_indices_iter.zip(cj_matrices.into_iter()) {
            let imp = Vector3::new(impulse[0][col]);
            impulse[0][col] =
                (imp + Matrix3::new(jac).transpose() * Vector3::new(impulse[1][row])).into()
        }
    }

    fn contact_normals(&self) -> Chunked3<Vec<f64>> {
        // Contacts occur at the vertex positions of the colliding mesh.
        let surf = &self.implicit_surface;
        let contact_points = self.contact_points.borrow_mut();

        let mut normal_coords = vec![0.0; surf.num_query_jacobian_entries()];
        surf.query_jacobian_values(contact_points.view().into(), &mut normal_coords);
        let mut normals = Chunked3::from_flat(normal_coords);

        // Normalize normals
        // Contact normals point away from the surface being collided against.
        // In this case the gradient is opposite of this direction.
        for n in normals.iter_mut() {
            let nml = Vector3::new(*n);
            let len = nml.norm();
            if len > 0.0 {
                *n = (nml / -len).into();
            }
        }

        normals
    }

    fn contact_radius(&self) -> f64 {
        self.implicit_surface.radius()
    }

    fn update_radius_multiplier(&mut self, rad: f64) {
        self.implicit_surface.update_radius_multiplier(rad);
    }

    fn update_max_step(&mut self, step: f64) {
        self.implicit_surface.update_max_step(step);
    }

    fn active_constraint_indices(&self) -> Vec<usize> {
        self.implicit_surface.nonempty_neighbourhood_indices()
    }

    fn update_neighbours(
        &mut self,
        object_pos: SubsetView<Chunked3<&[f64]>>,
        collider_pos: SubsetView<Chunked3<&[f64]>>,
    ) -> bool {
        let num_vertices_updated = self.update_surface_with_mesh_pos(object_pos);
        assert_eq!(
            num_vertices_updated,
            self.implicit_surface.surface_vertex_positions().len()
        );
        self.update_contact_points(collider_pos);

        let updated = {
            let contact_points = self.contact_points.borrow();
            self.implicit_surface.reset(contact_points.as_arrays())
        };

        self.project_friction_impulses([object_pos, collider_pos]);
        updated
    }
}

impl<'a> Constraint<'a, f64> for PointContactConstraint {
    type Input = [SubsetView<'a, Chunked3<&'a [f64]>>; 2]; // Object and collider vertices

    fn constraint_size(&self) -> usize {
        self.implicit_surface.num_neighbourhoods()
    }

    fn constraint_bounds(&self) -> (Vec<f64>, Vec<f64>) {
        let m = self.constraint_size();
        (vec![0.0; m], vec![2e19; m])
    }

    fn constraint(&mut self, _x0: Self::Input, x1: Self::Input, value: &mut [f64]) {
        debug_assert_eq!(value.len(), self.constraint_size());
        self.update_surface_with_mesh_pos(x1[0]);
        self.update_contact_points(x1[1]);

        let contact_points = self.contact_points.borrow_mut();

        let mut cbuf = self.constraint_buffer.borrow_mut();
        let radius = self.contact_radius();

        let surf = &self.implicit_surface;
        for (val, q) in cbuf.iter_mut().zip(contact_points.iter()) {
            // Clear potential value.
            let closest_sample = surf.nearest_neighbour_lookup(*q).unwrap();
            if closest_sample
                .nml
                .dot(Vector3::new(*q) - closest_sample.pos)
                > 0.0
            {
                *val = radius;
            } else {
                *val = -radius;
            }
        }

        surf.potential(contact_points.view().into(), &mut cbuf);

        //let bg_pts = self.background_points();
        //let collider_mesh = self.collision_object.borrow();
        //Self::fill_background_potential(&collider_mesh, &bg_pts, radius, &mut cbuf);

        let neighbourhood_sizes = surf.neighbourhood_sizes();

        //println!("cbuf = ");
        //for c in cbuf.iter() {
        //    print!("{:9.5} ", *c);
        //}
        //println!("");

        // Because `value` tracks only the values for which the neighbourhood is not empty.
        for ((_, new_v), v) in neighbourhood_sizes
            .iter()
            .zip(cbuf.iter())
            .filter(|&(&nbrhood_size, _)| nbrhood_size != 0)
            .zip(value.iter_mut())
        {
            *v = *new_v;
        }
        //dbg!(&value);
    }
}

impl ConstraintJacobian<'_, f64> for PointContactConstraint {
    fn constraint_jacobian_size(&self) -> usize {
        let num_obj = if !self.object_is_fixed {
            self.implicit_surface.num_surface_jacobian_entries()
        } else {
            0
        };

        let num_coll = if !self.collider_is_fixed {
            self.implicit_surface.num_query_jacobian_entries()
        } else {
            0
        };
        num_obj + num_coll
    }

    fn constraint_jacobian_indices_iter<'a>(
        &'a self,
    ) -> Result<Box<dyn Iterator<Item = MatrixElementIndex> + 'a>, Error> {
        let idx_iter = {
            let surf = &self.implicit_surface;
            let col_offset = surf.surface_vertex_positions().len() * 3;
            let obj_indices_iter = if !self.object_is_fixed {
                Some(surf.surface_jacobian_indices_iter())
            } else {
                None
            };

            let coll_indices_iter = if !self.collider_is_fixed {
                Some(surf.query_jacobian_indices_iter())
            } else {
                None
            };
            obj_indices_iter.into_iter().flatten().chain(
                coll_indices_iter
                    .into_iter()
                    .flatten()
                    .map(move |(row, col)| (row, col + col_offset)),
            )
        };

        let neighbourhood_indices = nonempty_neighbourhood_indices(&self.implicit_surface);
        Ok(Box::new(idx_iter.map(move |(row, col)| {
            assert!(neighbourhood_indices[row].is_valid());
            MatrixElementIndex {
                row: neighbourhood_indices[row].unwrap(),
                col,
            }
        })))
    }

    fn constraint_jacobian_values(
        &mut self,
        _x0: Self::Input,
        x1: Self::Input,
        values: &mut [f64],
    ) -> Result<(), Error> {
        debug_assert_eq!(values.len(), self.constraint_jacobian_size());

        self.update_surface_with_mesh_pos(x1[0]);
        self.update_contact_points(x1[1]);

        let contact_points = self.contact_points.borrow_mut();

        let num_obj_jac_nnz;

        if !self.object_is_fixed {
            num_obj_jac_nnz = self.implicit_surface.num_surface_jacobian_entries();

            self.implicit_surface.surface_jacobian_values(
                contact_points.view().into(),
                &mut values[..num_obj_jac_nnz],
            );
        } else {
            num_obj_jac_nnz = 0;
        }

        if !self.collider_is_fixed {
            self.implicit_surface.query_jacobian_values(
                contact_points.view().into(),
                &mut values[num_obj_jac_nnz..],
            );
        }
        Ok(())
    }
}

impl<'a> ConstraintHessian<'a, f64> for PointContactConstraint {
    type InputDual = &'a [f64];
    fn constraint_hessian_size(&self) -> usize {
        0 + if !self.object_is_fixed {
            self.implicit_surface
                .num_surface_hessian_product_entries()
                .unwrap_or(0)
        } else {
            0
        } + if !self.collider_is_fixed {
            self.implicit_surface.num_query_hessian_product_entries()
        } else {
            0
        }
    }

    fn constraint_hessian_indices_iter<'b>(
        &'b self,
    ) -> Result<Box<dyn Iterator<Item = MatrixElementIndex> + 'b>, Error> {
        let idx_iter = {
            let offset = self.implicit_surface.surface_vertex_positions().len() * 3;
            let obj_indices_iter = if !self.object_is_fixed {
                Some(
                    self.implicit_surface
                        .surface_hessian_product_indices_iter()?,
                )
            } else {
                None
            };
            let coll_indices_iter = if !self.collider_is_fixed {
                Some(self.implicit_surface.query_hessian_product_indices_iter())
            } else {
                None
            };
            obj_indices_iter.into_iter().flatten().chain(
                coll_indices_iter
                    .into_iter()
                    .flatten()
                    .map(move |(row, col)| (row + offset, col + offset)),
            )
        };
        Ok(Box::new(
            idx_iter.map(move |(row, col)| MatrixElementIndex { row, col }),
        ))
    }

    fn constraint_hessian_values(
        &mut self,
        _x0: Self::Input,
        x1: Self::Input,
        lambda: Self::InputDual,
        scale: f64,
        values: &mut [f64],
    ) -> Result<(), Error> {
        self.update_surface_with_mesh_pos(x1[0]);
        self.update_contact_points(x1[1]);
        let surf = &self.implicit_surface;
        let contact_points = self.contact_points.borrow();

        let mut obj_hess_nnz = 0;

        if !self.object_is_fixed {
            obj_hess_nnz = self
                .implicit_surface
                .num_surface_hessian_product_entries()
                .unwrap_or(0);

            surf.surface_hessian_product_scaled_values(
                contact_points.view().into(),
                lambda,
                scale,
                &mut values[..obj_hess_nnz],
            )?;
        }

        if !self.collider_is_fixed {
            surf.query_hessian_product_scaled_values(
                contact_points.view().into(),
                lambda,
                scale,
                &mut values[obj_hess_nnz..],
            );
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use geo::mesh::builder::*;

    /// Test the `fill_background_potential` function on a small grid.
    #[test]
    fn background_fill_test() {
        // Make a small grid.
        let mut grid = TriMesh::from(
            GridBuilder {
                rows: 4,
                cols: 6,
                orientation: AxisPlaneOrientation::ZX,
            }
            .build(),
        );

        let mut values = vec![0.0; grid.num_vertices()];
        let mut bg_pts = vec![true; grid.num_vertices()];
        let radius = 0.5;
        for ((p, v), bg) in grid
            .vertex_position_iter()
            .zip(values.iter_mut())
            .zip(bg_pts.iter_mut())
        {
            if p[0] == 0.0 {
                *bg = false;
            } else if p[0] < radius && p[0] > 0.0 {
                *v = radius;
                *bg = false;
            } else if p[0] > -radius && p[0] < 0.0 {
                *v = -radius;
                *bg = false;
            }
        }

        PointContactConstraint::fill_background_potential(&grid, &bg_pts, radius, &mut values);

        grid.set_attrib_data::<_, VertexIndex>("potential", &values)
            .expect("Failed to set potential field on grid");

        //geo::io::save_polymesh(&geo::mesh::PolyMesh::from(grid.clone()), &std::path::PathBuf::from("out/background_test.vtk")).unwrap();

        for (&p, &v) in grid.vertex_position_iter().zip(values.iter()) {
            if p[0] > 0.0 {
                assert!(v > 0.0);
            } else {
                assert!(v <= 0.0);
            }
        }
    }
}
