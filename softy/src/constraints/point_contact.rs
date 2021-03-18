use super::*;
use crate::constraint::*;
use crate::contact::*;
use crate::fem::opt::problem::{Tag, Var};
use crate::friction::*;
use crate::matrix::*;
use crate::objects::TriMeshShell;
use crate::Error;
use crate::TriMesh;
use crate::{CheckedIndex, Index};
use geo::bbox::BBox;
use geo::mesh::topology::*;
use geo::mesh::{Attrib, VertexPositions};
use geo::ops::*;
use implicits::*;
use lazycell::LazyCell;
use num_traits::Zero;
use rayon::iter::Either;
#[cfg(feature = "af")]
use reinterpret::*;
use tensr::*;

/// Data needed to build a mass matrix
#[derive(Clone, Debug, PartialEq)]
pub enum MassData {
    Dense(f64, Matrix3<f64>), // Rigid body data
    // TODO: make this a regular old vec<f64> no need to have triplets
    Sparse(Chunked3<Vec<f64>>), // Vertex masses
    Zero,                       // Infinite mass
}

type DBlockMatrix3<T> = Tensor![T; D D 3 3];

#[derive(Clone, Debug, PartialEq)]
enum MassMatrixInv<S, T, I> {
    /// Rigid body effective mass matrix on contact points
    ///
    /// The additional chunked Vec is temporarily there for faking rigid-rigid friction.
    Dense(DBlockMatrix3<T>, Chunked3<Vec<f64>>),
    /// Selected vertex masses
    Sparse(DiagonalBlockMatrixBase<S, I, U3>),
    /// Infinite mass
    Zero,
}

impl MassData {
    fn is_some(&self) -> bool {
        match self {
            MassData::Zero => false,
            _ => true,
        }
    }
}

/// Enforce a contact constraint on a mesh against animated vertices. This constraint prevents
/// vertices from occupying the same space as a smooth representation of the simulation mesh.
#[derive(Clone, Debug)]
pub struct PointContactConstraint {
    /// Implicit surface that represents the deforming object.
    pub implicit_surface: QueryTopo,
    /// Vertex positions on the collider mesh where contact occurs.
    pub contact_points: Chunked3<Vec<f64>>,

    /// Friction impulses applied during contact.
    pub frictional_contact: Option<FrictionalContact>,
    /// A mass inverse matrix data for the object.
    pub object_mass_data: MassData,
    /// A mass inverse matrix data for the collider.
    pub collider_mass_data: MassData,

    /// A flag indicating if the object is fixed. Otherwise it's considered
    /// to be deforming, and thus appropriate derivatives are computed.
    object_tag: Tag<f64>,

    /// A flag indicating if the collider is fixed. Otherwise it's considered
    /// to be deforming, and thus appropriate derivatives are computed.
    collider_tag: Tag<f64>,

    /// The distance above which to constrain collider contact points.
    ///
    /// This helps prevent interpenetration artifacts in the result.
    contact_offset: f64,

    /// The maximum distance between two points of the given geometry.
    ///
    /// This value is used to produce relative thresholds.
    problem_diameter: f64,

    /// Internal constraint function buffer used to store temporary constraint computations.
    ///
    /// For linearized constraints, this is used to store the initial constraint value.
    constraint_value: Vec<f64>,

    /// Constraint Jacobian in two blocks: first for object Jacobian and second for collider
    /// Jacobian. If one is fixed, it will be `None`. This is used only when the constraint is
    /// linearized.
    constraint_jacobian: LazyCell<[Option<DSBlockMatrix1x3>; 2]>,

    /// Vertex to vertex topology of the collider mesh along with a cotangent weight.
    collider_vertex_topo: Chunked<Vec<(usize, f64)>>,
}

impl PointContactConstraint {
    pub fn new(
        // Main object experiencing contact against its implicit surface representation.
        object: Var<&TriMesh, f64>,
        // Collision object consisting of points pushing against the solid object.
        collider: Var<&TriMesh, f64>,
        kernel: KernelType,
        friction_params: Option<FrictionParams>,
        contact_offset: f64,
        linearized: bool,
    ) -> Result<Self, Error> {
        let mut surface_builder = ImplicitSurfaceBuilder::new();
        let object_tag = object.tag();
        let collider_tag = collider.tag();

        surface_builder
            .trimesh(object.untag())
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
                object.untag().num_vertices()
            );

            if let implicits::MLS::Local(mls) = &surface {
                log::info!(
                    "Implicit Surface:\n\
                    Base radius: {}\n\
                    Linearized: {}\n\
                    Params: {:#?}",
                    mls.base_radius,
                    linearized,
                    friction_params
                );
            }

            let query_points = collider.untag().vertex_positions();

            // Construct mass matrices
            let object_mass_data = Self::mass_matrix_data(object)?;
            let collider_mass_data = Self::mass_matrix_data(collider)?;

            assert!(object_mass_data.is_some() || collider_mass_data.is_some());

            let object = object.untag();
            let collider = collider.untag();

            let mut bbox = BBox::empty();
            bbox.absorb(object.bounding_box());
            bbox.absorb(collider.bounding_box());

            let mut constraint = PointContactConstraint {
                implicit_surface: surface.query_topo(query_points),
                contact_points: Chunked3::from_array_vec(query_points.to_vec()),
                frictional_contact: friction_params.and_then(|fparams| {
                    if fparams.dynamic_friction > 0.0 {
                        Some(FrictionalContact::new(fparams))
                    } else {
                        None
                    }
                }),
                object_mass_data,
                collider_mass_data,
                object_tag,
                collider_tag,
                contact_offset,
                problem_diameter: bbox.diameter(),
                constraint_value: vec![0.0; query_points.len()],
                constraint_jacobian: LazyCell::new(),
                collider_vertex_topo: Self::build_vertex_topo(collider),
            };

            if linearized {
                constraint.linearize_constraint(
                    Subset::all(Chunked3::from_array_slice(object.vertex_positions())),
                    Subset::all(Chunked3::from_array_slice(query_points)),
                );
            }

            Ok(constraint)
        } else {
            Err(Error::InvalidImplicitSurface)
        }
    }

    fn object_is_fixed(&self) -> bool {
        self.object_tag == Tag::Fixed
    }
    fn collider_is_fixed(&self) -> bool {
        self.collider_tag == Tag::Fixed
    }

    fn mass_matrix_data(mesh: Var<&TriMesh, f64>) -> Result<MassData, Error> {
        match mesh {
            Var::Rigid(_, mass, inertia) => Ok(MassData::Dense(mass, inertia)),
            Var::Fixed(_) => Ok(MassData::Zero),
            Var::Variable(mesh) => mesh
                .attrib_as_slice::<MassType, VertexIndex>(MASS_ATTRIB)
                .map_err(|_| Error::InvalidParameter {
                    name: "Missing mass attribute or parameter".to_string(),
                })
                .and_then(|attrib| {
                    if !attrib.iter().all(|&x| x > 0.0) {
                        Err(Error::InvalidParameter {
                            name: "Zero mass".to_string(),
                        })
                    } else {
                        let data: Chunked3<Vec<_>> = if let Ok(fixed) =
                            mesh.attrib_iter::<FixedIntType, VertexIndex>(FIXED_ATTRIB)
                        {
                            // Fixed vertices have infinite mass, so zero inverse mass.
                            attrib
                                .iter()
                                .zip(fixed)
                                .map(
                                    |(&x, &fixed)| {
                                        if fixed != 0 {
                                            [0.0; 3]
                                        } else {
                                            [1.0 / x; 3]
                                        }
                                    },
                                )
                                .collect()
                        } else {
                            attrib.iter().map(|&x| [1.0 / x; 3]).collect()
                        };
                        Ok(MassData::Sparse(data))
                    }
                }),
        }
    }

    fn build_constraint_jacobian(
        &mut self,
        pos: [SubsetView<Chunked3<&[f64]>>; 2],
    ) -> [Option<DSBlockMatrix1x3>; 2] {
        self.update_surface_with_mesh_pos(pos[0]);
        self.update_contact_points(pos[1]);
        //{
        //    let mut indices = vec![1; contact_points.len() * 2];
        //    for i in 0..contact_points.len() {
        //        indices[2 * i + 1] = i;
        //    }

        //    let polymesh = geo::mesh::PolyMesh::new(contact_points.clone().into(), &indices);
        //    geo::io::save_polymesh(&polymesh, "./out/before_jac_points.vtk");
        //}

        let surf = &self.implicit_surface;
        let neighbourhood_indices = enumerate_nonempty_neighbourhoods_inplace(&surf);
        let row_correction = |((row, col), val)| {
            let idx: Index = neighbourhood_indices[row];
            assert!(idx.is_valid());
            (idx.unwrap(), col, [val])
        };

        let num_rows = surf.num_neighbourhoods();

        let obj_jac = if !self.object_is_fixed() {
            let num_cols = surf.surface_vertex_positions().len();
            let iter = surf
                .surface_jacobian_block_indices_iter()
                .zip(
                    self.implicit_surface
                        .surface_jacobian_block_iter(self.contact_points.view().into()),
                )
                .map(row_correction);
            Some(
                DSBlockMatrix1x3::from_block_triplets_iter_uncompressed(iter, num_rows, num_cols)
                    .pruned(|_, _, block| !block.is_zero()),
            )
        } else {
            None
        };

        let coll_jac = if !self.collider_is_fixed() {
            let num_cols = self.contact_points.len();
            let iter = surf
                .query_jacobian_block_indices_iter()
                .zip(
                    self.implicit_surface
                        .query_jacobian_block_iter(self.contact_points.view().into()),
                )
                .map(row_correction);
            Some(
                DSBlockMatrix1x3::from_block_triplets_iter_uncompressed(iter, num_rows, num_cols)
                    .pruned(|_, _, block| !block.is_zero()),
            )
        } else {
            None
        };

        [obj_jac, coll_jac]
    }

    /// Compute vertex to vertex topology of the entire collider mesh along with corresponding
    /// cotangent weights. This makes it easy to query neighbours when we are computing laplacians.
    fn build_vertex_topo(mesh: &TriMesh) -> Chunked<Vec<(usize, f64)>> {
        let mut topo = vec![Vec::with_capacity(5); mesh.num_vertices()];
        let pos = mesh.vertex_positions();

        let cot = |v0, v1, v2| {
            let p0 = Vector3::new(pos[v0]);
            let p1 = Vector3::new(pos[v1]);
            let p2 = Vector3::new(pos[v2]);
            let a: Vector3<f64> = p1 - p0;
            let b: Vector3<f64> = p2 - p0;
            let denom = a.cross(b).norm();
            if denom > 0.0 {
                a.dot(b) / denom
            } else {
                0.0
            }
        };

        // Accumulate all the neighbours
        mesh.face_iter().for_each(|&[v0, v1, v2]| {
            let t0 = cot(v0, v1, v2);
            let t1 = cot(v1, v2, v0);
            let t2 = cot(v2, v0, v1);
            topo[v0].push((v1, t2));
            topo[v0].push((v2, t1));
            topo[v1].push((v0, t2));
            topo[v1].push((v2, t0));
            topo[v2].push((v0, t1));
            topo[v2].push((v1, t0));
        });

        // Sort and dedup neighbourhoods
        for nbrhood in topo.iter_mut() {
            nbrhood.sort_by_key(|(idx, _)| *idx);
            nbrhood.dedup_by(|(v0, t0), (v1, t1)| {
                *v0 == *v1 && {
                    *t1 += *t0;
                    *t1 *= 0.5;
                    true
                }
            });
        }

        // Flatten data layout to make access faster.
        Chunked::from_nested_vec(topo)
    }

    /// Build a matrix that smoothes values at contact points with their neighbours by the given
    /// weight. For `weight = 0.0`, no smoothing is performed, and this matrix is the identity.
    ///
    /// The implementation uses cotangent weights. This is especially important here since we are
    /// using this matrix to smooth pressures, which are inherently area based.
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
                        let weighted_nbrhood = &self.collider_vertex_topo[active_idx];
                        let n = weighted_nbrhood
                            .iter()
                            .filter(|&(nbr_idx, _)| neighbourhood_indices[*nbr_idx].is_valid())
                            .count();
                        std::iter::repeat((valid_idx, weight / n as f64))
                            .zip(weighted_nbrhood.iter())
                            .filter_map(|((valid_idx, normalized_weight), (nbr_idx, w))| {
                                neighbourhood_indices[*nbr_idx]
                                    .into_option()
                                    .map(|valid_nbr| (valid_idx, valid_nbr, w * normalized_weight))
                            })
                            .chain(std::iter::once((valid_idx, valid_idx, 1.0 - weight)))
                    });
            // Don't need to sort or compress.
            DSMatrix::from_sorted_triplets_iter_uncompressed(triplets, size, size)
        } else {
            let neighbourhood_indices = enumerate_nonempty_neighbourhoods_inplace(&surf);
            let size = surf.num_neighbourhoods();
            let triplets = self
                .collider_vertex_topo
                .iter()
                .zip(neighbourhood_indices.iter())
                .filter_map(|(nbrhood, idx)| idx.into_option().map(|i| (nbrhood, i)))
                .flat_map(|(weighted_nbrhood, valid_idx)| {
                    let n = weighted_nbrhood
                        .iter()
                        .filter(|&(nbr_idx, _)| neighbourhood_indices[*nbr_idx].is_valid())
                        .count();
                    std::iter::repeat((valid_idx, weight / n as f64))
                        .zip(weighted_nbrhood.iter())
                        .filter_map(|((valid_idx, normalized_weight), (nbr_idx, w))| {
                            neighbourhood_indices[*nbr_idx]
                                .into_option()
                                .map(|valid_nbr| (valid_idx, valid_nbr, w * normalized_weight))
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
        x.clone_into_other(&mut self.contact_points);
    }

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
    pub fn in_contact_indices(
        &self,
        contact_impulse: &[f64],
        potential: &[f64],
    ) -> (Vec<usize>, Vec<usize>, Vec<f64>) {
        let surf = &self.implicit_surface;
        let query_points = &self.contact_points;
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

    // Note that this does NOT set the num_cols and num_rows fields of TripletContactJacobian.
    pub(crate) fn append_contact_jacobian_triplets(
        &self,
        jac_triplets: &mut TripletContactJacobian,
        active_contact_indices: &[usize],
        contact_offset: usize,
        object_offset: usize,
        collider_offset: usize,
    ) {
        let query_points = &self.contact_points;
        let surf = &self.implicit_surface;
        let active_contact_points = Select::new(active_contact_indices, query_points.view());

        // Compute contact Jacobian
        jac_triplets.append_selection(
            &surf,
            active_contact_points,
            contact_offset,
            object_offset,
            collider_offset,
        );
    }

    fn compute_contact_jacobian(&self, active_contact_indices: &[usize]) -> ContactJacobian {
        let query_points = &self.contact_points;
        let surf = &self.implicit_surface;
        let active_contact_points = Select::new(active_contact_indices, query_points.view());

        // Compute contact Jacobian
        let jac_triplets = TripletContactJacobian::from_selection(&surf, active_contact_points);
        let jac: ContactJacobian = jac_triplets.into();
        jac.into_tensor()
            .pruned(|_, _, block| !block.is_zero())
            .into_data()
    }

    fn compute_effective_mass_inv(
        &self,
        active_contact_indices: &[usize],
        rigid_motion: [Option<[[f64; 3]; 2]>; 2],
        jac: ContactJacobianView,
    ) -> Option<EffectiveMassInv> {
        // Construct mass matrices for object and collider.

        let collider_mass_inv = match &self.collider_mass_data {
            MassData::Sparse(mass_data) => {
                // Collider mass matrix is constructed at active contacts only.
                // Extract a subset of contact indices from the total mass_data:
                //
                // TODO: Taking a Subset of a Subset should be a function within `Subset`.
                MassMatrixInv::Sparse(DiagonalBlockMatrixBase::from_subset(
                    Subset::from_unique_ordered_indices(active_contact_indices, mass_data.view()),
                ))
            }
            MassData::Dense(mass, inertia) => {
                if active_contact_indices.is_empty() {
                    MassMatrixInv::Zero
                } else {
                    let query_points = &self.contact_points;
                    let translation = rigid_motion[1].unwrap()[0];
                    let rotation = rigid_motion[1].unwrap()[1];
                    MassMatrixInv::Dense(
                        TriMeshShell::rigid_effective_mass_inv(
                            *mass,
                            translation.into_tensor(),
                            rotation.into_tensor(),
                            *inertia,
                            Select::new(active_contact_indices, query_points.view()),
                        ),
                        Chunked3::from_flat(vec![*mass; active_contact_indices.len() * 3]),
                    )
                }
            }
            MassData::Zero => MassMatrixInv::Zero,
        };

        let mut jac_mass = jac.clone().into_owned().into_tensor();

        match &self.object_mass_data {
            MassData::Sparse(object_mass_data) => {
                let object_mass_inv =
                    DiagonalBlockMatrixBase::from_uniform(object_mass_data.view());
                jac_mass *= object_mass_inv.view();
            }
            MassData::Dense(mass, _) => {
                // TODO: Implement jac_mass *= object_mass_data.view();
                jac_mass *= DiagonalBlockMatrixBase::from_subset(Subset::all(Chunked3::from_flat(
                    vec![*mass; active_contact_indices.len() * 3],
                )));
            }
            MassData::Zero => {
                match collider_mass_inv {
                    MassMatrixInv::Sparse(mass_inv) => {
                        let data = DiagonalBlockMatrixBase::from_subset(mass_inv.0.into_owned());
                        return Some(data.into());
                    }
                    MassMatrixInv::Dense(_, mass_data) => {
                        // For now we will use the decoupled jacobian. This is needed here until rigid
                        // to rigid can be solved properly. Currently The singularity in the
                        // delassus operator is causing issues. Probably rigid friction needs a
                        // different solver.
                        return Some(DiagonalBlockMatrixBase::from_uniform(mass_data).into());
                        //return Some(mass_inv.into());
                    }
                    MassMatrixInv::Zero => return None,
                }
            }
        }

        let object_mass_inv = jac_mass.view() * jac.view().into_tensor().transpose();

        Some(match collider_mass_inv {
            MassMatrixInv::Sparse(collider_mass_inv) => object_mass_inv.view() + collider_mass_inv,
            MassMatrixInv::Dense(_collider_mass_inv, mass_data) => {
                // TODO: implement addition collider_mass_inv + object_mass_inv.view(),
                object_mass_inv.view() + DiagonalBlockMatrixBase::from_uniform(mass_data).view()
            }
            MassMatrixInv::Zero => {
                object_mass_inv.view()
                    + DiagonalBlockMatrixBase::from_uniform(Chunked3::from_flat(
                        vec![0.0; 3 * object_mass_inv.num_rows()],
                    ))
                    .view()
            }
        })
    }

    fn compute_predictor_impulse(
        v: [SubsetView<Chunked3<&[f64]>>; 2],
        active_contact_indices: &[usize],
        jac: ContactJacobianView,
        effective_mass_inv: EffectiveMassInvView,
    ) -> Chunked3<Vec<f64>> {
        let collider_velocity = Subset::from_unique_ordered_indices(active_contact_indices, v[1]);

        let mut object_velocity = jac.view().into_tensor() * v[0].into_tensor();
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
        object_velocity.into_data()
    }
}

/// Enumerate non-empty neighbourhoods in place.
fn enumerate_nonempty_neighbourhoods_inplace(surf: &QueryTopo) -> Vec<Index> {
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
        self.contact_points.len()
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

    fn smooth_collider_values(&self, mut values: SubsetView<&mut [f64]>) {
        if let Some(ref frictional_contact) = self.frictional_contact() {
            let weight = frictional_contact.params.smoothing_weight;
            let lap = self.build_contact_laplacian(weight, None);
            let mut contacts = vec![0.0; frictional_contact.collider_impulse.len()];
            let indices = frictional_contact.collider_impulse.indices();
            assert_eq!(indices.len(), contacts.len());
            for (&i, v) in zip!(indices.iter(), contacts.iter_mut()) {
                *v = values[i];
            }
            let res: Vec<f64> = (lap.expr() * contacts.expr()).eval();
            let res: Vec<f64> = (lap.expr() * res.expr()).eval();
            for (&i, &v) in zip!(indices.iter(), res.iter()) {
                values[i] = v;
            }
        }
    }

    #[cfg(feature = "af")]
    fn contact_jacobian_af(&self) -> af::Array<f64> {
        // Compute contact jacobian
        let surf = &self.implicit_surface;
        let query_points = &self.contact_points;

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

    fn collider_contact_normals(&mut self, mut out_normals: Chunked3<&mut [f64]>) {
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
        let remapped_normals_iter = crate::constraints::remap_values_iter(
            normals.into_iter(),
            [0.0; 3], // Default normal (there should not be any).
            query_indices.into_iter(),
            collider_impulse.selection().index_iter().cloned(),
        );

        for (&aci, nml) in zip!(
            collider_impulse.selection().index_iter(),
            remapped_normals_iter,
        ) {
            out_normals[aci] = nml;
        }
    }

    fn project_friction_impulses(&mut self, x: [SubsetView<Chunked3<&[f64]>>; 2]) {
        if self.frictional_contact.is_none() {
            return;
        }
        self.update_contact_pos(x);

        let normals = self.contact_normals();
        let query_indices = self.active_constraint_indices();

        let FrictionalContact {
            object_impulse,
            collider_impulse, // for active point contacts
            ..
        } = self.frictional_contact.as_mut().unwrap();

        // Only interested in normals at contact points on the collider impulse.
        let remapped_normals_iter = crate::constraints::remap_values_iter(
            normals.into_iter(),
            [0.0; 3], // Default normal (there should not be many).
            query_indices.into_iter(),
            collider_impulse.selection().indices.clone().into_iter(),
        );

        if remapped_normals_iter.len() == 0 {
            return;
        }

        // Project contact impulse
        ContactBasis::project_out_normal_component(
            remapped_normals_iter,
            collider_impulse.source_iter_mut().map(|(_, imp)| imp),
        );

        // Project object impulse
        ContactBasis::project_out_normal_component(
            self.implicit_surface.surface_vertex_normals().into_iter(),
            object_impulse.iter_mut().map(|(_, imp)| imp),
        );
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
        rigid_motion: [Option<[[f64; 3]; 2]>; 2],
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

        let smoothing_weight = self
            .frictional_contact
            .as_ref()
            .unwrap()
            .params
            .smoothing_weight;
        let smoothed_contact_impulse_n = if smoothing_weight > 0.0 {
            let lap = self.build_contact_laplacian(smoothing_weight, Some(&active_contact_indices));

            let smoothed_contact_impulse_n: Vec<f64> =
                (lap.expr() * orig_contact_impulse_n.expr()).eval();
            let smoothed_contact_impulse_n: Vec<f64> =
                (lap.expr() * smoothed_contact_impulse_n.expr()).eval();

            assert_eq!(
                orig_contact_impulse_n.len(),
                smoothed_contact_impulse_n.len()
            );
            smoothed_contact_impulse_n
        } else {
            orig_contact_impulse_n.to_vec()
        };

        let jac = self.compute_contact_jacobian(&active_contact_indices);
        let effective_mass_inv =
            self.compute_effective_mass_inv(&active_contact_indices, rigid_motion, jac.view());

        let FrictionalContact {
            contact_basis,
            params,
            object_impulse,
            collider_impulse, // for active point contacts
        } = self.frictional_contact.as_mut().unwrap();

        // A new set of contacts have been determined. We should remap the previous friction
        // impulses to match new impulses.
        let mut prev_friction_impulse: Chunked3<Vec<f64>> = if params.friction_forwarding > 0.0 {
            crate::constraints::remap_values_iter(
                collider_impulse.source_iter().map(|x| *x.1),
                [0.0; 3], // Previous impulse for unmatched contacts.
                collider_impulse.selection().index_iter().cloned(),
                active_contact_indices.iter().cloned(),
            )
            .collect()
        } else {
            Chunked3::from_array_vec(vec![[0.0; 3]; active_contact_indices.len()])
        };

        // Initialize the new friction impulse in physical space at active contacts.
        let mut friction_impulse =
            Chunked3::from_array_vec(vec![[0.0; 3]; active_contact_indices.len()]);

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
        let mut effective_mass_inv = effective_mass_inv.unwrap();

        let (min_mass_inv, max_mass_inv) = effective_mass_inv
            .storage()
            .iter()
            .fold((std::f64::INFINITY, 0.0_f64), |(min, max), &val| {
                (if val > 0.0 { min.min(val) } else { min }, max.max(val))
            });

        std::dbg!(max_mass_inv);
        std::dbg!(min_mass_inv);
        let mass_inv_scale = 0.5 * (max_mass_inv + min_mass_inv);

        let mut contact_impulse: Chunked3<Vec<f64>> = contact_basis
            .from_normal_space(&smoothed_contact_impulse_n)
            .collect();

        // Prepare true predictor for the friction solve.
        let mut predictor_impulse = Self::compute_predictor_impulse(
            v,
            &active_contact_indices,
            jac.view(),
            effective_mass_inv.view(),
        );

        // Rescale the effective mass. It is important to note that this does not change the
        // solution, but it effects the performance of the optimization significantly.
        effective_mass_inv
            .as_mut_data()
            .storage_mut()
            .iter_mut()
            .for_each(|x| *x /= mass_inv_scale);

        // Project out the normal component.
        contact_basis.project_to_tangent_space(predictor_impulse.iter_mut());

        // Friction impulse to be subtracted.
        let prev_step_friction_impulse = prev_friction_impulse.clone();

        let predictor_impulse: Chunked3<Vec<f64>> =
            (predictor_impulse.expr() + contact_impulse.expr() + prev_friction_impulse.expr())
                .eval();
        let mut contact_impulse_n = smoothed_contact_impulse_n.clone();

        // Euclidean coords
        // Switch between implicit solver and explicit solver here.
        let success = loop {
            //println!("predictor: {:?}", predictor_impulse.view());
            let friction_predictor: Chunked3<Vec<f64>> =
                (predictor_impulse.expr() - contact_impulse.expr()).eval();
            //println!("f_predictor: {:?}", friction_predictor.view());
            match crate::friction::polar_solver::FrictionPolarSolver::new(
                friction_predictor.view().into(),
                &contact_impulse_n,
                &contact_basis,
                effective_mass_inv.view(),
                mass_inv_scale,
                *params,
            ) {
                Ok(mut solver) => {
                    log::debug!("Solving Friction");

                    match solver.step() {
                        Ok(FrictionSolveResult {
                            solution: r_t,
                            iterations,
                            ..
                        }) => {
                            log::info!("Friction iteration count: {}", iterations);
                            friction_impulse = contact_basis.from_tangent_space(&r_t).collect();
                        }
                        Err(Error::FrictionSolveError {
                            status: ipopt::SolveStatus::MaximumIterationsExceeded,
                            result:
                                FrictionSolveResult {
                                    solution: r_t,
                                    iterations,
                                    ..
                                },
                        }) => {
                            // This is not ideal, but the next outer iteration will hopefully fix this.
                            log::warn!("Friction iterations exceeded: {}", iterations);
                            friction_impulse = contact_basis.from_tangent_space(&r_t).collect();
                        }
                        Err(err) => {
                            log::error!("{}", err);
                            break false;
                        }
                    }
                }
                Err(err) => {
                    log::error!("Failed to create friction solver: {}", err);
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
                    log::debug!("Solving Contact");

                    match solver.step() {
                        Ok(r_n) => {
                            contact_impulse_n.copy_from_slice(&r_n);
                            contact_impulse.clear();
                            contact_impulse
                                .extend(contact_basis.from_normal_space(&contact_impulse_n));
                        }
                        Err(err) => {
                            log::error!("Failed contact solve: {}", err);
                            break false;
                        }
                    }
                }
                Err(err) => {
                    log::error!("Failed to create contact solver: {}", err);
                    break false;
                }
            }

            //println!("c_after: {:?}", contact_impulse_n);
            //println!("c_after_full: {:?}", contact_impulse.view());

            let f_prev = prev_friction_impulse.view().into_tensor();
            let f_cur = friction_impulse.view().into_tensor();
            //println!("prev friction impulse: {:?}", f_prev.norm());
            //println!("cur friction impulse: {:?}", f_cur.norm());

            let f_delta: Chunked3<Vec<f64>> = (f_prev.expr() - f_cur.expr()).eval();
            let rel_err_numerator: f64 = f_delta
                .expr()
                .dot((effective_mass_inv.view() * f_delta.view().into_tensor()).expr());

            let rel_err_denominator: f64 = f_prev
                .expr()
                .dot::<f64, _>((effective_mass_inv.view() * f_prev.view()).expr());

            if rel_err_denominator > 0.0 {
                let rel_err = rel_err_numerator / rel_err_denominator;

                log::debug!("Friction relative error: {}", rel_err);
                if rel_err < 1e-4 {
                    friction_steps = 0;
                    break true;
                }
            } else if rel_err_numerator <= 0.0 {
                log::warn!(
                    "Friction relative error is NaN: num = {:?}, denom = {:?}",
                    rel_err_numerator,
                    rel_err_denominator
                );
                friction_steps = 0;
                break true;
            } else {
                log::debug!("Friction relative error is infinite");
            }

            // Update prev_friction_impulse for computing error subsequent iterations.
            // Note that this should not and does not affect the "prev_step_friction_impulse"
            // variable which is used in friciton forwarding and set outside the loop.
            prev_friction_impulse = friction_impulse.clone();

            friction_steps -= 1;

            if friction_steps == 0 {
                break true;
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
        //    .from_normal_space(&smoothed_contact_impulse_n)
        //    .collect();
        //let impulse: Chunked3<Vec<f64>> = (
        //    friction_impulse.expr()
        //    + contact_impulse.expr()
        //    - prev_contact_impulse.expr()
        //    ).eval();
        let forwarded_impulse: Chunked3<Vec<f64>> =
            (friction_impulse.expr() * params.friction_forwarding).eval();
        // Correct friction_impulse by subtracting previous friction impulse
        let impulse_corrector: Chunked3<Vec<f64>> = (friction_impulse.expr()
            //+ contact_impulse.expr()
            //- prev_contact_impulse.expr()
            - prev_step_friction_impulse.expr())
        .eval();
        let mut object_friction_impulse_tensor =
            jac.view().into_tensor().transpose() * forwarded_impulse.view().into_tensor();
        object_friction_impulse_tensor.negate();

        let mut object_impulse_corrector_tensor =
            jac.view().into_tensor().transpose() * impulse_corrector.view().into_tensor();
        object_impulse_corrector_tensor.negate();

        *object_impulse = Chunked3::from_flat((
            object_impulse_corrector_tensor.into_data().into_storage(),
            object_friction_impulse_tensor.into_data().into_storage(),
        ));

        *collider_impulse = Sparse::from_dim(
            active_contact_indices.clone(),
            self.contact_points.len(),
            Chunked3::from_flat((
                impulse_corrector.into_storage(),
                forwarded_impulse.into_storage(),
            )),
        );

        if friction_steps > 0 {
            friction_steps - 1
        } else {
            0
        }
    }

    fn add_mass_weighted_frictional_contact_impulse_to_object(
        &self,
        mut object_vel: SubsetView<Chunked3<&mut [f64]>>,
    ) {
        if let Some(ref frictional_contact) = self.frictional_contact {
            if frictional_contact.object_impulse.is_empty() {
                return;
            }
            match &self.object_mass_data {
                MassData::Sparse(masses) => {
                    let mass_mtx = DiagonalBlockMatrixView::view(masses.view());
                    let corrector = Chunked3::from_flat(
                        frictional_contact.object_impulse.view().into_storage().0,
                    );
                    let add_vel = mass_mtx.view() * corrector.into_tensor();
                    *&mut object_vel.expr_mut() += add_vel.expr();
                }
                MassData::Dense(mass, _) => {
                    let corrector = Chunked3::from_flat(
                        frictional_contact.object_impulse.view().into_storage().0,
                    );
                    //*&mut corrector.expr_mut() *= *mass;
                    //corrector.into_tensor()
                    *&mut object_vel.expr_mut() += corrector.expr() * *mass;
                }
                _ => {}
            };
        }
    }

    fn add_mass_weighted_frictional_contact_impulse_to_collider(
        &self,
        collider_vel: SubsetView<Chunked3<&mut [f64]>>,
    ) {
        if let Some(ref frictional_contact) = self.frictional_contact {
            if frictional_contact.collider_impulse.is_empty() {
                return;
            }

            let indices = frictional_contact.collider_impulse.indices();

            let corrector = Chunked3::from_flat(
                frictional_contact
                    .collider_impulse
                    .source()
                    .view()
                    .into_storage()
                    .0,
            );

            let mut out_vel = Subset::from_unique_ordered_indices(indices.as_slice(), collider_vel);

            match &self.collider_mass_data {
                MassData::Sparse(masses) => {
                    let collider_mass_inv = DiagonalBlockMatrixView::from_subset(
                        Subset::from_unique_ordered_indices(indices.as_slice(), masses.view()),
                    );
                    let add_vel = collider_mass_inv * corrector.into_tensor();
                    *&mut out_vel.expr_mut() += add_vel.expr();
                }
                MassData::Dense(mass, _) => {
                    *&mut out_vel.expr_mut() += corrector.expr() * *mass;
                }
                _ => {}
            }
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

            for (i, (&cr, _)) in frictional_contact.collider_impulse.indexed_source_iter() {
                out[1][i] = (Vector3::new(out[1][i]) + Vector3::new(cr) * multiplier).into();
            }
        }
    }

    fn add_friction_impulse_to_object(
        &self,
        mut grad: SubsetView<Chunked3<&mut [f64]>>,
        multiplier: f64,
    ) {
        if let Some(ref frictional_contact) = self.frictional_contact() {
            if !frictional_contact.object_impulse.is_empty() && !grad.is_empty() {
                for (i, (_, &r)) in frictional_contact.object_impulse.iter().enumerate() {
                    grad[i] = (Vector3::new(grad[i]) + Vector3::new(r) * multiplier).into();
                }
            }
        }
    }

    fn add_friction_impulse_to_collider(
        &self,
        mut grad: SubsetView<Chunked3<&mut [f64]>>,
        multiplier: f64,
    ) {
        if let Some(ref frictional_contact) = self.frictional_contact() {
            if !frictional_contact.collider_impulse.is_empty() && !grad.is_empty() {
                for (i, (_, &r)) in frictional_contact.collider_impulse.indexed_source_iter() {
                    grad[i] = (Vector3::new(grad[i]) + Vector3::new(r) * multiplier).into();
                }
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

            for (i, (_, &r)) in frictional_contact.collider_impulse.indexed_source_iter() {
                if let Some(i) = i.into() {
                    dissipation += Vector3::new(v[1][i]).dot(Vector3::new(r));
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

        let query_points = &self.contact_points;
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

    fn contact_normals(&self) -> Vec<[f64; 3]> {
        // Contacts occur at the vertex positions of the colliding mesh.
        let surf = &self.implicit_surface;
        let contact_points = &self.contact_points;

        let mut normal_coords = vec![0.0; surf.num_query_jacobian_entries()];
        surf.query_jacobian_values(contact_points.view().into(), &mut normal_coords);
        let mut normals = Chunked3::from_flat(normal_coords).into_arrays();

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

        let updated = { self.implicit_surface.reset(self.contact_points.as_arrays()) };

        if self.constraint_jacobian.filled() {
            self.linearize_constraint(object_pos, collider_pos);
        }
        self.project_friction_impulses([object_pos, collider_pos]);
        updated
    }

    fn linearize_constraint(
        &mut self,
        object_pos: SubsetView<Chunked3<&[f64]>>,
        collider_pos: SubsetView<Chunked3<&[f64]>>,
    ) {
        let jac =
        //let (lap, jac) =
        //    if let Some(smoothing_weight) = self
        //    .frictional_contact
        //    .as_ref()
        //    .map(|fc| fc.params.smoothing_weight)
        //{
        //    let lap = self.build_contact_laplacian(smoothing_weight);

        //    let [obj_jac, coll_jac] = self.build_constraint_jacobian([object_pos, collider_pos]);
        //    let jac = [
        //        obj_jac
        //            .map(|j| lap.view() * j.view())
        //        coll_jac
        //            .map(|j| lap.view() * j.view())
        //    ];
        //    (Some(lap), jac)
        //} else {
         //   (
         //       None,
                self.build_constraint_jacobian([object_pos, collider_pos]);
        //   )
        //};

        self.constraint_jacobian.replace(jac);

        let num_non_zero_constraints = self.implicit_surface.num_neighbourhoods();
        let mut c0 = vec![0.0; num_non_zero_constraints];
        self.implicit_surface
            .local_potential(self.contact_points.view().into(), c0.as_mut_slice());
        //if let Some(lap) = lap {
        //    self.constraint_value = (lap.expr() * c0.expr()).eval();
        //} else {
        c0.iter_mut().for_each(|c| *c -= self.contact_offset);
        self.constraint_value = c0;
        //}
    }
    fn is_linear(&self) -> bool {
        self.constraint_jacobian.filled()
    }
}

pub(crate) type Input<'a> = [SubsetView<'a, Chunked3<&'a [f64]>>; 2]; // Object and collider vertices

impl PointContactConstraint {
    pub(crate) fn constraint_size(&self) -> usize {
        self.implicit_surface.num_neighbourhoods()
    }

    pub(crate) fn constraint_bounds(&self) -> (Vec<f64>, Vec<f64>) {
        let m = self.constraint_size();
        (vec![0.0; m], vec![2e19; m])
    }

    pub(crate) fn constraint(&mut self, x: Input, value: &mut [f64]) {
        debug_assert_eq!(value.len(), self.constraint_size());
        if let Some(jac) = self.constraint_jacobian.borrow() {
            // Constraint is linearized

            let x0_coll = &self.contact_points;
            let x0_obj = self.implicit_surface.surface_vertex_positions();

            value.copy_from_slice(self.constraint_value.as_slice());

            let dx0: Chunked3<Vec<f64>> = (x[0].expr() - x0_obj.expr()).eval();
            let dx1: Chunked3<Vec<f64>> = (x[1].expr() - x0_coll.expr()).eval();

            if let Some(jac) = &jac[0] {
                *&mut value.expr_mut() += (jac.view() * *dx0.view().as_tensor()).expr();
            }
            if let Some(jac) = &jac[1] {
                *&mut value.expr_mut() += (jac.view() * *dx1.view().as_tensor()).expr();
            }

            return;
        }

        // Constraint not linearized, compute the true constraint.
        self.update_surface_with_mesh_pos(x[0]);
        self.update_contact_points(x[1]);

        let radius = self.contact_radius();

        let surf = &self.implicit_surface;
        for (val, q) in self
            .constraint_value
            .iter_mut()
            .zip(self.contact_points.iter())
        {
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

        surf.potential(
            self.contact_points.view().into(),
            &mut self.constraint_value,
        );

        //let bg_pts = self.background_points();
        //let collider_mesh = self.collision_object.borrow();
        //Self::fill_background_potential(&collider_mesh, &bg_pts, radius, &mut cbuf);

        let neighbourhood_sizes = surf.neighbourhood_sizes();

        // Because `value` tracks only the values for which the neighbourhood is not empty.
        for ((_, new_v), v) in neighbourhood_sizes
            .iter()
            .zip(self.constraint_value.iter())
            .filter(|&(&nbrhood_size, _)| nbrhood_size != 0)
            .zip(value.iter_mut())
        {
            *v = *new_v - self.contact_offset;
        }
    }

    pub(crate) fn object_constraint_jacobian_size(&self) -> usize {
        if let Some(jac) = self.constraint_jacobian.borrow() {
            jac[0].as_ref().map_or(0, |jac| jac.num_non_zeros())
        } else {
            if !self.object_is_fixed() {
                self.implicit_surface.num_surface_jacobian_entries()
            } else {
                0
            }
        }
    }

    pub(crate) fn collider_constraint_jacobian_size(&self) -> usize {
        if let Some(jac) = self.constraint_jacobian.borrow() {
            jac[1].as_ref().map_or(0, |jac| jac.num_non_zeros())
        } else {
            if !self.collider_is_fixed() {
                self.implicit_surface.num_query_jacobian_entries()
            } else {
                0
            }
        }
    }

    pub(crate) fn constraint_jacobian_size(&self) -> usize {
        self.object_constraint_jacobian_size() + self.collider_constraint_jacobian_size()
    }

    pub(crate) fn linearized_constraint_jacobian_indices<'a>(
        jac: DSBlockMatrix1x3View<'a>,
    ) -> impl Iterator<Item = MatrixElementIndex> + 'a {
        jac.as_data()
            .into_iter()
            .enumerate()
            .flat_map(move |(row_idx, row)| {
                row.into_iter().flat_map(move |(col_idx, _)| {
                    (0..3).map(move |component_idx| {
                        MatrixElementIndex::new(row_idx, 3 * col_idx + component_idx)
                    })
                })
            })
    }

    /// Remap jacobian row indices to coincide with surface trimesh indices.
    pub(crate) fn jacobian_index_row_adapter(
        &self,
        iter: impl Iterator<Item = (usize, usize)>,
        is_fixed: bool,
    ) -> impl Iterator<Item = MatrixElementIndex> {
        let neighbourhood_indices =
            enumerate_nonempty_neighbourhoods_inplace(&self.implicit_surface);
        if is_fixed { None } else { Some(iter) }
            .into_iter()
            .flatten()
            .map(move |(row, col)| {
                assert!(neighbourhood_indices[row].is_valid());
                MatrixElementIndex {
                    row: neighbourhood_indices[row].unwrap(),
                    col,
                }
            })
    }

    pub(crate) fn object_constraint_jacobian_indices_iter<'a>(
        &'a self,
    ) -> impl Iterator<Item = MatrixElementIndex> + 'a {
        if let Some(jac) = self.constraint_jacobian.borrow() {
            Either::Left(
                jac[0]
                    .as_ref()
                    .map(|jac| Self::linearized_constraint_jacobian_indices(jac.view()))
                    .into_iter()
                    .flatten(),
            )
        } else {
            let surf = &self.implicit_surface;
            Either::Right(self.jacobian_index_row_adapter(
                surf.surface_jacobian_indices_iter(),
                self.object_is_fixed(),
            ))
        }
    }

    pub(crate) fn collider_constraint_jacobian_indices_iter<'a>(
        &'a self,
    ) -> impl Iterator<Item = MatrixElementIndex> + 'a {
        if let Some(jac) = self.constraint_jacobian.borrow() {
            Either::Left(
                jac[1]
                    .as_ref()
                    .map(|jac| Self::linearized_constraint_jacobian_indices(jac.view()))
                    .into_iter()
                    .flatten(),
            )
        } else {
            let surf = &self.implicit_surface;
            Either::Right(self.jacobian_index_row_adapter(
                surf.query_jacobian_indices_iter(),
                self.collider_is_fixed(),
            ))
        }
    }

    pub(crate) fn object_constraint_jacobian_blocks_iter<'a>(
        &'a self,
    ) -> impl Iterator<Item = (usize, usize, [f64; 3])> + 'a {
        if let Some(jac) = self.constraint_jacobian.borrow() {
            Either::Left(
                jac[0]
                    .as_ref()
                    .map(|jac| {
                        jac.view().as_data().into_iter().enumerate().flat_map(
                            move |(row_idx, row)| {
                                row.into_iter().map(move |(block_col_idx, block)| {
                                    (row_idx, block_col_idx, block.into_arrays()[0])
                                })
                            },
                        )
                    })
                    .into_iter()
                    .flatten(),
            )
        } else {
            let surf = &self.implicit_surface;
            let iter = surf
                .surface_jacobian_block_indices_iter()
                .zip(surf.surface_jacobian_block_iter(self.contact_points.view().into()));
            let neighbourhood_indices = enumerate_nonempty_neighbourhoods_inplace(surf);
            Either::Right(
                if self.object_is_fixed() {
                    None
                } else {
                    Some(iter)
                }
                .into_iter()
                .flatten()
                .map(move |((row, col), block)| {
                    assert!(neighbourhood_indices[row].is_valid());
                    (neighbourhood_indices[row].unwrap(), col, block)
                }),
            )
        }
    }

    pub(crate) fn collider_constraint_jacobian_blocks_iter<'a>(
        &'a self,
    ) -> impl Iterator<Item = (usize, usize, [f64; 3])> + 'a {
        if let Some(jac) = self.constraint_jacobian.borrow() {
            Either::Left(
                jac[1]
                    .as_ref()
                    .map(|jac| {
                        jac.view().as_data().into_iter().enumerate().flat_map(
                            move |(row_idx, row)| {
                                row.into_iter().map(move |(block_col_idx, block)| {
                                    (row_idx, block_col_idx, block.into_arrays()[0])
                                })
                            },
                        )
                    })
                    .into_iter()
                    .flatten(),
            )
        } else {
            let surf = &self.implicit_surface;
            let iter = surf
                .query_jacobian_block_indices_iter()
                .zip(surf.query_jacobian_block_iter(self.contact_points.view().into()));
            let neighbourhood_indices = enumerate_nonempty_neighbourhoods_inplace(surf);
            Either::Right(
                if self.collider_is_fixed() {
                    None
                } else {
                    Some(iter)
                }
                .into_iter()
                .flatten()
                .map(move |((row, col), block)| {
                    assert!(neighbourhood_indices[row].is_valid());
                    (neighbourhood_indices[row].unwrap(), col, block)
                }),
            )
        }
    }

    #[allow(dead_code)]
    pub(crate) fn constraint_jacobian_indices_iter<'a>(
        &'a self,
    ) -> impl Iterator<Item = MatrixElementIndex> + 'a {
        let col_offset = self.implicit_surface.surface_vertex_positions().len() * 3;
        let obj_indices_iter = self.object_constraint_jacobian_indices_iter();
        let coll_indices_iter = self
            .collider_constraint_jacobian_indices_iter()
            .map(move |idx| idx + (0, col_offset).into());

        obj_indices_iter.chain(coll_indices_iter)
    }

    pub(crate) fn object_constraint_jacobian_values_iter<'a>(
        &'a self,
    ) -> impl Iterator<Item = f64> + 'a {
        if let Some(jac) = self.constraint_jacobian.borrow() {
            Either::Left(
                jac[0]
                    .as_ref()
                    .map(|jac| {
                        jac.view()
                            .into_data()
                            .into_iter()
                            .flat_map(move |row| row.into_iter())
                            .flat_map(|(_, val)| {
                                let arr = val.into_arrays()[0];
                                (0..3).map(move |i| arr[i])
                            })
                    })
                    .into_iter()
                    .flatten(),
            )
        } else {
            Either::Right(
                if !self.object_is_fixed() {
                    Some(
                        self.implicit_surface
                            .surface_jacobian_values_iter(self.contact_points.view().into()),
                    )
                } else {
                    None
                }
                .into_iter()
                .flatten(),
            )
        }
    }

    pub(crate) fn collider_constraint_jacobian_values_iter<'a>(
        &'a self,
    ) -> impl Iterator<Item = f64> + 'a {
        if let Some(jac) = self.constraint_jacobian.borrow() {
            Either::Left(
                jac[1]
                    .as_ref()
                    .map(|jac| {
                        jac.view()
                            .into_data()
                            .into_iter()
                            .flat_map(move |row| row.into_iter())
                            .flat_map(|(_, val)| {
                                let arr = val.into_arrays()[0];
                                (0..3).map(move |i| arr[i])
                            })
                    })
                    .into_iter()
                    .flatten(),
            )
        } else {
            Either::Right(
                if !self.collider_is_fixed() {
                    Some(
                        self.implicit_surface
                            .query_jacobian_values_iter(self.contact_points.view().into()),
                    )
                } else {
                    None
                }
                .into_iter()
                .flatten(),
            )
        }
    }

    #[allow(dead_code)]
    pub(crate) fn constraint_jacobian_values_iter<'a>(
        &'a mut self,
        x: Input,
    ) -> impl Iterator<Item = f64> + 'a {
        if !self.is_linear() {
            // Must not update the surface if constraint is linearized.
            self.update_surface_with_mesh_pos(x[0]);
            self.update_contact_points(x[1]);
        }

        self.object_constraint_jacobian_values_iter()
            .chain(self.collider_constraint_jacobian_values_iter())
    }

    pub(crate) fn object_constraint_hessian_size(&self) -> usize {
        if self.object_is_fixed() || self.constraint_jacobian.filled() {
            0
        } else {
            self.implicit_surface
                .num_surface_hessian_product_entries()
                .unwrap_or(0)
        }
    }

    pub(crate) fn collider_constraint_hessian_size(&self) -> usize {
        if self.collider_is_fixed() || self.constraint_jacobian.filled() {
            0
        } else {
            self.implicit_surface.num_query_hessian_product_entries()
        }
    }

    #[allow(dead_code)]
    pub(crate) fn constraint_hessian_size(&self) -> usize {
        self.object_constraint_hessian_size() + self.collider_constraint_hessian_size()
    }

    pub(crate) fn object_constraint_hessian_indices_iter<'b>(
        &'b self,
    ) -> impl Iterator<Item = MatrixElementIndex> + 'b {
        if self.object_is_fixed() || self.constraint_jacobian.filled() {
            None
        } else {
            self.implicit_surface
                .surface_hessian_product_indices_iter()
                .ok()
        }
        .into_iter()
        .flatten()
        .map(MatrixElementIndex::from)
    }

    pub(crate) fn collider_constraint_hessian_indices_iter<'b>(
        &'b self,
    ) -> impl Iterator<Item = MatrixElementIndex> + 'b {
        if self.collider_is_fixed() || self.constraint_jacobian.filled() {
            None
        } else {
            Some(self.implicit_surface.query_hessian_product_indices_iter())
        }
        .into_iter()
        .flatten()
        .map(MatrixElementIndex::from)
    }

    #[allow(dead_code)]
    pub(crate) fn constraint_hessian_indices_iter<'b>(
        &'b self,
    ) -> impl Iterator<Item = MatrixElementIndex> + 'b {
        let offset = self.implicit_surface.surface_vertex_positions().len() * 3;
        let obj_indices_iter = self.object_constraint_hessian_indices_iter();
        let coll_indices_iter = self.collider_constraint_hessian_indices_iter();
        obj_indices_iter.chain(coll_indices_iter.map(move |idx| idx + (offset, offset).into()))
    }

    // Assumes surface and contact points are upto date.
    pub(crate) fn object_constraint_hessian_values_iter<'a>(
        &'a self,
        lambda: &'a [f64],
    ) -> impl Iterator<Item = f64> + 'a {
        if self.object_is_fixed() || self.constraint_jacobian.filled() {
            None
        } else {
            let surf = &self.implicit_surface;
            surf.surface_hessian_product_values_iter(self.contact_points.view().into(), lambda)
                .ok()
        }
        .into_iter()
        .flatten()
    }

    // Assumes surface and contact points are upto date.
    pub(crate) fn collider_constraint_hessian_values_iter<'a>(
        &'a self,
        lambda: &'a [f64],
    ) -> impl Iterator<Item = f64> + 'a {
        if self.collider_is_fixed() || self.constraint_jacobian.filled() {
            None
        } else {
            let surf = &self.implicit_surface;
            Some(surf.query_hessian_product_values_iter(self.contact_points.view().into(), lambda))
        }
        .into_iter()
        .flatten()
    }

    #[allow(dead_code)]
    pub(crate) fn constraint_hessian_values_iter<'a>(
        &'a mut self,
        x: Input,
        lambda: &'a [f64],
    ) -> impl Iterator<Item = f64> + 'a {
        if !self.is_linear() {
            // Must not update the surface if constraint is linearized.
            self.update_surface_with_mesh_pos(x[0]);
            self.update_contact_points(x[1]);
        }
        self.object_constraint_hessian_values_iter(lambda)
            .chain(self.collider_constraint_hessian_values_iter(lambda))
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

        let radius = self.contact_radius();

        let surf = &self.implicit_surface;
        for (val, q) in self
            .constraint_value
            .iter_mut()
            .zip(self.contact_points.iter())
        {
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

        surf.potential(
            self.contact_points.view().into(),
            &mut self.constraint_value,
        );

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
            .zip(self.constraint_value.iter())
            .filter(|&(&nbrhood_size, _)| nbrhood_size != 0)
            .zip(value.iter_mut())
        {
            *v = *new_v;
        }
        //dbg!(&value);
    }
}

impl ContactConstraintJacobian<'_, f64> for PointContactConstraint {
    fn constraint_jacobian_size(&self) -> usize {
        let num_obj = if !self.object_is_fixed() {
            self.implicit_surface.num_surface_jacobian_entries()
        } else {
            0
        };

        let num_coll = if !self.collider_is_fixed() {
            self.implicit_surface.num_query_jacobian_entries()
        } else {
            0
        };
        num_obj + num_coll
    }

    fn constraint_jacobian_indices_iter<'a>(
        &'a self,
    ) -> Box<dyn Iterator<Item = MatrixElementIndex> + 'a> {
        let idx_iter = {
            let surf = &self.implicit_surface;
            let col_offset = surf.surface_vertex_positions().len() * 3;
            let obj_indices_iter = if !self.object_is_fixed() {
                Some(surf.surface_jacobian_indices_iter())
            } else {
                None
            };

            let coll_indices_iter = if !self.collider_is_fixed() {
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

        let neighbourhood_indices =
            enumerate_nonempty_neighbourhoods_inplace(&self.implicit_surface);
        Box::new(idx_iter.map(move |(row, col)| {
            assert!(neighbourhood_indices[row].is_valid());
            MatrixElementIndex {
                row: neighbourhood_indices[row].unwrap(),
                col,
            }
        }))
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

        let num_obj_jac_nnz;

        if !self.object_is_fixed() {
            num_obj_jac_nnz = self.implicit_surface.num_surface_jacobian_entries();

            self.implicit_surface.surface_jacobian_values(
                self.contact_points.view().into(),
                &mut values[..num_obj_jac_nnz],
            );
        } else {
            num_obj_jac_nnz = 0;
        }

        if !self.collider_is_fixed() {
            self.implicit_surface.query_jacobian_values(
                self.contact_points.view().into(),
                &mut values[num_obj_jac_nnz..],
            );
        }
        Ok(())
    }
}

impl<'a> ContactConstraintHessian<'a, f64> for PointContactConstraint {
    type InputDual = &'a [f64];
    fn constraint_hessian_size(&self) -> usize {
        0 + if !self.object_is_fixed() {
            self.implicit_surface
                .num_surface_hessian_product_entries()
                .unwrap_or(0)
        } else {
            0
        } + if !self.collider_is_fixed() {
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
            let obj_indices_iter = if !self.object_is_fixed() {
                Some(
                    self.implicit_surface
                        .surface_hessian_product_indices_iter()?,
                )
            } else {
                None
            };
            let coll_indices_iter = if !self.collider_is_fixed() {
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

        let mut obj_hess_nnz = 0;

        if !self.object_is_fixed() {
            obj_hess_nnz = self
                .implicit_surface
                .num_surface_hessian_product_entries()
                .unwrap_or(0);

            surf.surface_hessian_product_scaled_values(
                self.contact_points.view().into(),
                lambda,
                scale,
                &mut values[..obj_hess_nnz],
            )?;
        }

        if !self.collider_is_fixed() {
            surf.query_hessian_product_scaled_values(
                self.contact_points.view().into(),
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
