use crate::attrib_defines::*;
use crate::constraint::*;
use crate::constraints::{volume::VolumeConstraint, ContactConstraint};
use crate::energy::*;
use crate::energy_models::{elasticity::*, gravity::Gravity, inertia::Inertia};
use crate::matrix::*;
use crate::objects::*;
use crate::PointCloud;
use geo::mesh::{topology::*, Attrib, VertexPositions};
use ipopt::{self, Number};
use log::{debug, error, trace};
use std::cell::RefCell;
use utils::soap::Vector3;
use utils::{soap::*, zip};

const FORWARD_FRICTION: bool = true;

#[derive(Clone)]
pub struct Solution {
    /// This is the solution of the solve.
    pub primal_variables: Vec<f64>,
    /// Lower bound multipliers.
    pub lower_bound_multipliers: Vec<f64>,
    /// Upper bound multipliers.
    pub upper_bound_multipliers: Vec<f64>,
    /// Constraint (Lagrange) multipliers.
    pub constraint_multipliers: Vec<f64>,
}

/// Create an empty solution.
impl Default for Solution {
    fn default() -> Solution {
        Solution {
            primal_variables: Vec::new(),
            lower_bound_multipliers: Vec::new(),
            upper_bound_multipliers: Vec::new(),
            constraint_multipliers: Vec::new(),
        }
    }
}

/// Materialize a solution from the references got from Ipopt.
impl<'a> From<ipopt::Solution<'a>> for Solution {
    fn from(sol: ipopt::Solution<'a>) -> Solution {
        let mut mysol = Solution::default();
        mysol.update(sol);
        mysol
    }
}

impl Solution {
    /// Initialize a solution with all variables and multipliers set to zero.
    pub fn reset(&mut self, num_variables: usize, num_constraints: usize) -> &mut Self {
        self.clear();
        let x = vec![0.0; num_variables];
        self.primal_variables.extend_from_slice(&x);
        self.lower_bound_multipliers.extend_from_slice(&x);
        self.upper_bound_multipliers.extend_from_slice(&x);
        self.constraint_multipliers
            .extend_from_slice(&vec![0.0; num_constraints]);
        self
    }

    /// Clear all solution vectors.
    pub fn clear(&mut self) {
        self.primal_variables.clear();
        self.lower_bound_multipliers.clear();
        self.upper_bound_multipliers.clear();
        self.constraint_multipliers.clear();
    }

    /// Update allocated solution vectors with new data from Ipopt.
    pub fn update<'a>(&mut self, sol: ipopt::Solution<'a>) -> &mut Self {
        self.clear();
        self.primal_variables
            .extend_from_slice(sol.primal_variables);
        self.lower_bound_multipliers
            .extend_from_slice(sol.lower_bound_multipliers);
        self.upper_bound_multipliers
            .extend_from_slice(sol.upper_bound_multipliers);
        self.constraint_multipliers
            .extend_from_slice(sol.constraint_multipliers);
        self
    }

    ///// for the next solve. For this reason we must remap the old multipliers to the new set of
    ///// constraints. Constraint multipliers that are new in the next solve will have a zero value.
    ///// This function works on a subset of all multipliers. The caller gives a slice of the
    ///// multipliers for which this function produces a new Vec of multipliers correspnding to the
    ///// new constraints, where the old multipliers are copied as available.
    ///// The values in `new_indices` and `old_indices` are required to be sorted.
    /////
    ///// NOTE: Most efficient to replace the entire constraint_multipliers vector in the warm start.
    //pub fn remap_constraint_multipliers(
    //    constraint_multipliers: &[f64],
    //    old_indices: &[usize],
    //    new_indices: &[usize],
    //) -> Vec<f64> {
    //    remap_values(constraint_multipliers, 0.0, old_indices, new_indices)
    //}
}

/// The index of the object subject to the appropriate contact constraint.
/// This enum helps us map from the particular contact constraint to the
/// originating simulation object (shell or solid).
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum SourceIndex {
    Solid(usize),
    Shell(usize),
}

impl SourceIndex {
    pub fn get(&self) -> usize {
        match self {
            SourceIndex::Solid(idx) | SourceIndex::Shell(idx) => *idx,
        }
    }
}

/// A struct that keeps track of which objects are being affected by the contact
/// constraints.
#[derive(Debug)]
pub struct FrictionalContactConstraint {
    pub object_index: SourceIndex,
    pub collider_index: SourceIndex,
    pub constraint: Box<RefCell<dyn ContactConstraint>>,
}

/// A `Vertex` is a single element of the `VertexSet`.
#[derive(Clone, Debug)]
pub struct Vertex {
    prev_pos: [f64; 3],
    prev_vel: [f64; 3],
    cur_pos: [f64; 3],
    cur_vel: [f64; 3],
}

/// An enum that tags data as static (Fixed) or changing (Variable).
/// `Var` is short for "variability".
#[derive(Copy, Clone, Debug)]
pub enum Var<T> {
    Fixed(T),
    Variable(T),
}

impl<T> Var<T> {
    #[inline]
    pub fn map<U, F: FnOnce(T) -> U>(self, f: F) -> Var<U> {
        match self {
            Var::Fixed(x) => Var::Fixed(f(x)),
            Var::Variable(x) => Var::Variable(f(x)),
        }
    }

    #[inline]
    pub fn is_fixed(&self) -> bool {
        match self {
            Var::Fixed(_) => true,
            Var::Variable(_) => false,
        }
    }

    /// Effectively untag the underlying data by converting into the inner type.
    #[inline]
    pub fn untag(self) -> T {
        match self {
            Var::Fixed(t) | Var::Variable(t) => t,
        }
    }
}

pub type MeshVertexData<D> = Subset<D>;
pub type VertexData<D> = Chunked<Chunked<D>>;
pub type VertexView<'i, D> = ChunkedView<'i, ChunkedView<'i, D>>;

pub type MeshVertexData3<T> = MeshVertexData<Chunked3<T>>;
pub type VertexData3<D> = VertexData<Chunked3<D>>;
pub type VertexView3<'i, D> = VertexView<'i, Chunked3<D>>;

/// This set collects all simulated vertex data. Meshes that are not simulated are excluded.
/// The data is chunked into solids/shells, then into a subset of individual
/// meshes, and finally into x,y,z coordinates. Rigid shells have 6 degrees of
/// freedom, 3 for position and 3 for rotation, so a rigid shell will correspond
/// to 6 floats in each of these vectors. This can be more generally
/// interpreted as generalized coordinates.
/// This struct is responsible for mapping between input mesh vertices and
/// generalized coordinates.
// TODO: Factor out common chunking structure so it's not repeated for each
// chunked collection.
#[derive(Clone, Debug)]
pub struct ObjectData {
    /// Generalized coordinates from the previous time step. Sometimes referred
    /// to as `q` in literature.
    pub prev_x: VertexData3<Vec<f64>>,
    /// Generalized coordinate derivative from the previous time step. Referred
    /// to as `\dot{q}` in literature.
    pub prev_v: VertexData3<Vec<f64>>,
    /// Workspace vector to compute intermediate displacements.
    pub cur_x: RefCell<VertexData3<Vec<f64>>>,
    /// Workspace vector to rescale variable values before performing computations on them.
    pub cur_v: RefCell<VertexData3<Vec<f64>>>,

    /// Saved initial conditions from previous step in case we need to revert.
    pub prev_prev_x: Vec<f64>,
    /// Saved initial conditions from previous step in case we need to revert.
    pub prev_prev_v: Vec<f64>,

    /// Workspace positions for all meshes for which the generalized coordinates
    /// don't coincide with vertex positions. These are used to pass concrete
    /// positions (as opposed to generalized coordinates) to constraint
    /// functions.
    pub pos: VertexData3<Vec<f64>>,
    /// Workspace velocities for all meshes for which the generalized
    /// coordinates don't coincide with vertex positions. These are used to pass
    /// concrete velocities (as opposed to generalized coordinates) to
    /// constraint functions.
    pub vel: VertexData3<Vec<f64>>,

    /// Tetrahedron mesh representing a soft solid computational domain.
    pub solids: Vec<TetMeshSolid>,
    /// Shell object represented by a triangle mesh.
    pub shells: Vec<TriMeshShell>,
}

impl ObjectData {
    /// Build a `VertexData` struct with an zero entry for each vertex of each
    /// mesh.
    pub fn build_vertex_data(&self) -> VertexData<Vec<f64>> {
        let mut mesh_sizes = Vec::new();
        mesh_sizes.extend(self.solids.iter().map(|solid| solid.tetmesh.num_vertices()));
        mesh_sizes.extend(self.shells.iter().map(|shell| shell.trimesh.num_vertices()));
        let out = vec![0.0; mesh_sizes.iter().sum::<usize>()];

        let out = Chunked::from_sizes(mesh_sizes, out);
        let num_solids = self.solids.len();
        let num_shells = self.shells.len();

        Chunked::from_offsets(vec![0, num_solids, num_solids + num_shells], out)
    }

    /// Build a `VertexData3` struct with an zero entry for each vertex of each
    /// mesh.
    pub fn build_vertex_data3(&self) -> VertexData3<Vec<f64>> {
        let mut mesh_sizes = Vec::new();
        mesh_sizes.extend(self.solids.iter().map(|solid| solid.tetmesh.num_vertices()));
        mesh_sizes.extend(self.shells.iter().map(|shell| shell.trimesh.num_vertices()));
        let out = Chunked3::from_array_vec(vec![[0.0; 3]; mesh_sizes.iter().sum::<usize>()]);

        let out = Chunked::from_sizes(mesh_sizes, out);
        let num_solids = self.solids.len();
        let num_shells = self.shells.len();

        Chunked::from_offsets(vec![0, num_solids, num_solids + num_shells], out)
    }

    pub fn cur_pos<'x>(
        &'x self,
        cur_x: VertexView3<'x, &'x [f64]>,
        src_idx: SourceIndex,
    ) -> MeshVertexData3<&'x [f64]> {
        // First determine which variable set to use for the current pos.
        // E.g. rigid meshes use the `pos` set, while deformable meshes use the
        // `cur_x` field.
        let ObjectData {
            pos,
            solids,
            shells,
            ..
        } = self;
        Self::mesh_vertex_subset_impl(cur_x, pos.view(), src_idx, solids, shells)
    }

    pub fn prev_pos(&self, src_idx: SourceIndex) -> MeshVertexData3<&[f64]> {
        let ObjectData {
            prev_x,
            pos,
            solids,
            shells,
            ..
        } = self;
        Self::mesh_vertex_subset_impl(prev_x.view(), pos.view(), src_idx, solids, shells)
    }

    pub fn cur_vel<'x>(
        &'x self,
        cur_v: VertexView3<'x, &'x [f64]>,
        src_idx: SourceIndex,
    ) -> MeshVertexData3<&'x [f64]> {
        // First determine which variable set to use for the current pos.
        // E.g. rigid meshes use the `pos` set, while deformable meshes use the
        // `cur_x` field.
        let ObjectData {
            vel,
            solids,
            shells,
            ..
        } = self;
        Self::mesh_vertex_subset_impl(cur_v, vel.view(), src_idx, solids, shells)
    }

    pub fn prev_vel(&self, src_idx: SourceIndex) -> MeshVertexData3<&[f64]> {
        let ObjectData {
            prev_v,
            vel,
            solids,
            shells,
            ..
        } = self;
        Self::mesh_vertex_subset_impl(prev_v.view(), vel.view(), src_idx, solids, shells)
    }

    pub fn cur_pos_mut<'x>(
        &'x mut self,
        cur_x: VertexView3<'x, &'x mut [f64]>,
        alt_x: VertexView3<'x, &'x mut [f64]>,
        src_idx: SourceIndex,
    ) -> MeshVertexData3<&'x mut [f64]> {
        let ObjectData { solids, shells, .. } = self;
        Self::mesh_vertex_subset_mut_impl(cur_x, alt_x, src_idx, solids, shells)
    }

    pub fn prev_pos_mut(&mut self, src_idx: SourceIndex) -> MeshVertexData3<&mut [f64]> {
        let ObjectData {
            prev_x,
            pos,
            solids,
            shells,
            ..
        } = self;
        Self::mesh_vertex_subset_mut_impl(
            prev_x.view_mut(),
            pos.view_mut(),
            src_idx,
            solids,
            shells,
        )
    }

    pub fn cur_vel_mut<'x>(
        &'x mut self,
        cur_v: VertexView3<'x, &'x mut [f64]>,
        alt_v: VertexView3<'x, &'x mut [f64]>,
        src_idx: SourceIndex,
    ) -> MeshVertexData3<&'x mut [f64]> {
        // First determine which variable set to use for the current pos.
        // E.g. rigid meshes use the `pos` set, while deformable meshes use the
        // `cur_x` field.
        let ObjectData { solids, shells, .. } = self;
        Self::mesh_vertex_subset_mut_impl(cur_v, alt_v, src_idx, solids, shells)
    }

    pub fn prev_vel_mut(&mut self, src_idx: SourceIndex) -> MeshVertexData3<&mut [f64]> {
        let ObjectData {
            prev_v,
            vel,
            solids,
            shells,
            ..
        } = self;
        Self::mesh_vertex_subset_mut_impl(
            prev_v.view_mut(),
            vel.view_mut(),
            src_idx,
            solids,
            shells,
        )
    }

    pub fn mesh_vertex_subset<'s, 'x, T: Clone + 'x>(
        &self,
        x: VertexView3<'x, &'x [T]>,
        alt: VertexView3<'x, &'x [T]>,
        source: SourceIndex,
    ) -> MeshVertexData3<&'x [T]> {
        Self::mesh_vertex_subset_impl(x, alt, source, &self.solids, &self.shells)
    }

    pub fn mesh_vertex_subset_mut<'s, 'x, D: 'x, Alt>(
        &self,
        x: VertexView<'x, D>,
        alt: Alt,
        source: SourceIndex,
    ) -> MeshVertexData<D>
    where
        D: Set + RemovePrefix,
        std::ops::Range<usize>: IsolateIndex<D, Output = D>,
        Alt: Into<Option<VertexView<'x, D>>>,
    {
        Self::mesh_vertex_subset_mut_impl(x, alt, source, &self.solids, &self.shells)
    }

    fn mesh_vertex_subset_impl<'x, T: Clone + 'x>(
        x: VertexView3<'x, &'x [T]>,
        alt: VertexView3<'x, &'x [T]>,
        source: SourceIndex,
        solids: &[TetMeshSolid],
        shells: &[TriMeshShell],
    ) -> MeshVertexData3<&'x [T]> {
        match source {
            SourceIndex::Solid(i) => Subset::from_unique_ordered_indices(
                solids[i].surface().indices.to_vec(),
                x.at(0).at(i),
            ),
            SourceIndex::Shell(i) => {
                let props = shells[i].material.properties;

                // Determine source data.
                let x = match props {
                    ShellProperties::Deformable { .. } => x,
                    _ => alt,
                };

                Subset::all(x.at(1).at(i))
            }
        }
    }

    // TODO: refactor this function together with the function above.
    fn mesh_vertex_subset_mut_impl<'x, D: 'x, Alt>(
        x: VertexView<'x, D>,
        alt: Alt,
        source: SourceIndex,
        solids: &[TetMeshSolid],
        shells: &[TriMeshShell],
    ) -> MeshVertexData<D>
    where
        D: Set + RemovePrefix,
        std::ops::Range<usize>: IsolateIndex<D, Output = D>,
        Alt: Into<Option<VertexView<'x, D>>>,
    {
        match source {
            SourceIndex::Solid(i) => Subset::from_unique_ordered_indices(
                solids[i].surface().indices.to_vec(),
                x.isolate(0).isolate(i),
            ),
            SourceIndex::Shell(i) => {
                let props = shells[i].material.properties;

                // Determine source data.
                let x = match props {
                    ShellProperties::Deformable { .. } => x,
                    _ => alt.into().unwrap_or(x),
                };

                Subset::all(x.isolate(1).isolate(i))
            }
        }
    }

    /// Split a given array into a pair of mutable views.
    /// This works when contacts happen on two different objects, however
    /// if/when we implement self contact, this needs to be inspected carefully.
    fn mesh_vertex_subset_split_mut<'x, T: Clone + 'x, Alt>(
        &self,
        x: VertexView3<'x, &'x mut [T]>,
        alt: Alt,
        source: [SourceIndex; 2],
    ) -> [MeshVertexData3<&'x mut [T]>; 2]
    where
        Alt: Into<Option<VertexView3<'x, &'x mut [T]>>>,
    {
        Self::mesh_vertex_subset_split_mut_impl(x, alt, source, &self.solids, &self.shells)
    }

    // TODO: Refactor this monstrosity
    fn mesh_vertex_subset_split_mut_impl<'x, T: Clone + 'x, Alt>(
        x: VertexView3<'x, &'x mut [T]>,
        alt: Alt,
        source: [SourceIndex; 2],
        solids: &[TetMeshSolid],
        shells: &[TriMeshShell],
    ) -> [MeshVertexData3<&'x mut [T]>; 2]
    where
        Alt: Into<Option<VertexView3<'x, &'x mut [T]>>>,
    {
        match source[0] {
            SourceIndex::Solid(i) => match source[1] {
                SourceIndex::Solid(j) => {
                    if i < j {
                        let (l, r) = x.isolate(0).split_at(j);
                        [
                            Subset::from_unique_ordered_indices(
                                solids[i].surface().indices.to_vec(),
                                l.isolate(i),
                            ),
                            Subset::from_unique_ordered_indices(
                                solids[j].surface().indices.to_vec(),
                                r.isolate(0),
                            ),
                        ]
                    } else {
                        assert_ne!(i, j); // This needs special handling for self contact.
                        let (l, r) = x.isolate(0).split_at(i);
                        [
                            Subset::from_unique_ordered_indices(
                                solids[i].surface().indices.to_vec(),
                                r.isolate(0),
                            ),
                            Subset::from_unique_ordered_indices(
                                solids[j].surface().indices.to_vec(),
                                l.isolate(j),
                            ),
                        ]
                    }
                }
                SourceIndex::Shell(j) => {
                    let props = shells[j].material.properties;

                    // Determine source data.
                    let x = match props {
                        ShellProperties::Deformable { .. } => x,
                        _ => match alt.into() {
                            Some(alt) => {
                                return [
                                    Subset::from_unique_ordered_indices(
                                        solids[i].surface().indices.to_vec(),
                                        x.isolate(0).isolate(i),
                                    ),
                                    Subset::all(alt.isolate(1).isolate(j)),
                                ];
                            }
                            None => x,
                        },
                    };

                    let (l, r) = x.split_at(1);
                    [
                        Subset::from_unique_ordered_indices(
                            solids[i].surface().indices.to_vec(),
                            l.isolate(0).isolate(i),
                        ),
                        Subset::all(r.isolate(0).isolate(j)),
                    ]
                }
            },
            SourceIndex::Shell(i) => {
                let props = shells[i].material.properties;

                // Determine source data.
                let x = match props {
                    ShellProperties::Deformable { .. } => x,
                    _ => match alt.into() {
                        Some(alt) => {
                            return match source[1] {
                                SourceIndex::Solid(j) => [
                                    Subset::all(alt.isolate(1).isolate(i)),
                                    Subset::from_unique_ordered_indices(
                                        solids[j].surface().indices.to_vec(),
                                        x.isolate(0).isolate(j),
                                    ),
                                ],
                                SourceIndex::Shell(j) => {
                                    let x = match props {
                                        ShellProperties::Deformable { .. } => x,
                                        _ => alt, // Both non-deformable shells.
                                    };

                                    if i < j {
                                        let (l, r) = x.isolate(1).split_at(j);
                                        [Subset::all(l.isolate(i)), Subset::all(r.isolate(0))]
                                    } else {
                                        assert_ne!(i, j); // This needs special handling for self contact.
                                        let (l, r) = x.isolate(1).split_at(i);
                                        [Subset::all(r.isolate(0)), Subset::all(l.isolate(j))]
                                    }
                                }
                            };
                        }
                        None => x,
                    },
                };

                match source[1] {
                    SourceIndex::Solid(j) => {
                        let (l, r) = x.split_at(1);
                        [
                            Subset::all(r.isolate(0).isolate(i)),
                            Subset::from_unique_ordered_indices(
                                solids[j].surface().indices.to_vec(),
                                l.isolate(0).isolate(j),
                            ),
                        ]
                    }
                    SourceIndex::Shell(j) => {
                        if i < j {
                            let (l, r) = x.isolate(1).split_at(j);
                            [Subset::all(l.isolate(i)), Subset::all(r.isolate(0))]
                        } else {
                            assert_ne!(i, j); // This needs special handling for self contact.
                            let (l, r) = x.isolate(1).split_at(i);
                            [Subset::all(r.isolate(0)), Subset::all(l.isolate(j))]
                        }
                    }
                }
            }
        }
    }

    pub fn mesh_surface_vertex_count(&self, source: SourceIndex) -> usize {
        match source {
            SourceIndex::Solid(i) => self.solids[i].surface().indices.len(),
            SourceIndex::Shell(i) => self.shells[i].trimesh.num_vertices(),
        }
    }

    /// Translate a surface mesh coordinate index into the corresponding
    /// simulation coordinate index in our global array.
    pub fn source_coordinates(
        &self,
        object_index: SourceIndex,
        collider_index: SourceIndex,
        coord: usize,
    ) -> usize {
        let surf_vtx_idx = coord / 3;

        // Retrieve tetmesh coordinates when it's determined that the source is a solid tetmesh.
        let tetmesh_coordinates = |mesh_index, surf_vtx_idx| {
            let offset = self.prev_v.view().at(0).offset_value(mesh_index);
            3 * (offset + self.solids[mesh_index].surface().indices[surf_vtx_idx]) + coord % 3
        };

        // Retrieve trimesh coordinates when it's determined that the source is a shell trimesh.
        let trimesh_coordinates = |mesh_index| {
            let prev_v = self.prev_v.view().at(1);
            assert!(!prev_v.is_empty());
            3 * prev_v.offset_value(mesh_index) + coord
        };

        let num_object_surface_indices: usize;

        match object_index {
            SourceIndex::Solid(obj_i) => {
                num_object_surface_indices = self.solids[obj_i].surface().indices.len();
                if surf_vtx_idx < num_object_surface_indices {
                    return tetmesh_coordinates(obj_i, surf_vtx_idx);
                }
            }
            SourceIndex::Shell(obj_i) => {
                num_object_surface_indices = self.shells[obj_i].trimesh.num_vertices();
                if surf_vtx_idx < num_object_surface_indices {
                    return trimesh_coordinates(obj_i);
                }
            }
        }

        // If we haven't returned yet, this means that surf_vtx_idx corresponds
        // to collider surface.

        // Adjust surf_vtx_idx to be in the right range:
        let surf_vtx_idx = surf_vtx_idx - num_object_surface_indices;

        match collider_index {
            SourceIndex::Solid(coll_i) => {
                assert!(surf_vtx_idx < self.solids[coll_i].surface().indices.len());
                tetmesh_coordinates(coll_i, surf_vtx_idx)
            }
            SourceIndex::Shell(coll_i) => {
                assert!(surf_vtx_idx < self.shells[coll_i].trimesh.num_vertices());
                trimesh_coordinates(coll_i)
            }
        }
    }

    /// Produce an iterator over the given slice of scaled variables.
    pub(crate) fn scaled_variables_iter<'a>(
        unscaled_var: &'a [Number],
        scale: Number,
    ) -> impl Iterator<Item = Number> + 'a {
        unscaled_var.iter().map(move |&val| val * scale)
    }

    pub fn update_current_velocity(
        &self,
        uv: &[Number],
        scale: f64,
    ) -> std::cell::Ref<'_, VertexData3<Vec<f64>>> {
        {
            let mut sv = self.cur_v.borrow_mut();
            let sv = sv.view_mut().into_flat();
            for (output, input) in sv.iter_mut().zip(Self::scaled_variables_iter(uv, scale)) {
                *output = input;
            }
        }
        self.cur_v.borrow()
    }

    // Update `cur_x` using implicit integration with the given velocity `v`.
    fn integrate_step(&self, v: VertexView3<&[f64]>, dt: f64) {
        debug_assert!(dt > 0.0);
        let ObjectData { cur_x, prev_x, .. } = self;
        let mut cur_x = cur_x.borrow_mut();
        let cur_x = cur_x.view_mut().into_flat();
        let v = v.view().into_flat();
        let prev_x = prev_x.view().into_flat();
        // In static simulations, velocity is simply displacement.
        debug_assert_eq!(v.len(), cur_x.len());
        debug_assert_eq!(prev_x.len(), cur_x.len());
        cur_x
            .iter_mut()
            .zip(prev_x.iter().zip(v.iter()))
            .for_each(|(x1, (&x0, &v))| *x1 = x0 + v * dt);
    }

    /// Update the solid meshes with the given global array of vertex positions
    /// for all solids. Note that we set velocities only, since the positions will be updated
    /// automatically from the ipopt solution.
    pub fn update_solid_vertices(
        &mut self,
        new_pos: Chunked3<&[f64]>,
        time_step: f64,
    ) -> Result<(), crate::Error> {
        // All solids have prev_x coincident with pos so we use prev_x directly here.
        let mut prev_x = self.prev_x.view_mut().isolate(0);
        let mut prev_v = self.prev_v.view_mut().isolate(0);

        // All solids are simulated, so the input point set must have the same
        // size as our internal vertex set. If these are mismatched, then there
        // was an issue with constructing the solid meshes. This may not
        // necessarily be an error, we are just being conservative here.
        if new_pos.len() != prev_x.view().data().len() {
            // We got an invalid point cloud
            return Err(crate::Error::SizeMismatch);
        }

        debug_assert!(time_step > 0.0);
        let dt_inv = 1.0 / time_step;

        // Get the tetmesh and prev_pos so we can update the fixed vertices.
        for (solid, mut prev_pos, mut prev_vel) in
            zip!(self.solids.iter(), prev_x.iter_mut(), prev_v.iter_mut())
        {
            // Get the vertex index of the original vertex (point in given point cloud).
            // This is done because meshes can be reordered when building the
            // solver. This attribute maintains the link between the caller and
            // the internal mesh representation. This way the user can still
            // update internal meshes as needed between solves.
            let source_index_iter = solid
                .tetmesh
                .attrib_iter::<SourceIndexType, VertexIndex>(SOURCE_INDEX_ATTRIB)?;
            let new_pos_iter = source_index_iter.map(|&idx| new_pos.get(idx));

            // Only update fixed vertices, if no such attribute exists, return an error.
            let fixed_iter = solid
                .tetmesh
                .attrib_iter::<FixedIntType, VertexIndex>(FIXED_ATTRIB)?;
            prev_pos
                .iter_mut()
                .zip(prev_vel.iter_mut())
                .zip(new_pos_iter)
                .zip(fixed_iter)
                .filter_map(|(data, &fixed)| if fixed != 0i8 { Some(data) } else { None })
                .for_each(|((pos, vel), new_pos)| {
                    // Update the vertices we find in the given `new_pos` collection, not all may
                    // still be there.
                    if let Some(&new_pos) = new_pos {
                        *vel.as_mut_tensor() =
                            (*new_pos.as_tensor() - *(*pos).as_tensor()) * dt_inv;
                        //*pos = new_pos; // automatically updated via solve.
                    }
                });

            // TODO: Compute new velocity from previous step. This is an explicit velocity estimate.
            // TODO: determine if this is sufficient, or do we need the user to
            // specify a more accurate velocity estimate for fixed vertices.
        }
        Ok(())
    }

    /// Update the shell meshes with the given global array of vertex positions
    /// and velocities for all shells.
    pub fn update_shell_vertices(&mut self, new_pos: Chunked3<&[f64]>) -> Result<(), crate::Error> {
        // Some shells are simulated on a per vertex level, some are rigid or
        // fixed, so we will update `prev_x` for the former and `pos` for the
        // latter.
        let ObjectData { prev_x, pos, .. } = self;

        let mut prev_x = prev_x.view_mut().isolate(1);
        //let mut prev_v = prev_x.view_mut().isolate(1);
        let mut pos = pos.view_mut().isolate(1);

        //let dt_inv = if self.time_step > 0.0 { 1.0 / self.time_step } else { 1.0 };

        // Get the trimesh and prev_x/pos so we can update the fixed vertices.
        for (shell, (mut prev_x, mut pos)) in self
            .shells
            .iter()
            .zip(prev_x.iter_mut().zip(pos.iter_mut()))
        {
            // Get the vertex index of the original vertex (point in given point cloud).
            // This is done because meshes can be reordered when building the
            // solver. This attribute maintains the link between the caller and
            // the internal mesh representation. This way the user can still
            // update internal meshes as needed between solves.
            let source_index_iter = shell
                .trimesh
                .attrib_iter::<SourceIndexType, VertexIndex>(SOURCE_INDEX_ATTRIB)?;
            let new_pos_iter = source_index_iter.map(|&idx| new_pos.get(idx));

            match shell.material.properties {
                ShellProperties::Deformable { .. } => {
                    // Only update fixed vertices, if no such attribute exists, return an error.
                    let fixed_iter = shell
                        .trimesh
                        .attrib_iter::<FixedIntType, VertexIndex>(FIXED_ATTRIB)?;
                    prev_x
                        .iter_mut()
                        .zip(new_pos_iter)
                        .zip(fixed_iter)
                        .filter_map(|(pair, &fixed)| if fixed != 0i8 { Some(pair) } else { None })
                        .for_each(|(pos, new_pos)| {
                            // It's possible that the new vector of positions is missing some
                            // vertices that were fixed before, so we try to update those we
                            // actually find in the `new_pos` collection.
                            if let Some(&new_pos) = new_pos {
                                *pos = new_pos;
                            }
                        });
                }
                ShellProperties::Rigid { .. } => {
                    // A rigid mesh has different degrees of freedom than its vertex positions.
                    // Here we extract a rigid motion from the fixed vertices
                    // and apply it to the rest of the mesh.

                    // TODO: do a least squares fit for a rotation from current
                    // configuration to the given configuration of fixed
                    // vertices.

                    // Steps:
                    // 1. Apply the current rigid transform (prev_x/prev_v) to
                    //    `shell` and store in `pos/vel` to figure out the
                    //    current vertex positions.
                    // 2. Compute the rigid rotation between that and the given
                    //    set of fixed points.
                    // 3. Add this rotation and translation to (prev_x/prev_v)
                    return Err(crate::Error::UnimplementedFeature {
                        description: String::from("Rigid body update"),
                    });
                }
                ShellProperties::Fixed => {
                    // This mesh is fixed and doesn't obey any physics. Simply
                    // copy the positions and velocities over.
                    pos.iter_mut().zip(new_pos_iter).for_each(|(pos, new_pos)| {
                        if let Some(&new_pos) = new_pos {
                            *pos = new_pos;
                        }
                    });
                }
            }
        }
        Ok(())
    }

    /// Advance prev_* variables to cur_* variables and update the referenced meshes.
    pub fn advance(&mut self, and_velocity: bool) {
        let ObjectData {
            prev_x,
            prev_v,
            cur_v,
            cur_x,
            prev_prev_x,
            prev_prev_v,
            solids,
            shells,
            ..
        } = self;

        let cur_v = cur_v.borrow();
        let cur_x = cur_x.borrow();

        {
            let prev_x_flat_view = prev_x.view_mut().into_flat();
            let prev_v_flat_view = prev_v.view_mut().into_flat();

            // Update prev pos
            zip!(
                prev_prev_x.iter_mut(),
                prev_x_flat_view.iter_mut(),
                cur_x.view().into_flat().iter()
            )
            .for_each(|(prev_prev, prev, &cur)| {
                *prev_prev = *prev;
                *prev = cur;
            });

            // Update prev vel
            if and_velocity {
                zip!(
                    prev_prev_v.iter_mut(),
                    prev_v_flat_view.iter_mut(),
                    cur_v.view().into_flat().iter()
                )
                .for_each(|(prev_prev, prev, &cur)| {
                    *prev_prev = *prev;
                    *prev = cur
                });
            } else {
                // Clear velocities. This ensures that any non-zero initial velocities are cleared
                // for subsequent steps.
                prev_prev_v
                    .iter_mut()
                    .zip(prev_v_flat_view.iter_mut())
                    .for_each(|(prev, v)| {
                        *prev = *v;
                        *v = 0.0
                    });
            }
        }

        Self::update_meshes_with(solids, shells, prev_x.view(), prev_v.view());
    }

    pub fn revert_prev_step(&mut self) {
        let ObjectData {
            prev_x,
            prev_v,
            prev_prev_x,
            prev_prev_v,
            ..
        } = self;

        {
            let prev_x_flat_view = prev_x.view_mut().into_flat();
            let prev_v_flat_view = prev_v.view_mut().into_flat();

            prev_prev_x
                .iter()
                .zip(prev_x_flat_view.iter_mut())
                .for_each(|(prev, cur)| *cur = *prev);

            prev_prev_v
                .iter()
                .zip(prev_v_flat_view.iter_mut())
                .for_each(|(prev, cur)| *cur = *prev);
        }
    }

    pub fn update_meshes_with(
        solids: &mut [TetMeshSolid],
        shells: &mut [TriMeshShell],
        x: VertexView3<&[f64]>,
        v: VertexView3<&[f64]>,
    ) {
        // Update mesh vertex positions and velocities
        for (i, solid) in solids.iter_mut().enumerate() {
            let verts = solid.tetmesh.vertex_positions_mut();
            verts.copy_from_slice(x.at(0).at(i).into());
            solid
                .tetmesh
                .attrib_as_mut_slice::<_, VertexIndex>(VELOCITY_ATTRIB)
                .expect("Missing velocity attribute")
                .copy_from_slice(v.at(0).at(i).into());
        }
        for (i, shell) in shells.iter_mut().enumerate() {
            match shell.material.properties {
                ShellProperties::Deformable { .. } => {
                    let verts = shell.trimesh.vertex_positions_mut();
                    verts.copy_from_slice(x.at(1).at(i).into());
                    shell
                        .trimesh
                        .attrib_as_mut_slice::<_, VertexIndex>(VELOCITY_ATTRIB)
                        .expect("Missing velocity attribute")
                        .copy_from_slice(v.at(1).at(i).into());
                }
                ShellProperties::Rigid { .. } => {
                    unimplemented!();
                }
                ShellProperties::Fixed => {
                    // Nothing to do here, fixed meshes aren't moved by
                    // simulation.
                }
            }
        }
    }
}

/// This struct encapsulates the non-linear problem to be solved by a non-linear solver like Ipopt.
/// It is meant to be owned by the solver.
pub(crate) struct NonLinearProblem {
    /// A model of the current problem. This includes all primal variables and
    /// any additional mesh data required for simulation.
    pub object_data: ObjectData,
    /// One way contact constraints between a pair of objects.
    pub frictional_contacts: Vec<FrictionalContactConstraint>,
    /// Constraint on the total volume.
    pub volume_constraints: Vec<(usize, RefCell<VolumeConstraint>)>,
    /// Gravitational potential energy.
    pub gravity: [f64; 3],
    /// The time step defines the amount of time elapsed between steps (calls to `advance`).
    /// If the time step is zero, objects don't exhibit inertia.
    pub time_step: f64,
    /// Interrupt callback that interrupts the solver (making it return prematurely) if the closure
    /// returns `false`.
    pub interrupt_checker: Box<dyn FnMut() -> bool>,
    /// Count the number of iterations.
    pub iterations: usize,
    /// Solution data. This is kept around for warm starts.
    pub warm_start: Solution,
    pub initial_residual_error: f64,
    pub iter_counter: RefCell<usize>,

    /// The maximum size (diameter) of a simulated object (deformable or rigid).
    pub max_size: f64,
    pub max_modulus: f64,
}

impl NonLinearProblem {
    pub fn variable_scale(&self) -> f64 {
        // This scaling makes variables unitless.
        utils::approx_power_of_two64(self.max_size / self.time_step())
    }

    fn impulse_inv_scale(&self) -> f64 {
        utils::approx_power_of_two64(
            1.0 / (self.time_step() * self.max_size * self.max_size * self.max_modulus),
        )
    }

    fn volume_constraint_scale(&self) -> f64 {
        1.0
    }

    fn contact_constraint_scale(&self) -> f64 {
        1.0
    }

    fn time_step(&self) -> f64 {
        if self.is_static() {
            1.0
        } else {
            self.time_step
        }
    }

    /// Check if this problem represents a static simulation. In this case
    /// inertia is ignored and velocities are treated as displacements.
    fn is_static(&self) -> bool {
        self.time_step == 0.0
    }

    /// Produce an iterator over the given slice of scaled variables.
    pub fn scaled_variables_iter<'a>(
        &self,
        unscaled_var: &'a [Number],
    ) -> impl Iterator<Item = Number> + 'a {
        ObjectData::scaled_variables_iter(unscaled_var, self.variable_scale())
    }

    /// Save Ipopt solution for warm starts.
    pub fn update_warm_start(&mut self, solution: ipopt::Solution) {
        self.warm_start.update(solution);
    }

    ///// Clear the warm start using the sizes in the given solution.
    //pub fn clear_warm_start(&mut self, solution: ipopt::Solution) {
    //    self.warm_start.reset(
    //        solution.primal_variables.len(),
    //        solution.constraint_multipliers.len(),
    //    );
    //}

    /// Reset solution used for warm starts. Note that if the number of constraints has changed,
    /// then this method will set the warm start to have the new number of constraints.
    pub fn reset_warm_start(&mut self) {
        use ipopt::{BasicProblem, ConstrainedProblem};
        self.warm_start
            .reset(self.num_variables(), self.num_constraints());
    }

    /// Get the current iteration count and reset it.
    pub fn pop_iteration_count(&mut self) -> usize {
        let iter = self.iterations;
        // Reset count
        self.iterations = 0;
        iter
    }

    /// Intermediate callback for `Ipopt`.
    pub fn intermediate_cb(&mut self, data: ipopt::IntermediateCallbackData) -> bool {
        if data.iter_count == 0 {
            // Record the initial max of dual and primal infeasibility.
            if data.inf_du > 0.0 {
                self.initial_residual_error = data.inf_du;
            }
        }

        self.iterations += 1;
        !(self.interrupt_checker)()
    }

    /// Get the minimum contact radius among all contact problems. If there are
    /// no contacts, simply return `None`.
    pub fn min_contact_radius(&self) -> Option<f64> {
        self.frictional_contacts
            .iter()
            .map(|fc| fc.constraint.borrow().contact_radius())
            .min_by(|a, b| a.partial_cmp(b).expect("Detected NaN contact radius"))
    }

    /// Save an intermediate state of the solve. This is used for debugging.
    #[allow(dead_code)]
    pub fn save_intermediate(&mut self, uv: &[f64], step: usize) {
        let v = self.update_current_velocity(uv);
        let x = self.compute_step(v.view());
        let mut solids = self.object_data.solids.clone();
        let mut shells = self.object_data.shells.clone();
        ObjectData::update_meshes_with(&mut solids, &mut shells, x.view(), v.view());
        geo::io::save_tetmesh(
            &solids[0].tetmesh,
            &std::path::PathBuf::from(format!("./out/predictor_{}.vtk", step)),
        )
        .unwrap();
    }

    /// Update the solid meshes with the given points.
    pub fn update_solid_vertices(&mut self, pts: &PointCloud) -> Result<(), crate::Error> {
        let new_pos = Chunked3::from_array_slice(pts.vertex_positions());
        self.object_data
            .update_solid_vertices(new_pos.view(), self.time_step())
    }

    /// Update the shell meshes with the given points.
    pub fn update_shell_vertices(&mut self, pts: &PointCloud) -> Result<(), crate::Error> {
        let new_pos = Chunked3::from_array_slice(pts.vertex_positions());
        self.object_data.update_shell_vertices(new_pos.view())
    }

    pub fn update_current_velocity(
        &self,
        uv: &[Number],
    ) -> std::cell::Ref<'_, VertexData3<Vec<f64>>> {
        self.object_data
            .update_current_velocity(uv, self.variable_scale())
    }

    /// Compute the set of currently active constraints into the given `Chunked` `Vec`.
    pub fn compute_active_constraint_set(&self, active_set: &mut Chunked<Vec<usize>>) {
        // Disassemble chunked collection.
        let (offsets, active_set) = active_set.as_inner_mut();

        for i in 0..self.volume_constraints.len() {
            active_set.push(i);
            offsets.push(active_set.len());
        }

        let mut offset = active_set.len();
        for FrictionalContactConstraint { ref constraint, .. } in self.frictional_contacts.iter() {
            let fc_active_constraints = constraint.borrow().active_constraint_indices();
            for c in fc_active_constraints.into_iter() {
                active_set.push(c + offset);
            }
            offset += constraint.borrow().num_potential_contacts();
            offsets.push(active_set.len());
        }
    }

    /// Get the set of currently active constraints.
    pub fn active_constraint_set(&self) -> Chunked<Vec<usize>> {
        let mut active_set = Chunked::new();
        self.compute_active_constraint_set(&mut active_set);
        active_set
    }

    #[allow(dead_code)]
    pub fn clear_friction_impulses(&mut self) {
        for fc in self.frictional_contacts.iter_mut() {
            fc.constraint
                .borrow_mut()
                .clear_frictional_contact_impulse();
        }
    }

    /// Restore the constraint set.
    pub fn reset_constraint_set(&mut self) -> bool {
        self.update_constraint_set(None)
    }

    /// Check if the given constraint set is the same as the current one.
    pub fn is_same_as_constraint_set(&self, other_set: ChunkedView<&[usize]>) -> bool {
        let cur_set = self.active_constraint_set().into_flat();
        let other_set = other_set.into_flat();
        cur_set.len() == other_set.len()
            && cur_set
                .into_iter()
                .zip(other_set.iter())
                .all(|(cur, &other)| cur == other)
    }

    /// Update all stateful constraints with the most recent data.
    /// Return an estimate if any constraints have changed, though this estimate may have false
    /// negatives.
    pub fn update_constraint_set(&mut self, solution: Option<ipopt::Solution>) -> bool {
        let mut changed = false; // Report if anything has changed to the caller.

        let scale = self.variable_scale();

        let time_step = self.time_step();
        let NonLinearProblem {
            ref mut frictional_contacts,
            ref object_data,
            ..
        } = *self;

        // Update positions with the given solution (if any).
        let x = solution.map(|sol| {
            let v = object_data.update_current_velocity(sol.primal_variables, scale);
            object_data.integrate_step(v.view(), time_step);
            object_data.cur_x.borrow()
        });

        for FrictionalContactConstraint {
            object_index,
            collider_index,
            constraint,
        } in frictional_contacts.iter_mut()
        {
            if let Some(x) = x.as_ref() {
                let object_pos = object_data.cur_pos((*x).view(), *object_index);
                let collider_pos = object_data.cur_pos((*x).view(), *collider_index);
                changed |= constraint
                    .borrow_mut()
                    .update_neighbours(object_pos.view(), collider_pos.view());
            } else {
                let object_pos = object_data.prev_pos(*object_index);
                let collider_pos = object_data.prev_pos(*collider_index);
                changed |= constraint
                    .borrow_mut()
                    .update_neighbours(object_pos.view(), collider_pos.view());
            }
        }

        changed
    }

    /// Build a new set of multipliers from the old set and replace warm start multipliers with the
    /// new set.
    pub fn remap_warm_start(&mut self, old_constraint_set: ChunkedView<&[usize]>) {
        use crate::constraints::remap_values;
        let active_constraint_set = self.active_constraint_set();
        let new_values = active_constraint_set.data();
        let old_values = old_constraint_set.data();

        // Remap multipliers
        let new_multipliers = remap_values(
            self.warm_start.constraint_multipliers.iter().cloned(),
            0.0,
            old_values.iter().cloned(),
            new_values.iter().cloned(),
        );
        std::mem::replace(&mut self.warm_start.constraint_multipliers, new_multipliers);
    }

    ///// Update all stateful constraints with the most recent data. This also involves remapping any
    ///// multipliers that may have changed.
    ///// Return an estimate if any constraints have changed. This estimate may have false negatives.
    //pub fn update_constraints_and_remap(&mut self, solution: ipopt::Solution) -> bool {
    //    let mut changed = false; // Report if anything has changed to the caller.

    //    let offset = if self.volume_constraint.is_some() {
    //        1
    //    } else {
    //        0
    //    };

    //    // Build a new set of multipliers from the old set.
    //    let mut new_multipliers = self.warm_start.constraint_multipliers.clone();

    //    if let Some(ref mut scc) = self.smooth_contact_constraint {
    //        let (cache_changed, mapping) = scc.update_cache_with_mapping()
    //            .expect("Failed to update cached neighbourhoods");
    //        changed |= cache_changed;
    //        new_multipliers.resize(offset + mapping.len(), 0.0);
    //        dbg!(&new_multipliers);
    //        dbg!(&self.warm_start.constraint_multipliers);
    //        for (old_idx, new_mult) in zip!(mapping.into_iter(), new_multipliers.iter_mut())
    //            .filter_map(|(idx, mult)| idx.into_option().map(|i| (i, mult)))
    //        {
    //            *new_mult = self.warm_start.constraint_multipliers[offset + old_idx];
    //        }
    //        dbg!(&new_multipliers);
    //    }

    //    std::mem::replace(&mut self.warm_start.constraint_multipliers, new_multipliers);
    //    dbg!(&self.warm_start.constraint_multipliers);

    //    changed
    //}

    pub fn apply_frictional_contact_impulse(&mut self) {
        let NonLinearProblem {
            frictional_contacts,
            object_data:
                ObjectData {
                    vel,
                    cur_v,
                    solids,
                    shells,
                    ..
                },
            ..
        } = self;

        let mut cur_v = cur_v.borrow_mut();
        for fc in frictional_contacts.iter() {
            // TODO: It is unfortunate that we have to leak abstraction here.
            // Possibly we have to hide object_data behind a RefCell and use
            // borrow splitting.
            let [mut obj_vel, mut coll_vel] = ObjectData::mesh_vertex_subset_split_mut_impl(
                cur_v.view_mut(),
                vel.view_mut(),
                [fc.object_index, fc.collider_index],
                solids,
                shells,
            );

            fc.constraint
                .borrow()
                .add_mass_weighted_frictional_contact_impulse([
                    obj_vel.view_mut(),
                    coll_vel.view_mut(),
                ]);
        }
    }

    /// Commit velocity by advancing the internal state by the given unscaled velocity `uv`.
    /// If `and_velocity` is `false`, then only positions are advanced, and velocities are reset.
    /// This emulates a critically damped, or quasi-static simulation.
    pub fn advance(&mut self, uv: &[f64], and_velocity: bool, and_warm_start: bool) {
        self.update_current_velocity(uv);
        {
            self.apply_frictional_contact_impulse();

            let cur_v = self.object_data.cur_v.borrow();
            self.compute_step(cur_v.view());
        }

        self.object_data.advance(and_velocity);

        if !and_warm_start {
            self.reset_warm_start();
        }
    }

    /// Advance object data one step back.
    pub fn revert_prev_step(&mut self) {
        self.object_data.revert_prev_step();
        self.reset_warm_start();
        // Clear any frictional impulsesl
        for fc in self.frictional_contacts.iter() {
            if let Some(friction_data) = fc.constraint.borrow_mut().frictional_contact_mut() {
                friction_data
                    .collider_impulse
                    .source_iter_mut()
                    .for_each(|(x, y)| {
                        *x = [0.0; 3];
                        *y = [0.0; 3]
                    });
                friction_data.object_impulse.iter_mut().for_each(|(x, y)| {
                    *x = [0.0; 3];
                    *y = [0.0; 3]
                });
            }
        }
    }

    pub fn update_max_step(&mut self, step: f64) {
        for fc in self.frictional_contacts.iter_mut() {
            fc.constraint.borrow_mut().update_max_step(step);
        }
    }
    pub fn update_radius_multiplier(&mut self, rad_mult: f64) {
        for fc in self.frictional_contacts.iter_mut() {
            fc.constraint
                .borrow_mut()
                .update_radius_multiplier(rad_mult);
        }
    }

    //pub fn precompute_linearized_constraints(&mut self) {
    //    let NonLinearProblem {
    //        ref mut frictional_contacts,
    //        ref object_data,
    //        ..
    //    } = *self;

    //    for FrictionalContactConstraint {
    //        object_index,
    //        collider_index,
    //        constraint,
    //    } in frictional_contacts.iter_mut()
    //    {
    //        let object_pos = object_data.prev_pos(*object_index);
    //        let collider_pos = object_data.prev_pos(*collider_index);
    //        constraint
    //            .borrow_mut()
    //            .linearize_constraint(object_pos.view(), collider_pos.view());
    //    }
    //}

    ///// Revert to the given old solution by the given displacement.
    //pub fn revert_to(
    //    &mut self,
    //    solution: Solution,
    //    old_prev_pos: Vec<Vector3<f64>>,
    //    old_prev_vel: Vec<Vector3<f64>>,
    //) {
    //    {
    //        // Reinterpret solver variables as positions in 3D space.
    //        let mut prev_pos = self.object_data.prev_pos.data().borrow_mut();
    //        let mut prev_vel = self.object_data.prev_vel.data().borrow_mut();

    //        std::mem::replace(&mut *prev_vel, old_prev_vel);
    //        std::mem::replace(&mut *prev_pos, old_prev_pos);

    //        let mut tetmesh = self.tetmesh.borrow_mut();
    //        let verts = tetmesh.vertex_positions_mut();
    //        verts.copy_from_slice(reinterpret_slice(prev_pos.as_slice()));

    //        std::mem::replace(&mut self.warm_start, solution);
    //    }

    //    // Since we transformed the mesh, we need to invalidate its neighbour data so it's
    //    // recomputed at the next time step (if applicable).
    //    //self.update_constraints();
    //}

    fn compute_constraint_violation(&self, unscaled_vel: &[f64], constraint: &mut [f64]) {
        use ipopt::ConstrainedProblem;
        let mut lower = vec![0.0; constraint.len()];
        let mut upper = vec![0.0; constraint.len()];
        self.constraint_bounds(&mut lower, &mut upper);
        assert!(self.constraint(unscaled_vel, constraint));
        for (c, (l, u)) in constraint
            .iter_mut()
            .zip(lower.into_iter().zip(upper.into_iter()))
        {
            assert!(l <= u); // sanity check

            // Subtract the appropriate bound from the constraint function:
            // If the constraint is lower than the lower bound, then the multiplier will be
            // non-zero and the result of the constraint force must be balanced by the objective
            // gradient under convergence. The same goes for the upper bound.
            if *c < l {
                *c -= l;
            } else if *c > u {
                *c -= u;
            } else {
                *c = 0.0; // Otherwise the constraint is satisfied so we set it to zero.
            }
        }
    }

    fn constraint_violation_norm(&self, unscaled_vel: &[f64]) -> f64 {
        use ipopt::ConstrainedProblem;
        let mut g = vec![0.0; self.num_constraints()];
        self.compute_constraint_violation(unscaled_vel, &mut g);
        crate::inf_norm(g)
    }

    /// Return the constraint violation and whether the neighbourhood data (sparsity) would be
    /// changed if we took this step.
    pub fn probe_contact_constraint_violation(&mut self, solution: ipopt::Solution) -> f64 {
        self.constraint_violation_norm(solution.primal_variables)
    }

    ///// Linearized constraint true violation measure.
    //pub fn linearized_constraint_violation_l1(&self, dx: &[f64]) -> f64 {
    //    let mut value = 0.0;

    //    if let Some(ref scc) = self.smooth_contact_constraint {
    //        let prev_pos = self.prev_pos.borrow();
    //        let x: &[Number] = reinterpret_slice(prev_pos.as_slice());

    //        let n = scc.constraint_size();

    //        let mut g = vec![0.0; n];
    //        scc.0.constraint(x, dx, &mut g);

    //        let (g_l, g_u) = scc.0.constraint_bounds();
    //        assert_eq!(g_l.len(), n);
    //        assert_eq!(g_u.len(), n);

    //        value += g
    //            .into_iter()
    //            .zip(g_l.into_iter().zip(g_u.into_iter()))
    //            .map(|(c, (l, u))| {
    //                assert!(l <= u);
    //                if c < l {
    //                    // below lower bound
    //                    l - c
    //                } else if c > u {
    //                    // above upper bound
    //                    c - u
    //                } else {
    //                    // Constraint not violated
    //                    0.0
    //                }
    //            })
    //            .sum::<f64>();
    //    }

    //    value
    //}

    ///// Linearized constraint model violation measure.
    //pub fn linearized_constraint_violation_model_l1(&self, dx: &[f64]) -> f64 {
    //    let mut value = 0.0;

    //    if let Some(ref scc) = self.smooth_contact_constraint {
    //        let prev_pos = self.object_data.prev_pos.data().borrow();
    //        let x: &[Number] = prev_pos.as_slice();

    //        let n = scc.constraint_size();

    //        let mut g = vec![0.0; n];
    //        scc.constraint(x, dx, &mut g);

    //        let (g_l, g_u) = scc.constraint_bounds();
    //        assert_eq!(g_l.len(), n);
    //        assert_eq!(g_u.len(), n);

    //        value += g
    //            .into_iter()
    //            .zip(g_l.into_iter().zip(g_u.into_iter()))
    //            .map(|(c, (l, u))| {
    //                assert!(l <= u);
    //                if c < l {
    //                    // below lower bound
    //                    l - c
    //                } else if c > u {
    //                    // above upper bound
    //                    c - u
    //                } else {
    //                    // Constraint not violated
    //                    0.0
    //                }
    //            })
    //            .sum::<f64>();
    //    }

    //    value
    //}

    /// A convenience function to integrate the given velocity by the internal time step. For
    /// implicit integration this boils down to a simple multiply by the time step.
    pub fn compute_step(
        &self,
        v: VertexView3<&[Number]>,
    ) -> std::cell::Ref<'_, VertexData3<Vec<f64>>> {
        self.object_data.integrate_step(v, self.time_step());
        self.object_data.cur_x.borrow()
    }

    /// Compute and return the objective value.
    pub fn objective_value(&self, v: VertexView3<&[Number]>) -> f64 {
        let cur_x = self.compute_step(v);
        let x1 = cur_x.view();
        let x0 = self.object_data.prev_x.view();
        let v0 = self.object_data.prev_v.view();

        let mut obj = 0.0;

        for (i, solid) in self.object_data.solids.iter().enumerate() {
            let x0 = x0.at(0).at(i).into_flat();
            let x1 = x1.at(0).at(i).into_flat();
            let v0 = v0.at(0).at(i).into_flat();
            let v = v.at(0).at(i).into_flat();
            obj += solid.elasticity().energy(x0, x1);
            obj += solid.gravity(self.gravity).energy(x0, x1);
            if !self.is_static() {
                obj += solid.inertia().energy(v0, v);
            }
        }

        for (i, shell) in self.object_data.shells.iter().enumerate() {
            let x0 = x0.at(1).at(i).into_flat();
            let x1 = x1.at(1).at(i).into_flat();
            //let v0 = v0.at(1).at(i).into_flat();
            //obj += shell.elasticity().energy(x0, x1);
            obj += shell.gravity(self.gravity).energy(x0, x1);
            //obj += shell.inertia().energy(v0, v);
        }

        // If time_step is 0.0, this is a pure static solve, which means that
        // there cannot be friction.
        if !self.is_static() {
            if FORWARD_FRICTION {
                for fc in self.frictional_contacts.iter() {
                    let obj_v = self.object_data.cur_vel(v, fc.object_index);
                    let col_v = self.object_data.cur_vel(v, fc.collider_index);
                    obj -= fc
                        .constraint
                        .borrow()
                        .frictional_dissipation([obj_v.view(), col_v.view()]);
                }
            }
        }

        obj
    }

    /// Convert a given array of contact forces to impulses.
    fn contact_impulse_magnitudes(forces: &[f64], scale: f64) -> Vec<f64> {
        forces.iter().map(|&cf| cf * scale).collect()
    }

    pub fn num_frictional_contacts(&self) -> usize {
        self.frictional_contacts.len()
    }

    /// Returns true if all friction solves have been completed/converged.
    /// This should be the case if and only if all elements in `friction_steps`
    /// are zero, which makes the return type simply a convenience.
    pub fn update_friction_impulse(
        &mut self,
        solution: ipopt::Solution,
        constraint_values: &[f64],
        friction_steps: &mut [u32],
    ) -> bool {
        if self.frictional_contacts.is_empty() {
            return true;
        }

        self.update_current_velocity(solution.primal_variables);

        let multiplier_impulse_scale = self.time_step() / self.impulse_inv_scale();
        let NonLinearProblem {
            ref mut frictional_contacts,
            ref volume_constraints,
            ref object_data,
            ..
        } = *self;

        let cur_v = object_data.cur_v.borrow();

        let mut is_finished = true;

        let mut constraint_offset = volume_constraints.len();

        for (fc_idx, fc) in frictional_contacts.iter_mut().enumerate() {
            let v = cur_v.view();
            let obj_prev_pos = self.object_data.prev_pos(fc.object_index);
            let col_prev_pos = self.object_data.prev_pos(fc.collider_index);
            let obj_vel = self.object_data.cur_vel(v, fc.object_index);
            let col_vel = self.object_data.cur_vel(v, fc.collider_index);

            let n = fc.constraint.borrow().constraint_size();
            let contact_impulse = Self::contact_impulse_magnitudes(
                &solution.constraint_multipliers[constraint_offset..constraint_offset + n],
                multiplier_impulse_scale,
            );

            debug!(
                "Maximum contact impulse: {}",
                crate::inf_norm(contact_impulse.iter().cloned())
            );
            let potential_values = &constraint_values[constraint_offset..constraint_offset + n];
            friction_steps[fc_idx] = fc
                .constraint
                .borrow_mut()
                .update_frictional_contact_impulse(
                    &contact_impulse,
                    [obj_prev_pos.view(), col_prev_pos.view()],
                    [obj_vel.view(), col_vel.view()],
                    potential_values,
                    friction_steps[fc_idx],
                );

            is_finished &= friction_steps[fc_idx] == 0;
            constraint_offset += n;
        }

        is_finished
    }

    /// Given a tetmesh, compute the strain energy per tetrahedron.
    fn compute_strain_energy_attrib(solid: &mut TetMeshSolid) {
        use geo::ops::ShapeMatrix;
        // Overwrite the "strain_energy" attribute.
        let mut strain = solid
            .tetmesh
            .remove_attrib::<CellIndex>(STRAIN_ENERGY_ATTRIB)
            .unwrap();
        strain
            .iter_mut::<f64>()
            .unwrap()
            .zip(zip!(
                    solid
                        .tetmesh
                        .attrib_iter::<LambdaType, CellIndex>(LAMBDA_ATTRIB)
                        .unwrap()
                        .map(|&x| f64::from(x)),
                    solid
                        .tetmesh
                        .attrib_iter::<MuType, CellIndex>(MU_ATTRIB)
                        .unwrap()
                        .map(|&x| f64::from(x)),
                    solid
                        .tetmesh
                        .attrib_iter::<RefVolType, CellIndex>(REFERENCE_VOLUME_ATTRIB)
                        .unwrap(),
                    solid
                        .tetmesh
                        .attrib_iter::<RefShapeMtxInvType, CellIndex>(
                            REFERENCE_SHAPE_MATRIX_INV_ATTRIB,
                        )
                        .unwrap(),
                    solid.tetmesh.tet_iter()
                ))
            .for_each(|(strain, (lambda, mu, &vol, &ref_shape_mtx_inv, tet))| {
                let shape_mtx = Matrix3::new(tet.shape_matrix()).transpose();
                *strain = NeoHookeanTetEnergy::new(shape_mtx, ref_shape_mtx_inv, vol, lambda, mu)
                    .elastic_energy()
            });

        solid
            .tetmesh
            .insert_attrib::<CellIndex>(STRAIN_ENERGY_ATTRIB, strain)
            .unwrap();
    }

    /// Given a tetmesh, compute the elastic forces per vertex, and save it at a vertex attribute.
    fn compute_elastic_forces_attrib(solid: &mut TetMeshSolid) {
        use geo::ops::ShapeMatrix;
        let mut forces_attrib = solid
            .tetmesh
            .remove_attrib::<VertexIndex>(ELASTIC_FORCE_ATTRIB)
            .unwrap();

        let mut forces =
            Chunked3::from_array_slice_mut(forces_attrib.as_mut_slice::<[f64; 3]>().unwrap());

        // Reset forces
        for f in forces.iter_mut() {
            *f = [0.0; 3];
        }

        let grad_iter =
            zip!(
            solid
                .tetmesh
                .attrib_iter::<LambdaType, CellIndex>(LAMBDA_ATTRIB)
                .unwrap().map(|&x| f64::from(x)),
            solid
                .tetmesh
                .attrib_iter::<MuType, CellIndex>(MU_ATTRIB)
                .unwrap().map(|&x| f64::from(x)),
            solid
                .tetmesh
                .attrib_iter::<RefVolType, CellIndex>(REFERENCE_VOLUME_ATTRIB)
                .unwrap(),
            solid
                .tetmesh
                .attrib_iter::<RefShapeMtxInvType, CellIndex>(REFERENCE_SHAPE_MATRIX_INV_ATTRIB,)
                .unwrap(),
            solid.tetmesh.tet_iter()
        )
            .map(|(lambda, mu, &vol, &ref_shape_mtx_inv, tet)| {
                let shape_mtx = Matrix3::new(tet.shape_matrix()).transpose();
                NeoHookeanTetEnergy::new(shape_mtx, ref_shape_mtx_inv, vol, lambda, mu)
                    .elastic_energy_gradient()
            });

        for (grad, cell) in grad_iter.zip(solid.tetmesh.cells().iter()) {
            for j in 0..4 {
                let f = Vector3::new(forces[cell[j]]);
                forces[cell[j]] = (f - grad[j]).into();
            }
        }

        // Reinsert forces back into the attrib map
        solid
            .tetmesh
            .insert_attrib::<VertexIndex>(ELASTIC_FORCE_ATTRIB, forces_attrib)
            .unwrap();
    }

    pub fn contact_potential(&self) -> VertexData<Vec<f64>> {
        let NonLinearProblem {
            object_data,
            frictional_contacts,
            ..
        } = self;

        let mut pot = object_data.build_vertex_data();

        let mut workspace_pot = Vec::new();

        for fc in frictional_contacts.iter() {
            let obj_x0 = object_data.prev_pos(fc.object_index);
            let coll_x0 = object_data.prev_pos(fc.collider_index);

            let active_constraint_indices = fc.constraint.borrow().active_constraint_indices();
            workspace_pot.clear();
            workspace_pot.resize(active_constraint_indices.len(), 0.0);

            fc.constraint.borrow_mut().constraint(
                [obj_x0.view(), coll_x0.view()],
                [obj_x0.view(), coll_x0.view()],
                workspace_pot.as_mut_slice(),
            );

            let mut pot_view_mut =
                object_data.mesh_vertex_subset_mut(pot.view_mut(), None, fc.collider_index);
            for (&aci, &pot) in active_constraint_indices.iter().zip(workspace_pot.iter()) {
                pot_view_mut[aci] = pot;
            }
        }

        pot
    }

    /// Return the stacked friction impulses: one for each vertex.
    pub fn friction_impulse(&self) -> VertexData3<Vec<f64>> {
        let NonLinearProblem {
            object_data,
            frictional_contacts,
            ..
        } = self;

        // Create a chunked collection for the output. This essentially combines
        // the structure in `pos`, which involved vertices that are not degrees
        // of freedom and `prev_x` which involves vertices that ARE degrees of
        // freedom.
        let mut impulse = object_data.build_vertex_data3();

        let mut obj_imp = Vec::new();
        let mut coll_imp = Vec::new();

        for fc in frictional_contacts.iter() {
            obj_imp.clear();
            obj_imp.resize(
                object_data.mesh_surface_vertex_count(fc.object_index),
                [0.0; 3],
            );

            coll_imp.clear();
            coll_imp.resize(
                object_data.mesh_surface_vertex_count(fc.collider_index),
                [0.0; 3],
            );

            let mut obj_imp = Chunked3::from_array_slice_mut(obj_imp.as_mut_slice());
            let mut coll_imp = Chunked3::from_array_slice_mut(coll_imp.as_mut_slice());

            fc.constraint
                .borrow()
                .add_friction_impulse([obj_imp.view_mut().into(), coll_imp.view_mut().into()], 1.0);

            let mut imp =
                object_data.mesh_vertex_subset_mut(impulse.view_mut(), None, fc.object_index);
            for (imp, obj_imp) in imp.iter_mut().zip(obj_imp.iter()) {
                *imp = *obj_imp;
            }

            let mut imp =
                object_data.mesh_vertex_subset_mut(impulse.view_mut(), None, fc.collider_index);
            for (imp, coll_imp) in imp.iter_mut().zip(coll_imp.iter()) {
                *imp = *coll_imp;
            }
        }
        impulse
    }

    /// Return the stacked contact impulses: one for each vertex.
    pub fn contact_impulse(&self) -> Chunked<Chunked<Chunked3<Vec<f64>>>> {
        let multiplier_impulse_scale = self.time_step() / self.impulse_inv_scale();
        let NonLinearProblem {
            object_data,
            frictional_contacts,
            volume_constraints,
            warm_start,
            ..
        } = self;

        let mut impulse = object_data.build_vertex_data3();

        let mut offset = volume_constraints.len();

        let mut obj_imp = Vec::new();
        let mut coll_imp = Vec::new();

        for fc in frictional_contacts.iter() {
            // Get contact force from the warm start.
            let n = fc.constraint.borrow().constraint_size();
            let contact_impulse = Self::contact_impulse_magnitudes(
                &warm_start.constraint_multipliers[offset..offset + n],
                multiplier_impulse_scale,
            );

            offset += n;

            obj_imp.clear();
            obj_imp.resize(
                object_data.mesh_surface_vertex_count(fc.object_index),
                [0.0; 3],
            );

            coll_imp.clear();
            coll_imp.resize(
                object_data.mesh_surface_vertex_count(fc.collider_index),
                [0.0; 3],
            );

            let mut obj_imp = Chunked3::from_array_slice_mut(obj_imp.as_mut_slice());
            let mut coll_imp = Chunked3::from_array_slice_mut(coll_imp.as_mut_slice());

            let obj_x0 = object_data.prev_pos(fc.object_index);
            let coll_x0 = object_data.prev_pos(fc.collider_index);

            fc.constraint.borrow_mut().add_contact_impulse(
                [obj_x0.view(), coll_x0.view()],
                &contact_impulse,
                [obj_imp.view_mut(), coll_imp.view_mut()],
            );

            let obj_imp_view = obj_imp.view();
            let coll_imp_view = coll_imp.view();

            let mut imp =
                object_data.mesh_vertex_subset_mut(impulse.view_mut(), None, fc.object_index);
            for (imp, obj_imp) in imp.iter_mut().zip(obj_imp_view.iter()) {
                *imp = *obj_imp;
            }

            let mut imp =
                object_data.mesh_vertex_subset_mut(impulse.view_mut(), None, fc.collider_index);
            for (imp, coll_imp) in imp.iter_mut().zip(coll_imp_view.iter()) {
                *imp = *coll_imp;
            }
        }
        impulse
    }

    /// Update the solid and shell meshes with relevant simulation data.
    pub fn update_mesh_data(&mut self) {
        let contact_impulse = self.contact_impulse();
        let friction_impulse = self.friction_impulse();
        let potential = self.contact_potential();
        for (idx, solid) in self.object_data.solids.iter_mut().enumerate() {
            // Write back friction and contact impulses
            solid
                .tetmesh
                .set_attrib_data::<FrictionImpulseType, VertexIndex>(
                    FRICTION_ATTRIB,
                    friction_impulse.view().at(0).at(idx).view().into(),
                )
                .ok();
            solid
                .tetmesh
                .set_attrib_data::<ContactImpulseType, VertexIndex>(
                    CONTACT_ATTRIB,
                    contact_impulse.view().at(0).at(idx).view().into(),
                )
                .ok();
            solid
                .tetmesh
                .set_attrib_data::<PotentialType, VertexIndex>(
                    POTENTIAL_ATTRIB,
                    potential.view().at(0).at(idx).view().into(),
                )
                .ok();

            // Write back elastic strain energy for visualization.
            Self::compute_strain_energy_attrib(solid);

            // Write back elastic forces on each node.
            Self::compute_elastic_forces_attrib(solid);
        }
        // TODO: Do the same for shells
    }

    /*
     * The following functions are there for debugging jacobians and hessians
     */

    #[allow(dead_code)]
    pub fn write_jacobian_img(&self, jac: &na::DMatrix<f64>) {
        use image::ImageBuffer;

        let nrows = jac.nrows();
        let ncols = jac.ncols();

        let ciel = 1.0; //jac.max();
        let floor = -1.0; //jac.min();

        let img = ImageBuffer::from_fn(ncols as u32, nrows as u32, |c, r| {
            let val = jac[(r as usize, c as usize)];
            let color = if val > 0.0 {
                [255, (255.0 * val / ciel) as u8, 0]
            } else if val < 0.0 {
                [0, (255.0 * (1.0 + val / floor)) as u8, 255]
            } else {
                [255, 0, 255]
            };
            image::Rgb(color)
        });

        img.save(format!("./out/jac_{}.png", self.iter_counter.borrow()))
            .expect("Failed to save Jacobian Image");
    }

    #[allow(dead_code)]
    pub fn print_jacobian_svd(&self, values: &[Number]) {
        use ipopt::{BasicProblem, ConstrainedProblem};
        use na::{base::storage::Storage, DMatrix};

        if values.is_empty() {
            return;
        }

        let mut rows = vec![0; values.len()];
        let mut cols = vec![0; values.len()];
        assert!(self.constraint_jacobian_indices(&mut rows, &mut cols));

        let nrows = self.num_constraints();
        let ncols = self.num_variables();
        let mut jac = DMatrix::<f64>::zeros(nrows, ncols);
        for ((&row, &col), &v) in rows.iter().zip(cols.iter()).zip(values.iter()) {
            jac[(row as usize, col as usize)] += v;
        }

        self.write_jacobian_img(&jac);

        use std::io::Write;

        let mut f =
            std::fs::File::create(format!("./out/jac_{}.txt", self.iter_counter.borrow())).unwrap();
        writeln!(&mut f, "jac = ").ok();
        for r in 0..nrows {
            for c in 0..ncols {
                if jac[(r, c)] != 0.0 {
                    write!(&mut f, "{:9.5}", jac[(r, c)]).ok();
                } else {
                    write!(&mut f, "    .    ",).ok();
                }
            }
            writeln!(&mut f).ok();
        }
        writeln!(&mut f).ok();

        let svd = na::SVD::new(jac, false, false);
        let s: &[Number] = svd.singular_values.data.as_slice();
        let cond = s.iter().max_by(|x, y| x.partial_cmp(y).unwrap()).unwrap()
            / s.iter().min_by(|x, y| x.partial_cmp(y).unwrap()).unwrap();
        debug!("Condition number of jacobian is: {}", cond);
    }

    /*
    #[allow(dead_code)]
    fn output_mesh(&self, x: &[Number], dx: &[Number], name: &str) -> Result<(), crate::Error> {
        let mut iter_counter = self.iter_counter.borrow_mut();
        let mut mesh = self.tetmesh.borrow().clone();
        let all_displacements: &[Vector3<f64>] = reinterpret_slice(dx);
        let all_positions: &[Vector3<f64>] = reinterpret_slice(x);
        mesh.vertex_positions_mut()
            .iter_mut()
            .zip(all_displacements.iter())
            .zip(all_positions.iter())
            .for_each(|((mp, d), p)| *mp = (*p + *d).into());
        *iter_counter += 1;
        geo::io::save_tetmesh(
            &mesh,
            &std::path::PathBuf::from(format!("./out/{}_{}.vtk", name, *iter_counter)),
        )?;

        if let Some(tri_mesh_ref) = self.kinematic_object.as_ref() {
            let obj = geo::mesh::PolyMesh::from(tri_mesh_ref.borrow().clone());
            geo::io::save_polymesh(
                &obj,
                &std::path::PathBuf::from(format!("./out/tri_{}_{}.vtk", name, *iter_counter)),
            )?;
        } else {
            return Err(crate::Error::NoKinematicMesh);
        }

        dbg!(*iter_counter);
        Ok(())
    }
    */

    #[allow(dead_code)]
    pub fn print_hessian_svd(&self, values: &[Number]) {
        use ipopt::{BasicProblem, ConstrainedProblem};
        use na::{base::storage::Storage, DMatrix};

        if values.is_empty() {
            return;
        }

        let mut rows = vec![0; values.len()];
        let mut cols = vec![0; values.len()];
        assert!(self.hessian_indices(&mut rows, &mut cols));

        let mut hess = DMatrix::<f64>::zeros(self.num_variables(), self.num_variables());
        for ((&row, &col), &v) in rows.iter().zip(cols.iter()).zip(values.iter()) {
            hess[(row as usize, col as usize)] += v;
        }

        let svd = na::SVD::new(hess, false, false);
        let s: &[Number] = svd.singular_values.data.as_slice();
        let cond_hess = s.iter().max_by(|x, y| x.partial_cmp(y).unwrap()).unwrap()
            / s.iter().min_by(|x, y| x.partial_cmp(y).unwrap()).unwrap();
        debug!("Condition number of hessian is {}", cond_hess);
    }

    /*
     * End of debugging functions
     */
}

/// Prepare the problem for Newton iterations.
impl ipopt::BasicProblem for NonLinearProblem {
    fn num_variables(&self) -> usize {
        self.object_data.prev_v.view().into_flat().len()
    }

    fn bounds(&self, uv_l: &mut [Number], uv_u: &mut [Number]) -> bool {
        // Any value greater than 1e19 in absolute value is considered unbounded (infinity).
        let bound = 2e19;

        // Fixed vertices have a predetermined velocity which is specified in the prev_v variable.
        // Unscale velocities so we can set the unscaled bounds properly.
        let uv_flat_view = self.object_data.prev_v.view().into_flat();
        let unscaled_vel = self
            .object_data
            .update_current_velocity(uv_flat_view, 1.0 / self.variable_scale());
        let solid_prev_uv = unscaled_vel.view().isolate(0);

        uv_l.iter_mut().for_each(|x| *x = -bound);
        uv_u.iter_mut().for_each(|x| *x = bound);

        let x0 = self.object_data.prev_x.view();
        let mut uv_l = Chunked::from_offsets(
            *x0.offsets(),
            Chunked::from_offsets(*x0.data().offsets(), Chunked3::from_flat(uv_l)),
        );
        let mut uv_u = Chunked::from_offsets(
            *x0.offsets(),
            Chunked::from_offsets(*x0.data().offsets(), Chunked3::from_flat(uv_u)),
        );

        for (i, (solid, uv)) in self
            .object_data
            .solids
            .iter()
            .zip(solid_prev_uv.iter())
            .enumerate()
        {
            if let Ok(fixed_verts) = solid
                .tetmesh
                .attrib_as_slice::<FixedIntType, VertexIndex>(FIXED_ATTRIB)
            {
                let mut uv_l = uv_l.view_mut().isolate(0).isolate(i);
                let mut uv_u = uv_u.view_mut().isolate(0).isolate(i);
                // Find and set fixed vertices.
                uv_l.iter_mut()
                    .zip(uv_u.iter_mut())
                    .zip(uv.iter())
                    .zip(fixed_verts.iter())
                    .filter(|&(_, &fixed)| fixed != 0)
                    .for_each(|(((l, u), uv), _)| {
                        *l = *uv;
                        *u = *uv;
                    });
            }
        }

        for (i, shell) in self.object_data.shells.iter().enumerate() {
            if let Ok(fixed_verts) = shell
                .trimesh
                .attrib_as_slice::<FixedIntType, VertexIndex>(FIXED_ATTRIB)
            {
                let mut uv_l = uv_l.view_mut().isolate(1).isolate(i);
                let mut uv_u = uv_u.view_mut().isolate(1).isolate(i);
                // Find and set fixed vertices.
                uv_l.iter_mut()
                    .zip(uv_u.iter_mut())
                    .zip(fixed_verts.iter())
                    .filter(|&(_, &fixed)| fixed != 0)
                    .for_each(|((l, u), _)| {
                        *l = [0.0; 3];
                        *u = [0.0; 3];
                    });
            }
        }
        let uv_l = uv_l.into_flat();
        let uv_u = uv_u.into_flat();
        debug_assert!(uv_l.iter().all(|&x| x.is_finite()) && uv_u.iter().all(|&x| x.is_finite()));
        true
    }

    fn initial_point(&self, x: &mut [Number]) -> bool {
        x.copy_from_slice(self.warm_start.primal_variables.as_slice());

        debug_assert!(x.iter().all(|&x| x.is_finite()));
        true
    }

    fn initial_bounds_multipliers(&self, z_l: &mut [Number], z_u: &mut [Number]) -> bool {
        z_l.copy_from_slice(self.warm_start.lower_bound_multipliers.as_slice());
        z_u.copy_from_slice(self.warm_start.upper_bound_multipliers.as_slice());
        debug_assert!(z_l.iter().all(|&z| z.is_finite()) && z_u.iter().all(|&z| z.is_finite()));
        true
    }

    fn objective(&self, uv: &[Number], obj: &mut Number) -> bool {
        let v = self.update_current_velocity(uv);
        *obj = self.objective_value(v.view()) * self.impulse_inv_scale();
        //debug_assert!(obj.is_finite());
        trace!("Objective value = {}", *obj);
        obj.is_finite()
    }

    fn objective_grad(&self, uv: &[Number], grad_f: &mut [Number]) -> bool {
        trace!(
            "Unscaled variable norm: {}",
            crate::inf_norm(uv.iter().cloned())
        );
        grad_f.iter_mut().for_each(|x| *x = 0.0); // clear gradient vector

        let v = self.update_current_velocity(uv);
        let v = v.view();

        // Copy the chunked structure from our object_data.
        // TODO: Refactor this into a function for Chunked and UniChunked types.
        let mut grad = Chunked::from_offsets(
            *v.offsets(),
            Chunked::from_offsets(*v.data().offsets(), Chunked3::from_flat(grad_f)),
        );

        let cur_pos = self.compute_step(v.view());
        let x1 = cur_pos.view();
        let x0 = self.object_data.prev_x.view();
        let v0 = self.object_data.prev_v.view();

        for (i, solid) in self.object_data.solids.iter().enumerate() {
            let x0 = x0.at(0).at(i).into_flat();
            let x1 = x1.at(0).at(i).into_flat();
            let g = grad.view_mut().isolate(0).isolate(i).into_flat();
            solid.elasticity().add_energy_gradient(x0, x1, g);
            solid.gravity(self.gravity).add_energy_gradient(x0, x1, g);
        }

        for (i, shell) in self.object_data.shells.iter().enumerate() {
            let x0 = x0.at(1).at(i).into_flat();
            let x1 = x1.at(1).at(i).into_flat();
            //let v0 = v0.at(1).at(i).into_flat();
            let g = grad.view_mut().isolate(1).isolate(i).into_flat();
            //shell.elasticity().energy(x0, x1, g);
            shell.gravity(self.gravity).add_energy_gradient(x0, x1, g);
        }

        if !self.is_static() {
            {
                let grad_flat = grad.view_mut().into_flat();
                // This is a correction to transform the above energy derivatives to
                // velocity gradients from position gradients.
                grad_flat.iter_mut().for_each(|g| *g *= self.time_step);
            }

            // Finally add inertia terms
            for (i, solid) in self.object_data.solids.iter().enumerate() {
                let v0 = v0.at(0).at(i).into_flat();
                let v = v.at(0).at(i).into_flat();
                let g = grad.view_mut().isolate(0).isolate(i).into_flat();
                solid.inertia().add_energy_gradient(v0, v, g);
            }

            for fc in self.frictional_contacts.iter() {
                // Since add_fricton_impulse is looking for a valid gradient, this
                // must involve only vertices that can change.
                assert!(match fc.object_index {
                    SourceIndex::Solid(_) => true,
                    SourceIndex::Shell(i) => match self.object_data.shells[i].material.properties {
                        ShellProperties::Fixed => match fc.collider_index {
                            SourceIndex::Solid(_) => true,
                            SourceIndex::Shell(i) => {
                                match self.object_data.shells[i].material.properties {
                                    ShellProperties::Fixed => false,
                                    ShellProperties::Rigid { .. } => false, //unimplemented,
                                    _ => true,
                                }
                            }
                        },
                        ShellProperties::Rigid { .. } => false, // unimplemented
                        _ => true,
                    },
                });

                let [mut obj_g, mut coll_g] = self.object_data.mesh_vertex_subset_split_mut(
                    grad.view_mut(),
                    None,
                    [fc.object_index, fc.collider_index],
                );

                if FORWARD_FRICTION {
                    fc.constraint
                        .borrow()
                        .add_friction_impulse([obj_g.view_mut(), coll_g.view_mut()], -1.0);
                }
            }
        }

        let grad_f = grad.into_flat();

        let scale = self.variable_scale() * self.impulse_inv_scale();
        grad_f.iter_mut().for_each(|g| *g *= scale);
        trace!(
            "Objective gradient norm: {}",
            crate::inf_norm(grad_f.iter().cloned())
        );

        debug_assert!(grad_f.iter().all(|&g| g.is_finite()));
        true
    }
}

impl ipopt::ConstrainedProblem for NonLinearProblem {
    fn num_constraints(&self) -> usize {
        let mut num = 0;
        for (_, vc) in self.volume_constraints.iter() {
            num += vc.borrow().constraint_size();
        }
        for fc in self.frictional_contacts.iter() {
            num += fc.constraint.borrow().constraint_size();
            //println!("num constraints  = {:?}", num);
        }
        num
    }

    fn num_constraint_jacobian_non_zeros(&self) -> usize {
        let mut num = 0;
        for (_, vc) in self.volume_constraints.iter() {
            num += vc.borrow().constraint_jacobian_size();
        }
        for fc in self.frictional_contacts.iter() {
            num += fc.constraint.borrow().constraint_jacobian_size();
        }
        num
    }

    fn initial_constraint_multipliers(&self, lambda: &mut [Number]) -> bool {
        // TODO: Move this is_empty logic to remapping contacts: The important observation here is
        // that contact set != constraint set because some methods don't enforce contacts via a
        // constraint.
        if !lambda.is_empty() {
            // The constrained points may change between updating the warm start and using it here.
            if lambda.len() != self.warm_start.constraint_multipliers.len() {
                // This is sometimes not caught for some reason, so we ouput an explicit error.
                error!(
                    "Number of multipliers ({}) does not match warm start ({})",
                    lambda.len(),
                    self.warm_start.constraint_multipliers.len()
                );
                assert_eq!(lambda.len(), self.warm_start.constraint_multipliers.len());
            }
            lambda.copy_from_slice(self.warm_start.constraint_multipliers.as_slice());
        }

        debug_assert!(lambda.iter().all(|l| l.is_finite()));
        true
    }

    fn constraint(&self, uv: &[Number], g: &mut [Number]) -> bool {
        let cur_v = self.update_current_velocity(uv);
        let cur_x = self.compute_step(cur_v.view());
        let x = cur_x.view();
        let x0 = self.object_data.prev_x.view();

        //self.output_mesh(x, dx, "mesh").unwrap_or_else(|err| println!("WARNING: failed to output mesh: {:?}", err));

        let mut count = 0; // Constraint counter

        for (solid_idx, vc) in self.volume_constraints.iter() {
            let n = vc.borrow().constraint_size();
            vc.borrow_mut().constraint(
                x0.at(0).at(*solid_idx).into_flat(),
                x.at(0).at(*solid_idx).into_flat(),
                &mut g[count..count + n],
            );

            let scale = self.volume_constraint_scale();
            g[count..count + n].iter_mut().for_each(|x| *x *= scale);

            count += n;
        }

        for fc in self.frictional_contacts.iter() {
            let n = fc.constraint.borrow().constraint_size();
            let obj_x0 = self.object_data.prev_pos(fc.object_index);
            let coll_x0 = self.object_data.prev_pos(fc.collider_index);
            let obj_x = self.object_data.cur_pos(x, fc.object_index);
            let coll_x = self.object_data.cur_pos(x, fc.collider_index);

            fc.constraint.borrow_mut().constraint(
                [obj_x0.view(), coll_x0.view()],
                [obj_x.view(), coll_x.view()],
                &mut g[count..count + n],
            );

            let scale = self.contact_constraint_scale();
            g[count..count + n].iter_mut().for_each(|x| *x *= scale);

            count += n;
        }

        assert_eq!(count, g.len());

        debug_assert!(g.iter().all(|g| g.is_finite()));
        true
    }

    fn constraint_bounds(&self, g_l: &mut [Number], g_u: &mut [Number]) -> bool {
        let mut count = 0; // Constraint counter
        for (_, vc) in self.volume_constraints.iter() {
            let mut bounds = vc.borrow().constraint_bounds();
            let n = vc.borrow().constraint_size();
            g_l[count..count + n].swap_with_slice(&mut bounds.0);
            g_u[count..count + n].swap_with_slice(&mut bounds.1);
            // In theory we need to divide constraint bounds by time step to be consistent with the
            // units of the constraint. However all constraint bounds are either zero or infinite,
            // so this is not actually necessary here.
            //let scale = self.volume_constraint_scale();
            //g_l[count..count + n].iter_mut().for_each(|x| *x *= scale);
            //g_u[count..count + n].iter_mut().for_each(|x| *x *= scale);
            count += n;
        }

        for fc in self.frictional_contacts.iter() {
            let mut bounds = fc.constraint.borrow().constraint_bounds();
            let n = fc.constraint.borrow().constraint_size();
            g_l[count..count + n].swap_with_slice(&mut bounds.0);
            g_u[count..count + n].swap_with_slice(&mut bounds.1);

            // In theory we need to divide constraint bounds by time step to be consistent with the
            // units of the constraint. However all constraint bounds are either zero or infinite,
            // so this is not actually necessary here.
            //let scale = self.contact_constraint_scale();
            //g_l[count..count + n].iter_mut().for_each(|x| *x *= scale);
            //g_u[count..count + n].iter_mut().for_each(|x| *x *= scale);

            count += n;
        }

        assert_eq!(count, g_l.len());
        assert_eq!(count, g_u.len());
        debug_assert!(g_l.iter().all(|x| x.is_finite()) && g_u.iter().all(|x| x.is_finite()));
        true
    }

    fn constraint_jacobian_indices(
        &self,
        rows: &mut [ipopt::Index],
        cols: &mut [ipopt::Index],
    ) -> bool {
        // This is used for counting offsets.
        let prev_v_solid = self.object_data.prev_v.view().at(0);

        let mut count = 0; // Constraint counter

        let mut row_offset = 0;
        for (solid_idx, vc) in self.volume_constraints.iter() {
            let vc = vc.borrow();
            let iter = vc.constraint_jacobian_indices_iter().unwrap();
            let col_offset = prev_v_solid.offset_value(*solid_idx) * 3;
            for MatrixElementIndex { row, col } in iter {
                rows[count] = (row + row_offset) as ipopt::Index;
                cols[count] = (col + col_offset) as ipopt::Index;
                count += 1;
            }
            row_offset += 1;
        }

        for fc in self.frictional_contacts.iter() {
            //use ipopt::BasicProblem;
            let nrows = fc.constraint.borrow().constraint_size();
            //let ncols = self.num_variables();
            //let mut jac = vec![vec![0; nrows]; ncols]; // col major

            let constraint = fc.constraint.borrow();
            let iter = constraint.constraint_jacobian_indices_iter().unwrap();
            for MatrixElementIndex { row, col } in iter {
                rows[count] = (row + row_offset) as ipopt::Index;
                cols[count] =
                    self.object_data
                        .source_coordinates(fc.object_index, fc.collider_index, col)
                        as ipopt::Index;
                //jac[col][row] = 1;
                count += 1;
            }
            row_offset += nrows;

            //println!("jac = ");
            //for r in 0..nrows {
            //    for c in 0..ncols {
            //        if jac[c][r] == 1 {
            //            print!(".");
            //        } else {
            //            print!(" ");
            //        }
            //    }
            //    println!("");
            //}
            //println!("");
        }

        assert_eq!(count, rows.len());
        assert_eq!(count, cols.len());

        true
    }

    fn constraint_jacobian_values(&self, uv: &[Number], vals: &mut [Number]) -> bool {
        let v = self.update_current_velocity(uv);
        let cur_pos = self.compute_step(v.view());
        let x = cur_pos.view();
        let x0 = self.object_data.prev_x.view();

        let mut count = 0; // Constraint counter

        for (solid_idx, vc) in self.volume_constraints.iter() {
            let n = vc.borrow().constraint_jacobian_size();
            vc.borrow_mut()
                .constraint_jacobian_values(
                    x0.at(0).at(*solid_idx).into_flat(),
                    x.at(0).at(*solid_idx).into_flat(),
                    &mut vals[count..count + n],
                )
                .ok();
            let scale = self.volume_constraint_scale();
            vals[count..count + n].iter_mut().for_each(|v| *v *= scale);
            count += n;
        }

        for fc in self.frictional_contacts.iter() {
            let n = fc.constraint.borrow().constraint_jacobian_size();
            let obj_x0 = self.object_data.prev_pos(fc.object_index);
            let coll_x0 = self.object_data.prev_pos(fc.collider_index);
            let obj_x = self.object_data.cur_pos(x, fc.object_index);
            let coll_x = self.object_data.cur_pos(x, fc.collider_index);

            fc.constraint
                .borrow_mut()
                .constraint_jacobian_values(
                    [obj_x0.view(), coll_x0.view()],
                    [obj_x.view(), coll_x.view()],
                    &mut vals[count..count + n],
                )
                .ok();
            //println!("jac g vals = {:?}", &vals[count..count+n]);
            let scale = self.contact_constraint_scale();
            vals[count..count + n].iter_mut().for_each(|v| *v *= scale);
            count += n;
        }
        assert_eq!(count, vals.len());
        let scale = self.time_step() * self.variable_scale();
        vals.iter_mut().for_each(|v| *v *= scale);

        //self.output_mesh(x, dx, "mesh").unwrap_or_else(|err| println!("WARNING: failed to output mesh: {:?}", err));
        //self.print_jacobian_svd(vals);

        debug_assert!(vals.iter().all(|x| x.is_finite()));
        true
    }

    fn num_hessian_non_zeros(&self) -> usize {
        let mut num = 0;
        for solid in self.object_data.solids.iter() {
            num += solid.elasticity().energy_hessian_size()
                + if !self.is_static() {
                    solid.inertia().energy_hessian_size()
                } else {
                    0
                };
        }
        //for shell in self.shells.iter() {
        //    num += shell.inertia().energy_hessian_size();
        //}
        for (_, vc) in self.volume_constraints.iter() {
            num += vc.borrow().constraint_hessian_size();
        }
        for fc in self.frictional_contacts.iter() {
            num += fc.constraint.borrow().constraint_hessian_size();
        }
        num
    }

    fn hessian_indices(&self, rows: &mut [ipopt::Index], cols: &mut [ipopt::Index]) -> bool {
        // This is used for counting offsets.
        let prev_v_solid = self.object_data.prev_v.view().at(0);

        let mut count = 0; // Constraint counter

        // Add energy indices
        for (solid_idx, solid) in self.object_data.solids.iter().enumerate() {
            let offset = prev_v_solid.offset_value(solid_idx) * 3;
            let elasticity = solid.elasticity();
            let n = elasticity.energy_hessian_size();
            elasticity.energy_hessian_rows_cols_offset(
                (offset, offset).into(),
                &mut rows[count..count + n],
                &mut cols[count..count + n],
            );

            count += n;

            if !self.is_static() {
                let inertia = solid.inertia();
                let n = inertia.energy_hessian_size();
                inertia.energy_hessian_rows_cols_offset(
                    (offset, offset).into(),
                    &mut rows[count..count + n],
                    &mut cols[count..count + n],
                );
                count += n;
            }
        }

        //for shell in self.shells.iter() {
        //    let inertia = shell.inertia();
        //    let n = inertia.energy_hessian_size();
        //    inertia.energy_hessian_rows_cols(&mut rows[count..count + n], &mut cols[count..count + n]);
        //    count += n;
        //}

        // Add volume constraint indices
        for (solid_idx, vc) in self.volume_constraints.iter() {
            let offset = prev_v_solid.offset_value(*solid_idx) * 3;
            for MatrixElementIndex { row, col } in vc.borrow().constraint_hessian_indices_iter() {
                rows[count] = (row + offset) as ipopt::Index;
                cols[count] = (col + offset) as ipopt::Index;
                count += 1;
            }
        }

        for fc in self.frictional_contacts.iter() {
            let constraint = fc.constraint.borrow();
            let iter = constraint.constraint_hessian_indices_iter().unwrap();
            for MatrixElementIndex { row, col } in iter {
                rows[count] =
                    self.object_data
                        .source_coordinates(fc.object_index, fc.collider_index, row)
                        as ipopt::Index;
                cols[count] =
                    self.object_data
                        .source_coordinates(fc.object_index, fc.collider_index, col)
                        as ipopt::Index;
                count += 1;
            }
        }
        assert_eq!(count, rows.len());
        assert_eq!(count, cols.len());

        true
    }
    fn hessian_values(
        &self,
        uv: &[Number],
        obj_factor: Number,
        lambda: &[Number],
        vals: &mut [Number],
    ) -> bool {
        let cur_vel = self.update_current_velocity(uv);
        let v = cur_vel.view();
        let cur_pos = self.compute_step(v.view());
        let x1 = cur_pos.view();
        let x0 = self.object_data.prev_x.view();
        let v0 = self.object_data.prev_v.view();

        if cfg!(debug_assertions) {
            // Initialize vals in debug builds.
            for v in vals.iter_mut() {
                *v = 0.0;
            }
        }

        // Correction to make the above hessian wrt velocity instead of displacement.
        let dt = self.time_step();

        let mut count = 0; // Constraint counter

        for (solid_idx, solid) in self.object_data.solids.iter().enumerate() {
            let x0 = x0.at(0).at(solid_idx).into_flat();
            let x1 = x1.at(0).at(solid_idx).into_flat();
            let v0 = v0.at(0).at(solid_idx).into_flat();
            let v = v.at(0).at(solid_idx).into_flat();
            let elasticity = solid.elasticity();
            let n = elasticity.energy_hessian_size();
            elasticity.energy_hessian_values(x0, x1, dt * dt, &mut vals[count..count + n]);
            count += n;

            if !self.is_static() {
                let inertia = solid.inertia();
                let n = inertia.energy_hessian_size();
                inertia.energy_hessian_values(v0, v, 1.0, &mut vals[count..count + n]);
                count += n;
            }
        }

        //for (shell_idx, shell) in self.shells.iter().enumerate() {
        //    let v0 = v0.at(1).at(shell_idx).into_flat();
        //    let v = v.at(1).at(shell_idx).into_flat();
        //    let inertia = shell.inertia();
        //    let n = inertia.energy_hessian_size();
        //    inertia.energy_hessian_values(v0, v, 1.0, &mut vals[count..count + n]);
        //    count += n;
        //}

        // Multiply energy hessian by objective factor.
        let factor =
            obj_factor * self.variable_scale() * self.variable_scale() * self.impulse_inv_scale();
        for v in vals[0..count].iter_mut() {
            *v *= factor;
        }

        let mut coff = 0;

        let c_scale = self.variable_scale() * self.variable_scale() * dt * dt;

        for (solid_idx, vc) in self.volume_constraints.iter() {
            let x0 = x0.at(0).at(*solid_idx).into_flat();
            let x1 = x1.at(0).at(*solid_idx).into_flat();
            let nc = vc.borrow().constraint_size();
            let nh = vc.borrow().constraint_hessian_size();
            vc.borrow_mut()
                .constraint_hessian_values(
                    x0,
                    x1,
                    &lambda[coff..coff + nc],
                    c_scale * self.volume_constraint_scale(),
                    &mut vals[count..count + nh],
                )
                .unwrap();

            count += nh;
            coff += nc;
        }

        for fc in self.frictional_contacts.iter() {
            let obj_x0 = self.object_data.prev_pos(fc.object_index);
            let coll_x0 = self.object_data.prev_pos(fc.collider_index);
            let obj_x = self.object_data.cur_pos(x1, fc.object_index);
            let coll_x = self.object_data.cur_pos(x1, fc.collider_index);
            let nc = fc.constraint.borrow().constraint_size();
            let nh = fc.constraint.borrow().constraint_hessian_size();
            fc.constraint
                .borrow_mut()
                .constraint_hessian_values(
                    [obj_x0.view(), coll_x0.view()],
                    [obj_x.view(), coll_x.view()],
                    &lambda[coff..coff + nc],
                    c_scale * self.contact_constraint_scale(),
                    &mut vals[count..count + nh],
                )
                .unwrap();

            count += nh;
            coff += nc;
        }

        assert_eq!(count, vals.len());
        assert_eq!(coff, lambda.len());
        //self.print_hessian_svd(vals);

        debug_assert!(vals.iter().all(|x| x.is_finite()));
        true
    }
}
