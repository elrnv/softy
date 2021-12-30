use super::point_contact::{enumerate_nonempty_neighborhoods_inplace, PointContactConstraint};
use crate::constraints::ContactSurface;
use crate::{CheckedIndex, Error, FrictionParams, Real, TriMesh, ORIGINAL_VERTEX_INDEX_ATTRIB};
use autodiff as ad;
use flatk::{Chunked3, SubsetView, View};
use geo::attrib::Attrib;
use geo::mesh::VertexMesh;
use geo::{NumVertices, VertexIndex};
use implicits::KernelType;
use rayon::iter::Either;
use rayon::prelude::*;
use tensr::{AsData, Scalar};

/// Same as `PointContactConstraint` but this constraint keeps track of where each vertex maps within
/// the global array
///
#[derive(Clone, Debug)]
pub struct IndexedPointContactConstraint<T = f64>
where
    T: Scalar,
{
    pub constraint: PointContactConstraint<T>,
    /// Indices of original vertices for the implicit surface.
    pub implicit_surface_vertex_indices: Vec<usize>,
    /// Indices of original vertices for the collider.
    pub collider_vertex_indices: Vec<usize>,
}

impl<T: Real> IndexedPointContactConstraint<T> {
    pub fn clone_cast<S: Real>(&self) -> IndexedPointContactConstraint<S> {
        IndexedPointContactConstraint {
            constraint: self.constraint.clone_cast(),
            implicit_surface_vertex_indices: self.implicit_surface_vertex_indices.clone(),
            collider_vertex_indices: self.collider_vertex_indices.clone(),
        }
    }

    pub fn new<VP: VertexMesh<f64>>(
        // Main object experiencing contact against its implicit surface representation.
        object: ContactSurface<&TriMesh, f64>,
        // Collision object consisting of points pushing against the solid object.
        collider: ContactSurface<&VP, f64>,
        kernel: KernelType,
        friction_params: Option<FrictionParams>,
        contact_offset: f64,
        linearized: bool,
    ) -> Result<Self, Error> {
        let implicit_surface_vertex_indices = object
            .mesh
            .attrib_clone_into_vec::<usize, VertexIndex>(ORIGINAL_VERTEX_INDEX_ATTRIB)
            .unwrap_or_else(|_| (0..object.mesh.num_vertices()).collect::<Vec<_>>());
        let collider_vertex_indices = collider
            .mesh
            .attrib_clone_into_vec::<usize, VertexIndex>(ORIGINAL_VERTEX_INDEX_ATTRIB)
            .unwrap_or_else(|_| (0..collider.mesh.num_vertices()).collect::<Vec<_>>());

        let constraint = PointContactConstraint::new(
            object,
            collider,
            kernel,
            friction_params,
            contact_offset,
            linearized,
        )?;

        Ok(IndexedPointContactConstraint {
            constraint,
            implicit_surface_vertex_indices,
            collider_vertex_indices,
        })
    }

    /// Constructs a clone of this constraint with autodiff variables.
    pub fn clone_as_autodiff<S: Real>(&self) -> IndexedPointContactConstraint<ad::FT<S>> {
        self.clone_cast::<ad::FT<S>>()
    }

    // Same as `update_collider_vertex_positions` but without knowledge about original vertex indices.
    pub fn update_surface_vertex_positions(&mut self, x: Chunked3<&[T]>) -> usize {
        let x = SubsetView::from_unique_ordered_indices(&self.implicit_surface_vertex_indices, x);
        self.constraint.update_surface_with_mesh_pos(x)
    }

    // Same as `update_collider_vertex_positions` but without knowledge about original vertex indices.
    pub fn update_collider_vertex_positions(&mut self, x: Chunked3<&[T]>) {
        let x = SubsetView::from_unique_ordered_indices(&self.collider_vertex_indices, x);
        self.constraint.update_collider_vertex_positions(x);
    }

    /// Prune contacts with zero contact_impulse and contacts without neighboring samples.
    /// This function outputs the indices of contacts as well as a pruned vector of impulses.
    pub fn in_contact_indices(
        &self,
        contact_impulse: &[T],
        potential: &[T],
    ) -> (Vec<usize>, Vec<usize>, Vec<T>) {
        self.constraint
            .in_contact_indices(contact_impulse, potential)
    }

    pub(crate) fn constraint_size(&self) -> usize {
        self.constraint.constraint_size()
    }

    // Helper function to construct subsets from x using internal intdices.
    pub(crate) fn input_and_constraint<'a>(
        &mut self,
        x: Chunked3<&'a [T]>,
    ) -> (
        [SubsetView<Chunked3<&'a [T]>>; 2],
        &mut PointContactConstraint<T>,
    ) {
        let x0 = SubsetView::from_unique_ordered_indices(&self.implicit_surface_vertex_indices, x);
        let x1 = SubsetView::from_unique_ordered_indices(&self.collider_vertex_indices, x);
        ([x0, x1], &mut self.constraint)
    }

    pub(crate) fn constraint_bounds(&self) -> (Vec<f64>, Vec<f64>) {
        self.constraint.constraint_bounds()
    }

    pub(crate) fn constraint<'a>(&mut self, x: Chunked3<&'a [T]>, value: &mut [T]) {
        let ([x0, x1], constraint) = self.input_and_constraint(x);
        constraint.constraint([x0, x1], value);
    }

    pub(crate) fn object_constraint_jacobian_blocks_iter<'a>(
        &'a self,
    ) -> impl Iterator<Item = (usize, usize, [T; 3])> + 'a {
        self.constraint
            .object_constraint_jacobian_blocks_iter()
            .map(move |(row, col, block)| (row, self.implicit_surface_vertex_indices[col], block))
    }

    pub(crate) fn collider_constraint_jacobian_blocks_iter<'a>(
        &'a self,
    ) -> impl Iterator<Item = (usize, usize, [T; 3])> + 'a {
        self.constraint
            .collider_constraint_jacobian_blocks_iter()
            .map(move |(row, col, block)| (row, self.collider_vertex_indices[col], block))
    }

    //TODO: reuse code from PointContactConstraint
    // Currently because constraint_jacobian is lazycell we cannot just call the method.
    pub(crate) fn object_constraint_jacobian_blocks_par_iter<'a>(
        &'a self,
    ) -> impl ParallelIterator<Item = (usize, usize, [T; 3])> + 'a {
        if let Some(jac) = self.constraint.constraint_jacobian.borrow() {
            Either::Left(
                jac[0]
                    .as_ref()
                    .map(|jac| {
                        jac.view()
                            .as_data()
                            .into_par_iter()
                            .enumerate()
                            .flat_map_iter(move |(row_idx, row)| {
                                row.into_iter().map(move |(block_col_idx, block)| {
                                    (row_idx, block_col_idx, block.into_arrays()[0])
                                })
                            })
                    })
                    .into_par_iter()
                    .flatten(),
            )
        } else {
            let surf = &self.constraint.implicit_surface;
            let iter = surf.surface_jacobian_indexed_block_par_iter(
                self.constraint.collider_vertex_positions.view().into(),
            );
            let neighborhood_indices = enumerate_nonempty_neighborhoods_inplace(surf);
            Either::Right(
                if self.constraint.object_is_fixed() {
                    None
                } else {
                    Some(iter)
                }
                .into_par_iter()
                .flatten()
                .map(move |(row, col, block)| {
                    assert!(neighborhood_indices[row].is_valid());
                    (neighborhood_indices[row].unwrap(), col, block)
                }),
            )
        }
    }

    //TODO: reuse code from PointContactConstraint
    // Currently because constraint_jacobian is lazycell we cannot just call the method.
    pub(crate) fn collider_constraint_jacobian_blocks_par_iter<'a>(
        &'a self,
    ) -> impl ParallelIterator<Item = (usize, usize, [T; 3])> + 'a {
        if let Some(jac) = self.constraint.constraint_jacobian.borrow() {
            Either::Left(
                jac[1]
                    .as_ref()
                    .map(|jac| {
                        jac.view()
                            .as_data()
                            .into_par_iter()
                            .enumerate()
                            .flat_map_iter(move |(row_idx, row)| {
                                row.into_iter().map(move |(block_col_idx, block)| {
                                    (row_idx, block_col_idx, block.into_arrays()[0])
                                })
                            })
                    })
                    .into_par_iter()
                    .flatten(),
            )
        } else {
            let surf = &self.constraint.implicit_surface;
            let iter = surf.query_jacobian_indexed_block_par_iter(
                self.constraint.collider_vertex_positions.view().into(),
            );
            let neighborhood_indices = enumerate_nonempty_neighborhoods_inplace(surf);
            Either::Right(
                if self.constraint.collider_is_fixed() {
                    None
                } else {
                    Some(iter)
                }
                .into_par_iter()
                .flatten()
                .map(move |(row, col, block)| {
                    assert!(neighborhood_indices[row].is_valid());
                    (neighborhood_indices[row].unwrap(), col, block)
                }),
            )
        }
    }
}

/// Computes the derivative of a cubic penalty function for contacts multiplied by `-κ`.
///
/// The penalty and its derivative alone:
/// ```verbatim
/// b(x;δ) = -((x-δ)^3)/δ if x < δ and 0 otherwise
/// db(x;δ) = -(3/δ)(x-δ)^2 if x < δ and 0 otherwise
/// ```
pub fn compute_contact_force_magnitude<S: Real>(
    // Input distance & Output force magnitude
    lambda: &mut [S],
    delta: f32,
    kappa: f32,
) {
    lambda.iter_mut().for_each(|lambda| {
        let d = *lambda;
        *lambda = if d.to_f32().unwrap() >= delta {
            S::zero()
        } else {
            let _2 = S::from(2.0).unwrap();
            let delta = S::from(delta).unwrap();
            let kappa = S::from(kappa).unwrap();
            -kappa * (_2 / delta) * (d - delta)
        }
    });
}
