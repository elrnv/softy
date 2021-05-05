use geo::ops::*;
use num_traits::Zero;
use rayon::prelude::IndexedParallelIterator;
use std::ops::Neg;
use tensr::{AsTensor, Scalar, Vector3};

pub use super::*;

/// A set of data stored on each sample for the implicit surface.
#[derive(Copy, Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Sample<T: Scalar> {
    /// Index of the sample in the original vector or array.
    pub index: usize,
    /// Position of the sample in 3D space.
    pub pos: Vector3<T>,
    /// Normal of the sample.
    pub nml: Vector3<T>,
    /// Velocity at the sample.
    pub vel: Vector3<T>,
    /// Value stored at the sample point.
    pub value: T,
}

/// Convert a sample of one type into a sample of another real type.
impl<T: Scalar> Sample<T> {
    /// Cast a sample into another real type. This function will unwrap internally, so it will
    /// panic if the conversion is invalid.
    pub fn cast<S: Scalar>(self) -> Sample<S> {
        Sample {
            index: self.index,
            pos: self.pos.mapd(|x| S::from(x).unwrap()),
            nml: self.nml.mapd(|x| S::from(x).unwrap()),
            vel: self.vel.mapd(|x| S::from(x).unwrap()),
            value: S::from(self.value).unwrap(),
        }
    }
}

/// Sample points that define the implicit surface including the point positions, normals and
/// values.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Samples<T> {
    /// Sample point positions defining the implicit surface.
    pub positions: Vec<[T; 3]>,
    /// Normals that define the field gradient at every sample point.
    pub normals: Vec<[T; 3]>,
    /// Velocities on the iso-surface at every sample point.
    pub velocities: Vec<Vector3<T>>,
    /// Field values at the interpolating points. These are the values to
    /// match by interpolating implicit surfaces. This means that, for example, the zero
    /// iso-surface will not necessarily pass through the given points.
    pub values: Vec<T>,
}

impl<T: Scalar> Samples<T> {
    /// Creates a clone of this `Samples` with all reals cast to the given type.
    pub fn clone_cast<S: Scalar>(&self) -> Samples<S> {
        Samples {
            positions: self
                .positions
                .iter()
                .map(|v| v.as_tensor().cast::<S>().into())
                .collect(),
            normals: self
                .normals
                .iter()
                .map(|v| v.as_tensor().cast::<S>().into())
                .collect(),
            velocities: self.velocities.iter().map(|v| v.cast::<S>()).collect(),
            values: self.values.iter().map(|&x| S::from(x).unwrap()).collect(),
        }
    }
}

impl<T: Scalar + Neg<Output = T>> Samples<T> {
    /// Construct samples centered at vertices. The normals are optionally given, or otherwise
    /// computed using an area weighted method.
    pub fn new_vertex_samples<V3>(
        triangles: &[[usize; 3]],
        vertices: &[V3],
        new_normals: Option<&[[T; 3]]>,
        values: Vec<T>,
    ) -> Self
    where
        V3: Into<[T; 3]> + Into<Vector3<T>> + Clone,
    {
        let positions = vec![[T::zero(); 3]; vertices.len()];
        let velocities = vec![Vector3::zero(); vertices.len()];
        let normals = vec![[T::zero(); 3]; vertices.len()];

        let mut samples = Samples {
            positions,
            normals,
            velocities,
            values,
        };

        samples.update_vertex_samples(triangles, vertices, new_normals);
        samples
    }

    /// Update samples centered at vertices. Normals are given optionally, otherwise they will be
    /// automatically computed using the area weighted method.
    pub fn update_vertex_samples<V3>(
        &mut self,
        triangles: &[[usize; 3]],
        new_vertices: &[V3],
        new_normals: Option<&[[T; 3]]>,
    ) where
        V3: Into<[T; 3]> + Into<Vector3<T>> + Clone,
    {
        let Samples {
            ref mut positions,
            ref mut normals,
            ..
        } = self;

        // Update positons
        for (pos, new_pos) in positions.iter_mut().zip(new_vertices.iter()) {
            *pos = new_pos.clone().into();
        }

        if let Some(nmls) = new_normals {
            assert_eq!(nmls.len(), new_vertices.len());
            for (nml, new_nml) in normals.iter_mut().zip(nmls.iter()) {
                *nml = *new_nml;
            }
        } else {
            geo::algo::compute_vertex_area_weighted_normals(new_vertices, triangles, normals);
        }
    }
}

impl<T: Real> Samples<T> {
    /// Utility function only used in tests for creating a dummy set of samples just from a set of
    /// points.
    #[cfg(test)]
    pub(crate) fn new_point_samples(positions: Vec<[T; 3]>) -> Self {
        let n = positions.len();
        Samples {
            positions,
            normals: vec![[T::zero(); 3]; n],
            velocities: vec![Vector3::zero(); n],
            values: vec![T::zero(); n],
        }
    }

    pub fn new_triangle_samples<V3>(
        triangles: &[[usize; 3]],
        vertices: &[V3],
        values: Vec<T>,
    ) -> Self
    where
        V3: Into<[T; 3]> + Into<Vector3<T>> + Clone,
    {
        let positions = vec![[T::zero(); 3]; triangles.len()];
        let normals = vec![[T::zero(); 3]; triangles.len()];
        let velocities = vec![Vector3::zero(); triangles.len()];

        let mut samples = Samples {
            positions,
            normals,
            velocities,
            values,
        };

        samples.update_triangle_samples(triangles, vertices);
        samples
    }

    /// Update samples centered at the centroids of the given triangles.
    pub fn update_triangle_samples<V3>(&mut self, triangles: &[[usize; 3]], vertices: &[V3])
    where
        V3: Into<[T; 3]> + Into<Vector3<T>> + Clone,
    {
        let Samples {
            ref mut positions,
            ref mut normals,
            ref mut velocities,
            ..
        } = self;

        let new_iter = triangles.iter().map(|tri_indices| {
            let tri = Triangle::from_indexed_slice(tri_indices, &vertices);
            let v = Vector3::new(tri[1].into()) - Vector3::new(tri[0].into()); // tangent direction
            (tri.centroid(), tri.area_normal(), v / v.norm())
        });

        for (((pos, nml), vel), (new_pos, new_nml, new_vel)) in (positions
            .iter_mut()
            .zip(normals.iter_mut())
            .zip(velocities.iter_mut()))
        .zip(new_iter)
        {
            *pos = From::<[T; 3]>::from(new_pos);
            *nml = new_nml;
            *vel = new_vel;
        }
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.positions.is_empty()
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.positions.len()
    }

    #[inline]
    pub fn iter<'a>(&'a self) -> impl Iterator<Item = Sample<T>> + Clone + 'a {
        let Samples {
            ref positions,
            ref normals,
            ref velocities,
            ref values,
        } = *self;
        positions
            .iter()
            .zip(normals.iter())
            .zip(velocities.iter())
            .zip(values.iter())
            .enumerate()
            .map(move |(i, (((&pos, &nml), &vel), &value))| Sample {
                index: i,
                pos: pos.into(),
                nml: nml.into(),
                vel,
                value,
            })
    }

    /// Consuming iterator.
    #[allow(clippy::should_implement_trait)] // waiting for impl trait on associated types
    #[inline]
    pub fn into_iter(self) -> impl Iterator<Item = Sample<T>> + Clone {
        let Samples {
            positions,
            normals,
            velocities,
            values,
        } = self;
        positions
            .into_iter()
            .zip(normals.into_iter())
            .zip(velocities.into_iter())
            .zip(values.into_iter())
            .enumerate()
            .map(move |(i, (((pos, nml), vel), value))| Sample {
                index: i,
                pos: pos.into(),
                nml: nml.into(),
                vel,
                value,
            })
    }
}

impl<T: Scalar + Send + Sync> Samples<T> {
    #[inline]
    pub fn par_iter<'a>(&'a self) -> impl IndexedParallelIterator<Item = Sample<T>> + Clone + 'a {
        let Samples {
            ref positions,
            ref normals,
            ref velocities,
            ref values,
        } = *self;
        positions
            .par_iter()
            .zip(normals.par_iter())
            .zip(velocities.par_iter())
            .zip(values.par_iter())
            .enumerate()
            .map(move |(i, (((&pos, &nml), &vel), &value))| Sample {
                index: i,
                pos: pos.into(),
                nml: nml.into(),
                vel,
                value,
            })
    }

    /// Consuming iterator.
    #[inline]
    pub fn into_par_iter(self) -> impl IndexedParallelIterator<Item = Sample<T>> + Clone {
        let Samples {
            positions,
            normals,
            velocities,
            values,
        } = self;
        positions
            .into_par_iter()
            .zip(normals.into_par_iter())
            .zip(velocities.into_par_iter())
            .zip(values.into_par_iter())
            .enumerate()
            .map(move |(i, (((pos, nml), vel), value))| Sample {
                index: i,
                pos: pos.into(),
                nml: nml.into(),
                vel,
                value,
            })
    }
}

/// A view into to the positions, normals and offsets of the sample points. This view need not be
/// contiguous as it often isnt.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct SamplesView<'i, 'd: 'i, T> {
    /// Indices into the sample points.
    indices: &'i [usize],
    /// Sample point positions defining the implicit surface.
    positions: &'d [[T; 3]],
    /// Normals that define the potential field gradient at every sample point.
    normals: &'d [[T; 3]],
    /// Vectors that identify a tangent to the iso-surface at every sample point.
    velocities: &'d [Vector3<T>],
    /// Field values at the interpolating points. These are the values to match by interpolating
    /// implicit surfaces. This means that, for example, the zero iso-surface will not necessarily
    /// pass through the given points.
    values: &'d [T],
}

impl<'i, 'd: 'i, T: Scalar> SamplesView<'i, 'd, T> {
    /// Create a view of samples with a given indices slice into the provided samples.
    #[inline]
    pub fn new(indices: &'i [usize], samples: &'d Samples<T>) -> Self {
        SamplesView {
            indices,
            positions: samples.positions.as_slice(),
            normals: samples.normals.as_slice(),
            velocities: samples.velocities.as_slice(),
            values: samples.values.as_slice(),
        }
    }

    /// Construct a new view from a given view using the same underlying data, but with a new set of
    /// indices (which need not be a subset of indices from this view).
    #[inline]
    pub fn from_view(indices: &'i [usize], view: Self) -> Self {
        SamplesView { indices, ..view }
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.indices.is_empty()
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.indices.len()
    }

    /// Function to get a sample at a specified index within the global arrays (not
    /// relative to the view).
    #[inline]
    pub fn at_index(&self, idx: usize) -> Sample<T> {
        let SamplesView {
            ref positions,
            ref normals,
            ref velocities,
            ref values,
            ..
        } = self;
        Sample {
            index: idx,
            pos: positions[idx].into(),
            nml: normals[idx].into(),
            vel: velocities[idx],
            value: values[idx],
        }
    }

    /// Get the sample at the given index within this view.
    #[inline]
    pub fn get(&self, index: usize) -> Sample<T> {
        self.at_index(self.indices[index])
    }

    #[inline]
    pub fn iter(&'i self) -> impl Iterator<Item = Sample<T>> + 'i {
        self.indices.iter().map(move |&i| self.at_index(i))
    }

    /// Consuming iterator.
    #[allow(clippy::should_implement_trait)] // waiting for impl trait on associated types
    #[inline]
    pub fn into_iter(self) -> impl Iterator<Item = Sample<T>> + 'i {
        self.indices.iter().map(move |&i| self.at_index(i))
    }
    #[inline]
    pub fn all_points(&'d self) -> &'d [[T; 3]] {
        self.positions
    }

    #[inline]
    pub fn all_normals(&'d self) -> &'d [[T; 3]] {
        self.normals
    }

    /// Get all the indices in this view.
    #[inline]
    pub fn indices(self) -> &'i [usize] {
        self.indices
    }
}

impl<'i, 'd: 'i, T: Scalar + Send + Sync> SamplesView<'i, 'd, T> {
    #[inline]
    pub fn par_iter(&'i self) -> impl IndexedParallelIterator<Item = Sample<T>> + 'i {
        self.indices.par_iter().map(move |&i| self.at_index(i))
    }

    /// Consuming parallel iterator.
    #[inline]
    pub fn into_par_iter(self) -> impl IndexedParallelIterator<Item = Sample<T>> + 'i {
        self.indices.into_par_iter().map(move |&i| self.at_index(i))
    }
}
