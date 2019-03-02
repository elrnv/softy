use geo::math::Vector3;
use geo::ops::*;
use geo::Real;
use rayon::prelude::IndexedParallelIterator;

pub use super::*;

/// A set of data stored on each sample for the implicit surface.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Sample<T: Real> {
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
impl<T: Real> Sample<T> {
    /// Cast a sample into another real type. This function will unwrap internally, so it will
    /// panic if the conversion is invalid.
    pub fn cast<S: Real>(self) -> Sample<S> {
        Sample {
            index: self.index,
            pos: self.pos.map(|x| S::from(x).unwrap()),
            nml: self.nml.map(|x| S::from(x).unwrap()),
            vel: self.vel.map(|x| S::from(x).unwrap()),
            value: S::from(self.value).unwrap(),
        }
    }
}

/// Sample points that define the implicit surface including the point positions, normals and
/// values.
#[derive(Clone, Debug, PartialEq)]
pub struct Samples<T: Real> {
    /// Sample point positions defining the implicit surface.
    pub points: Vec<Vector3<T>>,
    /// Normals that define the field gradient at every sample point.
    pub normals: Vec<Vector3<T>>,
    /// Velocities on the iso-surface at every sample point.
    pub velocities: Vec<Vector3<T>>,
    /// Field values at the interpolating points. These are the values to
    /// match by interpolating implicit surfaces. This means that, for example, the zero
    /// iso-surface will not necessarily pass through the given points.
    pub values: Vec<T>,
}

impl<T: Real + Send + Sync> Samples<T> {
    /// Construct samples centered at vertices. The normals are optionally given, or otherwise
    /// computed using an area weighted method.
    pub fn new_vertex_samples<V3>(
        triangles: &[[usize; 3]],
        vertices: &[V3],
        new_normals: Option<&[V3]>,
        values: Vec<T>,
    ) -> Self
    where
        V3: Into<Vector3<T>> + Clone,
    {
        let points = vec![Vector3::zeros(); vertices.len()];
        let velocities = vec![Vector3::zeros(); vertices.len()];
        let normals = vec![Vector3::zeros(); vertices.len()];

        let mut samples = Samples {
            points,
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
        new_normals: Option<&[V3]>,
    ) where
        V3: Into<Vector3<T>> + Clone,
    {
        let Samples {
            ref mut points,
            ref mut normals,
            ..
        } = self;

        // Update positons
        for (pos, new_pos) in points.iter_mut().zip(new_vertices.iter()) {
            *pos = new_pos.clone().into();
        }

        if let Some(nmls) = new_normals {
            assert_eq!(nmls.len(), new_vertices.len());
            for (nml, new_nml) in normals.iter_mut().zip(nmls.iter()) {
                *nml = new_nml.clone().into();
            }
        } else {
            ImplicitSurface::compute_vertex_area_normals(triangles, new_vertices, normals);
        }
    }
}

impl<T: Real> Samples<T> {
    /// Utility function only used in tests for creating a dummy set of samples just from a set of
    /// points.
    #[cfg(test)]
    pub(crate) fn new_point_samples(points: Vec<Vector3<T>>) -> Self {
        let n = points.len();
        Samples {
            points,
            normals: vec![Vector3::zeros(); n],
            velocities: vec![Vector3::zeros(); n],
            values: vec![T::zero(); n],
        }
    }

    pub fn new_triangle_samples<V3>(
        triangles: &[[usize; 3]],
        vertices: &[V3],
        values: Vec<T>,
    ) -> Self
    where
        V3: Into<Vector3<T>> + Clone,
    {
        let points = vec![Vector3::zeros(); triangles.len()];
        let normals = vec![Vector3::zeros(); triangles.len()];
        let velocities = vec![Vector3::zeros(); triangles.len()];

        let mut samples = Samples {
            points,
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
        V3: Into<Vector3<T>> + Clone,
    {
        let Samples {
            ref mut points,
            ref mut normals,
            ref mut velocities,
            ..
        } = self;

        let new_iter = triangles.iter().map(|tri_indices| {
            let tri = Triangle::from_indexed_slice(tri_indices, &vertices);
            let v = tri[1] - tri[0]; // tangent direction
            (tri.centroid(), tri.area_normal(), v / v.norm())
        });

        for (((pos, nml), vel), (new_pos, new_nml, new_vel)) in (points
            .iter_mut()
            .zip(normals.iter_mut())
            .zip(velocities.iter_mut()))
        .zip(new_iter)
        {
            *pos = new_pos;
            *nml = new_nml;
            *vel = new_vel;
        }
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.points.len()
    }

    #[inline]
    pub fn iter<'a>(&'a self) -> impl Iterator<Item = Sample<T>> + Clone + 'a {
        let Samples {
            ref points,
            ref normals,
            ref velocities,
            ref values,
        } = *self;
        points
            .iter()
            .zip(normals.iter())
            .zip(velocities.iter())
            .zip(values.iter())
            .enumerate()
            .map(move |(i, (((&pos, &nml), &vel), &value))| Sample {
                index: i,
                pos,
                nml,
                vel,
                value,
            })
    }

    /// Consuming iterator.
    #[allow(clippy::should_implement_trait)] // waiting for impl trait on associated types
    #[inline]
    pub fn into_iter(self) -> impl Iterator<Item = Sample<T>> + Clone {
        let Samples {
            points,
            normals,
            velocities,
            values,
        } = self;
        points
            .into_iter()
            .zip(normals.into_iter())
            .zip(velocities.into_iter())
            .zip(values.into_iter())
            .enumerate()
            .map(move |(i, (((pos, nml), vel), value))| Sample {
                index: i,
                pos,
                nml,
                vel,
                value,
            })
    }
}

impl<T: Real + Send + Sync> Samples<T> {
    #[inline]
    pub fn par_iter<'a>(&'a self) -> impl IndexedParallelIterator<Item = Sample<T>> + Clone + 'a {
        let Samples {
            ref points,
            ref normals,
            ref velocities,
            ref values,
        } = *self;
        points
            .par_iter()
            .zip(normals.par_iter())
            .zip(velocities.par_iter())
            .zip(values.par_iter())
            .enumerate()
            .map(move |(i, (((&pos, &nml), &vel), &value))| Sample {
                index: i,
                pos,
                nml,
                vel,
                value,
            })
    }

    /// Consuming iterator.
    #[inline]
    pub fn into_par_iter(self) -> impl IndexedParallelIterator<Item = Sample<T>> + Clone {
        let Samples {
            points,
            normals,
            velocities,
            values,
        } = self;
        points
            .into_par_iter()
            .zip(normals.into_par_iter())
            .zip(velocities.into_par_iter())
            .zip(values.into_par_iter())
            .enumerate()
            .map(move |(i, (((pos, nml), vel), value))| Sample {
                index: i,
                pos,
                nml,
                vel,
                value,
            })
    }
}

/// A view into to the positions, normals and offsets of the sample points. This view need not be
/// contiguous as it often isnt.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct SamplesView<'i, 'd: 'i, T: Real> {
    /// Indices into the sample points.
    indices: &'i [usize],
    /// Sample point positions defining the implicit surface.
    points: &'d [Vector3<T>],
    /// Normals that define the potential field gradient at every sample point.
    normals: &'d [Vector3<T>],
    /// Vectors that identify a tangent to the iso-surface at every sample point.
    velocities: &'d [Vector3<T>],
    /// Field values at the interpolating points. These are the values to match by interpolating
    /// implicit surfaces. This means that, for example, the zero iso-surface will not necessarily
    /// pass through the given points.
    values: &'d [T],
}

impl<'i, 'd: 'i, T: Real> SamplesView<'i, 'd, T> {
    /// Create a view of samples with a given indices slice into the provided samples.
    #[inline]
    pub fn new(indices: &'i [usize], samples: &'d Samples<T>) -> Self {
        SamplesView {
            indices,
            points: samples.points.as_slice(),
            normals: samples.normals.as_slice(),
            velocities: samples.velocities.as_slice(),
            values: samples.values.as_slice(),
        }
    }

    /// Construct a new view from this view using the same underlying data, but with a new set of
    /// indices (which need not be a subset of indices from this view).
    #[inline]
    pub fn from_view(self, indices: &'i [usize]) -> Self {
        SamplesView { indices, ..self }
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
            ref points,
            ref normals,
            ref velocities,
            ref values,
            ..
        } = self;
        Sample {
            index: idx,
            pos: points[idx],
            nml: normals[idx],
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
        self.indices.into_iter().map(move |&i| self.at_index(i))
    }
    #[inline]
    pub fn all_points(&'d self) -> &'d [Vector3<T>] {
        self.points
    }

    #[inline]
    pub fn all_normals(&'d self) -> &'d [Vector3<T>] {
        self.normals
    }
}

impl<'i, 'd: 'i, T: Real + Send + Sync> SamplesView<'i, 'd, T> {
    #[inline]
    pub fn par_iter(&'i self) -> impl IndexedParallelIterator<Item = Sample<T>> + 'i {
        self.indices.par_iter().map(move |&i| self.at_index(i))
    }

    /// Consuming iterator.
    #[inline]
    pub fn into_par_iter(self) -> impl IndexedParallelIterator<Item = Sample<T>> + 'i {
        self.indices.into_par_iter().map(move |&i| self.at_index(i))
    }
}
