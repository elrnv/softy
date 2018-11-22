use crate::geo::math::Vector3;
use crate::geo::Real;

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
    /// Offset stored at the sample point.
    pub off: f64,
}

/// Sample points that define the implicit surface including the point positions, normals and
/// offsets.
#[derive(Clone, Debug, PartialEq)]
pub struct Samples<T: Real> {
    /// Sample point positions defining the implicit surface.
    pub points: Vec<Vector3<T>>,
    /// Normals that define the potential field gradient at every sample point.
    pub normals: Vec<Vector3<T>>,

    /// Potential values at the interpolating points. These offsets indicate the values to
    /// match by interpolating implicit surfaces. This means that the zero iso-surface will not
    /// necessarily pass through the given points.
    pub offsets: Vec<f64>,
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
    /// Potential values at the interpolating points. These offsets indicate the values to
    /// match by interpolating implicit surfaces. This means that the zero iso-surface will not
    /// necessarily pass through the given points.
    offsets: &'d [f64],
}

impl<'i, 'd: 'i, T: Real> SamplesView<'i, 'd, T> {
    /// Create a view of samples with a given indices slice into the provided samples.
    #[inline]
    pub fn new(indices: &'i [usize], samples: &'d Samples<T>) -> Self {
        SamplesView {
            indices,
            points: samples.points.as_slice(),
            normals: samples.normals.as_slice(),
            offsets: samples.offsets.as_slice(),
        }
    }

    #[inline]
    pub fn from_view(indices: &'i [usize], samples: SamplesView<'i, 'd, T>) -> Self {
        SamplesView {
            indices,
            points: samples.points.clone(),
            normals: samples.normals.clone(),
            offsets: samples.offsets.clone(),
        }
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.indices.is_empty()
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.indices.len()
    }

    #[inline]
    pub fn iter(&'i self) -> impl Iterator<Item = Sample<T>> + 'i {
        let SamplesView {
            ref indices,
            ref points,
            ref normals,
            ref offsets,
        } = self;
        indices.iter().map(move |&i| Sample {
            index: i,
            pos: points[i],
            nml: normals[i],
            off: offsets[i],
        })
    }

    /// Consuming iterator.
    #[inline]
    pub fn into_iter(self) -> impl Iterator<Item = Sample<T>> + 'i {
        let SamplesView {
            indices,
            points,
            normals,
            offsets,
        } = self;
        indices.into_iter().map(move |&i| Sample {
            index: i,
            pos: points[i],
            nml: normals[i],
            off: offsets[i],
        })
    }

    #[inline]
    pub fn points(&'d self) -> &'d [Vector3<T>] {
        self.points
    }

    #[inline]
    pub fn normals(&'d self) -> &'d [Vector3<T>] {
        self.normals
    }
}
