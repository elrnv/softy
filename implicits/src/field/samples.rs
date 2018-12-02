use rayon::prelude::{IndexedParallelIterator};
use crate::geo::math::Vector3;
use crate::geo::ops::*;
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

/// Convert a sample of one type into a sample of another real type.
impl<T: Real> Sample<T> {
    /// Cast a sample into another real type. This function will unwrap internally, so it will
    /// panic if the conversion is invalid.
    pub fn cast<S: Real>(self) -> Sample<S> {
        Sample {
            index: self.index,
            pos: self.pos.map(|x| S::from(x).unwrap()),
            nml: self.nml.map(|x| S::from(x).unwrap()),
            off: self.off,
        }
    }
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

impl<T: Real> Samples<T> {
    pub fn new_triangle_samples<V3>(
        triangles: &[[usize; 3]],
        vertices: &[V3],
        offsets: Vec<f64>,
    ) -> Self
    where
        V3: Into<Vector3<T>> + Clone,
    {
        let points = vec![Vector3::zeros(); triangles.len()];
        let normals = vec![Vector3::zeros(); triangles.len()];

        let mut samples = Samples {
            points,
            normals,
            offsets,
        };

        samples.update_triangle_samples(triangles, vertices);
        samples
    }

    pub fn update_triangle_samples<V3>(&mut self, triangles: &[[usize; 3]], vertices: &[V3])
    where
        V3: Into<Vector3<T>> + Clone,
    {
        let Samples {
            ref mut points,
            ref mut normals,
            ..
        } = self;

        let new_pos_nml_iter = triangles.iter().map(|tri_indices| {
            let tri = Triangle::from_indexed_slice(tri_indices, &vertices);
            (tri.centroid(), tri.area_normal())
        });

        for ((pos, nml), (new_pos, new_nml)) in
            (points.iter_mut().zip(normals.iter_mut())).zip(new_pos_nml_iter)
        {
            *pos = new_pos;
            *nml = new_nml;
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
            ref offsets,
        } = *self;
        points
            .iter()
            .zip(normals.iter())
            .zip(offsets.iter())
            .enumerate()
            .map(move |(i, ((&pos, &nml), &off))| Sample {
                index: i,
                pos,
                nml,
                off,
            })
    }

    /// Consuming iterator.
    #[inline]
    pub fn into_iter(self) -> impl Iterator<Item = Sample<T>> + Clone {
        let Samples {
            points,
            normals,
            offsets,
        } = self;
        points
            .into_iter()
            .zip(normals.into_iter())
            .zip(offsets.into_iter())
            .enumerate()
            .map(move |(i, ((pos, nml), off))| Sample {
                index: i,
                pos,
                nml,
                off,
            })
    }
}

impl<T: Real + Send + Sync> Samples<T> {
    #[inline]
    pub fn par_iter<'a>(&'a self) -> impl IndexedParallelIterator<Item = Sample<T>> + Clone + 'a {
        let Samples {
            ref points,
            ref normals,
            ref offsets,
        } = *self;
        points
            .par_iter()
            .zip(normals.par_iter())
            .zip(offsets.par_iter())
            .enumerate()
            .map(move |(i, ((&pos, &nml), &off))| Sample {
                index: i,
                pos,
                nml,
                off,
            })
    }


    /// Consuming iterator.
    #[inline]
    pub fn into_par_iter(self) -> impl IndexedParallelIterator<Item = Sample<T>> + Clone {
        let Samples {
            points,
            normals,
            offsets,
        } = self;
        points
            .into_par_iter()
            .zip(normals.into_par_iter())
            .zip(offsets.into_par_iter())
            .enumerate()
            .map(move |(i, ((pos, nml), off))| Sample {
                index: i,
                pos,
                nml,
                off,
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

impl<'i, 'd: 'i, T: Real + Send + Sync> SamplesView<'i, 'd, T> {
    #[inline]
    pub fn par_iter(&'i self) -> impl IndexedParallelIterator<Item = Sample<T>> + 'i {
        let SamplesView {
            ref indices,
            ref points,
            ref normals,
            ref offsets,
        } = self;
        indices.par_iter().map(move |&i| Sample {
            index: i,
            pos: points[i],
            nml: normals[i],
            off: offsets[i],
        })
    }


    /// Consuming iterator.
    #[inline]
    pub fn into_par_iter(self) -> impl IndexedParallelIterator<Item = Sample<T>> + 'i {
        let SamplesView {
            indices,
            points,
            normals,
            offsets,
        } = self;
        indices.into_par_iter().map(move |&i| Sample {
            index: i,
            pos: points[i],
            nml: normals[i],
            off: offsets[i],
        })
    }

}
