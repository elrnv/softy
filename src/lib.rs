extern crate rayon;
extern crate nalgebra as na;
extern crate geometry as geo;

use rayon::prelude::*;
use geo::mesh::{attrib, topology::*, Attrib, PolyMesh};
use na::{dot, DMatrix, DVector, Dynamic, Matrix, MatrixVec, U1, Vector3};

type BasisMatrix = Matrix<f64, Dynamic, U1, MatrixVec<f64, Dynamic, U1>>;

pub struct Params {
    pub tolerance: f32,
}

#[allow(non_snake_case)]
pub fn compute_mls<F>(
    samples: &mut PolyMesh<f64>,
    surface: &mut PolyMesh<f64>,
    params: Params,
    interrupt: F,
) -> Result<(), Error> 
    where F: Fn() -> bool + Sync + Send,
{
    // Initialize potential with zeros.
    {
        samples.set_attrib::<_, VertexIndex>("potential", 0.0f32)?;
    }

    // Check that we have normals
    {
        surface.attrib_check::<[f32; 3], VertexIndex>("N")?;
    }

    let delta = params.tolerance as f64;

    let kernel = |r| {
        let w = 1.0 / (r * r + delta * delta);
        w * w
    };

    let w = |x, p| kernel((Vector3::from(x) - Vector3::from(p)).norm());

    let sample_pos = samples.vertices();
    let pos = surface.vertices();
    let nml: Vec<Vector3<f64>> = surface
        .attrib_iter::<[f32; 3], VertexIndex>("N")?
        .map(|x| Vector3::new(x[0] as f64, x[1] as f64, x[2] as f64))
        .collect();
    let n = surface.num_vertices();

    let B = BasisMatrix::from_element(n, 1.0);

    sample_pos.par_iter().zip(
        samples
            .attrib_as_mut_slice::<f32, VertexIndex>("potential")
            .unwrap()
            .par_iter_mut()
    ).for_each(|(q, potential)| {
        if interrupt() {
            return;
        }
        let W = DMatrix::from_fn(n, n, |i, j| if i == j { w(*q, pos[i]) } else { 0.0 });
        let S = DVector::from_fn(n, |i, _| {
            dot(
                &Vector3::from(nml[i]),
                &(Vector3::from(*q) - Vector3::from(pos[i])),
            )
        });

        let BTW = B.transpose() * W;
        let A = &BTW * &B;
        let b = &BTW * &S;

        if let Some(c) = A.lu().solve(&b) {
            *potential = c[0] as f32;
        }
    });

    if interrupt() {
        Err(Error::Interrupted)
    } else {
        Ok(())
    }
}

pub enum Error {
    Interrupted,
    MissingNormals,
    Failure,
}

impl From<attrib::Error> for Error {
    fn from(err: attrib::Error) -> Self {
        match err {
            attrib::Error::TypeMismatch => Error::MissingNormals,
            _ => Error::Failure,
        }
    }
}
