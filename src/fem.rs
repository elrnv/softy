use alga::general::Inverse;
use na::Point3;
use geo::topology::*;
use geo::prim::Tetrahedron;
use geo::mesh::{self, attrib, Attrib};
use geo::ops::ShapeMatrix;

pub type TetMesh = mesh::TetMesh<f64>;
pub type Tet = Tetrahedron<f64>;

/// Get reference tetrahedron.
pub fn ref_tet(tetmesh: &TetMesh, cidx: CellIndex) -> Option<Tet> {
    let attrib = tetmesh.attrib::<VertexIndex>("ref").ok()?;
    Some(Tet {
        a: attrib.get::<Point3<f64>, _>(tetmesh.cell_vertex(cidx, 0))?,
        b: attrib.get::<Point3<f64>, _>(tetmesh.cell_vertex(cidx, 1))?,
        c: attrib.get::<Point3<f64>, _>(tetmesh.cell_vertex(cidx, 2))?,
        d: attrib.get::<Point3<f64>, _>(tetmesh.cell_vertex(cidx, 3))?,
    })
}

pub fn run(mesh: &mut TetMesh) -> Result<(), Error> {
    // Prepare tet mesh for simulation.
    if !mesh.attrib_exists::<VertexIndex>("ref") {
        let verts = mesh.vertices();
        mesh.add_attrib_data::<_, VertexIndex>("ref", verts)?;
    }

    let mut ref_shape_matrix_inverses = Vec::new();
    for cidx in 0..mesh.num_cells() {
        let ref_shape_matrix = ref_tet(mesh, CellIndex::from(cidx)).unwrap().shape_matrix();
        ref_shape_matrix_inverses.push(ref_shape_matrix.inverse());
    }
    Ok(())
}

pub trait Energy<T> {
    fn energy(mesh: TetMesh);
}

#[derive(Debug)]
pub enum Error {
    AttribError(attrib::Error),
}

impl From<attrib::Error> for Error {
    fn from(err: attrib::Error) -> Error {
        Error::AttribError(err)
    }
}
