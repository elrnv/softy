use nodal_fem_nlp::NLP;
use energy::Energy;
use ipopt::Ipopt;
use alga::general::Inverse;
use na::{Matrix3, Point3, Vector3};
use geo::topology::*;
use geo::prim::Tetrahedron;
use geo::mesh::{self, attrib, Attrib};
use geo::mesh::attrib::Attribute;
use geo::ops::{ShapeMatrix, Volume};

pub type TetMesh = mesh::TetMesh<f64>;
pub type Tet = Tetrahedron<f64>;

/// Get reference tetrahedron.
//pub fn ref_tet(tetmesh: &TetMesh, cidx: CellIndex) -> Option<Tet> {
pub fn ref_tet(attrib: &Attribute<VertexIndex>, indices: &Vec<usize>, cidx: CellIndex) -> Option<Tet> {
    //let attrib = tetmesh.attrib::<VertexIndex>("ref").ok()?;
    let make_pt = |array: [f64;3]| {
        Point3::from_coordinates(Vector3::from(array))
    };
    let mb_i: Option<usize> = cidx.into();
    let i = mb_i.unwrap();
    Some(Tet {
        a: make_pt(attrib.get::<[f64;3], _>(VertexIndex::from(indices[4*i + 0]))?),
        b: make_pt(attrib.get::<[f64;3], _>(VertexIndex::from(indices[4*i + 1]))?),
        c: make_pt(attrib.get::<[f64;3], _>(VertexIndex::from(indices[4*i + 2]))?),
        d: make_pt(attrib.get::<[f64;3], _>(VertexIndex::from(indices[4*i + 3]))?),
        //a: make_pt(attrib.get::<[f64;3], _>(tetmesh.cell_vertex(cidx, 0))?),
        //b: make_pt(attrib.get::<[f64;3], _>(tetmesh.cell_vertex(cidx, 1))?),
        //c: make_pt(attrib.get::<[f64;3], _>(tetmesh.cell_vertex(cidx, 2))?),
        //d: make_pt(attrib.get::<[f64;3], _>(tetmesh.cell_vertex(cidx, 3))?),
    })
}

pub fn run(mesh: &mut TetMesh) -> Result<(), Error> {
    {
    let &mut TetMesh {
        ref mut vertices,
        ref mut indices,
        ref mut vertex_attributes,
        ref mut cell_attributes, ..
    } = mesh;


    vertex_attributes.entry("force".to_owned())
        .or_insert(Attribute::with_size(vertices.len(), Vector3::<f64>::zeros()));

    let ref_a = 
        vertex_attributes.entry("ref".to_owned())
            .or_insert(Attribute::from_vec(vertices.get().clone()));

    // Prepare tet mesh for simulation.
    //if !mesh.attrib_exists::<VertexIndex>("ref") {
    //    let verts = mesh.vertices();
    //    mesh.add_attrib_data::<_, VertexIndex>("ref", verts)?;
    //}
    {
        let mtx_a = cell_attributes.entry("ref_shape_mtx_inv".to_owned())
            .or_insert(Attribute::with_size(indices.len()/4, Matrix3::<f64>::zeros()));

        for (i, mtx) in mtx_a.iter_mut::<Matrix3<f64>>().unwrap().enumerate() {
            let cidx = CellIndex::from(i);
            let ref_shape_matrix = ref_tet(ref_a, indices, cidx).unwrap().shape_matrix();
            *mtx = ref_shape_matrix.inverse();
        }
    }

    {
        let vol_a = cell_attributes.entry("ref_volume".to_owned())
            .or_insert(Attribute::with_size(indices.len()/4, 0.0));
        for (i, vol) in vol_a.iter_mut::<f64>().unwrap().enumerate() {
            let cidx = CellIndex::from(i);
            *vol = ref_tet(ref_a, indices, cidx).unwrap().signed_volume();
        }
    }
    // create vertex force attribute
    //mesh.add_attrib::<Vector3<f64>, VertexIndex>("force", Vector3::zeros())?;
    //{
    //    let mtx_a = mesh.add_attrib::<Matrix3<f64>, CellIndex>("ref_shape_mtx_inv", Matrix3::zeros())?;
    //    for (i, mtx) in mtx_a.iter_mut::<Matrix3<f64>>().unwrap().enumerate() {
    //        let cidx = CellIndex::from(i);
    //        let ref_shape_matrix = ref_tet(mesh, cidx).unwrap().shape_matrix();
    //        *mtx = ref_shape_matrix.inverse();
    //    }
    //}

    //{
    //    let vol_a = mesh.add_attrib::<_, CellIndex>("ref_volume", 0.0)?;
    //    for (i, vol) in vol_a.iter_mut::<f64>().unwrap().enumerate() {
    //        let cidx = CellIndex::from(i);
    //        *vol = ref_tet(mesh,cidx).unwrap().signed_volume();
    //    }
    //}
    }

    let x = {
        let nlp = NLP { body: mesh };
        let mut ipopt = Ipopt::new_unconstrained(nlp);

        ipopt.set_option("tol", 1e-7);
        ipopt.set_option("mu_strategy", "adaptive");
        ipopt.set_option("sb", "yes");
        ipopt.set_option("print_level", 5);
        ipopt.set_intermediate_callback(Some(NLP::intermediate_cb));
        let (_r, _obj) = ipopt.solve();
        ipopt.solution().to_vec()
    };
    for (i, v) in mesh.vertex_iter_mut().enumerate() {
        *v = [x[3*i + 0], x[3*i + 1], x[3*i + 2]];
    }

    Ok(())
}


/// Consitutive model invariants and their derivatives wrt. the deformation gradient.
#[allow(non_snake_case)]
struct Invariants {
    I: f64,
    J: f64,
}

struct InvariantDerivatives {
    dIdF: Matrix3<f64>,
    dJdF: Matrix3<f64>,
}

impl Invariants {
    #[allow(non_snake_case)]
    fn new(F: &Matrix3<f64>) -> Self {
        Invariants {
            I: F.map(|x| x*x).iter().sum(), // tr(F^TF)
            J: F.determinant(),
        }
    }
}

impl InvariantDerivatives {
    fn new(F: &Matrix3<f64>) -> Self {
        InvariantDerivatives {
            dIdF: 2.0*F,
            dJdF: F.determinant()*F.inverse().transpose(),
        }
    }
}

/// Define energy for Neohookean materials.
impl Energy<f64> for TetMesh {
    #[allow(non_snake_case)]
    fn energy(&self) -> f64 {
        let mu = 5.4;
        let lambda = 263.1;
        self.attrib_iter::<f64,CellIndex>("ref_volume").unwrap()
            .zip(self.attrib_iter::<Matrix3<f64>,CellIndex>("ref_shape_mtx_inv").unwrap())
            .zip(self.tet_iter())
            .map(|((&vol, &Dminv), tet)| {
                let F = tet.shape_matrix()*Dminv;
                let i = Invariants::new(&F);
                if i.J <= 0.0 {
                    ::std::f64::INFINITY
                } else {
                    let logJ = i.J.log2();
                    vol*(0.5*mu*(i.I - 3.0) - mu*logJ + 0.5*lambda*logJ*logJ)
                }
            }).sum()
    }
    #[allow(non_snake_case)]
    fn energy_gradient(&self) -> Vec<Vector3<f64>> {
        let mu = 5.4;
        let lambda = 263.1;
        let force_iter = 
            self.attrib_iter::<f64,CellIndex>("ref_volume").unwrap()
            .zip(self.attrib_iter::<Matrix3<f64>,CellIndex>("ref_shape_mtx_inv").unwrap())
            .zip(self.tet_iter())
            .map(|((&vol, &Dminv), tet)| {
                let F = tet.shape_matrix()*Dminv;
            let i = Invariants::new(&F);
            if i.J <= 0.0 {
                ::std::f64::INFINITY*Matrix3::new(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
            } else {
                let F_inv_tr = F.inverse().transpose();
                let logJ = i.J.log2();
                -vol* (mu * F  + (lambda * logJ - mu) * F_inv_tr)*Dminv.transpose()
            }
            });

        // Transfer forces from cell-vertices to vertices themeselves
        let mut vtx_forces = Vec::new();
        vtx_forces.resize(self.num_verts(), Vector3::zeros());
        for (cell, forces) in self.cell_iter().zip(force_iter) {
            for i in 0..3 {
                vtx_forces[cell[i]] += forces.column(i);
                vtx_forces[cell[3]] -= forces.column(i);
            }
        }
        vtx_forces
    }
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
