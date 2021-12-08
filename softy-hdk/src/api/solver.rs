use geo::algo::split::TypedMesh;
use geo::attrib::*;
use softy::{fem, Error, Mesh, PointCloud, SimResult, TetMesh, TriMesh, SOURCE_INDEX_ATTRIB};

// NOTE: We avoid using associated types here because of a compiler bug:
// https://github.com/rust-lang/rust/issues/23856
// This bug makes using this trait very verbose with associated types.

pub trait Solver: Send {
    fn solve(&mut self) -> SimResult;
    fn mesh(&self) -> Mesh;
    fn solid_mesh(&self) -> TetMesh;
    fn shell_mesh(&self) -> TriMesh;
    fn num_vertices(&self) -> usize;
    fn update_vertices(&mut self, pts: &PointCloud) -> Result<(), Error>;
    fn set_interrupter(&mut self, interrupter: Box<dyn FnMut() -> bool + Send>);
}

impl Solver for fem::opt::Solver {
    #[inline]
    fn solve(&mut self) -> SimResult {
        self.step().into()
    }
    #[inline]
    fn mesh(&self) -> Mesh {
        let tetmesh = self.solid_mesh();
        let trimesh = self.shell_mesh();
        Mesh::merge_with_vertex_source(
            [&Mesh::from(tetmesh), &Mesh::from(trimesh)],
            SOURCE_INDEX_ATTRIB,
        )
        .expect("Missing source index attribute.")
    }
    #[inline]
    fn solid_mesh(&self) -> TetMesh {
        let mut tetmesh = TetMesh::merge_with_vertex_source(
            self.solids().iter().map(|s| &s.tetmesh),
            SOURCE_INDEX_ATTRIB,
        )
        .expect("Missing source index attribute.");
        tetmesh
            .reset_attrib_to_default::<i32, geo::CellIndex>("object_type", 0)
            .unwrap();
        tetmesh
    }
    #[inline]
    fn shell_mesh(&self) -> TriMesh {
        let mut trimesh = TriMesh::merge_with_vertex_source(
            self.shells().iter().map(|s| &s.trimesh),
            SOURCE_INDEX_ATTRIB,
        )
        .expect("Missing source index attribute.");
        trimesh
            .reset_attrib_to_default::<i32, geo::FaceIndex>("object_type", 1)
            .unwrap();
        trimesh
    }
    #[inline]
    fn num_vertices(&self) -> usize {
        use geo::mesh::topology::NumVertices;
        let num_solid_verts: usize = self.solids().iter().map(|x| x.tetmesh.num_vertices()).sum();
        let num_shell_verts: usize = self.shells().iter().map(|x| x.trimesh.num_vertices()).sum();
        num_solid_verts + num_shell_verts
    }
    #[inline]
    fn update_vertices(&mut self, pts: &PointCloud) -> Result<(), Error> {
        self.update_solid_vertices(pts)
            .and(self.update_shell_vertices(pts))
    }
    #[inline]
    fn set_interrupter(&mut self, interrupter: Box<dyn FnMut() -> bool + Send>) {
        self.set_interrupter(interrupter);
    }
}

impl<S> Solver for fem::nl::Solver<S, f64>
where
    S: Send + fem::nl::NLSolver<fem::nl::NLProblem<f64>, f64>,
{
    #[inline]
    fn solve(&mut self) -> SimResult {
        self.step().into()
    }
    #[inline]
    fn mesh(&self) -> Mesh {
        fem::nl::Solver::mesh(self)
    }
    #[inline]
    fn solid_mesh(&self) -> TetMesh {
        let mesh = self.mesh();
        let meshes = mesh.split_into_typed_meshes();
        for unimesh in meshes.into_iter() {
            if let TypedMesh::Tet(mut tetmesh) = unimesh {
                tetmesh
                    .reset_attrib_to_default::<i32, geo::CellIndex>("object_type", 0)
                    .unwrap();
                return tetmesh;
            }
        }
        return TetMesh::default();
    }
    #[inline]
    fn shell_mesh(&self) -> TriMesh {
        let mesh = self.mesh();
        let meshes = mesh.split_into_typed_meshes();
        for unimesh in meshes.into_iter() {
            if let TypedMesh::Tri(mut trimesh) = unimesh {
                trimesh
                    .reset_attrib_to_default::<i32, geo::FaceIndex>("object_type", 1)
                    .unwrap();
                return trimesh;
            }
        }
        return TriMesh::default();
    }
    #[inline]
    fn num_vertices(&self) -> usize {
        use geo::mesh::topology::NumVertices;
        self.mesh().num_vertices()
    }
    #[inline]
    fn update_vertices(&mut self, pts: &PointCloud) -> Result<(), Error> {
        self.update_vertices(pts)
    }
    #[inline]
    fn set_interrupter(&mut self, interrupter: Box<dyn FnMut() -> bool + Send>) {
        self.set_coarse_interrupter(interrupter);
    }
}
