use geo::mesh::attrib::*;
use softy::{fem, Error, PointCloud, SimResult, TetMesh, TriMesh, SOURCE_INDEX_ATTRIB};

// NOTE: We avoid using associated types here because of a compiler bug:
// https://github.com/rust-lang/rust/issues/23856
// This bug makes using this trait very verbose with associated types.

pub trait Solver: Send {
    fn solve(&mut self) -> SimResult;
    fn solid_mesh(&self) -> TetMesh;
    fn shell_mesh(&self) -> TriMesh;
    fn num_solid_vertices(&self) -> usize;
    fn num_shell_vertices(&self) -> usize;
    fn update_solid_vertices(&mut self, pts: &PointCloud) -> Result<(), Error>;
    fn update_shell_vertices(&mut self, pts: &PointCloud) -> Result<(), Error>;
    fn set_interrupter(&mut self, interrupter: Box<dyn FnMut() -> bool + Send>);
}

impl Solver for fem::opt::Solver {
    #[inline]
    fn solve(&mut self) -> SimResult {
        self.step().into()
    }
    #[inline]
    fn solid_mesh(&self) -> TetMesh {
        let mut tetmesh = TetMesh::merge_with_vertex_source(
            self.solids().iter().map(|s| &s.tetmesh),
            SOURCE_INDEX_ATTRIB,
        )
        .expect("Missing source index attribute.");
        tetmesh
            .set_attrib::<i32, geo::CellIndex>("object_type", 0)
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
            .set_attrib::<i32, geo::FaceIndex>("object_type", 1)
            .unwrap();
        trimesh
    }
    #[inline]
    fn num_solid_vertices(&self) -> usize {
        use geo::mesh::topology::NumVertices;
        self.solids().iter().map(|x| x.tetmesh.num_vertices()).sum()
    }
    #[inline]
    fn num_shell_vertices(&self) -> usize {
        use geo::mesh::topology::NumVertices;
        self.shells().iter().map(|x| x.trimesh.num_vertices()).sum()
    }
    #[inline]
    fn update_solid_vertices(&mut self, pts: &PointCloud) -> Result<(), Error> {
        self.update_solid_vertices(pts)
    }
    #[inline]
    fn update_shell_vertices(&mut self, pts: &PointCloud) -> Result<(), Error> {
        self.update_shell_vertices(pts)
    }
    #[inline]
    fn set_interrupter(&mut self, interrupter: Box<dyn FnMut() -> bool + Send>) {
        self.set_interrupter(interrupter);
    }
}

impl Solver for fem::nl::Solver<f64> {
    #[inline]
    fn solve(&mut self) -> SimResult {
        self.step().into()
    }
    #[inline]
    fn solid_mesh(&self) -> TetMesh {
        let mut tetmesh = TetMesh::merge_with_vertex_source(
            self.solids().iter().map(|s| &s.tetmesh),
            SOURCE_INDEX_ATTRIB,
        )
        .expect("Missing source index attribute.");
        tetmesh
            .set_attrib::<i32, geo::CellIndex>("object_type", 0)
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
            .set_attrib::<i32, geo::FaceIndex>("object_type", 1)
            .unwrap();
        trimesh
    }
    #[inline]
    fn num_solid_vertices(&self) -> usize {
        use geo::mesh::topology::NumVertices;
        self.solids().iter().map(|x| x.tetmesh.num_vertices()).sum()
    }
    #[inline]
    fn num_shell_vertices(&self) -> usize {
        use geo::mesh::topology::NumVertices;
        self.shells().iter().map(|x| x.trimesh.num_vertices()).sum()
    }
    #[inline]
    fn update_solid_vertices(&mut self, pts: &PointCloud) -> Result<(), Error> {
        self.update_solid_vertices(pts)
    }
    #[inline]
    fn update_shell_vertices(&mut self, pts: &PointCloud) -> Result<(), Error> {
        self.update_shell_vertices(pts)
    }
    #[inline]
    fn set_interrupter(&mut self, interrupter: Box<dyn FnMut() -> bool + Send>) {
        self.set_simple_interrupter(interrupter);
    }
}
