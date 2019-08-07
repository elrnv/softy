use softy::{fem, Error, PointCloud, SolveResult, TetMesh, TriMesh};

// NOTE: We avoid using associated types here because of a compiler bug:
// https://github.com/rust-lang/rust/issues/23856
// This bug makes using this trait very verbose with associated types.

pub trait Solver: Send {
    fn solve(&mut self) -> Result<SolveResult, Error>;
    fn solid_mesh(&self) -> TetMesh;
    fn shell_mesh(&self) -> TriMesh;
    fn num_solid_vertices(&self) -> usize;
    fn num_shell_vertices(&self) -> usize;
    fn update_solid_vertices(&mut self, pts: &PointCloud) -> Result<(), Error>;
    fn update_shell_vertices(&mut self, pts: &PointCloud) -> Result<(), Error>;
    fn set_interrupter(&mut self, interrupter: Box<dyn FnMut() -> bool>);
}

impl Solver for fem::Solver {
    #[inline]
    fn solve(&mut self) -> Result<SolveResult, Error> {
        self.step()
    }
    #[inline]
    fn solid_mesh(&self) -> TetMesh {
        use geo::algo::Merge;
        let mut tetmesh = TetMesh::default();
        for solid in self.solids().iter() {
            tetmesh.merge(solid.tetmesh.clone());
        }
        tetmesh
    }
    #[inline]
    fn shell_mesh(&self) -> TriMesh {
        use geo::algo::Merge;
        let mut trimesh = TriMesh::default();
        for shell in self.shells().iter() {
            trimesh.merge(shell.trimesh.clone());
        }
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
    fn set_interrupter(&mut self, interrupter: Box<dyn FnMut() -> bool>) {
        self.set_interrupter(interrupter);
    }
}
