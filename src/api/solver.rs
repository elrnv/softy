use softy::{Error, FemEngine, SolveResult, TetMesh, PointCloud};

// NOTE: We avoid using associated types here because of a compiler bug:
// https://github.com/rust-lang/rust/issues/23856
// This bug makes using this trait very verbose with associated types.

pub trait Solver: Send {
    fn solve(&mut self) -> Result<SolveResult, Error>;
    fn mesh_ref(&mut self) -> &TetMesh;
    fn update_mesh_vertices(&mut self, pts: &PointCloud) -> Result<(), Error>;
    fn set_interrupter(&mut self, interrupter: Box<FnMut() -> bool>);
}

impl Solver for FemEngine {
    #[inline]
    fn solve(&mut self) -> Result<SolveResult, Error> {
        self.step()
    }
    #[inline]
    fn mesh_ref(&mut self) -> &TetMesh {
        self.mesh_ref()
    }
    #[inline]
    fn update_mesh_vertices(&mut self, pts: &PointCloud) -> Result<(), Error> {
        self.update_mesh_vertices(pts)
    }
    #[inline]
    fn set_interrupter(&mut self, interrupter: Box<FnMut() -> bool>) {
        self.set_interrupter(interrupter);
    }
}
