use softy::{fem, SolveResult, Error, TetMesh, PointCloud};
use std::cell::Ref;

// NOTE: We avoid using associated types here because of a compiler bug:
// https://github.com/rust-lang/rust/issues/23856
// This bug makes using this trait very verbose with associated types.

pub trait Solver: Send {
    fn solve(&mut self) -> Result<SolveResult, Error>;
    fn borrow_mesh(&self) -> Ref<'_, TetMesh>;
    fn update_mesh_vertices(&mut self, pts: &PointCloud) -> Result<(), Error>;
    fn set_interrupter(&mut self, interrupter: Box<FnMut() -> bool>);
}

impl Solver for fem::Solver {
    #[inline]
    fn solve(&mut self) -> Result<SolveResult, Error> {
        self.step()
    }
    #[inline]
    fn borrow_mesh(&self) -> Ref<'_, TetMesh> {
        self.borrow_mesh()
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
