use softy::{fem, Error, PointCloud, SolveResult, TetMesh, TriMesh};
use std::cell::Ref;

// NOTE: We avoid using associated types here because of a compiler bug:
// https://github.com/rust-lang/rust/issues/23856
// This bug makes using this trait very verbose with associated types.

pub trait Solver: Send {
    fn solve(&mut self) -> Result<SolveResult, Error>;
    fn borrow_mesh(&self) -> Ref<'_, TetMesh>;
    fn try_borrow_kinematic_mesh(&self) -> Option<Ref<'_, TriMesh>>;
    fn update_solid_vertices(&mut self, pts: &PointCloud) -> Result<(), Error>;
    fn update_shell_vertices(&mut self, pts: &PointCloud) -> Result<(), Error>;
    fn set_interrupter(&mut self, interrupter: Box<FnMut() -> bool>);
}

impl Solver for fem::Solver {
    #[inline]
    fn solve(&mut self) -> Result<SolveResult, Error> {
        self.step()
    }
    #[inline]
    fn try_borrow_kinematic_mesh(&self) -> Option<Ref<'_, TriMesh>> {
        self.try_borrow_kinematic_mesh()
    }
    #[inline]
    fn borrow_mesh(&self) -> Ref<'_, TetMesh> {
        self.borrow_mesh()
    }
    #[inline]
    fn update_solid_vertices(&mut self, pts: &PointCloud) -> Result<(), Error> {
        self.update_mesh_vertices(pts)
    }
    #[inline]
    fn update_shell_vertices(&mut self, pts: &PointCloud) -> Result<(), Error> {
        self.update_kinematic_vertices(pts)
    }
    #[inline]
    fn set_interrupter(&mut self, interrupter: Box<FnMut() -> bool>) {
        self.set_interrupter(interrupter);
    }
}
