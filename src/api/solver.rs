use softy::{TetMesh, FemEngine, SolveResult, Error};

// NOTE: We avoid using associated types here because of a compiler bug:
// https://github.com/rust-lang/rust/issues/23856
// This bug makes using this trait very verbose with associated types.

pub trait Solver {
    fn solve(&mut self) -> Result<SolveResult, Error>;
    fn mesh_ref(&self) -> &TetMesh;
    fn set_interrupter(&mut self, interrupter: Box<FnMut() -> bool>);
}

impl Solver for FemEngine {
    fn solve(&mut self) -> Result<SolveResult, Error> {
        self.step()
    }
    fn mesh_ref(&self) -> &TetMesh {
        self.mesh_ref()
    }
    fn set_interrupter(&mut self, interrupter: Box<FnMut() -> bool>) {
        self.set_interrupter(interrupter);
    }
}
