use fem::{FemEngine, SolveResult, Error};

pub trait Solver {
    type ResultType;

    fn solve(&mut self) -> Self::ResultType;
}

impl<'a, F: FnMut() -> bool + Sync> Solver for FemEngine<'a, F> {
    type ResultType = Result<SolveResult, Error>;

    fn solve(&mut self) -> Self::ResultType {
        self.step()
    }
}
