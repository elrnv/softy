//use std::path::Path;
//use geo::io::save_tetmesh;
use energy::*;
use geo::topology::*;
use TetMesh;
use ipopt::*;

/// Non-linear problem.
pub struct NLP<'a, F: Fn() -> bool> {
    pub body: &'a mut TetMesh,
    pub interrupt_checker: F,
    pub energy_count: u32,
}

impl<'a, F: Fn() -> bool> NLP<'a, F> {
    pub fn new(tetmesh: &'a mut TetMesh, interrupt_checker: F) -> Self {
        NLP {
            body: tetmesh,
            interrupt_checker,
            energy_count: 0u32,
        }
    }
    pub fn intermediate_cb(
        &mut self,
        _alg_mod: Index,
        _iter_count: Index,
        _obj_value: Number,
        _inf_pr: Number,
        _inf_du: Number,
        _mu: Number,
        _d_norm: Number,
        _regularization_size: Number,
        _alpha_du: Number,
        _alpha_pr: Number,
        _ls_trials: Index,
    ) -> bool {
        (self.interrupt_checker)()
    }
    pub fn update(&mut self, x: &[Number]) {
        for (i, v) in self.body.vertex_iter_mut().enumerate() {
            *v = [x[3 * i + 0], x[3 * i + 1], x[3 * i + 2]];
        }
    }
}

impl<'a, F: Fn() -> bool> BasicProblem for NLP<'a, F> {
    fn num_variables(&self) -> usize {
        self.body.num_verts() * 3
    }

    fn bounds(&self) -> (Vec<Number>, Vec<Number>) {
        let n = self.num_variables();
        let mut lo = Vec::with_capacity(n);
        let mut hi = Vec::with_capacity(n);
        // unbounded
        lo.resize(n, -2e19);
        hi.resize(n, 2e19);
        (lo, hi)
    }

    fn initial_point(&self) -> Vec<Number> {
        self.body
            .vertex_iter()
            .flat_map(|x| x.iter().cloned())
            .collect()
    }

    fn objective(&mut self, x: &[Number], obj: &mut Number) -> bool {
        self.update(x);

        //save_tetmesh(
        //    self.body,
        //    Path::new(format!("./tetmesh_{}.vtk", self.energy_count).as_str()),
        //);
        self.energy_count += 1;

        *obj = self.body.energy();
        true
    }

    fn objective_grad(&mut self, x: &[Number], grad_f: &mut [Number]) -> bool {
        self.update(x);

        for (i, g) in self.body.energy_gradient().iter().enumerate() {
            grad_f[3 * i + 0] = g[0];
            grad_f[3 * i + 1] = g[1];
            grad_f[3 * i + 2] = g[2];
        }

        true
    }
}

impl<'a, F: Fn() -> bool> NewtonProblem for NLP<'a, F> {
    fn num_hessian_non_zeros(&self) -> usize {
        self.body.energy_hessian_size()
    }
    fn hessian_indices(&mut self, rows: &mut [Index], cols: &mut [Index]) -> bool {
        let hess = self.body.energy_hessian();
        for (i, &MatrixElementTriplet { ref idx, .. }) in hess.iter().enumerate() {
            rows[i] = idx.row as Index;
            cols[i] = idx.col as Index;
        }

        true
    }
    fn hessian_values(&mut self, x: &[Number], vals: &mut [Number]) -> bool {
        self.update(x);

        let hess = self.body.energy_hessian();
        for (i, elem) in hess.iter().enumerate() {
            vals[i] = elem.val as Number;
        }

        true
    }
}
