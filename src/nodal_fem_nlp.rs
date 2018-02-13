use energy::{Energy};
use geo::topology::*;
use TetMesh;
use ipopt::*;

pub struct NLP<'a> {
    pub body: &'a mut TetMesh,
}

impl<'a> NLP<'a> {
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
        _ls_trials: Index) -> bool
    {
        true
    }
    pub fn update(&mut self, x: &[Number]) {
        for (i, v) in self.body.vertex_iter_mut().enumerate() {
            *v = [x[3*i + 0], x[3*i + 1], x[3*i + 2]];
        }
    }
}

impl<'a> BasicProblem for NLP<'a> {
    fn num_variables(&self) -> usize {
        self.body.num_verts()*3
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
        self.body.vertex_iter().flat_map(|x| x.iter().cloned()).collect()
    }

    fn objective(&mut self, x: &[Number], obj: &mut Number) -> bool {
        self.update(x);

        *obj = self.body.energy();
        true
    }

    fn objective_grad(&mut self, x: &[Number], grad_f: &mut [Number]) -> bool {
        self.update(x);

        for (i, g) in self.body.energy_gradient().iter().enumerate() {
            grad_f[3*i + 0] = g[0];
            grad_f[3*i + 1] = g[1];
            grad_f[3*i + 2] = g[2];
        }

        true
    }
}


