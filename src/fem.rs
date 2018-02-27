use nodal_fem_nlp::NLP;
use energy::*;
use ipopt::Ipopt;
use geo::math::{Matrix3, Vector3};
use geo::topology::*;
use geo::prim::Tetrahedron;
use geo::mesh::{self, attrib, Attrib};
use geo::ops::{ShapeMatrix, Volume};

pub type TetMesh = mesh::TetMesh<f64>;
pub type Tet = Tetrahedron<f64>;

/// Get reference tetrahedron.
/// This routine assumes that there is a vertex attribute called `ref` of type `[f64;3]`.
pub fn ref_tet(tetmesh: &TetMesh, indices: &[usize]) -> Tet {
    let attrib = tetmesh.attrib::<VertexIndex>("ref").unwrap();
    Tet {
        a: (attrib.get::<[f64; 3], _>(indices[0]).unwrap()).into(),
        b: (attrib.get::<[f64; 3], _>(indices[1]).unwrap()).into(),
        c: (attrib.get::<[f64; 3], _>(indices[2]).unwrap()).into(),
        d: (attrib.get::<[f64; 3], _>(indices[3]).unwrap()).into(),
    }
}

pub fn run<F>(mesh: &mut TetMesh, check_interrupt: F) -> Result<(), Error>
where
    F: Fn() -> bool,
{
    // Prepare tet mesh for simulation.
    if !mesh.attrib_exists::<VertexIndex>("ref") {
        let verts = mesh.vertices();
        mesh.add_attrib_data::<_, VertexIndex>("ref", verts)?;
    } else {
        // Ensure that the existing reference attribute is of the right type.
        let ref_a = mesh.attrib::<VertexIndex>("ref")?;
        ref_a.check::<[f64; 3]>()?
    }

    // Remove attributes, to clear up names we will need.
    mesh.remove_attrib::<CellIndex>("ref_volume").ok();
    mesh.remove_attrib::<CellIndex>("ref_shape_mtx_inv").ok();
    mesh.remove_attrib::<VertexIndex>("force").ok();

    // Create vertex force attribute
    mesh.add_attrib::<_, VertexIndex>("force", [0.0; 3])?;

    {
        // compute reference element signed volumes
        let ref_volumes: Vec<f64> = mesh.cell_iter()
            .map(|cell| ref_tet(mesh, cell).signed_volume())
            .collect();
        if ref_volumes.iter().find(|&&x| x <= 0.0).is_some() {
            return Err(Error::InvertedReferenceElement);
        }
        mesh.add_attrib_data::<_, CellIndex>("ref_volume", ref_volumes)?;
    }

    {
        // compute reference shape matrix inverses
        let ref_shape_mtx_inverses: Vec<Matrix3<f64>> = mesh.cell_iter()
            .map(|cell| {
                let ref_shape_matrix = ref_tet(mesh, cell).shape_matrix();
                // We assume that ref_shape_matrices are non-singular.
                ref_shape_matrix.inverse().unwrap()
            })
            .collect();
        mesh.add_attrib_data::<_, CellIndex>("ref_shape_mtx_inv", ref_shape_mtx_inverses)?;
    }

    let x = {
        let nlp = NLP::new(mesh, check_interrupt);
        let mut ipopt = Ipopt::new_newton(nlp);

        ipopt.set_option("tol", 1e-9);
        ipopt.set_option("acceptable_tol", 1e-8);
        ipopt.set_option("max_iter", 800);
        ipopt.set_option("mu_strategy", "adaptive");
        ipopt.set_option("sb", "yes"); // removes the Ipopt welcome message
        ipopt.set_option("print_level", 0);
        //ipopt.set_option("derivative_test", "second-order");
        //ipopt.set_option("derivative_test_tol", 1e-4);
        //ipopt.set_option("point_perturbation_radius", 0.01);
        ipopt.set_option("nlp_scaling_max_gradient", 1e-5);
        ipopt.set_intermediate_callback(Some(NLP::intermediate_cb));
        let (_r, _obj) = ipopt.solve();
        ipopt.solution().to_vec()
    };
    for (i, v) in mesh.vertex_iter_mut().enumerate() {
        *v = [x[3 * i + 0], x[3 * i + 1], x[3 * i + 2]];
    }

    Ok(())
}

/// Define energy for Neohookean materials.
impl Energy<f64> for TetMesh {
    #[allow(non_snake_case)]
    fn energy(&self) -> f64 {
        let mu = 5.4;
        let lambda = 263.1;
        self.attrib_iter::<f64, CellIndex>("ref_volume")
            .unwrap()
            .zip(
                self.attrib_iter::<Matrix3<f64>, CellIndex>("ref_shape_mtx_inv")
                    .unwrap(),
            )
            .zip(self.tet_iter())
            .map(|((&vol, &Dm_inv), tet)| {
                let F = tet.shape_matrix() * Dm_inv;
                let I = F.clone().map(|x| x * x).sum(); // tr(F^TF)
                let J = F.determinant();
                if J <= 0.0 {
                    ::std::f64::INFINITY
                } else {
                    let logJ = J.ln();
                    vol * (0.5 * mu * (I - 3.0) - mu * logJ + 0.5 * lambda * logJ * logJ)
                }
            })
            .sum()
    }

    #[allow(non_snake_case)]
    fn energy_gradient(&self) -> Vec<Vector3<f64>> {
        let mu = 5.4;
        let lambda = 263.1;
        let force_iter = self.attrib_iter::<f64, CellIndex>("ref_volume")
            .unwrap()
            .zip(
                self.attrib_iter::<Matrix3<f64>, CellIndex>("ref_shape_mtx_inv")
                    .unwrap(),
            )
            .zip(self.tet_iter())
            .map(|((&vol, &Dm_inv), tet)| {
                let F = tet.shape_matrix() * Dm_inv;
                let J = F.determinant();
                if J <= 0.0 {
                    Matrix3::zeros()
                } else {
                    let F_inv_tr = F.inverse_transpose().unwrap();
                    let logJ = J.ln();
                    vol * (mu * F + (lambda * logJ - mu) * F_inv_tr) * Dm_inv.transpose()
                }
            });

        // Transfer forces from cell-vertices to vertices themeselves
        let mut vtx_grad = Vec::new();
        vtx_grad.resize(self.num_verts(), Vector3::zeros());
        for (cell, grad) in self.cell_iter().zip(force_iter) {
            for i in 0..3 {
                vtx_grad[cell[i]] += grad[i];
                vtx_grad[cell[3]] -= grad[i];
            }
        }
        vtx_grad
    }

    fn energy_hessian_size(&self) -> usize {
        78 * self.num_cells() // There are 4*6 + 3*9*4/2 = 78 triplets per tet (overestimate)
    }

    #[allow(non_snake_case)]
    fn energy_hessian(&self) -> Vec<MatrixElementTriplet<f64>> {
        let mu = 5.4;
        let lambda = 263.1;
        let hess_iter = self.attrib_iter::<f64, CellIndex>("ref_volume")
            .unwrap()
            .zip(
                self.attrib_iter::<Matrix3<f64>, CellIndex>("ref_shape_mtx_inv")
                    .unwrap(),
            )
            .zip(self.cell_iter())
            .zip(self.tet_iter());

        let mut hess = Vec::with_capacity(self.energy_hessian_size());

        {
            let mut push_elem = |row, col, val| {
                hess.push(MatrixElementTriplet::new(row, col, val));
            };

            for (((&vol, &Dm_inv), cell), tet) in hess_iter {
                let Ds = tet.shape_matrix();
                let F = Ds * Dm_inv;
                let J = F.determinant();
                if J > 0.0 {
                    let A = Dm_inv * Dm_inv.transpose();
                    // Theoretically we known Ds is invertible since F is, but it could have
                    // numerical differences.
                    let Ds_inv_tr = match Ds.inverse_transpose() {
                        Some(inv) => inv,
                        None => break,
                    };

                    let alpha = mu - lambda * J.ln();

                    // Off-diagonal elements
                    for col in 0..3 {
                        for row in 0..3 {
                            let mut last_hess = [0.0; 4];
                            for k in 0..3 {
                                // which vertex
                                let mut last_wrt_hess = 0.0;
                                for n in 0..3 {
                                    // with respect to which vertex
                                    let c_lambda = lambda * Ds_inv_tr[n][row] * Ds_inv_tr[k][col];
                                    let c_alpha = alpha * Ds_inv_tr[n][col] * Ds_inv_tr[k][row];
                                    let mut h = vol * (c_alpha + c_lambda);
                                    if col == row {
                                        h += vol * mu * A[k][n];
                                    }
                                    last_wrt_hess -= h;
                                    last_hess[n] -= h;

                                    // skip upper trianglar part of the global hessian.
                                    if (cell[n] == cell[k] && row >= col) || cell[n] > cell[k] {
                                        push_elem(3 * cell[n] + row, 3 * cell[k] + col, h);
                                    }
                                }

                                // with respect to last vertex
                                last_hess[3] -= last_wrt_hess;
                                if cell[3] > cell[k] {
                                    push_elem(3 * cell[3] + row, 3 * cell[k] + col, last_wrt_hess);
                                }
                            }

                            // last vertex
                            for n in 0..4 {
                                // with respect to which vertex
                                if (cell[n] == cell[3] && row >= col) || cell[n] > cell[3] {
                                    push_elem(3 * cell[n] + row, 3 * cell[3] + col, last_hess[n]);
                                }
                            }
                        }
                    }
                }
            }
        }
        hess
    }
}

#[derive(Debug)]
pub enum Error {
    AttribError(attrib::Error),
    InvertedReferenceElement,
}

impl From<attrib::Error> for Error {
    fn from(err: attrib::Error) -> Error {
        Error::AttribError(err)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn simple_tet_test() {
        let verts = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 0.0, 2.0],
            [1.0, 0.0, 2.0],
        ];
        let indices = vec![5, 2, 4, 0, 3, 2, 5, 0, 1, 0, 3, 5];
        let mut mesh = TetMesh::new(verts, indices);

        let ref_verts = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
        ];

        mesh.add_attrib_data::<_, VertexIndex>("ref", ref_verts)
            .ok();

        assert!(run(&mut mesh, || true).is_ok());
    }
}
