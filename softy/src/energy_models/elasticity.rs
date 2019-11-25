use utils::soap::{Real, Matrix3, Vector3};
use geo::prim::Tetrahedron;

//mod tet_inv_nh;
mod tet_nh;
mod tet_snh;
//mod tri_nh;

pub use tet_nh::*;
pub use tet_snh::*;
//pub use tet_inv_nh::*;
//pub use tri_nh::*;

/// Tetrahedron energy interface. Abstracting over tet energies is useful for damping
/// implementations like Rayleigh damping which depend on the elasticity model used.
pub trait TetEnergy<T: Real> {
    /// Constructor accepts:
    /// `Dx`: the deformed shape matrix
    /// `DX_inv`: the undeformed shape matrix
    /// `volume`: volume of the tetrahedron
    /// `lambda` and `mu`: Lamé parameters
    #[allow(non_snake_case)]
    fn new(Dx: Matrix3<T>, DX_inv: Matrix3<T>, volume: T, lambda: T, mu: T) -> Self;

    /// Compute the deformation gradient `F` for this tet.
    fn deformation_gradient(&self) -> Matrix3<T>;

    /// Compute the deformation gradient differential `dF` for this tet.
    fn deformation_gradient_differential(&self, tet_dx: &Tetrahedron<T>) -> Matrix3<T>;

    /// Elastic strain energy per element.
    /// This is a helper function that computes the strain energy given shape matrices, which can
    /// be obtained from a tet and its reference configuration.
    fn elastic_energy(&self) -> T;

    /// Elastic energy gradient per element vertex.
    /// This is a helper function that computes the energy gradient given shape matrices, which can
    /// be obtained from a tet and its reference configuration.
    fn elastic_energy_gradient(&self) -> [Vector3<T>; 4];

    /// Elasticity Hessian per element. This is represented by a 4x4 block matrix of 3x3 matrices. The
    /// total matrix is a lower triangular 12x12 matrix. The blocks are specified in row-major
    /// order to be consistent with the 3x3 Matrices.
    fn elastic_energy_hessian(&self) -> [[Matrix3<T>; 4]; 4];

    /// Elasticity Hessian per element with respect to deformation gradient. This is a 3x3 matrix
    /// of 3x3 blocks.
    fn elastic_energy_deformation_hessian(&self) -> [[Matrix3<T>; 3]; 3] { [[Matrix3::zeros();3];3] }

    /// Elasticity Hessian*displacement product tranpose per element. Represented by a 3x3 matrix
    /// where row `i` produces the hessian product contribution for the vertex `i` within the
    /// current element.
    fn elastic_energy_hessian_product_transpose(&self, dx: &Tetrahedron<T>) -> Matrix3<T>;
}

/// This trait defines an accessor for an elastic energy model. Elastic objects can implement this
/// trait to have a unified method for getting an elastic energy model.
pub trait Elasticity<'a, E> {
    fn elasticity(&'a self) -> E;
}


#[cfg(test)]
mod test_utils {
    use super::*;
    use approx::*;
    use autodiff::F;
    use utils::soap::*;
    use geo::ops::*;

    #[allow(non_snake_case)]
    pub(crate) fn tet_energy_gradient_tester<E: TetEnergy<F>>() {
        let verts = vec![[0.0; 3], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]];
        let deformed_verts = vec![[0.0; 3], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 2.0]];
        let cell = [0, 1, 2, 3];

        let verts: Vec<_> = verts.iter().map(|&v| Tensor::new(v).map(|x| F::cst(x)).data).collect();
        let mut deformed_verts: Vec<_> = deformed_verts.iter().map(|&v| Tensor::new(v).map(|x| F::cst(x)).data).collect();

        let tet_x0 = Tetrahedron::from_indexed_slice(&cell, &verts);

        let df = {
            let tet_x1 = Tetrahedron::from_indexed_slice(&cell, &deformed_verts);
            let DX_inv = Matrix3::new(tet_x0.clone().shape_matrix()).inverse_transpose().unwrap();
            let Dx = Matrix3::new(tet_x1.clone().shape_matrix()).transpose();
            let energy = E::new(Dx, DX_inv, F::cst(1.0), F::cst(1.0), F::cst(1.0));
            energy.elastic_energy_gradient()
        };

        // Print Gradient derivative
        eprintln!("Gradient:");
        for wrt_vtx_idx in 0..4 {
            for j in 0..3 {
                eprintln!("{:10.2e}", df[wrt_vtx_idx].data[j].value());
            }
        }
        eprintln!("");

        eprintln!("Testing with autodiff gradient:");
        let mut success = true;
        for vtx_idx in 0..4 {
            for i in 0..3 {
                deformed_verts[vtx_idx][i] = F::var(deformed_verts[vtx_idx][i]);
                let tet_x1 = Tetrahedron::from_indexed_slice(&cell, &deformed_verts);
                let DX_inv = Matrix3::new(tet_x0.clone().shape_matrix()).inverse_transpose().unwrap();
                let Dx = Matrix3::new(tet_x1.shape_matrix()).transpose();
                let energy = E::new(Dx, DX_inv, F::cst(1.0), F::cst(1.0), F::cst(1.0));

                let f = energy.elastic_energy();
                eprintln!("(vtx, i, f) = ({}, {}, {:10.2e})", vtx_idx, i, f.deriv());
                success &= relative_eq!(df[vtx_idx].data[i].value(), f.deriv(), max_relative = 1e-7, epsilon = 1e-7);
                deformed_verts[vtx_idx][i] = F::cst(deformed_verts[vtx_idx][i]);
            }
        }
        assert!(success);
    }

    #[allow(non_snake_case)]
    pub(crate) fn tet_energy_hessian_tester<E: TetEnergy<F>>() {
        let verts = vec![[0.0; 3], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]];
        //let deformed_verts = vec![[0.2; 3], [0.01, 1.2, 0.01], [1.01, 0.2, 0.01], [0.01, 0.2, 2.0]];
        let deformed_verts = vec![[0.0; 3], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 2.0]];
        let cell = [0, 1, 2, 3];

        let verts: Vec<_> = verts.iter().map(|&v| Tensor::new(v).map(|x| F::cst(x)).data).collect();
        let mut deformed_verts: Vec<_> = deformed_verts.iter().map(|&v| Tensor::new(v).map(|x| F::cst(x)).data).collect();

        let tet_x0 = Tetrahedron::from_indexed_slice(&cell, &verts);

        let ddf = {
            let tet_x1 = Tetrahedron::from_indexed_slice(&cell, &deformed_verts);
            let DX_inv = Matrix3::new(tet_x0.clone().shape_matrix()).inverse_transpose().unwrap();
            let Dx = Matrix3::new(tet_x1.clone().shape_matrix()).transpose();
            let energy = E::new(Dx, DX_inv, F::cst(1.0), F::cst(1.0), F::cst(1.0));
            energy.elastic_energy_hessian()
        };

        // Print hessian
        for n in 0..4 {
            for i in 0..3 {
                for k in 0..4 {
                    for j in 0..3 {
                        eprint!("{:10.2e}", ddf[n][k][i][j].value());
                    }
                    eprint!("\t");
                }
                eprintln!("");
            }
        }
        eprintln!("");

        let mut success = true;
        let mut autodiff_h = [[Matrix3::zeros(); 4]; 4];

        for vtx_idx in 0..4 {
            for i in 0..3 {
                deformed_verts[vtx_idx][i] = F::var(deformed_verts[vtx_idx][i]);
                let tet_x1 = Tetrahedron::from_indexed_slice(&cell, &deformed_verts);
                let DX_inv = Matrix3::new(tet_x0.clone().shape_matrix()).inverse_transpose().unwrap();
                let Dx = Matrix3::new(tet_x1.shape_matrix()).transpose();
                let energy = E::new(Dx, DX_inv, F::cst(1.0), F::cst(1.0), F::cst(1.0));
                let df = energy.elastic_energy_gradient();
                dbg!(&vtx_idx);
                dbg!(&i);

                // Print Gradient derivative
                for wrt_vtx_idx in 0..4 {
                    for j in 0..3 {
                        autodiff_h[wrt_vtx_idx][vtx_idx].data[j][i] = df[wrt_vtx_idx].data[j].deriv();
                    }
                }

                for wrt_vtx_idx in 0..4 {
                    for j in 0..3 {
                        if vtx_idx < wrt_vtx_idx || (vtx_idx == wrt_vtx_idx && j >= i) {
                            success &= relative_eq!(ddf[wrt_vtx_idx][vtx_idx].data[j][i].value(), df[wrt_vtx_idx].data[j].deriv(), max_relative = 1e-7, epsilon = 1e-7);
                        }
                    }
                }
                deformed_verts[vtx_idx][i] = F::cst(deformed_verts[vtx_idx][i]);
            }
        }

        for n in 0..4 {
            for i in 0..3 {
                for k in 0..4 {
                    for j in 0..3 {
                        eprint!("{:10.2e}", autodiff_h[n][k][i][j]);
                    }
                    eprint!("\t");
                }
                eprintln!("");
            }
        }
        eprintln!("");
        assert!(success);
    }

    #[allow(non_snake_case)]
    pub(crate) fn tet_energy_hessian_product_tester<E: TetEnergy<f64>>() {
        let verts = vec![[0.0; 3], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]];
        let cell = [0, 1, 2, 3];

        let tet_x0 = Tetrahedron::from_indexed_slice(&cell, &verts);

        let DX_inv = Matrix3::new(tet_x0.clone().shape_matrix()).inverse_transpose().unwrap();
        let Dx = Matrix3::new(tet_x0.clone().shape_matrix()).transpose();
        let energy = E::new(Dx, DX_inv, 1.0, 1.0, 1.0);
        let ddf = energy.elastic_energy_hessian();

        // Print hessian
        for n in 0..4 {
            for i in 0..3 {
                for k in 0..4 {
                    for j in 0..3 {
                        eprint!("{:10.2e}", ddf[n][k][i][j]);
                    }
                    eprint!("\t");
                }
                eprintln!("");
            }
        }
        eprintln!("");

        let mut d_verts = vec![[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]];

        for wrt_vtx in 0..3 {
            for wrt_i in 0..3 {
                d_verts[wrt_vtx][wrt_i] = 1.0;
                let tet_dx = Tetrahedron::from_indexed_slice(&cell, &d_verts);

                let h = energy.elastic_energy_hessian_product_transpose(&tet_dx);
                // Print hessian flat
                for wrt_vtx_idx in 0..3 {
                    for j in 0..3 {
                        eprintln!("{:10.2e}", h[wrt_vtx_idx][j]);
                    }
                }
                eprintln!("");
                for vtx in 0..3 {
                    for i in 0..3 {
                        if wrt_vtx < vtx || (wrt_vtx == vtx && i >= wrt_i) {
                            assert_relative_eq!(h[vtx][i], ddf[vtx][wrt_vtx][i][wrt_i], max_relative = 1e-7, epsilon = 1e-7);
                        }
                    }
                }
                d_verts[wrt_vtx][wrt_i] = 0.0;
            }
        }
    }
}
