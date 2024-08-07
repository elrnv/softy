use geo::prim::{Tetrahedron, Triangle};
use na::Scalar;
use tensr::{Matrix2, Matrix2x3, Matrix3, Vector3};

//mod tet_inv_nh;
mod tet_nh;
mod tet_snh;
mod tetsolid_nh;
mod tri_nh;
mod trishell_nh;

pub use tet_nh::*;
pub use tet_snh::*;
//pub use tet_inv_nh::*;
pub use tetsolid_nh::*;
pub use tri_nh::*;
pub use trishell_nh::*;

/// Element energy interface.
pub trait LinearElementEnergy<T> {
    type Element;
    type ShapeMatrix;
    type RefShapeMatrix;
    type Gradient;
    type Hessian;

    /// Constructor accepts:
    /// `Dx`: the deformed shape matrix
    /// `DX_inv`: the undeformed shape matrix
    /// `volume`: volume of the tetrahedron
    /// `lambda` and `mu`: Lamé parameters
    #[allow(non_snake_case)]
    fn new(
        Dx: Self::ShapeMatrix,
        DX_inv: Self::RefShapeMatrix,
        volume: T,
        lambda: T,
        mu: T,
    ) -> Self;

    /// Compute the deformation gradient differential `dF` for this element.
    fn deformation_gradient_differential(&self, dx: &Self::Element) -> Self::ShapeMatrix;

    /// Elastic strain energy per element.
    /// This is a helper function that computes the strain energy given shape matrices, which can
    /// be obtained from a tet and its reference configuration.
    fn energy(&self) -> T;

    /// Elastic energy gradient per element vertex.
    /// This is a helper function that computes the energy gradient given shape matrices, which can
    /// be obtained from a tet and its reference configuration.
    fn energy_gradient(&self) -> Self::Gradient;

    /// Elasticity Hessian per element. This is represented by a 4x4 block matrix of 3x3 matrices. The
    /// total matrix is a lower triangular 12x12 matrix. The blocks are specified in row-major
    /// order to be consistent with the 3x3 Matrices.
    fn energy_hessian(&self) -> Self::Hessian;

    /// Elasticity Hessian*displacement product tranpose per element. Represented by a 3x3 matrix
    /// where row `i` produces the hessian product contribution for the vertex `i` within the
    /// current element.
    fn energy_hessian_product_transpose(&self, dx: &Self::Element) -> Self::ShapeMatrix;
}

pub trait TetEnergy<T: Scalar>:
    LinearElementEnergy<
    T,
    Element = Tetrahedron<T>,
    ShapeMatrix = Matrix3<T>,
    RefShapeMatrix = Matrix3<T>,
    Gradient = [Vector3<T>; 4],
    Hessian = [[Matrix3<T>; 4]; 4],
>
{
}
impl<T: Scalar, E> TetEnergy<T> for E where
    E: LinearElementEnergy<
        T,
        Element = Tetrahedron<T>,
        ShapeMatrix = Matrix3<T>,
        RefShapeMatrix = Matrix3<T>,
        Gradient = [Vector3<T>; 4],
        Hessian = [[Matrix3<T>; 4]; 4],
    >
{
}
pub trait TriEnergy<T: Scalar>:
    LinearElementEnergy<
    T,
    Element = Triangle<T>,
    ShapeMatrix = Matrix2x3<T>,
    RefShapeMatrix = Matrix2<T>,
    Gradient = [Vector3<T>; 3],
    Hessian = [[Matrix3<T>; 3]; 3],
>
{
}
impl<T: Scalar, E> TriEnergy<T> for E where
    E: LinearElementEnergy<
        T,
        Element = Triangle<T>,
        ShapeMatrix = Matrix2x3<T>,
        RefShapeMatrix = Matrix2<T>,
        Gradient = [Vector3<T>; 3],
        Hessian = [[Matrix3<T>; 3]; 3],
    >
{
}

/// This trait defines an accessor for an elastic energy model. Elastic objects can implement this
/// trait to have a unified method for getting an elastic energy model.
pub trait Elasticity<'a, E> {
    fn elasticity(&'a self) -> E;
}

#[cfg(test)]
mod test_utils {
    use super::*;
    use crate::objects::trishell::TriangleElements;
    use approx::*;
    use autodiff::F1 as F;
    use geo::ops::*;

    #[allow(non_snake_case)]
    pub(crate) fn tet_energy_gradient_tester<E: TetEnergy<F>>() {
        let verts = vec![[0.0; 3], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]];
        let deformed_verts = vec![[0.0; 3], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 2.0]];
        let cell = [0, 1, 2, 3];

        let verts: Vec<_> = verts
            .iter()
            .map(|&v| Vector3::new(v).mapd(|x| F::cst(x)))
            .collect();
        let mut deformed_verts: Vec<_> = deformed_verts
            .iter()
            .map(|&v| Vector3::new(v).mapd(|x| F::cst(x)))
            .collect();

        let tet_x0 = Tetrahedron::from_indexed_slice(&cell, &verts);

        let df = {
            let tet_x1 = Tetrahedron::from_indexed_slice(&cell, &deformed_verts);
            let DX_inv = Matrix3::new(tet_x0.clone().shape_matrix())
                .inverse()
                .unwrap();
            let Dx = Matrix3::new(tet_x1.clone().shape_matrix());
            let energy = E::new(Dx, DX_inv, F::cst(1.0), F::cst(1.0), F::cst(1.0));
            energy.energy_gradient()
        };

        // Print Gradient derivative
        eprintln!("Gradient:");
        for wrt_vtx_idx in 0..4 {
            for j in 0..3 {
                eprintln!("{:10.2e}", df[wrt_vtx_idx][j].value());
            }
        }
        eprintln!();

        eprintln!("Testing with autodiff gradient:");
        let mut success = true;
        for vtx_idx in 0..4 {
            for i in 0..3 {
                deformed_verts[vtx_idx][i] = F::var(deformed_verts[vtx_idx][i]);
                let tet_x1 = Tetrahedron::from_indexed_slice(&cell, &deformed_verts);
                let DX_inv = Matrix3::new(tet_x0.clone().shape_matrix())
                    .inverse()
                    .unwrap();
                let Dx = Matrix3::new(tet_x1.shape_matrix());
                let energy = E::new(Dx, DX_inv, F::cst(1.0), F::cst(1.0), F::cst(1.0));

                let f = energy.energy();
                eprintln!("(vtx, i, f) = ({}, {}, {:10.2e})", vtx_idx, i, f.deriv());
                success &= relative_eq!(
                    df[vtx_idx][i].value(),
                    f.deriv(),
                    max_relative = 1e-7,
                    epsilon = 1e-7
                );
                deformed_verts[vtx_idx][i] = F::cst(deformed_verts[vtx_idx][i]);
            }
        }
        assert!(success);
    }

    #[allow(non_snake_case)]
    pub(crate) fn tet_energy_hessian_tester<E: TetEnergy<F>>() {
        let verts = vec![[0.0; 3], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]];
        let deformed_verts = vec![
            [0.2; 3],
            [0.01, 1.2, 0.01],
            [1.01, 0.2, 0.01],
            [0.01, 0.2, 2.0],
        ];
        //let deformed_verts = vec![[0.0; 3], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 2.0]];
        let cell = [0, 1, 2, 3];

        let verts: Vec<_> = verts
            .iter()
            .map(|&v| Vector3::new(v).mapd(|x| F::cst(x)))
            .collect();
        let mut deformed_verts: Vec<_> = deformed_verts
            .iter()
            .map(|&v| Vector3::new(v).mapd(|x| F::cst(x)))
            .collect();

        let tet_x0 = Tetrahedron::from_indexed_slice(&cell, &verts);

        let ddf = {
            let tet_x1 = Tetrahedron::from_indexed_slice(&cell, &deformed_verts);
            let DX_inv = Matrix3::new(tet_x0.clone().shape_matrix())
                .inverse()
                .unwrap();
            let Dx = Matrix3::new(tet_x1.clone().shape_matrix());
            let energy = E::new(Dx, DX_inv, F::cst(1.0), F::cst(1.0), F::cst(1.0));
            energy.energy_hessian()
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
                eprintln!();
            }
        }
        eprintln!();

        let mut success = true;
        let mut autodiff_h = [[Matrix3::<f64>::zeros(); 4]; 4];

        for vtx_idx in 0..4 {
            for i in 0..3 {
                deformed_verts[vtx_idx][i] = F::var(deformed_verts[vtx_idx][i]);
                let tet_x1 = Tetrahedron::from_indexed_slice(&cell, &deformed_verts);
                let DX_inv = Matrix3::new(tet_x0.clone().shape_matrix())
                    .inverse()
                    .unwrap();
                let Dx = Matrix3::new(tet_x1.shape_matrix());
                let energy = E::new(Dx, DX_inv, F::cst(1.0), F::cst(1.0), F::cst(1.0));
                let df = energy.energy_gradient();
                dbg!(&vtx_idx);
                dbg!(&i);

                // Print Gradient derivative
                for wrt_vtx_idx in 0..4 {
                    for j in 0..3 {
                        autodiff_h[wrt_vtx_idx][vtx_idx][j][i] = df[wrt_vtx_idx][j].deriv();
                    }
                }

                for wrt_vtx_idx in 0..4 {
                    for j in 0..3 {
                        if vtx_idx < wrt_vtx_idx || (vtx_idx == wrt_vtx_idx && j >= i) {
                            success &= relative_eq!(
                                ddf[wrt_vtx_idx][vtx_idx][j][i].value(),
                                df[wrt_vtx_idx][j].deriv(),
                                max_relative = 1e-7,
                                epsilon = 1e-7
                            );
                        }
                    }
                }
                deformed_verts[vtx_idx][i] = F::cst(deformed_verts[vtx_idx][i]);
            }
        }

        for n in 0..4 {
            for i in 0..3 {
                for k in 0usize..4 {
                    for j in 0usize..3 {
                        eprint!("{:10.2e}", autodiff_h[n][k][i][j]);
                    }
                    eprint!("\t");
                }
                eprintln!();
            }
        }
        eprintln!();
        assert!(success);
    }

    #[allow(non_snake_case)]
    pub(crate) fn tet_energy_hessian_product_tester<E: TetEnergy<f64>>() {
        let verts = vec![[0.0; 3], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]];
        let deformed_verts = vec![
            [0.2; 3],
            [0.01, 1.2, 0.01],
            [1.01, 0.2, 0.01],
            [0.01, 0.2, 2.0],
        ];
        let cell = [0, 1, 2, 3];

        let tet_x0 = Tetrahedron::from_indexed_slice(&cell, &verts);

        let DX_inv = Matrix3::new(tet_x0.clone().shape_matrix())
            .inverse()
            .unwrap();
        let tet_x1 = Tetrahedron::from_indexed_slice(&cell, &deformed_verts);
        let Dx = Matrix3::new(tet_x1.clone().shape_matrix());
        let energy = E::new(Dx, DX_inv, 1.0, 1.0, 1.0);
        let ddf = energy.energy_hessian();

        // Print hessian
        for n in 0..4 {
            for i in 0..3 {
                for k in 0..4 {
                    for j in 0..3 {
                        eprint!("{:10.2e}", ddf[n][k][i][j]);
                    }
                    eprint!("\t");
                }
                eprintln!();
            }
        }
        eprintln!();

        let mut d_verts = vec![
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ];

        for wrt_vtx in 0..3 {
            for wrt_i in 0..3 {
                d_verts[wrt_vtx][wrt_i] = 1.0;
                let tet_dx = Tetrahedron::from_indexed_slice(&cell, &d_verts);

                let h = energy.energy_hessian_product_transpose(&tet_dx);
                // Print hessian flat
                for wrt_vtx_idx in 0..3 {
                    for j in 0..3 {
                        eprintln!("{:10.2e}", h[wrt_vtx_idx][j]);
                    }
                }
                eprintln!();
                for vtx in 0..3 {
                    for i in 0..3 {
                        if wrt_vtx < vtx || (wrt_vtx == vtx && i >= wrt_i) {
                            assert_relative_eq!(
                                h[vtx][i],
                                ddf[vtx][wrt_vtx][i][wrt_i],
                                max_relative = 1e-7,
                                epsilon = 1e-7
                            );
                        }
                    }
                }
                d_verts[wrt_vtx][wrt_i] = 0.0;
            }
        }
    }

    #[allow(non_snake_case)]
    pub(crate) fn tri_energy_gradient_tester<E: TriEnergy<F>>() {
        let verts = vec![[0.0; 3], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]];
        let deformed_verts = vec![[0.0; 3], [0.0, 1.0, 0.0], [2.0, 0.0, 0.0]];
        let face = [0, 1, 2];

        let verts: Vec<_> = verts
            .iter()
            .map(|&v| Vector3::new(v).mapd(|x| F::cst(x)))
            .collect();
        let mut deformed_verts: Vec<_> = deformed_verts
            .iter()
            .map(|&v| Vector3::new(v).mapd(|x| F::cst(x)))
            .collect();

        let tri_x0 = Triangle::from_indexed_slice(&face, &verts);

        let df = {
            let tri_x1 = Triangle::from_indexed_slice(&face, &deformed_verts);
            let DX_inv = TriangleElements::isotropic_tri_shape_matrix(Matrix2x3::new(
                tri_x0.clone().shape_matrix(),
            ))
            .inverse()
            .unwrap();
            let Dx = Matrix2x3::new(tri_x1.clone().shape_matrix());
            let energy = E::new(Dx, DX_inv, F::cst(1.0), F::cst(1.0), F::cst(1.0));
            energy.energy_gradient()
        };

        // Print Gradient derivative
        eprintln!("Gradient:");
        for wrt_vtx_idx in 0..3 {
            for j in 0..3 {
                eprintln!("{:10.2e}", df[wrt_vtx_idx][j].value());
            }
        }
        eprintln!();

        eprintln!("Testing with autodiff gradient:");
        let mut success = true;
        for vtx_idx in 0..3 {
            for i in 0..3 {
                deformed_verts[vtx_idx][i] = F::var(deformed_verts[vtx_idx][i]);
                let tri_x1 = Triangle::from_indexed_slice(&face, &deformed_verts);
                let DX_inv = TriangleElements::isotropic_tri_shape_matrix(Matrix2x3::new(
                    tri_x0.clone().shape_matrix(),
                ))
                .inverse()
                .unwrap();
                let Dx = Matrix2x3::new(tri_x1.shape_matrix());
                let energy = E::new(Dx, DX_inv, F::cst(1.0), F::cst(1.0), F::cst(1.0));

                let f = energy.energy();
                eprintln!("(vtx, i, f) = ({}, {}, {:10.2e})", vtx_idx, i, f.deriv());
                success &= relative_eq!(
                    df[vtx_idx][i].value(),
                    f.deriv(),
                    max_relative = 1e-7,
                    epsilon = 1e-7
                );
                deformed_verts[vtx_idx][i] = F::cst(deformed_verts[vtx_idx][i]);
            }
        }
        assert!(success);
    }

    #[allow(non_snake_case)]
    pub(crate) fn tri_energy_hessian_tester<E: TriEnergy<F>>() {
        let verts = vec![[0.0; 3], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]];
        let deformed_verts = vec![[0.2; 3], [0.01, 1.2, 0.01], [2.01, 0.2, 0.01]];
        //let deformed_verts = vec![[0.0; 3], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 2.0]];
        let face = [0, 1, 2];

        let verts: Vec<_> = verts
            .iter()
            .map(|&v| Vector3::new(v).mapd(|x| F::cst(x)))
            .collect();
        let mut deformed_verts: Vec<_> = deformed_verts
            .iter()
            .map(|&v| Vector3::new(v).mapd(|x| F::cst(x)))
            .collect();

        let tri_x0 = Triangle::from_indexed_slice(&face, &verts);

        let ddf = {
            let tri_x1 = Triangle::from_indexed_slice(&face, &deformed_verts);
            let DX_inv = TriangleElements::isotropic_tri_shape_matrix(Matrix2x3::new(
                tri_x0.clone().shape_matrix(),
            ))
            .inverse()
            .unwrap();
            let Dx = Matrix2x3::new(tri_x1.clone().shape_matrix());
            let energy = E::new(Dx, DX_inv, F::cst(1.0), F::cst(1.0), F::cst(1.0));
            energy.energy_hessian()
        };

        // Print hessian
        for n in 0..3 {
            for i in 0..3 {
                for k in 0..3 {
                    for j in 0..3 {
                        eprint!("{:10.2e}", ddf[n][k][i][j].value());
                    }
                    eprint!("\t");
                }
                eprintln!();
            }
        }
        eprintln!();

        let mut success = true;
        let mut autodiff_h = [[Matrix3::<f64>::zeros(); 3]; 3];

        for vtx_idx in 0..3 {
            for i in 0..3 {
                deformed_verts[vtx_idx][i] = F::var(deformed_verts[vtx_idx][i]);
                let tri_x1 = Triangle::from_indexed_slice(&face, &deformed_verts);
                let DX_inv = TriangleElements::isotropic_tri_shape_matrix(Matrix2x3::new(
                    tri_x0.clone().shape_matrix(),
                ))
                .inverse()
                .unwrap();
                let Dx = Matrix2x3::new(tri_x1.shape_matrix());
                let energy = E::new(Dx, DX_inv, F::cst(1.0), F::cst(1.0), F::cst(1.0));
                let df = energy.energy_gradient();
                dbg!(&vtx_idx);
                dbg!(&i);

                // Print Gradient derivative
                for wrt_vtx_idx in 0..3 {
                    for j in 0..3 {
                        autodiff_h[wrt_vtx_idx][vtx_idx][j][i] = df[wrt_vtx_idx][j].deriv();
                    }
                }

                for wrt_vtx_idx in 0..3 {
                    for j in 0..3 {
                        if vtx_idx < wrt_vtx_idx || (vtx_idx == wrt_vtx_idx && j >= i) {
                            success &= relative_eq!(
                                ddf[wrt_vtx_idx][vtx_idx][j][i].value(),
                                df[wrt_vtx_idx][j].deriv(),
                                max_relative = 1e-7,
                                epsilon = 1e-7
                            );
                        }
                    }
                }
                deformed_verts[vtx_idx][i] = F::cst(deformed_verts[vtx_idx][i]);
            }
        }

        for n in 0..3 {
            for i in 0..3 {
                for k in 0usize..3 {
                    for j in 0usize..3 {
                        eprint!("{:10.2e}", autodiff_h[n][k][i][j]);
                    }
                    eprint!("\t");
                }
                eprintln!();
            }
        }
        eprintln!();
        assert!(success);
    }

    #[allow(non_snake_case)]
    pub(crate) fn tri_energy_hessian_product_tester<E: TriEnergy<f64>>() {
        let verts = vec![[0.0; 3], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]];
        let deformed_verts = vec![[0.2; 3], [0.01, 1.2, 0.01], [2.01, 0.2, 0.01]];
        let face = [0, 1, 2];

        let tri_x0 = Triangle::from_indexed_slice(&face, &verts);

        let DX_inv = TriangleElements::isotropic_tri_shape_matrix(Matrix2x3::new(
            tri_x0.clone().shape_matrix(),
        ))
        .inverse()
        .unwrap();
        let tri_x1 = Triangle::from_indexed_slice(&face, &deformed_verts);
        let Dx = Matrix2x3::new(tri_x1.clone().shape_matrix());
        let energy = E::new(Dx, DX_inv, 1.0, 1.0, 1.0);
        let ddf = energy.energy_hessian();

        // Print hessian
        for n in 0..3 {
            for i in 0..3 {
                for k in 0..3 {
                    for j in 0..3 {
                        eprint!("{:10.2e}", ddf[n][k][i][j]);
                    }
                    eprint!("\t");
                }
                eprintln!();
            }
        }
        eprintln!();

        let mut d_verts = vec![[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]];

        for wrt_vtx in 0..2 {
            for wrt_i in 0..3 {
                d_verts[wrt_vtx][wrt_i] = 1.0;
                let tri_dx = Triangle::from_indexed_slice(&face, &d_verts);

                let h = energy.energy_hessian_product_transpose(&tri_dx);
                // Print hessian flat
                for wrt_vtx_idx in 0..2 {
                    for j in 0..3 {
                        eprintln!("{:10.2e}", h[wrt_vtx_idx][j]);
                    }
                }
                eprintln!();
                for vtx in 0..2 {
                    for i in 0..3 {
                        if wrt_vtx < vtx || (wrt_vtx == vtx && i >= wrt_i) {
                            assert_relative_eq!(
                                h[vtx][i],
                                ddf[vtx][wrt_vtx][i][wrt_i],
                                max_relative = 1e-7,
                                epsilon = 1e-7
                            );
                        }
                    }
                }
                d_verts[wrt_vtx][wrt_i] = 0.0;
            }
        }
    }
}
