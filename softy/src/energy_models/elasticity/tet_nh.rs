//! Neo-Hookean energy model for tetrahedral meshes.

//use std::path::Path;
//use geo::io::save_tetmesh;
use super::LinearElementEnergy;
use crate::Real;
use geo::ops::*;
use geo::prim::Tetrahedron;
use num_traits::{Float, Zero};
use tensr::*;
use unroll::unroll_for_loops;

/// Per-tetrahedron Neo-Hookean energy model. This struct stores conveniently precomputed values
/// for tet energy computation. It encapsulates tet specific energy computation.
#[allow(non_snake_case)]
pub struct NeoHookeanTetEnergy<T: Real> {
    Dx: Matrix3<T>,
    DX_inv: Matrix3<T>,
    volume: T,
    lambda: T,
    mu: T,
}

impl<T: Real> NeoHookeanTetEnergy<T> {
    /// Compute the deformation gradient `F` for this tet.
    #[allow(non_snake_case)]
    #[inline]
    pub fn deformation_gradient(&self) -> Matrix3<T> {
        self.DX_inv * self.Dx
    }
}

impl<T: Real> LinearElementEnergy<T> for NeoHookeanTetEnergy<T> {
    type Element = Tetrahedron<T>;
    type ShapeMatrix = Matrix3<T>;
    type RefShapeMatrix = Matrix3<T>;
    type Gradient = [Vector3<T>; 4];
    type Hessian = [[Matrix3<T>; 4]; 4];

    #[allow(non_snake_case)]
    fn new(Dx: Matrix3<T>, DX_inv: Matrix3<T>, volume: T, lambda: T, mu: T) -> Self {
        NeoHookeanTetEnergy {
            Dx,
            DX_inv,
            volume,
            lambda,
            mu,
        }
    }

    /// Compute the deformation gradient differential `dF` for this tet.
    #[allow(non_snake_case)]
    #[inline]
    fn deformation_gradient_differential(&self, tet_dx: &Tetrahedron<T>) -> Matrix3<T> {
        // Build differential dDx
        let dDx = Matrix3::new(tet_dx.shape_matrix());
        self.DX_inv * dDx
    }

    /// Elastic strain energy per element.
    ///
    /// This is a helper function that computes the strain energy given shape matrices, which can
    /// be obtained from a tet and its reference configuration.
    #[allow(non_snake_case)]
    #[inline]
    fn energy(&self) -> T {
        let NeoHookeanTetEnergy {
            volume, mu, lambda, ..
        } = *self;
        let F = self.deformation_gradient();
        let I = F.frob_norm_squared(); // tr(F^TF)
        let J = F.determinant();
        if J <= T::zero() {
            T::infinity()
        } else {
            let logJ = Float::ln(J);
            let half = T::from(0.5).unwrap();
            volume
                * (half * mu * (I - T::from(3.0).unwrap()) - mu * logJ
                    + half * lambda * logJ * logJ)
        }
    }

    /// Elastic energy gradient per element vertex.
    ///
    /// This is a helper function that computes the energy gradient given shape matrices, which can
    /// be obtained from a tet and its reference configuration.
    #[allow(non_snake_case)]
    #[inline]
    fn energy_gradient(&self) -> [Vector3<T>; 4] {
        let NeoHookeanTetEnergy {
            DX_inv,
            volume,
            mu,
            lambda,
            ..
        } = *self;
        let F = self.deformation_gradient();
        let J = F.determinant();
        if J <= T::zero() {
            [Vector3::zero(); 4]
        } else {
            let F_inv_tr = F.inverse_transpose().unwrap();
            let logJ = Float::ln(J);
            let H = DX_inv.transpose() * (F * mu + F_inv_tr * (lambda * logJ - mu)) * volume;
            [H[0], H[1], H[2], -H[0] - H[1] - H[2]]
        }
    }

    /// Elasticity Hessian per element.
    ///
    /// This is represented by a 4x4 block matrix of 3x3 matrices. The total
    /// matrix is a lower triangular 12x12 matrix. The blocks are specified in
    /// row-major order to be consistent with the 3x3 Matrices.
    #[allow(non_snake_case)]
    #[unroll_for_loops]
    #[inline]
    fn energy_hessian(&self) -> [[Matrix3<T>; 4]; 4] {
        let NeoHookeanTetEnergy {
            DX_inv,
            volume,
            lambda,
            mu,
            ..
        } = *self;

        let mut local_hessians = [[Matrix3::zeros(); 4]; 4];

        let F = self.deformation_gradient();
        let J = F.determinant();
        if J <= T::zero() {
            return local_hessians;
        }

        let A = DX_inv.transpose() * DX_inv;

        // Theoretically we know Dx is invertible since F is, but it could have
        // numerical differences, so we check anyways.
        let Dx_inv = match self.Dx.inverse() {
            Some(inv) => inv,
            None => return local_hessians,
        };

        let alpha = mu - lambda * Float::ln(J);

        // Off-diagonal elements
        for row in 0..3 {
            for col in 0..3 {
                let mut last_hess = T::zero();
                for n in 0..3 {
                    // which vertex
                    for k in 0..3 {
                        // with respect to which vertex
                        let c_lambda = lambda * Dx_inv[row][n] * Dx_inv[col][k];
                        let c_alpha = alpha * Dx_inv[col][n] * Dx_inv[row][k];
                        let mut h = volume * (c_alpha + c_lambda);
                        if col == row {
                            h += volume * mu * A[n][k];
                        }

                        // skip upper trianglar part
                        if (n == k && row >= col) || n > k {
                            local_hessians[n][k][row][col] = h;
                        }
                        // with respect to last vertex
                        local_hessians[3][k][row][col] -= h;
                        last_hess += h;
                    }
                }

                // last vertex
                if row >= col {
                    local_hessians[3][3][row][col] = last_hess;
                }
            }
        }

        local_hessians
    }

    /// Elasticity Hessian*displacement product per element.
    ///
    /// Respresented by a 3x3 matrix where row `i` produces the hessian product
    /// contribution for the vertex `i` within the current element.  The
    /// contribution to the last vertex is given by the negative sum of all the
    /// rows.
    #[allow(non_snake_case)]
    #[inline]
    fn energy_hessian_product_transpose(&self, tet_dx: &Tetrahedron<T>) -> Matrix3<T> {
        let NeoHookeanTetEnergy {
            DX_inv,
            volume,
            lambda,
            mu,
            ..
        } = *self;
        let F = self.deformation_gradient();
        let dF = self.deformation_gradient_differential(tet_dx);
        let J = F.determinant();
        if J > T::zero() {
            let alpha = mu - lambda * Float::ln(J);

            let F_inv_tr = F.inverse_transpose().unwrap();
            let dF_tr_F_inv_tr = dF.transpose() * F_inv_tr;

            let dP = dF * mu
                + F_inv_tr * dF_tr_F_inv_tr * alpha
                + F_inv_tr * (dF_tr_F_inv_tr.trace() * lambda);

            DX_inv.transpose() * dP * volume
        } else {
            Matrix3::zero()
        }
    }
}
