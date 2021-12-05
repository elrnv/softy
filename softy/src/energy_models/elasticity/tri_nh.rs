#![allow(clippy::just_underscores_and_digits)]
//! Neo-Hookean energy model for a single triangle.

use num_traits::Zero;
use unroll::unroll_for_loops;

use crate::objects::shell::interior_edge::*;
use geo::ops::*;
use geo::prim::Triangle;
use tensr::*;

use super::LinearElementEnergy;

// TODO: Remove reinterpret when bytemuck implements Pod for all arrays.

/// Per-triangle Neo-Hookean energy model. This struct stores conveniently precomputed values
/// for triangle energy computation. It encapsulates triangle specific energy computation.
#[allow(non_snake_case)]
pub struct NeoHookeanTriEnergy<T> {
    Dx: Matrix2x3<T>,
    DX_inv: Matrix2<T>,
    area: T,
    lambda: T,
    mu: T,
}

impl<T: Real> NeoHookeanTriEnergy<T> {
    /// Compute the deformation gradient `F` for this triangle.
    #[allow(non_snake_case)]
    #[inline]
    fn deformation_gradient(&self) -> Matrix2x3<T> {
        self.DX_inv * self.Dx
    }

    /// A helper function to compute the energy Hessian in the context of a single Hessian product
    /// or the entire Hessian block.
    #[allow(non_snake_case)]
    #[inline]
    fn stiffness_product_tensor(
        &self,
        F_inv_tr: Matrix2x3<T>,
        C_inv_tr: Matrix2<T>,
        n: Vector3<T>,
        mu: T,
        lambda: T,
        alpha: T,
    ) -> [[Matrix2x3<T>; 3]; 3] {
        let F_inv: Matrix3x2<_> = F_inv_tr.transpose();
        let DX_inv_tr = self.DX_inv.transpose();
        let dF = Matrix3x2::from_rows([
            [T::one(), T::zero()].into(),
            [T::zero(), T::one()].into(),
            [-T::one(), -T::one()].into(),
        ]) * DX_inv_tr;

        let FinvT_dFT: [[Matrix2<_>; 3]; 3] = [
            [
                F_inv[0] * dF[0].transpose(),
                F_inv[1] * dF[0].transpose(),
                F_inv[2] * dF[0].transpose(),
            ],
            [
                F_inv[0] * dF[1].transpose(),
                F_inv[1] * dF[1].transpose(),
                F_inv[2] * dF[1].transpose(),
            ],
            [
                F_inv[0] * dF[2].transpose(),
                F_inv[1] * dF[2].transpose(),
                F_inv[2] * dF[2].transpose(),
            ],
        ];

        let dP = |x: usize, i: usize| {
            let mut out = FinvT_dFT[i][x] * F_inv_tr * alpha
                + F_inv_tr * (FinvT_dFT[i][x].trace() * lambda)
                - ((C_inv_tr * dF[i] * n[x]) * alpha) * n.transpose();

            out[0][x] += dF[i][0] * mu;
            out[1][x] += dF[i][1] * mu;
            out
        };

        [
            [dP(0, 0), dP(1, 0), dP(2, 0)],
            [dP(0, 1), dP(1, 1), dP(2, 1)],
            [dP(0, 2), dP(1, 2), dP(2, 2)],
        ]
    }
}

impl<T: Real> LinearElementEnergy<T> for NeoHookeanTriEnergy<T> {
    type Element = Triangle<T>;
    type ShapeMatrix = Matrix2x3<T>;
    type RefShapeMatrix = Matrix2<T>;
    type Gradient = [Vector3<T>; 3];
    type Hessian = [[Matrix3<T>; 3]; 3];

    #[allow(non_snake_case)]
    #[inline]
    fn new(Dx: Self::ShapeMatrix, DX_inv: Self::RefShapeMatrix, area: T, lambda: T, mu: T) -> Self {
        NeoHookeanTriEnergy {
            Dx,
            DX_inv,
            area,
            lambda,
            mu,
        }
    }

    /// Compute the deformation gradient differential `dF` for this triangle.
    #[allow(non_snake_case)]
    #[inline]
    fn deformation_gradient_differential(&self, tri_dx: &Triangle<T>) -> Matrix2x3<T> {
        // Build differential dDx
        let dDx = Matrix2x3::new(tri_dx.shape_matrix());
        self.DX_inv * dDx
    }

    /// Elastic strain energy per element.
    /// This is a helper function that computes the strain energy given shape matrices, which can
    /// be obtained from a triangle and its reference configuration.
    #[allow(non_snake_case)]
    #[inline]
    fn energy(&self) -> T {
        let NeoHookeanTriEnergy {
            area, mu, lambda, ..
        } = *self;
        let F = self.deformation_gradient();
        let C = F * F.transpose();
        let C_det = C.determinant();
        let I = C[0][0] + C[1][1]; // trace
        if C_det <= T::zero() {
            T::infinity()
        } else {
            let half = T::from(0.5).unwrap();
            let log_C_det = C_det.ln();
            let _2 = T::from(2.0).unwrap();
            area * half
                * (mu * (I - _2 - log_C_det)
                    + T::from(0.25).unwrap() * lambda * log_C_det * log_C_det)
        }
    }

    /// Elastic energy gradient per element vertex.
    /// This is a helper function that computes the energy gradient given shape matrices, which can
    /// be obtained from a triangle and its reference configuration.
    #[allow(non_snake_case)]
    #[inline]
    fn energy_gradient(&self) -> [Vector3<T>; 3] {
        let NeoHookeanTriEnergy {
            DX_inv,
            area,
            mu,
            lambda,
            ..
        } = *self;
        let F = self.deformation_gradient();
        let C = F * F.transpose();
        let C_det = C.determinant();
        if C_det <= T::zero() {
            [Vector3::zero(); 3]
        } else {
            let F_inv_tr = C.inverse().unwrap() * F;
            let logJ = T::from(0.5).unwrap() * C_det.ln();
            let P = F * mu + F_inv_tr * (lambda * logJ - mu);
            let H = DX_inv.transpose() * P * area;
            [H[0], H[1], -H[0] - H[1]]
        }
    }

    /// Elasticity Hessian per element. This is represented by a 4x4 block matrix of 3x3 matrices. The
    /// total matrix is a lower triangular 12x12 matrix. The blocks are specified in row-major
    /// order to be consistent with the 3x3 Matrices.
    #[allow(non_snake_case)]
    #[unroll_for_loops]
    #[inline]
    fn energy_hessian(&self) -> [[Matrix3<T>; 3]; 3] {
        let mut hess = [[Matrix3::zeros(); 3]; 3];

        let NeoHookeanTriEnergy {
            DX_inv,
            area,
            lambda,
            mu,
            ..
        } = *self;
        let F = self.deformation_gradient();
        let C = F * F.transpose();
        let C_det = C.determinant();
        if C_det > T::zero() {
            let alpha = mu - lambda * T::from(0.5).unwrap() * C_det.ln();

            let C_inv_tr = C.inverse_transpose().unwrap();
            let F_inv_tr: Matrix2x3<_> = C_inv_tr.transpose() * F;
            let n = F[0].cross(F[1]).normalized();

            let dP = self.stiffness_product_tensor(F_inv_tr, C_inv_tr, n, mu, lambda, alpha);

            //let mut tri_dx = Triangle(
            //    [T::zero(); 3].into(),
            //    [T::zero(); 3].into(),
            //    [T::zero(); 3].into(),
            //);

            for i in 0..3 {
                // vertex
                for row in 0..3 {
                    //tri_dx[i][row] = T::one();
                    let h = DX_inv.transpose() * dP[i][row] * area;
                    //let h = self.energy_hessian_product_transpose(&tri_dx);
                    for j in 0..2 {
                        // vertex
                        for col in 0..3 {
                            // component
                            if i > j || (i == j && row >= col) {
                                hess[i][j][row][col] += h[j][col];
                                if i == 2 {
                                    hess[i][2][row][col] -= h[j][col];
                                }
                            }
                        }
                    }

                    //tri_dx[i][row] = T::zero();
                }
            }
        }

        hess
    }

    /// Elasticity Hessian*displacement product per element. Respresented by a 3x3 matrix where row `i`
    /// produces the hessian product contribution for the vertex `i` within the current element.
    /// The contribution to the last vertex is given by the negative sum of all the rows.
    #[allow(non_snake_case)]
    #[inline]
    fn energy_hessian_product_transpose(&self, tri_dx: &Triangle<T>) -> Matrix2x3<T> {
        let NeoHookeanTriEnergy {
            DX_inv,
            area,
            lambda,
            mu,
            ..
        } = *self;
        let F = self.deformation_gradient();
        let dF = self.deformation_gradient_differential(tri_dx);
        let C = F * F.transpose();
        let C_det = C.determinant();
        if C_det > T::zero() {
            let alpha = mu - lambda * T::from(0.5).unwrap() * C_det.ln();

            let C_inv = C.inverse().unwrap();
            let F_inv: Matrix3x2<_> = (C_inv * F).transpose();
            let dF_F_inv: Matrix2<_> = dF * F_inv;
            let n = F[0].cross(F[1]).normalized();

            let dP_tr: Matrix3x2<_> = dF.transpose() * mu
                + F_inv * dF_F_inv * alpha
                + F_inv * (dF_F_inv.trace() * lambda)
                - n * (n.transpose() * dF.transpose() * C_inv) * alpha;

            DX_inv.transpose() * dP_tr.transpose() * area
        } else {
            Matrix2x3::zero()
        }
    }
}

pub struct DiscreteShellBendingEnergy<'a, T> {
    cur_pos: &'a [[T; 3]],
    faces: &'a [[usize; 3]],
    edge: InteriorEdge,
    prev_theta: T,
    ref_theta: T,
    ref_shape: T,
    stiffness: T,
}

impl<T: Real> DiscreteShellBendingEnergy<'_, T> {
    /// Compute the bending energy.
    #[inline]
    pub fn energy(&self) -> T {
        let theta = self
            .edge
            .incremental_angle(self.prev_theta, self.cur_pos, |f, i| self.faces[f][i]);
        let theta_strain = theta - self.ref_theta;
        T::from(0.5).unwrap() * self.ref_shape * self.stiffness * theta_strain * theta_strain
    }

    /// Compute energy derivative with respect to theta.
    ///
    /// If `W` is the energy then this value corresponds to `∂W/∂θ`.
    #[inline]
    pub fn energy_angle_derivative(&self) -> T {
        let theta = self
            .edge
            .incremental_angle(self.prev_theta, self.cur_pos, |f, i| self.faces[f][i]);
        self.ref_shape * self.stiffness * (theta - self.ref_theta)
    }

    /// Compute the bending energy gradient.
    #[inline]
    pub fn energy_gradient(&self) -> Matrix4x3<T> {
        // Compute energy derivative with respect to theta.
        let dw_dth = self.energy_angle_derivative();

        let DiscreteShellBendingEnergy {
            cur_pos,
            faces,
            edge,
            ..
        } = *self;

        // Theta derivative with respect to x.
        let dth_dx = Matrix4x3::new(edge.edge_angle_gradient(cur_pos, |f, i| faces[f][i]));
        dth_dx * dw_dth
    }

    /// Compute the bending energy Hessian terms.
    #[inline]
    pub fn energy_hessian(&self) -> (Matrix4x3<T>, T, T, ([[T; 6]; 4], [[[T; 3]; 3]; 5])) {
        // Compute energy derivative with respect to theta.
        let dw_dth = self.energy_angle_derivative(); // ∂W/∂θ

        let DiscreteShellBendingEnergy {
            cur_pos,
            faces,
            edge,
            ref_shape,
            stiffness,
            ..
        } = *self;

        // ∂θ/∂x ∂²W/∂θ² ∂θ/∂xᵀ + ∂W/∂θ ∂²θ/∂x²
        let dth_dx = edge
            .edge_angle_gradient(cur_pos, |f, i| faces[f][i])
            .into_tensor(); // ∂θ/∂x
        let d2w_dth2 = stiffness * ref_shape; // ∂²W/∂θ²

        let d2th_dx2 = edge.edge_angle_hessian(cur_pos, |f, i| faces[f][i]); // ∂²θ/∂x²

        (dth_dx, d2w_dth2, dw_dth, d2th_dx2)
    }
}

/// In contrast to `DiscreteShellBendingEnergy`, this energy prevents inversions altogether.
#[allow(dead_code)]
pub struct DiscreteShellTanBendingEnergy<'a, T> {
    cur_pos: &'a [[T; 3]],
    faces: &'a [[usize; 3]],
    edge: InteriorEdge,
    prev_theta: T,
    ref_theta: T,
    ref_shape: T,
    stiffness: T,
}

#[allow(dead_code)]
impl<T: Real> DiscreteShellTanBendingEnergy<'_, T> {
    /// Compute the bending energy.
    #[inline]
    fn energy(&self) -> T {
        let half = T::from(0.5).unwrap();
        let _2 = T::from(2.0).unwrap();
        let theta = self
            .edge
            .incremental_angle(self.prev_theta, self.cur_pos, |f, i| self.faces[f][i]);
        let theta_strain = theta - self.ref_theta;
        let tan_strain = (theta_strain * half).tan() * _2;
        self.ref_shape * self.stiffness * tan_strain * tan_strain * half
    }

    /// Compute energy derivative with respect to θ.
    ///
    /// If `W` is the energy then this value corresponds to `∂W/∂θ`.
    #[inline]
    fn energy_angle_derivative(&self) -> T {
        let half = T::from(0.5).unwrap();
        let _2 = T::from(2.0).unwrap();
        let theta = self
            .edge
            .incremental_angle(self.prev_theta, self.cur_pos, |f, i| self.faces[f][i]);
        let theta_strain = theta - self.ref_theta;
        let tan_strain = (theta_strain * half).tan() * _2;
        let tan_strain_dth_sqrt = T::one() / (theta_strain * half).cos();
        let tan_strain_dth = tan_strain_dth_sqrt * tan_strain_dth_sqrt;
        self.ref_shape * self.stiffness * tan_strain * tan_strain_dth
    }

    /// Compute the bending energy gradient.
    #[inline]
    fn energy_gradient(&self) -> Matrix4x3<T> {
        // Compute energy derivative with respect to theta.
        let dw_dth = self.energy_angle_derivative();

        let DiscreteShellTanBendingEnergy {
            cur_pos,
            faces,
            edge,
            ..
        } = *self;

        // Theta derivative with respect to x.
        let dth_dx = Matrix4x3::new(edge.edge_angle_gradient(cur_pos, |f, i| faces[f][i]));
        dth_dx * dw_dth
    }

    /// Compute the bending energy Hessian terms.
    #[inline]
    fn energy_hessian(&self) -> (Matrix4x3<T>, T, T, ([[T; 6]; 4], [[[T; 3]; 3]; 5])) {
        // Compute energy derivative with respect to theta.
        let dw_dth = self.energy_angle_derivative(); // ∂W/∂θ

        let DiscreteShellTanBendingEnergy {
            cur_pos,
            faces,
            edge,
            prev_theta,
            ref_theta,
            ref_shape,
            stiffness,
        } = *self;

        // ∂θ/∂x ∂²W/∂θ² ∂θ/∂xᵀ + ∂W/∂θ ∂²θ/∂x²
        let dth_dx = edge
            .edge_angle_gradient(cur_pos, |f, i| faces[f][i])
            .into_tensor(); // ∂θ/∂x

        let half = T::from(0.5).unwrap();
        let _2 = T::from(2.0).unwrap();
        let theta = edge.incremental_angle(prev_theta, cur_pos, |f, i| faces[f][i]);
        let theta_strain = theta - ref_theta;
        let tan_strain = (theta_strain * half).tan() * _2;
        let sec = T::one() / (theta_strain * half).cos();
        let tan_strain_dth = sec * sec;
        let tan_strain_ddth = tan_strain_dth * (tan_strain_dth + tan_strain * tan_strain * half);
        let d2w_dth2 = stiffness * ref_shape * tan_strain_ddth; // ∂²W/∂θ²

        let d2th_dx2 = edge.edge_angle_hessian(cur_pos, |f, i| faces[f][i]); // ∂²θ/∂x²

        (dth_dx, d2w_dth2, dw_dth, d2th_dx2)
    }
}
