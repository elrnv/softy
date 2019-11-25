//! Stable Neo-Hookean energy model for tetrahedral meshes.
//! See http://graphics.pixar.com/library/StableElasticity/paper.pdf for details.

//use std::path::Path;
//use geo::io::save_tetmesh;
use super::TetEnergy;
use geo::ops::*;
use geo::prim::Tetrahedron;
use num_traits::Zero;
use utils::soap::*;
use super::tet_nh::TetMeshElasticity;
use unroll::unroll_for_loops;

/// Per-tetrahedron Neo-Hookean energy model. This struct stores conveniently precomputed values
/// for tet energy computation. It encapsulates tet specific energy computation.
#[allow(non_snake_case)]
pub struct StableNeoHookeanTetEnergy<T: Real> {
    Dx: Matrix3<T>,
    DX_inv: Matrix3<T>,
    volume: T,
    lambda: T,
    mu: T,
}

impl<T: Real> TetEnergy<T> for StableNeoHookeanTetEnergy<T> {
    #[allow(non_snake_case)]
    fn new(Dx: Matrix3<T>, DX_inv: Matrix3<T>, volume: T, lambda: T, mu: T) -> Self {
        StableNeoHookeanTetEnergy {
            Dx,
            DX_inv,
            volume,
            lambda,
            mu,
        }
    }

    /// Compute the deformation gradient `F` for this tet.
    #[allow(non_snake_case)]
    #[inline]
    fn deformation_gradient(&self) -> Matrix3<T> {
        self.Dx * self.DX_inv
    }

    /// Compute the deformation gradient differential `dF` for this tet.
    #[allow(non_snake_case)]
    #[inline]
    fn deformation_gradient_differential(&self, tet_dx: &Tetrahedron<T>) -> Matrix3<T> {
        // Build differential dDx
        let dDx = Matrix3::new(tet_dx.shape_matrix()).transpose();
        dDx * self.DX_inv
    }

    /// Elastic strain energy per element.
    /// This is a helper function that computes the strain energy given shape matrices, which can
    /// be obtained from a tet and its reference configuration.
    #[allow(non_snake_case)]
    #[inline]
    fn elastic_energy(&self) -> T {
        let StableNeoHookeanTetEnergy {
            volume, mu, lambda, ..
        } = *self;
        let F = self.deformation_gradient();
        let I = F.frob_norm_squared(); // tr(F^TF)
        let J = F.determinant();
        let half = T::from(0.5).unwrap();
        let J_minus_1 = J - T::one();
        let J_minus_alpha =  J_minus_1 - T::from(0.75).unwrap() * mu / lambda;
        volume
            * half * (mu * (I - T::from(3.0).unwrap() - (I + T::one()).ln())
                + lambda * J_minus_alpha*J_minus_alpha)
    }

    /// Elastic energy gradient per element vertex.
    /// This is a helper function that computes the energy gradient given shape matrices, which can
    /// be obtained from a tet and its reference configuration.
    #[allow(non_snake_case)]
    #[inline]
    fn elastic_energy_gradient(&self) -> [Vector3<T>; 4] {
        let StableNeoHookeanTetEnergy {
            DX_inv,
            volume,
            mu,
            lambda,
            ..
        } = *self;
        let F = self.deformation_gradient();
        let I = F.frob_norm_squared(); // tr(F^TF)
        let J = F.determinant();
        let alpha = T::one() + T::from(0.75).unwrap() * mu / lambda;
        let dJdF = Matrix3::new([F[1].cross(F[2]).data, F[2].cross(F[0]).data, F[0].cross(F[1]).data]);
        let P = F * (mu * (I / (I + T::one()))) + dJdF * (lambda * (J - alpha));
        //let PT = (F.transpose() * mu + F_inv * (lambda * logJ - mu));
        let H = DX_inv * P.transpose() * volume;
        [H[0], H[1], H[2], -H[0] - H[1] - H[2]]
    }

    /// Elasticity Hessian per element with respect to deformation gradient. This is a 3x3 matrix
    /// of 3x3 blocks.
    #[allow(non_snake_case)]
    #[unroll_for_loops]
    #[inline]
    fn elastic_energy_deformation_hessian(&self) -> [[Matrix3<T>; 3]; 3] {
        let StableNeoHookeanTetEnergy { lambda, mu, ..  } = *self;
        let F = self.deformation_gradient();
        let I = F.frob_norm_squared(); // tr(F^TF)
        let I_plus_1 = I + T::one();
        let J = F.determinant();
        let J_minus_alpha =  J - T::one() - T::from(0.75).unwrap() * mu / lambda;

        let Ti = Matrix9::identity();
        let f = F.vec();
        let M = f * f.transpose();
        let dJdF = Matrix3::new([F[1].cross(F[2]).data, F[2].cross(F[0]).data, F[0].cross(F[1]).data]);
        let g = dJdF.vec();
        let G = g * g.transpose();
        let zero = [[T::zero(); 3]; 3];
        let H_blocks = [
            [zero, (-F[2]).skew().data, F[1].skew().data],
            [F[2].skew().data, zero, (-F[0]).skew().data],
            [(-F[1]).skew().data, F[0].skew().data, zero]
        ];
        let mut H = Matrix9::<T>::zero();
        for r in 0..3 {
            for c in 0..3 {
                for i in 0..3 {
                    for j in 0..3 {
                        H[3*r + i][3*c + j] = H_blocks[r][c][i][j];
                    }
                }
            }
        }

        let dfdF = Ti * (mu * I / I_plus_1)
            + M * (mu * T::from(2.0).unwrap() / (I_plus_1 * I_plus_1))
            + G * lambda
            + H * lambda * J_minus_alpha;

        let mut out = [[Matrix3::zero(); 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                out[i][j] = dfdF[3*i + j].mtx()
            }
        }
        out
    }

    /// Elasticity Hessian per element. This is represented by a 4x4 block matrix of 3x3 matrices. The
    /// total matrix is a lower triangular 12x12 matrix. The blocks are specified in row-major
    /// order.
    #[allow(non_snake_case)]
    #[unroll_for_loops]
    #[inline]
    fn elastic_energy_hessian(&self) -> [[Matrix3<T>; 4]; 4] {
        let StableNeoHookeanTetEnergy {
            DX_inv,
            volume,
            ..
        } = *self;

        let dfdF = self.elastic_energy_deformation_hessian();

        let mut local_hessians = [[Matrix3::zeros(); 4]; 4];

        for r in 0..3 { // vertex
            for c in 0..3 { // vertex
                for i in 0..3 { // component
                    for j in 0..3 { // component
                        // F contraction
                        for k in 0..3 {
                            local_hessians[r][c][i][j] += volume * dfdF[j][k][i].dot(DX_inv[r]) * DX_inv[c][k];
                        }
                    }
                }
            }
        }

        // Last row
        for r in 0..3 {
            for c in 0..3 {
                local_hessians[3][c] -= local_hessians[r][c];
            }
        }

        // Last matrix
        for c in 0..3 {
            local_hessians[3][3] -= local_hessians[3][c];
        }

        local_hessians
    }

    /// Elasticity Hessian*displacement product per element. Respresented by a 3x3 matrix where row `i`
    /// produces the hessian product contribution for the vertex `i` within the current element.
    /// The contribution to the last vertex is given by the negative sum of all the rows.
    #[allow(non_snake_case)]
    #[inline]
    fn elastic_energy_hessian_product_transpose(&self, tet_dx: &Tetrahedron<T>) -> Matrix3<T> {
        let StableNeoHookeanTetEnergy {
            DX_inv,
            volume,
            lambda,
            mu,
            ..
        } = *self;
        let F = self.deformation_gradient();
        let dF = self.deformation_gradient_differential(tet_dx);
        let J = F.determinant();
        let I = F.frob_norm_squared(); // tr(F^TF)
        let I_plus_1 = I + T::one();
        let dJdF = Matrix3::new([F[1].cross(F[2]).data, F[2].cross(F[0]).data, F[0].cross(F[1]).data]);
        let alpha = T::one() + T::from(0.75).unwrap() * mu / lambda;

        let zero = Matrix3::zero().data;
        let H_blocks = [
            [zero, (-F[2]).skew().data, F[1].skew().data],
            [F[2].skew().data, zero, (-F[0]).skew().data],
            [(-F[1]).skew().data, F[0].skew().data, zero]
        ];
        let mut H = Matrix9::<T>::zero();
        for r in 0..3 {
            for c in 0..3 {
                for i in 0..3 {
                    for j in 0..3 {
                        H[3*r + i][3*c + j] = H_blocks[r][c][i][j];
                    }
                }
            }
        }
        let h_mtx = (H * dF.vec()).mtx();

        let FdFtrace = F.vec().dot(dF.vec());

        let dP = (dF * (I / I_plus_1)
                  + F * (FdFtrace * T::from(2.0).unwrap() / (I_plus_1 * I_plus_1))) * mu
            + (dJdF * dJdF.vec().dot(dF.vec())  + h_mtx * (J - alpha)) * lambda;

        DX_inv * dP.transpose() * volume
    }
}

pub type TetMeshStableNeoHookean<'a, T> = TetMeshElasticity<'a, StableNeoHookeanTetEnergy<T>>;

#[cfg(test)]
mod tests {
    use crate::objects::material::*;
    use super::*;
    use crate::energy_models::test_utils::*;
    use crate::energy_models::elasticity::test_utils::*;
    use crate::fem::SolverBuilder;
    use crate::objects::TetMeshSolid;
    use geo::mesh::VertexPositions;

    fn material() -> SolidMaterial {
        SolidMaterial::new(0).with_elasticity(ElasticityParameters {
            lambda: 5.4,
            mu: 263.1,
        })
    }

    fn test_solids() -> Vec<TetMeshSolid> {
        let material = material();

        test_meshes()
            .into_iter()
            .map(|mut tetmesh| {
                // Prepare attributes relevant for elasticity computations.
                SolverBuilder::prepare_deformable_mesh_vertex_attributes(&mut tetmesh).unwrap();
                SolverBuilder::prepare_deformable_tetmesh_attributes(&mut tetmesh).unwrap();
                let mut solid = TetMeshSolid::new(tetmesh, material);
                SolverBuilder::prepare_elasticity_attributes(&mut solid).unwrap();
                solid
            })
            .collect()
    }

    fn build_energies(solids: &[TetMeshSolid]) -> Vec<(TetMeshStableNeoHookean<autodiff::F>, Vec<[f64; 3]>)> {
        solids
            .iter()
            .map(|solid| {
                (
                    TetMeshStableNeoHookean::new(solid),
                    solid.tetmesh.vertex_positions().to_vec(),
                )
            })
            .collect()
    }


    #[test]
    fn tet_energy_gradient() {
        tet_energy_gradient_tester::<StableNeoHookeanTetEnergy<autodiff::F>>();
    }

    #[test]
    fn tet_energy_hessian() {
        tet_energy_hessian_tester::<StableNeoHookeanTetEnergy<autodiff::F>>();
    }

    #[test]
    fn tet_energy_hessian_product() {
        tet_energy_hessian_product_tester::<StableNeoHookeanTetEnergy<f64>>();
    }

    #[test]
    fn gradient() {
        let solids = test_solids();
        gradient_tester(build_energies(&solids), EnergyType::Position);
    }

    #[test]
    fn hessian() {
        let solids = test_solids();
        hessian_tester(build_energies(&solids), EnergyType::Position);
    }
}