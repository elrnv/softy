//use std::path::Path;
//use geo::io::save_tetmesh;
use crate::attrib_names::*;
use crate::energy::*;
use crate::geo::math::{Matrix3, Vector3};
use crate::geo::mesh::{topology::*, Attrib};
use crate::geo::ops::*;
use crate::matrix::*;
use crate::TetMesh;
use rayon::prelude::*;
use reinterpret::*;
use std::{cell::RefCell, rc::Rc};

/// Per-tetrahedron Neo-Hookean energy model. This struct stores conveniently precomputed values
/// for tet energy computation. It encapsulates tet specific energy computation.
#[allow(non_snake_case)]
pub struct NeoHookeanTetEnergy {
    Dx: Matrix3<f64>,
    DX_inv: Matrix3<f64>,
    volume: f64,
    lambda: f64,
    mu: f64,
}

impl NeoHookeanTetEnergy {
    pub const NUM_HESSIAN_TRIPLETS: usize = 78; // There are 4*6 + 3*9*4/2 = 78 triplets per tet (overestimate)

    /// Compute the deformation gradient `F` for this tet.
    #[allow(non_snake_case)]
    #[inline]
    pub fn deformation_gradient(&self) -> Matrix3<f64> {
        self.Dx * self.DX_inv
    }

    /// Compute the deformation gradient differential `dF` for this tet.
    #[allow(non_snake_case)]
    #[inline]
    fn deformation_gradient_differential(&self, dx: &[Vector3<f64>; 4]) -> Matrix3<f64> {
        // Build differential dDx
        let dDx = Matrix3([
            (dx[0] - dx[3]).into(),
            (dx[1] - dx[3]).into(),
            (dx[2] - dx[3]).into(),
        ]);
        dDx * self.DX_inv
    }

    /// Elastic strain energy per element.
    /// This is a helper function that computes the strain energy given shape matrices, which can
    /// be obtained from a tet and its reference configuration.
    #[allow(non_snake_case)]
    #[inline]
    pub fn elastic_energy(&self) -> f64 {
        let NeoHookeanTetEnergy {
            volume, mu, lambda, ..
        } = *self;
        let F = self.deformation_gradient();
        let I = F.norm_squared(); // tr(F^TF)
        let J = F.determinant();
        if J <= 0.0 {
            ::std::f64::INFINITY
        } else {
            let logJ = J.ln();
            volume * (0.5 * mu * (I - 3.0) - mu * logJ + 0.5 * lambda * logJ * logJ)
        }
    }

    /// Elastic energy gradient per element vertex.
    /// This is a helper function that computes the energy gradient given shape matrices, which can
    /// be obtained from a tet and its reference configuration.
    #[allow(non_snake_case)]
    #[inline]
    pub fn elastic_energy_gradient(&self) -> [Vector3<f64>; 4] {
        let NeoHookeanTetEnergy {
            DX_inv,
            volume,
            mu,
            lambda,
            ..
        } = *self;
        let F = self.deformation_gradient();
        let J = F.determinant();
        if J <= 0.0 {
            [Vector3::zeros(); 4]
        } else {
            let F_inv_tr = F.inverse_transpose().unwrap();
            let logJ = J.ln();
            let H = volume * (mu * F + (lambda * logJ - mu) * F_inv_tr) * DX_inv.transpose();
            [H[0], H[1], H[2], -H[0] - H[1] - H[2]]
        }
    }

    /// Elasticity Hessian per element. This is represented by a 4x4 block matrix 3x3 matrices. The
    /// total matrix is a lower triangular 12x12 matrix. The blocks are specified in column-major
    /// order to be consistent with the 3x3 Matrices.
    #[allow(non_snake_case)]
    #[inline]
    fn elastic_energy_hessian(&self) -> [[Matrix3<f64>; 4]; 4] {
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
        if J <= 0.0 {
            return local_hessians;
        }

        let A = DX_inv * DX_inv.transpose();

        // Theoretically we known Dx is invertible since F is, but it could have
        // numerical differences, so we check anyways.
        let Dx_inv_tr = match self.Dx.inverse_transpose() {
            Some(inv) => inv,
            None => return local_hessians,
        };

        let alpha = mu - lambda * J.ln();

        // Off-diagonal elements
        for col in 0..3 {
            for row in 0..3 {
                let mut last_hess = 0.0;
                for k in 0..3 {
                    // which vertex
                    let mut last_wrt_hess = 0.0;
                    for n in 0..3 {
                        // with respect to which vertex
                        let c_lambda = lambda * Dx_inv_tr[n][row] * Dx_inv_tr[k][col];
                        let c_alpha = alpha * Dx_inv_tr[n][col] * Dx_inv_tr[k][row];
                        let mut h = volume * (c_alpha + c_lambda);
                        if col == row {
                            h += volume * mu * A[k][n];
                        }
                        last_wrt_hess -= h;

                        // skip upper trianglar part
                        if (n == k && row >= col) || n > k {
                            local_hessians[k][n][col][row] = h;
                        }
                    }

                    // with respect to last vertex
                    last_hess -= last_wrt_hess;
                    local_hessians[k][3][col][row] = last_wrt_hess;
                }

                // last vertex
                if row >= col {
                    local_hessians[3][3][col][row] = last_hess;
                }
            }
        }

        local_hessians
    }
}

impl TetEnergy<f64> for NeoHookeanTetEnergy {
    #[allow(non_snake_case)]
    fn new(Dx: Matrix3<f64>, DX_inv: Matrix3<f64>, volume: f64, lambda: f64, mu: f64) -> Self {
        NeoHookeanTetEnergy {
            Dx,
            DX_inv,
            volume,
            lambda,
            mu,
        }
    }

    /// Elasticity Hessian*displacement product per element. Respresented by a 3x3 matrix where column `i`
    /// produces the hessian product contribution for the vertex `i` within the current element.
    /// The contribution to the last vertex is given by the negative sum of all the columns.
    #[allow(non_snake_case)]
    #[inline]
    fn elastic_energy_hessian_product(&self, dx: &[Vector3<f64>; 4]) -> Matrix3<f64> {
        let NeoHookeanTetEnergy {
            DX_inv,
            volume,
            lambda,
            mu,
            ..
        } = *self;
        let F = self.deformation_gradient();
        let dF = self.deformation_gradient_differential(dx);
        let J = F.determinant();
        if J > 0.0 {
            let alpha = mu - lambda * J.ln();

            let F_inv_tr = F.inverse_transpose().unwrap();
            let F_inv = F_inv_tr.transpose();

            let dP = mu * dF
                + alpha * F_inv_tr * dF.transpose() * F_inv_tr
                + lambda * (F_inv * dF).trace() * F_inv_tr;

            volume * dP * DX_inv.transpose()
        } else {
            Matrix3::zeros()
        }
    }
}

/// A possibly non-linear elastic energy for tetrahedral meshes.
#[derive(Debug, PartialEq)]
pub struct ElasticTetMeshEnergy {
    /// The discretization of the solid domain using a tetrahedral mesh.
    pub tetmesh: Rc<RefCell<TetMesh>>,
    /// Material parameters.
    material: MaterialModel,

    // Workspace fields for storing intermediate results
    /// Energy hessian indices. Keep this memory around and reuse it to avoid unnecessary allocations.
    energy_hessian_indices: Vec<MatrixElementIndex>,
    /// Energy hessian values. Keep this memory around and reuse it to avoid unnecessary allocations.
    energy_hessian_values: Vec<f64>,
}

/// A builder for the `ElasticTetMeshEnergy` struct. The only required field is the `tetmesh` which
/// is specified in the `new` method.
pub struct ElasticTetMeshEnergyBuilder {
    tetmesh: Rc<RefCell<TetMesh>>,
    material: Option<MaterialModel>,
}

/// The material model including elasticity Lame parameters as well as dynamics specific parameter
/// like the damping coefficient.
#[derive(Copy, Clone, Debug, PartialEq)]
struct MaterialModel {
    /// First Lame parameter. Measured in Pa = N/m² = kg/(ms²).
    pub lambda: f64,
    /// Second Lame parameter. Measured in Pa = N/m² = kg/(ms²).
    pub mu: f64,
    /// Coefficient measuring the amount of artificial viscosity as dictated by the Rayleigh
    /// damping model. This coefficient should incorporate the timestep.
    pub damping: f64,
}

impl Default for MaterialModel {
    fn default() -> Self {
        MaterialModel {
            lambda: 5.4,
            mu: 263.1,
            damping: 0.0,
        }
    }
}

impl ElasticTetMeshEnergyBuilder {
    /// Create a new Neo-Hookean energy model defining a non-linear problem that can be solved
    /// using a non-linear solver like Ipopt.
    /// This function takes a tetrahedron mesh specifying a discretization of the solid domain
    pub fn new(tetmesh: Rc<RefCell<TetMesh>>) -> Self {
        ElasticTetMeshEnergyBuilder {
            tetmesh,
            material: None,
        }
    }

    /// Set the elastic material properties of the volumetric solid discretized by the tetmesh.
    pub fn material(mut self, lambda: f64, mu: f64, damping: f64) -> Self {
        self.material = Some(MaterialModel {
            lambda,
            mu,
            damping,
        });
        self
    }

    pub fn build(&self) -> ElasticTetMeshEnergy {
        let ElasticTetMeshEnergyBuilder { tetmesh, material } = self.clone();

        ElasticTetMeshEnergy {
            tetmesh: tetmesh.clone(),
            material: material.unwrap_or(MaterialModel::default()),
            energy_hessian_indices: Vec::new(),
            energy_hessian_values: Vec::new(),
        }
    }
}

impl ElasticTetMeshEnergy {
    /// Helper for distributing local Hessian entries into the global Hessian matrix.
    /// This function provides the order of Hessian matrix non-zeros.
    /// `indices` is a map from the local tet vertex indices to their position in the global
    /// tetmesh
    /// `local_hess` is the function that computes a local hessian matrix for a pair of vertex
    /// indices
    /// `value` is the mapping function that would compute the hessian value. In particular
    /// `value` takes 3 pairs of (row, col) indices in this order:
    ///     - vertex indices
    ///     - local matrix indices
    /// as well as the local hessian matrix computed by `local_hess`.
    #[inline]
    fn hessian_for_each<H, L, F>(mut local_hess: L, mut value: F)
    where
        L: FnMut(usize, usize) -> H,
        F: FnMut(usize, (usize, usize), (usize, usize), &mut H),
    {
        let mut i = 0; // triplet index for the tet. there should be 78 in total
        for k in 0..4 {
            for n in k..4 {
                let mut h = local_hess(n, k);
                for col in 0..3 {
                    let start = if n == k { col } else { 0 };
                    for row in start..3 {
                        value(i, (n, k), (row, col), &mut h);
                        i += 1;
                    }
                }
            }
        }

        assert_eq!(i, NeoHookeanTetEnergy::NUM_HESSIAN_TRIPLETS)
    }
}

/// Define energy for ElasticTetMeshEnergy materials.
impl Energy<f64> for ElasticTetMeshEnergy {
    #[allow(non_snake_case)]
    fn energy(&mut self, dx: &[f64]) -> f64 {
        let ElasticTetMeshEnergy {
            ref tetmesh,
            material:
                MaterialModel {
                    lambda,
                    mu,
                    damping,
                },
            ..
        } = *self;

        let tetmesh = tetmesh.borrow();

        let disp: &[Vector3<f64>] = reinterpret_slice(dx);

        tetmesh
            .attrib_iter::<f64, CellIndex>(REFERENCE_VOLUME_ATTRIB)
            .unwrap()
            .zip(
                tetmesh
                    .attrib_iter::<Matrix3<f64>, CellIndex>(REFERENCE_SHAPE_MATRIX_INV_ATTRIB)
                    .unwrap(),
            )
            .zip(tetmesh.cell_iter())
            .zip(tetmesh.tet_iter())
            .map(|(((&vol, &DX_inv), cell), tet)| {
                let Dx = tet.shape_matrix();
                let tet_energy = NeoHookeanTetEnergy::new(Dx, DX_inv, vol, lambda, mu);
                // elasticity
                tet_energy.elastic_energy() + 0.5 * damping * {
                    let dx = [disp[cell[0]], disp[cell[1]], disp[cell[2]], disp[cell[3]]];
                    let dH = tet_energy.elastic_energy_hessian_product(&dx);
                    // damping (viscosity)
                    (dH[0].dot(dx[0]) + dH[1].dot(dx[1]) + dH[2].dot(dx[2])
                        - (dx[3].transpose() * dH).sum())
                }
            })
            .sum()
    }
}

impl EnergyGradient<f64> for ElasticTetMeshEnergy {
    #[allow(non_snake_case)]
    fn add_energy_gradient(&self, dx: &[f64], grad_f: &mut [f64]) {
        let ElasticTetMeshEnergy {
            ref tetmesh,
            material:
                MaterialModel {
                    lambda,
                    mu,
                    damping,
                },
            ..
        } = *self;

        let tetmesh = tetmesh.borrow();
        let disp: &[Vector3<f64>] = reinterpret_slice(dx);

        debug_assert_eq!(grad_f.len(), dx.len());

        let gradient: &mut [Vector3<f64>] = reinterpret_mut_slice(grad_f);

        let force_iter = tetmesh
            .attrib_iter::<f64, CellIndex>(REFERENCE_VOLUME_ATTRIB)
            .unwrap()
            .zip(
                tetmesh
                    .attrib_iter::<Matrix3<f64>, CellIndex>(REFERENCE_SHAPE_MATRIX_INV_ATTRIB)
                    .unwrap(),
            )
            .zip(tetmesh.tet_iter())
            .map(|((&vol, &DX_inv), tet)| {
                let tet_energy =
                    NeoHookeanTetEnergy::new(tet.shape_matrix(), DX_inv, vol, lambda, mu);
                tet_energy.elastic_energy_gradient()
            });

        // Transfer forces from cell-vertices to vertices themeselves
        for ((((&vol, &DX_inv), tet), cell), grad) in tetmesh
            .attrib_iter::<f64, CellIndex>(REFERENCE_VOLUME_ATTRIB)
            .unwrap()
            .zip(
                tetmesh
                    .attrib_iter::<Matrix3<f64>, CellIndex>(REFERENCE_SHAPE_MATRIX_INV_ATTRIB)
                    .unwrap(),
            )
            .zip(tetmesh.tet_iter())
            .zip(tetmesh.cell_iter())
            .zip(force_iter)
        {
            let dx = [
                // current displacement
                disp[cell[0]],
                disp[cell[1]],
                disp[cell[2]],
                disp[cell[3]],
            ];

            for i in 0..4 {
                gradient[cell[i]] += grad[i];
            }

            // Needed for damping.
            let tet_energy = NeoHookeanTetEnergy::new(tet.shape_matrix(), DX_inv, vol, lambda, mu);
            let dH = tet_energy.elastic_energy_hessian_product(&dx);
            for i in 0..3 {
                // Damping
                gradient[cell[i]] += damping * dH[i];
                gradient[cell[3]] -= damping * dH[i];
            }
        }
    }
}

impl EnergyHessian<f64> for ElasticTetMeshEnergy {
    fn energy_hessian_size(&self) -> usize {
        NeoHookeanTetEnergy::NUM_HESSIAN_TRIPLETS * self.tetmesh.borrow().num_cells()
    }
    fn energy_hessian_indices(&mut self) -> &[MatrixElementIndex] {
        let num_hess_triplets = self.energy_hessian_size();
        let ElasticTetMeshEnergy {
            ref tetmesh,
            energy_hessian_indices: ref mut hess,
            ..
        } = *self;

        let tetmesh = tetmesh.borrow();

        // Ensure there are enough entries in our hessian index buffer.
        hess.resize(num_hess_triplets, MatrixElementIndex { row: 0, col: 0 });

        {
            // Break up the hessian indices into chunks of elements for each tet.
            let hess_chunks: &mut [[MatrixElementIndex;
                                       NeoHookeanTetEnergy::NUM_HESSIAN_TRIPLETS]] =
                reinterpret_mut_slice(hess);

            let hess_iter = hess_chunks.par_iter_mut().zip(tetmesh.cells().par_iter());

            hess_iter.for_each(|(tet_hess, cell)| {
                Self::hessian_for_each(
                    |_, _| (),
                    |i, (n, k), (row, col), _| {
                        let mut global_row = 3 * cell[n] + row;
                        let mut global_col = 3 * cell[k] + col;
                        if cell[n] < cell[k] {
                            // In the upper triangular part of the global matrix, transpose
                            ::std::mem::swap(&mut global_row, &mut global_col);
                        }
                        tet_hess[i] = MatrixElementIndex {
                            row: global_row,
                            col: global_col,
                        };
                    },
                );
            });
        }
        hess
    }

    #[allow(non_snake_case)]
    fn energy_hessian_values(&mut self, _dx: &[f64]) -> &[f64] {
        let num_hess_triplets = self.energy_hessian_size();
        let ElasticTetMeshEnergy {
            ref tetmesh,
            material:
                MaterialModel {
                    lambda,
                    mu,
                    damping,
                },
            energy_hessian_values: ref mut hess,
            ..
        } = *self;

        let tetmesh = &*tetmesh.borrow();

        // Ensure there are enough entries in our hessian index buffer.
        hess.resize(num_hess_triplets, 0.0);

        {
            // Break up the hessian triplets into chunks of elements for each tet.
            let hess_chunks: &mut [[f64; NeoHookeanTetEnergy::NUM_HESSIAN_TRIPLETS]] =
                reinterpret_mut_slice(hess);

            let hess_iter = hess_chunks
                .par_iter_mut()
                .zip(
                    tetmesh
                        .attrib_as_slice::<f64, CellIndex>(REFERENCE_VOLUME_ATTRIB)
                        .unwrap()
                        .par_iter(),
                )
                .zip(
                    tetmesh
                        .attrib_as_slice::<Matrix3<f64>, CellIndex>(
                            REFERENCE_SHAPE_MATRIX_INV_ATTRIB,
                        )
                        .unwrap()
                        .par_iter(),
                )
                .zip(tetmesh.cells().par_iter());

            hess_iter.for_each(|(((tet_hess, &vol), &DX_inv), cell)| {
                let tet = tetmesh.tet_from_indices(cell);
                let Dx = tet.shape_matrix();
                let tet_energy = NeoHookeanTetEnergy::new(Dx, DX_inv, vol, lambda, mu);

                let factor = 1.0 + damping;

                let local_hessians = tet_energy.elastic_energy_hessian();

                Self::hessian_for_each(
                    |n, k| factor * local_hessians[k][n],
                    |i, _, (row, col), h| tet_hess[i] = h[col][row],
                );
            });
        }
        hess
    }
}