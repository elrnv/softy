use super::FrictionParams;
use crate::contact::*;
use crate::energy_models::tet_nh::ElasticTetMeshEnergy;
use geo::math::{Matrix3, Vector3, Vector2};
use reinterpret::*;

use utils::zip;

pub struct ElasticEnergyParams {
    pub energy_model: ElasticTetMeshEnergy,
    pub time_step: f64,
}

/// A Contact jacobian matrix.
pub enum ContactJacobian<'a, CJI> {
    /// A selection matrix, where the selected surface vertex indices are provided by a `usize`
    /// slice.
    Selection {
        nrows: usize,
        ncols: usize,
        indices: &'a [usize],
    },
    /// A block matrix as provided by the `Matrix3<f64>` slice and a corresponding iterator over
    /// rows and columns of these blocks.
    Full {
        nrows: usize,
        ncols: usize,
        blocks: &'a [Matrix3<f64>],
        block_indices: CJI,
    },
}

impl<'a, CJI> ContactJacobian<'a, CJI> {
    fn matrix_sprs(&self) -> sprs::CsMat<f64> {
        match self {
            ContactJacobian::Selection { nrows, ncols, indices } => {
                let nnz = surf_indices.len();
                let values = vec![1.0; nnz];
                let rows: Vec<_> = (0..nnz).collect();
                let cols: Vec<_> = surf_indices.to_vec();
                sprs::TriMat::from_triplets((nrows, ncols), rows, cols, values).to_csr()
            }
            ContactJacobian::Full { nrows, ncols, blocks, block_indices } => {
                let values: &[f64] = reinterpret_slice(blocks);
                let index_iter = block_index_iter.flat_map(move |(row_mtx, col_mtx)| {
                    (0..3)
                        .flat_map(move |j| (0..3).map(move |i| (3 * row_mtx + i, 3 * col_mtx + j)))
                });

                let (rows, cols) = index_iter.unzip();
                sprs::TriMat::from_triplets((nrows, ncols), rows, cols, values).to_csr()
            }
        }
    }
}

/// Elastic Friction solver. This solver uses the elasticity model when computing friction to
/// propagate the frictional contact forces through the solid. This is critical in several
/// scenarios like pinching.
pub struct ElasticFrictionSolver<'a, CJI> {
    /// A set of generalized velocities.
    velocity: &'a [Vector3<f64>],
    /// A set of contact forces for each contact point.
    contact_impulse: &'a [f64],
    /// Basis defining the normal and tangent space at each point of contact.
    contact_basis: &'a ContactBasis,
    /// Friction coefficient.
    mu: f64,
    /// Contact Jacobian is a sparse matrix that maps vectors from generalized coordinates to
    /// physical coordinates (e.g. from vertices to contact points).  If the `None` is specified,
    /// it is assumed that the contact Jacobian is the identity matrix, meaning that contacts occur
    /// at vertex positions.
    contact_jacobian: ContactJacobian<'a, CJI>,
    /// Vertex masses.
    masses: &'a [f64],

    elastic_energy: Option<ElasticEnergyParams>,
}

impl<'a> ElasticFrictionSolver<'a, std::iter::Empty<(usize, usize)>> {
    /// Build a new solver for the friction problem. The given `velocity` is a stacked vector of
    /// velocities for each surface vertex. `contact_impulse` is the normal component of the
    /// predictor frictional contact impulse at each contact point.  Finally, `mu` is the friction
    /// coefficient.
    pub fn selection_contact_jacobian(
        velocity: &'a [[f64; 3]],
        contact_impulse: &'a [f64],
        contact_basis: &'a ContactBasis,
        masses: &'a [f64],
        params: FrictionParams,
        contact_jacobian: &'a [usize],
        elastic_energy: Option<ElasticEnergyParams>,
    ) -> ElasticFrictionSolver<'a, std::iter::Empty<(usize, usize)>> {
        ElasticFrictionSolver {
            velocity: reinterpret_slice(velocity),
            contact_impulse,
            contact_basis,
            mu: params.dynamic_friction,
            contact_jacobian: ContactJacobian::Selection {
                nrows: contact_impulse.len(),
                ncols: velocity.len(),
                indices: contact_jacobian,
            },
            masses,
            elastic_energy,
        }
    }
}

impl<'a, CJI: Iterator<Item = (usize, usize)>> ElasticFrictionSolver<'a, CJI> {
    /// Build a new solver for the friction problem. The given `velocity` is a stacked vector of
    /// tangential velocities for each contact point in contact space. `contact_impulse` is the
    /// normal component of the predictor frictional contact impulse at each contact point.
    /// Finally, `mu` is the friction coefficient.
    pub fn new(
        velocity: &'a [[f64; 3]],
        contact_impulse: &'a [f64],
        contact_basis: &'a ContactBasis,
        masses: &'a [f64],
        params: FrictionParams,
        contact_jacobian: (&'a [Matrix3<f64>], CJI),
        elastic_energy: Option<ElasticEnergyParams>,
    ) -> ElasticFrictionSolver<'a, CJI> {
        ElasticFrictionSolver {
            velocity: reinterpret_slice(velocity),
            contact_impulse,
            contact_basis,
            mu: params.dynamic_friction,
            contact_jacobian: ContactJacobian::Full {
                nrows: contact_impulse.len(),
                ncols: velocity.len(),
                blocks: contact_jacobian.0,
                block_indices: contact_jacobian.1,
            }),
            masses,
            elastic_energy,
        }
    }

    /// Solve one step.
    pub fn step(&self) -> Vec<[f64; 3]> {
        let ElasticFrictionSolver {
            velocity,
            masses,
            contact_impulse,
            contact_basis,
            contact_jacobian,
            mu,
            ..
        } = self;
        let mut friction_impulse = vec![Vector3::zeros(); velocity.len()];
        let mut mass_inv_triplets = sprs::TriMat::with_capacity((masses.len(), masses.len()), masses.len());
        for (i, &m) in masses.iter().enumerate() {
            mass_inv_triplets.add_triplet(i, i, 1.0/m);
        }
        let mass_inv = mass_inv_triplets.to_csr();
        let basis = contact_basis.tangent_basis_matrix_sprs();
        let jacobian = contact_jacobian.matrix_sprs();
        let basis_tr_jac = &basis.transpose_view() * &jacobian;
        let left = &basis_tr_jac * &mass_inv;
        let delassus = &left * &basis_tr_jac.transpose_view();

        let flat_velocity: &[f64] = reinterpret_slice(velocity);
        let vel = ndarray::Array::from_iter(flat_velocity.iter().map(|&v| -v));

        let mut rhs = &basis_tr_jac * &vel;
        sprs::linalg::trisolve::lsolve_csr_dense_rhs(delassus.view(), rhs.view_mut().into_slice().unwrap())
            .expect("Failed to solve for impulse.");

        friction_impulse = reinterpret_vec(rhs.to_vec());

        reinterpret_vec(friction_impulse)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Error;
    use approx::*;

    /// A point mass slides across a 2D surface in the positive x direction.
    #[test]
    fn sliding_point() -> Result<(), Error> {
        let mass = 10.0;
        let (velocity, impulse) = sliding_point_tester(0.000001, mass)?;

        // Check that the point still has velocity in the positive x direction
        dbg!(&velocity);
        dbg!(&impulse);
        assert!(velocity[0] > 0.8);

        // Sanity check that no perpendicular velocities or impulses were produced in the process.
        assert_relative_eq!(velocity[1], 0.0, max_relative = 1e-6);
        assert_relative_eq!(impulse[1], 0.0, max_relative = 1e-6);

        // Sanity check that no transverse velocities or impulses are produced.
        assert_relative_eq!(velocity[2], 0.0, max_relative = 1e-6);
        assert_relative_eq!(impulse[2], 0.0, max_relative = 1e-6);

        Ok(())
    }

    #[test]
    fn sticking_point() -> Result<(), Error> {
        let mass = 10.0;
        let (velocity, impulse) = sliding_point_tester(1.5, mass)?;
        // Check that the point gets stuck
        dbg!(&impulse);
        assert_relative_eq!(velocity[0], 0.0, max_relative = 1e-6, epsilon = 1e-8);

        // Sanity check that no perpendicular velocities or impulses were produced in the process
        assert_relative_eq!(velocity[1], 0.0, max_relative = 1e-6);
        assert_relative_eq!(impulse[1], 0.0, max_relative = 1e-6);

        // Sanity check that no transverse velocities or impulses are produced.
        assert_relative_eq!(velocity[2], 0.0, max_relative = 1e-6);
        assert_relative_eq!(impulse[2], 0.0, max_relative = 1e-6);
        Ok(())
    }

    fn sliding_point_tester(mu: f64, mass: f64) -> Result<(Vector3<f64>, Vector3<f64>), Error> {
        let params = FrictionParams {
            dynamic_friction: mu,
            inner_iterations: 30,
            tolerance: 1e-10,
            print_level: 5,
        };

        let velocity = vec![[1.0, 0.0, 0.0]]; // one point sliding right.
        let contact_impulse = vec![10.0 * mass];
        let masses = vec![mass; 1];

        let mut contact_basis = ContactBasis::new();
        contact_basis.update_from_normals(vec![[0.0, 1.0, 0.0]]);

        let solver = ElasticFrictionSolver::selection_contact_jacobian(
            &velocity,
            &contact_impulse,
            &contact_basis,
            &masses,
            &masses,
            params,
            &[0],
            None,
        );
        let solution = solver.step();

        let impulse = Vector3(solution[0]);
        let final_velocity = Vector3(velocity[0]) + impulse / mass;

        Ok((final_velocity, impulse))
    }
}
