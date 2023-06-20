use crate::energy::*;
use crate::matrix::*;
use crate::objects::tetsolid::*;
use crate::objects::trishell::*;
use crate::Real;
use flatk::zip;
use geo::ops::*;
use geo::prim::{Tetrahedron, Triangle};
use tensr::Vector3;

/// This trait defines a convenient accessor for the specific gravity implementation for a given
/// object.
pub trait Gravity<'a, E> {
    fn gravity(&'a self, g: [f64; 3]) -> E;
}

/*
 * Gravity for `TetSolid`s.
 */

pub struct TetSolidGravity<'a> {
    solid: &'a TetElements,
    g: Vector3<f64>,
}

impl<'a> TetSolidGravity<'a> {
    pub fn new(solid: &'a TetElements, gravity: [f64; 3]) -> TetSolidGravity<'a> {
        TetSolidGravity {
            solid,
            g: gravity.into(),
        }
    }
}

/// Define energy for gravity.
/// Gravity is a position based energy.
impl<T: Real> Energy<T> for TetSolidGravity<'_> {
    /// Since gravity depends on position, `x` is expected to be a position quantity.
    fn energy(&self, x: &[T], _: &[T], _dqdv: T) -> T {
        let pos1: &[Vector3<T>] = bytemuck::cast_slice(x);
        let tet_elems = &self.solid;
        let tet_iter = tet_elems
            .tets
            .iter()
            .map(|cell| Tetrahedron::from_indexed_slice(cell, pos1));

        let g = self.g.cast::<T>();

        zip!(
            tet_elems.ref_volume.iter(),
            tet_elems.density.iter().map(|&x| f64::from(x)),
            tet_iter
        )
        .map(|(&vol, density, tet)| {
            // We really want mass here. Since mass is conserved we can rely on reference
            // volume and density.
            g.dot(Vector3::new(tet.centroid())) * T::from(-vol * density).unwrap()
        })
        .sum()
    }
}

impl<X: Real, T: Real> EnergyGradient<X, T> for TetSolidGravity<'_> {
    /// Add the gravity gradient to the given global vector.
    fn add_energy_gradient(&self, _x: &[X], _dx: &[T], grad: &mut [T], _dqdv: T) {
        debug_assert_eq!(grad.len(), _x.len());

        let tet_elems = &self.solid;
        let gradient: &mut [Vector3<T>] = bytemuck::cast_slice_mut(grad);

        let g = self.g.cast::<T>();

        // Transfer forces from cell-vertices to vertices themeselves
        for (&vol, density, cell) in zip!(
            tet_elems.ref_volume.iter(),
            tet_elems.density.iter().map(|&x| f64::from(x)),
            tet_elems.tets.iter()
        ) {
            for i in 0..4 {
                // Energy gradient is in opposite direction to the force hence minus here.
                gradient[cell[i]] -= g * T::from(0.25 * vol * density).unwrap();
            }
        }
    }
}

impl EnergyHessianTopology for TetSolidGravity<'_> {
    fn energy_hessian_size(&self) -> usize {
        0
    }
    fn num_hessian_diagonal_nnz(&self) -> usize {
        0
    }
    fn energy_hessian_indices_offset(&self, _: MatrixElementIndex, _: &mut [MatrixElementIndex]) {}
}
impl<T: Real> EnergyHessian<T> for TetSolidGravity<'_> {
    fn energy_hessian_values(&self, _x0: &[T], _x1: &[T], _scale: T, _vals: &mut [T], _dqdv: T) {}
    fn add_energy_hessian_diagonal(
        &self,
        _x0: &[T],
        _x1: &[T],
        _scale: T,
        _diag: &mut [T],
        _dqdv: T,
    ) {
    }
}

/*
 * Gravity for `TriShell`s
 */

pub struct SoftTriShellGravity<'a> {
    shell: &'a TriShell,
    g: Vector3<f64>,
}

impl<'a> SoftTriShellGravity<'a> {
    pub fn new(shell: &'a TriShell, gravity: [f64; 3]) -> SoftTriShellGravity<'a> {
        SoftTriShellGravity {
            shell,
            g: gravity.into(),
        }
    }
}

/// Define energy for gravity.
/// Gravity is a position based energy.
impl<T: Real> Energy<T> for SoftTriShellGravity<'_> {
    /// Since gravity depends on position, `x` is expected to be a position quantity.
    fn energy(&self, x: &[T], _dx: &[T], _dqdv: T) -> T {
        let pos1: &[Vector3<T>] = bytemuck::cast_slice(x);
        let tri_elems = &self.shell.triangle_elements;
        let tri_iter = tri_elems
            .triangles
            .iter()
            .map(|face| Triangle::from_indexed_slice(face, pos1));

        let g = self.g.cast::<T>();

        zip!(
            tri_elems.ref_area.iter(),
            tri_elems.density.iter().map(|&x| f64::from(x)),
            tri_iter
        )
        .map(|(&area, density, tri)| {
            // We really want mass here. Since mass is conserved we can rely on reference
            // volume and density.
            g.dot(Vector3::new(tri.centroid())) * T::from(-area * density).unwrap()
        })
        .sum()
    }
}

impl<T: Real> EnergyGradient<T, T> for SoftTriShellGravity<'_> {
    /// Add the gravity gradient to the given global vector.
    fn add_energy_gradient(&self, _x: &[T], _v: &[T], grad: &mut [T], _dqdv: T) {
        debug_assert_eq!(grad.len(), _x.len());

        let tri_elems = &self.shell.triangle_elements;
        let gradient: &mut [Vector3<T>] = bytemuck::cast_slice_mut(grad);

        let g = self.g.cast::<T>();

        let third = 1.0 / 3.0;

        // Transfer forces from cell-vertices to vertices themeselves
        for (&area, density, face) in zip!(
            tri_elems.ref_area.iter(),
            tri_elems.density.iter().map(|&x| f64::from(x)),
            tri_elems.triangles.iter()
        ) {
            for i in 0..3 {
                // Energy gradient is in opposite direction to the force hence minus here.
                gradient[face[i]] -= g * T::from(third * area * density).unwrap();
            }
        }
    }
}

impl EnergyHessianTopology for SoftTriShellGravity<'_> {
    fn energy_hessian_size(&self) -> usize {
        0
    }
    fn num_hessian_diagonal_nnz(&self) -> usize {
        0
    }
    fn energy_hessian_indices_offset(&self, _: MatrixElementIndex, _: &mut [MatrixElementIndex]) {}
}

impl<T: Real> EnergyHessian<T> for SoftTriShellGravity<'_> {
    fn energy_hessian_values(&self, _x: &[T], _dx: &[T], _scale: T, _vals: &mut [T], _dqdv: T) {}
    fn add_energy_hessian_diagonal(
        &self,
        _x: &[T],
        _dx: &[T],
        _scale: T,
        _diag: &mut [T],
        _dqdv: T,
    ) {
    }
}

/*
 * Gravity for trimesh shells
 */

/// A constant directional force.
pub struct RigidShellGravity {
    mass: f64,
    g: Vector3<f64>,
}

impl RigidShellGravity {
    pub fn new(gravity: [f64; 3], mass: f64) -> RigidShellGravity {
        RigidShellGravity {
            g: gravity.into(),
            mass,
        }
    }
}

/// Define energy for gravity.
/// Gravity is a position based energy.
impl<T: Real> Energy<T> for RigidShellGravity {
    /// Since gravity depends on position, `x` is expected to be a position quantity.
    fn energy(&self, x: &[T], _dx: &[T], _dqdv: T) -> T {
        let pos = Vector3::new([x[0], x[1], x[2]]);
        T::from(-self.mass).unwrap() * self.g.cast::<T>().dot(pos)
    }
}

impl<X: Real, T: Real> EnergyGradient<X, T> for RigidShellGravity {
    /// Add the gravity gradient to the given global vector.
    fn add_energy_gradient(&self, _x: &[X], _dx: &[T], grad: &mut [T], _dqdv: T) {
        use tensr::AsMutTensor;
        debug_assert_eq!(grad.len(), _x.len());

        *grad[0..3].as_mut_tensor() -= self.g.cast::<T>() * T::from(self.mass).unwrap();
    }
}

impl EnergyHessianTopology for RigidShellGravity {
    fn energy_hessian_size(&self) -> usize {
        0
    }
    fn num_hessian_diagonal_nnz(&self) -> usize {
        0
    }
    fn energy_hessian_indices_offset(&self, _: MatrixElementIndex, _: &mut [MatrixElementIndex]) {}
}

impl<T: Real> EnergyHessian<T> for RigidShellGravity {
    fn energy_hessian_values(&self, _x0: &[T], _x1: &[T], _scale: T, _vals: &mut [T], _dqdv: T) {}
    fn add_energy_hessian_diagonal(
        &self,
        _x0: &[T],
        _x1: &[T],
        _scale: T,
        _diag: &mut [T],
        _dqdv: T,
    ) {
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::energy_models::test_utils::*;
    use crate::fem::nl::SolverBuilder;
    use crate::objects::{tetsolid::TetElements, *};
    use geo::mesh::{Mesh, VertexPositions};

    mod solid {
        use crate::SolidMaterial;

        use super::*;

        fn solid_material() -> SolidMaterial {
            SolidMaterial::new(0)
                .with_elasticity(Elasticity::from_lame(
                    5.4,
                    263.1,
                    ElasticityModel::NeoHookean,
                ))
                .with_density(1000.0)
                .with_damping(2.0)
        }

        fn test_solids() -> Vec<(TetElements, Vec<[f64; 3]>)> {
            let material = solid_material();

            test_tetmeshes()
                .into_iter()
                .map(|tetmesh| Mesh::from(tetmesh))
                .flat_map(|mut mesh| {
                    // Prepare attributes relevant for elasticity computations.
                    SolverBuilder::init_cell_vertex_ref_pos_attribute(&mut mesh).unwrap();
                    let materials = vec![material.into()];
                    let vertex_types =
                        crate::fem::nl::state::sort_mesh_vertices_by_type(&mut mesh, &materials);
                    let build_tet_elements = |model: ElasticityModel| {
                        TetElements::try_from_mesh_and_materials(
                            model,
                            &mesh,
                            &materials,
                            vertex_types.as_slice(),
                            false,
                        )
                        .unwrap()
                    };
                    std::iter::once((
                        build_tet_elements(ElasticityModel::NeoHookean),
                        mesh.vertex_positions().to_vec(),
                    ))
                    .chain(std::iter::once((
                        build_tet_elements(ElasticityModel::StableNeoHookean),
                        mesh.vertex_positions().to_vec(),
                    )))
                })
                .collect()
        }

        fn build_energies(
            solids: &[(TetElements, Vec<[f64; 3]>)],
        ) -> Vec<(TetSolidGravity, &[[f64; 3]])> {
            solids
                .iter()
                .map(|(solid, pos)| {
                    (
                        TetSolidGravity::new(solid, [0.0, -9.81, 0.0]),
                        pos.as_slice(),
                    )
                })
                .collect()
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

    mod soft_shell {
        use super::*;
        use crate::objects::trishell::TriShell;
        use crate::SoftShellMaterial;

        fn soft_shell_material() -> SoftShellMaterial {
            SoftShellMaterial::new(0)
                .with_elasticity(Elasticity::from_lame(
                    5.4,
                    263.1,
                    ElasticityModel::NeoHookean,
                ))
                .with_density(1000.0)
                .with_damping(1.0)
                .with_bending_stiffness(2.0)
        }

        fn test_shells() -> Vec<(TriShell, Vec<[f64; 3]>)> {
            let material = soft_shell_material();

            test_trimeshes()
                .into_iter()
                .map(|trimesh| {
                    let mut mesh = Mesh::from(trimesh);
                    let materials = vec![material.into()];
                    SolverBuilder::init_cell_vertex_ref_pos_attribute(&mut mesh).unwrap();
                    let vertex_types =
                        crate::fem::nl::state::sort_mesh_vertices_by_type(&mut mesh, &materials);
                    (
                        TriShell::try_from_mesh_and_materials(&mesh, &materials, &vertex_types)
                            .unwrap(),
                        mesh.vertex_positions().to_vec(),
                    )
                })
                .collect()
        }

        fn build_energies(
            shells: &[(TriShell, Vec<[f64; 3]>)],
        ) -> Vec<(SoftTriShellGravity, &[[f64; 3]])> {
            shells
                .iter()
                .map(|(shell, pos)| {
                    (
                        SoftTriShellGravity::new(shell, [0.0, -9.81, 0.0]),
                        pos.as_slice(),
                    )
                })
                .collect()
        }

        #[test]
        fn gradient() {
            let shells = test_shells();
            gradient_tester(build_energies(&shells), EnergyType::Position);
        }

        #[test]
        fn hessian() {
            let shells = test_shells();
            hessian_tester(build_energies(&shells), EnergyType::Position);
        }
    }
}
