use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use geo::ops::*;
use geo::prim::{Tetrahedron, Triangle};
use num_traits::Zero;
use softy::energy_models::elasticity::{
    NeoHookeanTetEnergy, NeoHookeanTriEnergy, StableNeoHookeanTetEnergy, TetEnergy, TriEnergy,
};
use softy::objects::TriMeshShell;
use tensr::*;

/// Compute the tet energy hessian directly using the energy_hessian implementation.
#[allow(non_snake_case)]
pub(crate) fn tet_energy_hessian_direct<E: TetEnergy<f64>>() -> [[Matrix3<f64>; 4]; 4] {
    let verts = vec![[0.0; 3], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]];
    let cell = [0, 1, 2, 3];

    let tet_x0 = Tetrahedron::from_indexed_slice(&cell, &verts);

    let DX_inv = Matrix3::new(tet_x0.clone().shape_matrix())
        .inverse_transpose()
        .unwrap();
    let Dx = Matrix3::new(tet_x0.clone().shape_matrix()).transpose();
    let energy = E::new(Dx, DX_inv, 1.0, 1.0, 1.0);
    energy.energy_hessian()
}

/// Compute the tet energy hessian using products.
#[allow(non_snake_case)]
pub(crate) fn tet_energy_hessian_product<E: TetEnergy<f64>>() -> [[Matrix3<f64>; 4]; 4] {
    let verts = vec![[0.0; 3], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]];
    let cell = [0, 1, 2, 3];

    let tet_x0 = Tetrahedron::from_indexed_slice(&cell, &verts);

    let DX_inv = Matrix3::new(tet_x0.clone().shape_matrix())
        .inverse_transpose()
        .unwrap();
    let Dx = Matrix3::new(tet_x0.clone().shape_matrix()).transpose();
    let energy = E::new(Dx, DX_inv, 1.0, 1.0, 1.0);

    let mut d_verts = [
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
    ];

    let mut hess = [[Matrix3::zero(); 4]; 4];
    for wrt_vtx in 0..3 {
        for wrt_i in 0..3 {
            d_verts[wrt_vtx][wrt_i] = 1.0;
            let tet_dx = Tetrahedron::from_indexed_slice(&cell, &d_verts[..]);

            let h = energy.energy_hessian_product_transpose(&tet_dx);
            for vtx in 0..3 {
                for i in 0..3 {
                    if wrt_vtx < vtx || (wrt_vtx == vtx && i >= wrt_i) {
                        hess[vtx][wrt_vtx][i][wrt_i] = h[vtx][i];
                    }
                    hess[3][wrt_vtx][i][wrt_i] -= h[vtx][i];
                    if i >= wrt_i {
                        hess[3][3][i][wrt_i] += h[vtx][i];
                    }
                }
            }
            d_verts[wrt_vtx][wrt_i] = 0.0;
        }
    }

    hess
}

/// Compute the triangle energy hessian directly using the energy_hessian implementation.
#[allow(non_snake_case)]
pub(crate) fn tri_energy_hessian_direct<E: TriEnergy<f64>>() -> [[Matrix3<f64>; 3]; 3] {
    let verts = vec![[0.0; 3], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]];
    let tri = [0, 1, 2];

    let tri_x0 = Triangle::from_indexed_slice(&tri, &verts);

    let shape = Matrix2x3::new(tri_x0.clone().shape_matrix());
    let DX_inv = TriMeshShell::isotropic_tri_shape_matrix(shape)
        .inverse()
        .unwrap();
    let Dx = Matrix2x3::new(tri_x0.clone().shape_matrix());
    let energy = E::new(Dx, DX_inv, 1.0, 1.0, 1.0);
    energy.energy_hessian()
}

/// Compute the triangle energy hessian using products.
#[allow(non_snake_case)]
pub(crate) fn tri_energy_hessian_product<E: TriEnergy<f64>>() -> [[Matrix3<f64>; 3]; 3] {
    let verts = vec![[0.0; 3], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]];
    let tri = [0, 1, 2];

    let tri_x0 = Triangle::from_indexed_slice(&tri, &verts);

    let shape = Matrix2x3::new(tri_x0.clone().shape_matrix());
    let DX_inv = TriMeshShell::isotropic_tri_shape_matrix(shape)
        .inverse()
        .unwrap();
    let Dx = Matrix2x3::new(tri_x0.clone().shape_matrix());
    let energy = E::new(Dx, DX_inv, 1.0, 1.0, 1.0);

    let mut d_verts = [
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
    ];

    let mut hess = [[Matrix3::zero(); 3]; 3];
    for wrt_vtx in 0..2 {
        for wrt_i in 0..3 {
            d_verts[wrt_vtx][wrt_i] = 1.0;
            let tri_dx = Triangle::from_indexed_slice(&tri, &d_verts[..]);

            let h = energy.energy_hessian_product_transpose(&tri_dx);
            for vtx in 0..2 {
                for i in 0..3 {
                    if wrt_vtx < vtx || (wrt_vtx == vtx && i >= wrt_i) {
                        hess[vtx][wrt_vtx][i][wrt_i] = h[vtx][i];
                    }
                    hess[2][wrt_vtx][i][wrt_i] -= h[vtx][i];
                    if i >= wrt_i {
                        hess[2][2][i][wrt_i] += h[vtx][i];
                    }
                }
            }
            d_verts[wrt_vtx][wrt_i] = 0.0;
        }
    }

    hess
}

fn energy_hessian(c: &mut Criterion) {
    let mut group = c.benchmark_group("Tet Energy Hessian");

    //// Print hessian
    //for n in 0..4 {
    //    for i in 0..3 {
    //        for k in 0..4 {
    //            for j in 0..3 {
    //                eprint!("{:10.2e}", hess[n][k][i][j]);
    //            }
    //            eprint!("\t");
    //        }
    //        eprintln!("");
    //    }
    //}
    //eprintln!("");
    assert_eq!(
        tet_energy_hessian_direct::<NeoHookeanTetEnergy<f64>>(),
        tet_energy_hessian_product::<NeoHookeanTetEnergy<f64>>()
    );

    group.bench_function(BenchmarkId::new("Direct", "NH"), |b| {
        b.iter(|| tet_energy_hessian_direct::<NeoHookeanTetEnergy<f64>>())
    });

    group.bench_function(BenchmarkId::new("Product", "NH"), |b| {
        b.iter(|| tet_energy_hessian_product::<NeoHookeanTetEnergy<f64>>())
    });

    group.bench_function(BenchmarkId::new("Direct", "SNH"), |b| {
        b.iter(|| tet_energy_hessian_direct::<StableNeoHookeanTetEnergy<f64>>())
    });

    group.bench_function(BenchmarkId::new("Product", "SNH"), |b| {
        b.iter(|| tet_energy_hessian_product::<StableNeoHookeanTetEnergy<f64>>())
    });

    group.finish();

    let mut group = c.benchmark_group("Tri Energy Hessian");

    group.bench_function(BenchmarkId::new("Direct", "Tri NH"), |b| {
        b.iter(|| tri_energy_hessian_direct::<NeoHookeanTriEnergy<f64>>())
    });

    group.bench_function(BenchmarkId::new("Product", "Tri NH"), |b| {
        b.iter(|| tri_energy_hessian_product::<NeoHookeanTriEnergy<f64>>())
    });

    group.finish();
}

criterion_group!(benches, energy_hessian);
criterion_main!(benches);
