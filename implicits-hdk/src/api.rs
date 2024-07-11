use crate::{Action, Params};
/**
 * Application specific code goes here.
 * The Rust cook entry point is defined here.
 * This file is intended to be completely free from C FFI except for POD types, which must be
 * designated as `#[repr(C)]`.
 */
use geo::{
    self,
    attrib::*,
    mesh::{topology::*, VertexPositions},
};
use hdkrs::interop::CookResult;
use implicits::{self, ImplicitSurface};

fn project_vertices(
    query_mesh: &mut geo::mesh::PointCloud<f64>,
    surface: &mut geo::mesh::PolyMesh<f64>,
    params: Params,
) -> Result<bool, implicits::Error> {
    let mut surf = implicits::mls_from_polymesh(surface, params.into())?;
    surf.reverse_par(); // Reverse normals for compatibility with HDK
    let query_surf = surf.query_topo(query_mesh.vertex_positions());

    let pos = query_mesh.vertex_positions_mut();
    let converged = if params.project_below {
        query_surf.project_to_below_par(f64::from(params.iso_value), 1e-4, pos)
    } else {
        query_surf.project_to_above_par(f64::from(params.iso_value), 1e-4, pos)
    };

    Ok(converged)
}

/// Main entry point to Rust code.
pub fn cook<F>(
    query_mesh: &mut geo::mesh::PointCloud<f64>,
    surface: &mut geo::mesh::PolyMesh<f64>,
    bg_samples: &mut geo::mesh::PointCloud<f64>,
    params: Params,
    check_interrupt: F,
) -> CookResult
where
    F: FnMut() -> bool + Sync + Send + 'static,
{
    let interrupt_checker = std::sync::Arc::new(std::sync::Mutex::new(check_interrupt));
    let interrupt = move || {
        if let Ok(mut interrupt) = interrupt_checker.lock() {
            interrupt()
        } else {
            // If we cannot acquire the lock, let the computation go on uninterrupted.
            false
        }
    };

    match params.action {
        Action::ComputePotential => {
            if params.debug {
                let res = implicits::surface_from_polymesh_with_samples(
                    surface,
                    bg_samples,
                    params.into(),
                )
                .and_then(|mut surf| {
                    surf.reverse_par(); // reverse polygons for compatibility with hdk
                    surf.compute_potential_on_mesh(query_mesh, interrupt)
                });
                convert_to_cookresult(res.map(|_| true))
            } else {
                let res = implicits::surface_from_polymesh_with_samples(
                    surface,
                    bg_samples,
                    params.into(),
                )
                .and_then(|mut surf| {
                    surf.reverse_par(); // reverse polygons for compatibility with hdk
                                        // Get or create a new potential attribute.
                    let potential_attrib = query_mesh
                        .remove_attrib::<VertexIndex>("potential")
                        .ok()
                        .unwrap_or_else(|| {
                            Attribute::direct_from_vec(vec![0.0f32; query_mesh.num_vertices()])
                        });

                    let mut potential = potential_attrib
                        .into_data()
                        .cast_into_vec::<f64>()
                        .unwrap_or_else(|| vec![0.0f64; query_mesh.num_vertices()]);

                    match surf {
                        ImplicitSurface::MLS(mls) => mls
                            .query_topo(query_mesh.vertex_positions())
                            .potential_par_interrupt(
                                query_mesh.vertex_positions(),
                                &mut potential,
                                interrupt,
                            ),
                        ImplicitSurface::Hrbf(hrbf) => ImplicitSurface::compute_hrbf_on_mesh(
                            query_mesh,
                            &hrbf.surf_base.samples,
                            interrupt,
                        ),
                    }
                    .ok();

                    query_mesh.insert_attrib_data::<_, VertexIndex>("potential", potential)?;
                    Ok(())
                });
                convert_to_cookresult(res.map(|_| true))
            }
        }
        Action::Project => {
            let res = project_vertices(query_mesh, surface, params);
            convert_to_cookresult(res)
        }
        _ => CookResult::Error("Unrecognized action".to_string()),
    }
}

fn convert_to_cookresult(res: Result<bool, implicits::Error>) -> CookResult {
    match res {
        Ok(true) => CookResult::Success("".to_string()),
        Ok(false) => CookResult::Warning("Projection failed to converge".to_string()),
        Err(implicits::Error::Interrupted) => {
            CookResult::Error("Execution was interrupted".to_string())
        }
        Err(implicits::Error::MissingNormals) => {
            CookResult::Error("Vertex normals are missing or have the wrong type".to_string())
        }
        Err(implicits::Error::MissingNeighborData) => {
            CookResult::Error("Missing neighbor data for derivative computations".to_string())
        }
        Err(implicits::Error::Failure) => CookResult::Error("Internal Error".to_string()),
        Err(implicits::Error::UnsupportedKernel) => {
            CookResult::Error("Given kernel is not supported yet".to_string())
        }
        Err(implicits::Error::InvalidBackgroundConstruction) => {
            CookResult::Error("Invalid Background field".to_string())
        }
        Err(implicits::Error::UnsupportedSampleType) => CookResult::Error(
            "Given sample type is not supported by the chosen configuration".to_string(),
        ),
        Err(implicits::Error::IO { source }) => {
            CookResult::Error(format!("IO Error: {:?}", source))
        }
    }
}
