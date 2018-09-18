use softy;
use geo;
use hdkrs::interop::CookResult;

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct MaterialProperties {
    pub bulk_modulus: f32,
    pub shear_modulus: f32,
    pub density: f32,
    pub damping: f32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct SimParams {
    pub material: MaterialProperties,
    pub time_step: f32,
    pub gravity: f32,
    pub tolerance: f32,
}

impl Into<softy::SimParams> for SimParams {
    fn into(self) -> softy::SimParams {
        let SimParams {
            material: MaterialProperties {
                bulk_modulus,
                shear_modulus,
                density,
                damping,
            },
            time_step,
            gravity,
            tolerance,
        } = self;
        softy::SimParams {
            material: softy::MaterialProperties {
                bulk_modulus,
                shear_modulus,
                density,
                damping,
            },
            time_step: if time_step > 0.0 { Some(time_step) } else { None },
            gravity: [0.0, -gravity, 0.0],
            tolerance,
        }
    }
}

/// Main entry point to Rust code.
pub fn cook<F>(
    tetmesh: Option<&mut geo::mesh::TetMesh<f64>>,
    polymesh: Option<&mut geo::mesh::PolyMesh<f64>>,
    params: SimParams,
    check_interrupt: F,
    ) -> CookResult
where
    F: Fn() -> bool + Sync + Send,
{
    let res = softy::sim(tetmesh, polymesh, params.into(), check_interrupt);
    convert_to_cookresult(res)
}

fn convert_to_cookresult(res: softy::SimResult) -> CookResult {
    match res {
        softy::SimResult::Success(msg) => CookResult::Success(msg),
        softy::SimResult::Warning(msg) => CookResult::Warning(msg),
        softy::SimResult::Error(msg) => CookResult::Error(msg),
    }
}

