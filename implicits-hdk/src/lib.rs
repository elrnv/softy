mod api;

use std::pin::Pin;

pub use cimplicits::iso_default_params;
pub use cimplicits::ISO_Params;

#[cxx::bridge(namespace = "implicits")]
mod ffi {
    /// The particular iso-surface related action to be taken.
    #[derive(Copy, Clone, Debug)]
    pub enum Action {
        /// Compute the implicit field at a given set of query points.
        ComputePotential,
        /// Project the given set of points to one side of the iso-surface.
        Project,
    }

    /// A C interface for passing parameters from SOP parameters to the Rust code.
    #[derive(Copy, Clone, Debug)]
    pub struct Params {
        pub action: Action,
        pub iso_value: f32,      // Only used for projection
        pub project_below: bool, // Only used for projection
        pub debug: bool,         // Only used for potential computation
        pub iso_params: ISO_Params,
    }

    #[namespace = ""]
    extern "C++" {
        include!("hdkrs/src/lib.rs.h");
        include!("cimplicits.h");
        type ISO_Params = cimplicits::ISO_Params;
        type GU_Detail = hdkrs::ffi::GU_Detail;
    }

    extern "Rust" {
        type PointCloud;
        type PolyMesh;
        fn num_verts(self: &PointCloud) -> usize;
        fn build_polymesh(detail: &GU_Detail) -> Result<Box<PolyMesh>>;
        fn build_pointcloud(detail: &GU_Detail) -> Result<Box<PointCloud>>;
        fn update_points(detail: Pin<&mut GU_Detail>, ptcloud: &PointCloud);
        fn cook(
            querymesh: Pin<&mut PointCloud>,
            polymesh: Pin<&mut PolyMesh>,
            params: Params,
            interrupt_checker: UniquePtr<InterruptChecker>,
        ) -> CookResult;
        fn default_params() -> Params;
    }

    #[namespace = "hdkrs"]
    extern "C++" {
        type InterruptChecker = hdkrs::ffi::InterruptChecker;
        type CookResult = hdkrs::ffi::CookResult;
    }
}

use ffi::*;

// TODO: At the time of this writing it is not clear how to expose Rust types over the cxx boundary from hdkrs,
// So we do this manually here. These are simply wrappers.

#[derive(Clone, Debug, PartialEq)]
#[repr(transparent)]
pub struct PointCloud(pub hdkrs::PointCloud);
#[derive(Clone, Debug, PartialEq)]
#[repr(transparent)]
pub struct PolyMesh(pub hdkrs::PolyMesh);

impl PointCloud {
    fn num_verts(&self) -> usize {
        use geo::mesh::topology::NumVertices;
        self.0 .0.num_vertices()
    }
}

fn build_polymesh(detail: &GU_Detail) -> Result<Box<PolyMesh>, cxx::Exception> {
    Ok(Box::new(PolyMesh(*hdkrs::ffi::build_polymesh(detail)?)))
}

fn build_pointcloud(detail: &GU_Detail) -> Result<Box<PointCloud>, cxx::Exception> {
    Ok(Box::new(PointCloud(*hdkrs::ffi::build_pointcloud(detail)?)))
}

fn update_points(detail: Pin<&mut GU_Detail>, pts: &PointCloud) {
    hdkrs::ffi::update_points(detail, &pts.0);
}

impl Default for Params {
    fn default() -> Self {
        Params {
            action: Action::ComputePotential,
            iso_value: 0.0,
            project_below: false,
            debug: false,
            iso_params: ISO_Params::default(),
        }
    }
}

/// Construct a default set of input parameters.
///
/// It is safer to use this function than populating the fields manually since it guarantees that
/// every field is properly initialized.
pub fn default_params() -> Params {
    Default::default()
}

impl Into<implicits::Params> for Params {
    fn into(self) -> implicits::Params {
        self.iso_params.into()
    }
}

/// Main entry point from Houdini SOP.
///
/// The purpose of this function is to cleanup the inputs for use in Rust code.
pub fn cook(
    querymesh: Pin<&mut PointCloud>,
    polymesh: Pin<&mut PolyMesh>,
    params: Params,
    mut interrupt_checker: cxx::UniquePtr<InterruptChecker>,
) -> CookResult {
    api::cook(
        &mut querymesh.get_mut().0 .0,
        &mut polymesh.get_mut().0 .0,
        params.into(),
        move || interrupt_checker.pin_mut().check_interrupt(),
    )
    .into()
}
