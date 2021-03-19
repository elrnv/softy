use cxx_build::CFG;
use std::path::{Path, PathBuf};

fn main() {
    CFG.include_prefix = "implicits";

    let out_path = PathBuf::from(std::env::var("OUT_DIR").unwrap());

    let target_dir = out_path
        .parent()
        .and_then(Path::parent)
        .and_then(Path::parent)
        .unwrap();

    let mut build = cxx_build::bridge("src/lib.rs");

    // For cimplicits.h include
    build.include(target_dir);

    cmake::Config::new(".")
        .no_build_target(true)
        .init_c_cfg(build.clone())
        .init_cxx_cfg(build)
        .build();

    println!("cargo:rerun-if-changed=src/lib.rs");
    println!("cargo:rustc-link-lib=static=cxxbridge");
    println!("cargo:rustc-link-search=native={}", out_path.display());
}
