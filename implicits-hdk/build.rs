use cxx_build::CFG;
use std::path::{Path, PathBuf};

fn main() {
    CFG.include_prefix = "implicits";

    let build = cxx_build::bridge("src/lib.rs");
    cmake::Config::new(".")
        .no_build_target(true)
        .init_c_cfg(build.clone())
        .init_cxx_cfg(build)
        .build();

    let out_dir = std::env::var("OUT_DIR").unwrap();

    println!("cargo:rerun-if-changed=src/lib.rs");
    println!("cargo:rustc-link-lib=static=cxxbridge");
    println!("cargo:rustc-link-search=native={}", out_dir);
}
