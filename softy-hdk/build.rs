use cxx_build::CFG;

fn main() {
    CFG.include_prefix = "softy";

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

//extern crate cbindgen;
//
//use std::env;
//use std::fs;
//use std::path::PathBuf;
//fn main() {
//    let crate_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
//
//    let package_name = env::var("CARGO_PKG_NAME").unwrap();
//    let header_file = format!("{}.h", package_name);
//    let output_file = target_dir().join(header_file.clone()).display().to_string();
//
//    let mut config: cbindgen::Config = Default::default();
//
//    config.include_guard = Some(String::from("SOFTY_CAPI_H"));
//    config.line_length = 80;
//    config.tab_width = 4;
//    config.cpp_compat = true;
//    config.language = cbindgen::Language::Cxx;
//
//    cbindgen::generate_with_config(&crate_dir, config)
//        .expect("Unable to generate bindings")
//        .write_to_file(&output_file);
//
//    // Copy artifact to where CMake can find it easily.
//    fs::copy(&output_file, &cmake_target_dir().join(header_file.clone())).unwrap();
//}
//
//fn target_dir() -> PathBuf {
//    let out_dir = env::var("OUT_DIR").unwrap();
//    PathBuf::from(out_dir)
//}
//
//fn cmake_target_dir() -> PathBuf {
//    let target_dir = target_dir();
//    // Point to where the library is stored.
//    PathBuf::from(
//        target_dir
//            .parent()
//            .unwrap()
//            .parent()
//            .unwrap()
//            .parent()
//            .unwrap(),
//    )
//}
