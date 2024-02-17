extern crate cbindgen;

use std::env;
use std::fs;
use std::path::PathBuf;

fn main() {
    let crate_dir = env::var("CARGO_MANIFEST_DIR").unwrap();

    let header_file = "cimplicits.h";
    let output_file = target_dir().join(header_file).display().to_string();

    let mut config: cbindgen::Config = Default::default();

    config.include_guard = Some(String::from("CIMPLICITS_H"));
    config.line_length = 80;
    config.tab_width = 4;
    config.language = cbindgen::Language::C;
    config.cpp_compat = true;

    cbindgen::generate_with_config(&crate_dir, config)
        .expect("Unable to generate bindings")
        .write_to_file(&output_file);

    // Copy artifact to where CMake can find it easily.
    fs::copy(&output_file, &cmake_target_dir().join(header_file)).unwrap();

    if cfg!(target_os = "linux") {
        // Set soname on linux to make this lib more portable.
        println!("cargo:rustc-cdylib-link-arg=-Wl,-soname=libcimplicits.so");
    }
}

fn target_dir() -> PathBuf {
    let out_dir = env::var("OUT_DIR").unwrap();
    PathBuf::from(out_dir)
}

fn cmake_target_dir() -> PathBuf {
    let target_dir = target_dir();
    // Point to where the library is stored.
    PathBuf::from(
        target_dir
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .parent()
            .unwrap(),
    )
}
