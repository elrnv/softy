extern crate cbindgen;

use std::env;
use std::fs;
use std::path::PathBuf;

fn main() {
    let crate_dir = env::var("CARGO_MANIFEST_DIR").unwrap();

    let package_name = env::var("CARGO_PKG_NAME").unwrap();
    let header_file = format!("{}.h", package_name);
    let output_file = target_dir().join(header_file.clone()).display().to_string();

    let mut config: cbindgen::Config = Default::default();

    config.include_guard = Some(String::from("IMPLICITS_HDK_H"));
    //config.namespace = Some(String::from("hdkrs"));
    config.line_length = 80;
    config.tab_width = 4;
    config.language = cbindgen::Language::Cxx;
    config.parse.parse_deps = true;
    config.parse.include = Some(vec!["cimplicits".to_string()]);
    config.parse.extra_bindings = vec!["cimplicits".to_string()];

    cbindgen::generate_with_config(&crate_dir, config)
        .expect("Unable to generate bindings")
        .write_to_file(&output_file);

    // Copy artifact to where CMake can find it easily.
    fs::copy(&output_file, &cmake_target_dir().join(header_file.clone())).unwrap();
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
