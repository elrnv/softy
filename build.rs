extern crate cbindgen;

use std::env;
use std::path::PathBuf;

fn main() {
    let crate_dir = env::var("CARGO_MANIFEST_DIR").unwrap();

    let package_name = env::var("CARGO_PKG_NAME").unwrap();
    let output_file = target_dir()
        .join(format!("{}.h", package_name))
        .display()
        .to_string();

    //let mut parse_config: cbindgen::ParseConfig = Default::default();
    //parse_config.parse_deps = true;

    let mut config: cbindgen::Config = Default::default();

    config.include_guard = Some(String::from("TEST_CAPI_H"));
    config.namespace = Some(String::from("hdkrs"));
    config.line_length = 80;
    config.tab_width = 4;
    config.language = cbindgen::Language::Cxx;
    //config.parse = parse_config;

    cbindgen::generate_with_config(&crate_dir, config)
        .expect("Unable to generate bindings")
        .write_to_file(&output_file);
}

fn target_dir() -> PathBuf {
    let crate_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
    PathBuf::from(crate_dir).join("target")
}
