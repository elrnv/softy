extern crate cbindgen;

use std::env;

fn main() {
    let crate_dir = env::var("CARGO_MANIFEST_DIR").unwrap();

    let mut config: cbindgen::Config = Default::default();

    config.include_guard = Some(String::from("TEST_CAPI_H"));
    config.namespace = Some(String::from("test"));
    config.line_length = 80;
    config.tab_width = 4;
    config.language = cbindgen::Language::Cxx;

    println!("crate dir = {:?}", crate_dir);

    cbindgen::generate_with_config(&crate_dir, config)
        .expect("Unable to generate bidnings")
        .write_to_file("testhdk.h");
}
