# Softy

Simulation tools and libraries for animating rigid and soft objects (including cloth) subject to
frictional contacts against smooth implicit surfaces.

This repository includes a reference implementations for [\[1\]] and [\[2\]].


# Overview

The included tools and libraries are organized in directories as follows
 - [implicits](implicits)  --- A Rust crate to generate local and global implicit surfaces.
 - [implicits-hdk](implicits-hdk)  --- A [Houdini](https://sidefx.com/products/houdini) plugin for
   the [implicits](implicits) Rust crate.
 - [cimplicits](cimplicits) --- A C wrapper for the [implicits](implicits) crate.
 - [softy](softy)  --- A Rust crate for simulating interacting soft (both solid and shell) and rigid objects.
 - [softy-hdk](softy-hdk)  --- A [Houdini](https://sidefx.com/products/houdini) plugin for
   the [softy](softy) Rust crate.

These tools are mainly written in [Rust](https://www.rust-lang.org) with C/C++ wrappers.


# Installation

There are two installation scripts available: `install_min.sh` and `install_all.sh`.
Installing HDK plugins written in Rust that use other C++ libraries like MKL or TBB may conflict with Houdini's libraries. For this reasons `install_min.sh` is there to install only the necessary plugin and command-line tool to create and run simulations. If one wants to only run the included examples, simply install `softy-eval` and run the included configuration files, see the corresponding [`README.md`](softy-eval/README.md) for instructions.

Most of the code included here is written in Rust, so you will need to have the Rust toolchain installed. Check [rust-lang.org](https://www.rust-lang.org/learn/get-started) for specific instructions. It is typical and recommended to install the `rustup` tool and then use it to install `cargo`.

Next, install the [`cargo-hdk`](https://crates.io/cargo-hdk) tool with
```
cargo install cargo-hdk
```
This tool helps orchestrate building and installing Rust plugins for Houdini's C++ API called the Houdini Development Kit (HDK).

Before installing any of the plugins, you must source the Houdini environment variables found in the Houdini installation directory.
For details on specific steps needed to build an HDK plugin see
the [HDK Getting Started Guide](https://www.sidefx.com/docs/hdk/_h_d_k__intro__getting_started.html). Follow all
the steps listed there up until actually compiling the plugin with `hcustom` and running it.

Finally, run either of the provided install scripts (macOS or Linux only) to install the plugins:
- `install_min.sh` installs `scene-hdk` for creating configurations using Houdini and `softy-eval` for running simulations using the generated configuration files with softy.
- `install_all.sh` installs all included plugins and command-line tools.


# References

 - [\[1\] Egor Larionov, Ye Fan, and Dinesh K. Pai. 2021. Frictional Contact on Smooth Elastic Solids.][\[1\]],
 - [\[2\] Egor Larionov, Andreas Longva, Uri M. Ascher, Jan Bender, Dinesh K. Pai. 2024. 
   Implicit Frictional Dynamics with Soft Constraints.][\[2\]],


# C API conventions

This section outlines the naming conventions for any public C symbols exposed from Rust libraries.

Let `pfx` be the desired prefix for the library at hand. For instance we may chose `hr` as the
prefix for hdkrs. The following conventions apply to all public facing C API symbols. This includes
Rust code with a C public API. Private symbols in Rust code need not be prefixed.

 * Function names should be in `snake_case` and prefixed with `pfx_`.
 * Struct names should be in `TitleCase` and prefixed with `PFX_`.
 * C Enums should be in `TitleCase` and prefixed with `PFX`. Note that some Rust enums will be bound
   to C structs, in which case their prefix should be `PFX_`.
 * Enum variants should be in `SCREAMING_SNAKE_CASE` and prefixed with `PFX_`.

Note that we must set the language for cbindgen to be `Cxx` or set `cpp_compat=true`. Otherwise,
without proper `extern` annotations in the generated C headers, the exported function symbols will
be mangled by the C++ compiler and effectively lost.

Ideally we would want to wrap any raw C calls in safer C++ wrappers.


# License

This repository is licensed under the [Mozilla Public License, v. 2.0](https://mozilla.org/MPL/2.0/).


# Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for details.

[\[1\]]: https://doi.org/10.1145/3446663
[\[2\]]: https://arxiv.org/abs/2211.10618