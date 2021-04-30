# Softy

Simulation tools and libraries for animating rigid and soft objects (including cloth) subject to
frictional contacts against smooth implicit surfaces.

This repository includes a reference implementation for [\[1\]].

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

## Implicit Surfaces

TBD

## Finite element simulation of soft tissues and cloth

TBD

## Houdini Plugins

TBD


# References

 - [\[1\] Egor Larionov, Ye Fan, and Dinesh K. Pai. 2021. Frictional Contact on Smooth Elastic Solids. ACM
   Trans. Graph. 40, 2, Article 15 (April 2021), 17 pages. DOI:https://doi.org/10.1145/3446663][\[1\]],


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
