
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

Note that we must set the language for cbindgen to be `Cxx`. Otherwise, without proper `extern`
annotations in the generated C headers, the exported function symbols will be mangled by the C++
compiler and effectively lost.

Ideally we would want to wrap any raw C calls in safer C++ wrappers.

# License

This repository is licensed under the [Mozilla Public License, v. 2.0](https://mozilla.org/MPL/2.0/).

# Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for details.
