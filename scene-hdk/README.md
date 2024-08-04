# `scene-hdk`

A Houdini SOP plugin for creating scene configuration files that can be simulated with `softy-eval`.

# Installation

Before building anything, you must source the Houdini environment variables found in the Houdini installation directory.
For details on specific steps needed to build an HDK plugin see
the [HDK Getting Started Guide](https://www.sidefx.com/docs/hdk/_h_d_k__intro__getting_started.html). Follow all
the steps listed there up until actually compiling the plugin with `hcustom` and running it.

Build and install this plugin using the [`cargo-hdk`](https://crates.io/cargo-hdk) tool using the following command:
```
cargo hdk --release
```

# Usage

If the installation above is successful, the Scene SOP will be available in the next start of Houdini.

# Motivation

This crate exists as a fallback to the main `softy-hdk` crate. Linking all the libraries needed by `softy` may conflict with the Houdini installation, which can cause various issues on different platforms. To avoid having to troubleshoot these, users are encouraged to instead build this `scene-hdk` plugin if they wish to construct the scenes for `softy` in Houdini and run the resulting saved configuration through command-line using `softy-eval`. This workflow is less streamlined but it decouples Houdini and `softy` builds.