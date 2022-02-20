# `scene-hdk`

A Houdini SOP plugin for creating scene configuration files that can be simulated with `softy`.

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