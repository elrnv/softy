# `softy-hdk`

A Houdini SOP plugin for simulation of solids and cloth with frictional contact.

## Requirements

This plugin has so far been tested to work on GNU/Linux.

### Debian/Ubuntu

The following command installs the libraries required for a successful build.

```
sudo apt install libssl-dev libclang-dev libopenblas-dev gfortran pkg-config`
```


## Installation

Before building anything, you must source the Houdini environment variables found in the Houdini installation directory.
For details on specific steps needed to build an HDK plugin see
the [HDK Getting Started Guide](https://www.sidefx.com/docs/hdk/_h_d_k__intro__getting_started.html). Follow all
the steps listed there up until actually compiling the plugin with `hcustom` and running it.

Build and install this plugin using the [`cargo-hdk`](https://crates.io/cargo-hdk) tool using the following command:
```
cargo hdk --release
```


## Usage

The Softy SOP compiled in the `hdk` directory computes a single time step of the simulation.
Additionally, it is able to perform a static solve when the time step is set to zero.

In the `hda` subdirectory, you will find `softy_solver.hda`, which is a convenient wrapper for the Solver
SOP, which forwards the Softy SOP parameters to the Solver SOP. This effectively evaluates Softy SOP
for each keyframe. This is the recommended way to use the Softy SOP.