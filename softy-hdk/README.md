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

### Windows

Building on Windows requires a few extra steps.

1. Install Houdini 19.0 or later.

2. Launch Houdini to install an appropriate license. If this step is skipped the command-line steps below may faily or hang indefinitely.

3. Install Visual Studio 2019

4. Install CMake. This needs to be at a newer version (3.24.1 or higher) than provided by Visual Studio 2019.

5. Download and install [Intel MKL](https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl-download.html).

6. Install `cargo-hdk` as above by running
```
cargo install cargo-hdk
```
from command line.

7. Start Houdini's Command Line Tools from the start menu, or run
```
> C:\"Program Files\Side Effects Software\Houdini 19.5.303"\bin\hcmd.exe
```
from command line. (replace 19.5.303 with the correct version).

8. Initialize Intel MKL environment in the same shell with
```
> C:\"Program Files (x86)\Intel\oneAPI\setvars.bat" intel64 vs2019
```

9. Navigate to the `softy-hdk` subfolder (the one containing this README file) and run
```
cargo hdk --release
```
to build the HDK Houdini plugin.


## Usage

The Softy SOP compiled in the `hdk` directory computes a single time step of the simulation.
Additionally, it is able to perform a static solve when the time step is set to zero.

In the `hda` subdirectory, you will find `softy_solver.hda`, which is a convenient wrapper for the Solver
SOP, which forwards the Softy SOP parameters to the Solver SOP. This effectively evaluates Softy SOP
for each keyframe. This is the recommended way to use the Softy SOP.
