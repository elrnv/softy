# `softy-hdk`

A Houdini SOP plugin for simulation of solids and cloth with frictional contact.

## Requirements

This plugin has so far been tested to work on GNU/Linux.

### Debian/Ubuntu

The following command installs the libraries required for a successful build.

```
sudo apt install libssl-dev libclang-dev libopenblas-dev gfortran pkg-config`
```

# Usage

The Softy SOP compiled through in the `hdk` directory computes a single time step of the simulation.
Additionaly it is able to perform a static solve.

In the `hda` subdirectory, you will `softy_solver.hda`, which is a convenient wrapper for the Solver
SOP, which forwards the Softy SOP parameters to the Solver SOP. This effectively evaluates Softy SOP
for each keyframe. This is the recommended way to use the Softy SOP.

