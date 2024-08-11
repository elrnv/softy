# `softy-eval`

A command-line tool for running the `softy` simulator on a given scene.


## Installation

Ensure that you have the Rust toolchain installed. Check
[rust-lang.org](https://www.rust-lang.org/learn/get-started) for specific
instructions. It is typical and recommended to install the `rustup` tool and
then use it to install `cargo`.

Then this CLI tool can be installed with the following command

```
cargo install --path .
```

when run from the `softy-eval` subdirectory (otherwise replace `.` with the
actual path to the directory containing the corresponding `Cargo.toml` file).


## Usage

Running the example configurations is as simple as

```
softy <path to scene> --steps <number of steps to simulate> <output path>
```

The supported scene configuration types are `.json`, `.ron`, and `.sfrb`.
The latter is a custom format combining `.ron` for the simulation parameters
followed by [bincode](https://github.com/bincode-org/bincode) for compact
binary.

Supported outputs are `.glb`/`.gltf` and `.vtk`/`.vtu`.


## Examples

The following examples from [\[2\]] can be reproduced with a single command:
- `grasp` - the Figure 1 example (bottom row) can be generated with
```
softy grasp.sfrb --steps 600 ./grasp.glb
```

- `tire` - the Figure 11 example can be generated with
(this example takes a long time to complete)
```
softy tire.sfrb --steps 1500 ./tire.glb
```


[\[2\]]: https://arxiv.org/abs/2211.10618

