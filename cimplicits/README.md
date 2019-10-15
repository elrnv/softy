# `cimplicits`

A library for computing implicit surfaces from unstructured point sets.

All paths in this document are with respect to the directory containing this `README.md` file, which
should coincide with the `cimplicits` project root directory.

## Building

To build the library with optimizations enabled simply run

```
cargo build --release
```

To make sure everything works correctly, build and run the tests with

```
cargo test --release
```

This will also build the library if the previous build command wasn't run.

All build artifacts can be found under `../target/release`.

## Installation as a Conan package

The following instructions describe how to install the build artifacts as a Conan package.
It is assumed that the library has been built as described above.

In summary the steps below describe how to copy a file from the build directory to the Conan package
directory.

Let `$build_dir` correspond to the build directory (e.g. for release builds this is `../target/release`).
Also let `$conan_dir` correspond to the conan package root directory.
It is assumed that the Conan package contains binaries only and has the following subdirectories already
created:

 - `$conan_dir/include` containing header files
 - `$conan_dir/lib ` containing library binaries

### Linux

First lets define the build directory 
Go to the target build directory:

```
cd ../target/release
```

Copy the header file found in the build directory to the conan root:

```
cp $build_dir/cimplicits.h $conan_dir/include/.
```

Copy the binaries to the `lib` directory.

```
cp $build_dir/libcimplicits.so $conan_dir/lib/.
```

We can compile the conan package with

```
cd $conan_dir
conan create . $user/stable 
```

where `$user` refers to the Conan username.
