#!/bin/sh

# Install softy-eval
echo Installing softy-eval.......
cargo install --path softy-eval

# Install scene-hdk
if [ $? -eq 0 ]; then
cd ..
echo Done
echo Installing scene-hdk........
cd scene-hdk
rm -r hdk/build_release
cargo hdk --release
echo Done

else
cd ..
echo Failed
fi
