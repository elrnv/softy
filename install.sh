#!/bin/sh

# Install softy-eval
echo Installing softy-eval.......
cargo install --path softy-eval

# Install softy-hdk
if [ $? -eq 0 ]; then
echo Done
echo Installing softy-hdk........
cd softy-hdk
rm -r hdk/build_release
cargo hdk --release

# Install implicits-hdk
if [ $? -eq 0 ]; then
cd ..
echo Done
echo Installing implicits-hdk....
cd implicits-hdk
rm -r hdk/build_release
cargo hdk --release

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

else
cd ..
echo Failed
fi

else
echo Failed
fi

