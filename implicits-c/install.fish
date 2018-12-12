#!/usr/local/bin/fish

set target_dir (dirname (status --current-filename))/../target

set build_type release

if [ (count $argv) -eq 0 ]
    set target_dir $target_dir/$build_type
else
    if [ $argv[1] = "debug" ]
        set build_type debug
    else if [ $argv[1] = "release" ]
        set build_type release
    else
        echo "Invalid option: $argv[1]"
        exit 1
    end
end

echo "Target dir = $target_dir"
echo "Build type = $build_type"

set platform (uname -s)

if [ $platform = "Darwin" ]
    pushd $target_dir/$build_type

    cp implicits.h ~/vmr/packages/implicits/include/

    if [ $status -eq 0 ]
        echo "Implicits header successfully installed."
    else
        echo "Error installing implicits header."
        exit -1
    end

    cp libimplicits.dylib ~/vmr/packages/implicits/lib/

    if [ $status -eq 0 ]
        echo "Implicits library successfully installed for mac."
    else
        echo "Error installing library for mac."
        exit 2
    end

    popd
    pushd $target_dir/x86_64-pc-windows-gnu/$build_type

    cp implicits.dll ~/vmr/packages/implicits/lib/

    if [ $status -ne 0 ]
        echo "Error installing dll for windows."
        exit 3
    end

    # Create the import library from the generated dll (we know it exists because the previous line
    # passed)
    x86_64-w64-mingw32-dlltool -z implicits.def --export-all-symbol implicits.dll
    x86_64-w64-mingw32-dlltool -A -d implicits.def -l libimplicits.a

    cp implicits.lib ~/vmr/packages/implicits/lib/

    if [ $status -eq 0 ]
        echo "Implicits library successfully installed for windows."
    else
        echo "Error installing import library for windows."
        exit 4
    end

    popd

    # finish linux installation by copying from the vm-share folder
    pushd $HOME/vm-share
    cp libimplicits.so ~/vmr/packages/implicits/lib/

    if [ $status -eq 0 ]
        echo "Implicits library successfully installed for linux."
    else
        echo "Error installing library for linux."
        exit 5
    end

    popd

else if [ $platform = "Linux" ]
    # we are in a vm, just copy to the vm share folder

    pushd $target_dir

    sudo cp libimplicits.so /media/sf_vm-share/

    if [ $status -eq 0 ]
        echo "Implicits library successfully staged for linux."
    else
        echo "Error staging library linux."
        exit 6
    end

    popd

else
    echo "Unsupported platform: $platform"
    exit 7
end

