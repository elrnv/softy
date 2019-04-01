#!/usr/bin/fish

set target_dir (dirname (status --current-filename))/../target

set build_type release

set vmr_dir $HOME/vmr/packages/implicits

set implicits_lib cimplicits

if [ (count $argv) -eq 0 ]
    set build_type release
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


function install_header
    pushd $target_dir/$build_type

    cp cimplicits.h $vmr_dir/include/

    if [ $status -eq 0 ]
        echo "Implicits header successfully installed."
    else
        echo "Error installing $implicits_lib header."
        exit -1
    end

    popd
end

function install_for_windows
    pushd $target_dir/x86_64-pc-windows-gnu/$build_type
    cp $implicits_lib.dll $vmr_dir/lib/

    if [ $status -ne 0 ]
        echo "Error installing dll for windows."
        exit 3
    end

    # Create the import library from the generated dll (we know it exists because the previous line
    # passed)
    x86_64-w64-mingw32-dlltool -z $implicits_lib.def --export-all-symbol $implicits_lib.dll
    x86_64-w64-mingw32-dlltool -A -d $implicits_lib.def -l $implicits_lib.lib

    cp $implicits_lib.lib $vmr_dir/lib/

    if [ $status -eq 0 ]
        echo "Implicits library successfully installed for windows."
    else
        echo "Error installing import library for windows."
        exit 4
    end

    popd
end

if [ $platform = "Darwin" ]
    install_header

    pushd $target_dir/$build_type

    cp lib$implicits_lib.dylib $vmr_dir/lib/

    if [ $status -eq 0 ]
        echo "Implicits library successfully installed for mac."
    else
        echo "Error installing library for mac."
        exit 2
    end

    popd

    install_for_windows

    # finish linux installation by copying from the vm-share folder
    pushd $HOME/vm-share
    cp lib$implicits_lib.so $vmr_dir/lib/

    if [ $status -eq 0 ]
        echo "Implicits library successfully installed for linux."
    else
        echo "Error installing library for linux."
        exit 5
    end

    popd

else if [ $platform = "Linux" ]
    # Check if we are in a vm, then we need to copy to vm folder
    set chassis (hostnamectl status | grep Chassis | sed "s/.*Chassis: \(.*\)/\1/g")
    if [ $chassis = "vm" ]
        # We are in a vm, just copy to the vm share folder

        pushd $target_dir/$build_type

        sudo cp lib$implicits_lib.so /media/sf_vm-share/

        if [ $status -eq 0 ]
            echo "Implicits library successfully staged for linux."
        else
            echo "Error staging library linux."
            exit 6
        end

        popd
    else
        # We are on desktop

        install_header

        pushd $target_dir/$build_type

        cp lib$implicits_lib.so $vmr_dir/lib/

        if [ $status -eq 0 ]
            echo "Implicits library successfully installed for Linux."
        else
            echo "Error staging library Linux."
            exit 8
        end

        popd

        install_for_windows
    end
else
    echo "Unsupported platform: $platform"
    exit 7
end

