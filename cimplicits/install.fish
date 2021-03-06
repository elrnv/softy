#!/usr/bin/env fish

set target_dir (dirname (status --current-filename))/../target

set build_type release

set vmr_dir $HOME/vmr/packages/cimplicits

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
    pushd $target_dir/x86_64-pc-windows-msvc/$build_type
    cp $implicits_lib.dll $vmr_dir/lib/

    if [ $status -ne 0 ]
        echo "Error installing dll for windows."
        exit 3
    end

    cp $implicits_lib.dll.lib $vmr_dir/lib/$implicits_lib.lib

    if [ $status -eq 0 ]
        echo "Implicits library successfully added to conan package for windows."
    else
        echo "Error adding import library for windows."
        exit 4
    end

    popd
end

function conan_install
    pushd $vmr_dir
    conan create . VitalMechanics/stable

    if [ $status -eq 0 ]
        echo "Implicits library successfully installed for $platform."
    else
        echo "Error installing library for $platform."
        exit 8
    end

    popd
end

if [ $platform = "Darwin" ]
    install_header

    pushd $target_dir/$build_type

    cp lib$implicits_lib.dylib $vmr_dir/lib/

    if [ $status -ne 0 ]
        echo "Error copying library for mac."
        exit 2
    end

    # Strip all installed rpaths since they reveal directory unrelated structures and are basically
    # useless
    #set rpaths (otool -l $vmr_dir/lib/lib$implicits_lib.dylib | grep RPATH -A2 | grep path | sed "s/.*path\ \(.*\) (.*).*/\1/g")
    #if [ $status -ne 0 ]
    #    echo "Error getting rust rpaths in lib$implicits_lib.dylib"
    #    exit 2
    #end

    #for rpath in $rpaths
    #    install_name_tool -delete_rpath $rpath $vmr_dir/lib/lib$implicits_lib.dylib
    #    if [ $status -ne 0 ]
    #        echo "Error removing rpath $rpath from lib$implicits_lib.dylib"
    #        exit 2
    #    end
    #end

    # Set cimplicits to be portable.
    install_name_tool -id @rpath/lib$implicits_lib.dylib $vmr_dir/lib/lib$implicits_lib.dylib

    conan_install

    if [ $status -eq 0 ]
        echo "Implicits library successfully installed for mac."
    else
        echo "Error installing library for mac."
        exit 2
    end

    popd

    # finish linux installation by copying from the vm-share folder
    #pushd $HOME/vm-share
    #cp lib$implicits_lib.so $vmr_dir/lib/

    #if [ $status -eq 0 ]
    #    echo "Implicits library successfully installed for linux."
    #else
    #    echo "Error installing library for linux."
    #    exit 5
    #end

    #popd

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
            echo "Implicits library successfully exported for $platform."
        else
            echo "Error exporting library for $platform."
            exit 7
        end

        conan_install

        popd

        install_for_windows
    end
else
    echo "Unsupported platform: $platform"
    exit 7
end

