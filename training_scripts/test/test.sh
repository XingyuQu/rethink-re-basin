#!/bin/bash

cd ../
folder_path="."
exclude_files="cifar10_fixup_resnet20.sh cifar10_vgg11_4x.sh cifar10_vgg11_8x.sh cifar10_vgg16_4x.sh cifar10_vgg16_8x.sh"

for file in "$folder_path"/*.sh; do
    if [ -f "$file" ]; then
        file_name=$(basename "$file")
        
        if [[ ! " $exclude_files " =~ " $file_name " ]]; then
            echo "Running $file"
            bash "$file"
        else
            echo "Skipping $file (Excluded)"
        fi
    fi
done