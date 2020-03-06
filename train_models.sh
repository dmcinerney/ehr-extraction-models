#!/bin/bash

read -s -p "Enter Password: " pswd
echo
checkpoints="checkpoints"
mkdir "${checkpoints}/code_supervision_only_description_unfrozen"
python train.py code_supervision_only_description_unfrozen --save_checkpoint_folder "${checkpoints}/code_supervision_only_description_unfrozen" --password $pswd
mkdir "${checkpoints}/code_supervision_only_linearization_description_unfrozen"
python train.py code_supervision_only_linearization_description_unfrozen --save_checkpoint_folder "${checkpoints}/code_supervision_only_linearization_description_unfrozen" --password $pswd
mkdir "${checkpoints}/code_supervision_unfrozen"
python train.py code_supervision_unfrozen --save_checkpoint_folder "${checkpoints}/code_supervision_unfrozen" --password $pswd
