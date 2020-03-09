#!/bin/bash

PARAMS=()
while (( "$#" )); do
  case "$1" in
    -e|--email)
      E=true
      shift 1
      ;;
    --) # end argument parsing
      shift
      break
      ;;
    -*|--*=) # unsupported flags
      echo "Error: Unsupported flag $1" >&2
      exit 1
      ;;
    *) # preserve positional arguments
      PARAMS=( ${PARAMS[@]} $1 )
      shift
      ;;
  esac
done
if [ ${#PARAMS[*]} -ne 1 ]
then
    echo "Error: number of args isn't right"
    exit 1
fi
CHECKPOINTS=${PARAMS[0]}
if [ $E ]
then
    read -s -p "Enter Password: " pswd
    echo
    ARGS=( "--sender_password" "$pswd" "-e" )
else
    ARGS=()
fi
#mkdir "${CHECKPOINTS}/code_supervision_only_description_unfrozen"
python train.py code_supervision_only_description_unfrozen --save_checkpoint_folder "${CHECKPOINTS}/code_supervision_only_description_unfrozen" "${ARGS[@]}" --load_checkpoint_folder "${CHECKPOINTS}/code_supervision_only_description_unfrozen"
#python train.py code_supervision_only_description_unfrozen --save_checkpoint_folder "${CHECKPOINTS}/code_supervision_only_description_unfrozen" "${ARGS[@]}"
mkdir "${CHECKPOINTS}/code_supervision_only_linearization_description_unfrozen"
python train.py code_supervision_only_linearization_description_unfrozen --save_checkpoint_folder "${CHECKPOINTS}/code_supervision_only_linearization_description_unfrozen" "${ARGS[@]}"
mkdir "${CHECKPOINTS}/code_supervision_unfrozen"
python train.py code_supervision_unfrozen --save_checkpoint_folder "${CHECKPOINTS}/code_supervision_unfrozen" "${ARGS[@]}"
