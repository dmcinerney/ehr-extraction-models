#!/bin/bash
E=false
D="cuda:0"
PARAMS=()
while (( "$#" )); do
  case "$1" in
    -e|--email)
      E=true
      shift 1
      ;;
    -d|--device)
      D=$2
      shift 2
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
ARGS=( "--device" "$D" )
if [ $E ]
then
    read -s -p "Enter Password: " pswd
    echo
    ARGS=( ${ARGS[@]} "--sender_password" "$pswd" "-e" )
fi
#mkdir "${CHECKPOINTS}/code_supervision_only_description_unfrozen"
#python train.py code_supervision_only_description_unfrozen --save_checkpoint_folder "${CHECKPOINTS}/code_supervision_only_description_unfrozen" "${ARGS[@]}"
#python train.py code_supervision_only_description_unfrozen --save_checkpoint_folder "${CHECKPOINTS}/code_supervision_only_description_unfrozen" "${ARGS[@]}" --load_checkpoint_folder "${CHECKPOINTS}/code_supervision_only_description_unfrozen"
#python test.py code_supervision_only_description_unfrozen  "${CHECKPOINTS}/code_supervision_only_description_unfrozen" "${ARGS[@]}" -s --data_file /home/jered/Documents/data/Dataset_10-11-2019/preprocessed/reports_and_codes_expanded/val_supervised_custom/supervised.data --hierarchy /home/jered/Documents/data/Dataset_10-11-2019/preprocessed/reports_and_codes_expanded/val_supervised_custom/hierarchy.pkl --results_folder "${CHECKPOINTS}/code_supervision_only_description_unfrozen/supervised_results"
#python test.py code_supervision_only_description_unfrozen  "${CHECKPOINTS}/code_supervision_only_description_unfrozen" "${ARGS[@]}"
#mkdir "${CHECKPOINTS}/code_supervision_only_linearization_description_unfrozen"
#python train.py code_supervision_only_linearization_description_unfrozen --save_checkpoint_folder "${CHECKPOINTS}/code_supervision_only_linearization_description_unfrozen" "${ARGS[@]}"
#python train.py code_supervision_only_linearization_description_unfrozen --save_checkpoint_folder "${CHECKPOINTS}/code_supervision_only_linearization_description_unfrozen" "${ARGS[@]}" --load_checkpoint_folder "${CHECKPOINTS}/code_supervision_only_linearization_description_unfrozen"
#python test.py code_supervision_only_linearization_description_unfrozen  "${CHECKPOINTS}/code_supervision_only_linearization_description_unfrozen" "${ARGS[@]}" -s --data_file /home/jered/Documents/data/Dataset_10-11-2019/preprocessed/reports_and_codes_expanded/val_supervised_custom/supervised.data --hierarchy /home/jered/Documents/data/Dataset_10-11-2019/preprocessed/reports_and_codes_expanded/val_supervised_custom/hierarchy.pkl --results_folder "${CHECKPOINTS}/code_supervision_only_linearization_description_unfrozen/supervised_results"
#python test.py code_supervision_only_linearization_description_unfrozen  "${CHECKPOINTS}/code_supervision_only_linearization_description_unfrozen" "${ARGS[@]}"
#mkdir "${CHECKPOINTS}/code_supervision_unfrozen"
#python train.py code_supervision_unfrozen --save_checkpoint_folder "${CHECKPOINTS}/code_supervision_unfrozen" "${ARGS[@]}"
python train.py code_supervision_unfrozen --save_checkpoint_folder "${CHECKPOINTS}/code_supervision_unfrozen" "${ARGS[@]}" --load_checkpoint_folder "${CHECKPOINTS}/code_supervision_unfrozen"
python test.py code_supervision_unfrozen  "${CHECKPOINTS}/code_supervision_unfrozen" "${ARGS[@]}" -s --data_file /home/jered/Documents/data/Dataset_10-11-2019/preprocessed/reports_and_codes_expanded/val_supervised/supervised.data --results_folder "${CHECKPOINTS}/code_supervision_unfrozen/supervised_results"
python test.py code_supervision_unfrozen  "${CHECKPOINTS}/code_supervision_unfrozen" "${ARGS[@]}"
