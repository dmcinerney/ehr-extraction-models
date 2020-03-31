#!/bin/bash
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
#BASE_DATASET_PATH="/home/jered/Documents/data/mimic-iii-clinical-database-1.4/preprocessed/reports_and_codes_expanded"
BASE_DATASET_PATH="/home/jered/Documents/data/Dataset_10-11-2019/preprocessed/reports_and_codes_expanded"
#BASE_DATASET_PATH="/home/dzm44/Documents/data/Dataset_10-11-2019/preprocessed/reports_and_codes_expanded"
#Supervised args
SA=( "${BASE_DATASET_PATH}/test_supervised_custom" "${BASE_DATASET_PATH}/test_supervised" )
#Train supervised args
TSA=( "--supervised_data_dir" "--results_folder" )

#mkdir "${CHECKPOINTS}/tfidf_similarity"
#python test.py tfidf_similarity "${CHECKPOINTS}/tfidf_similarity" "${ARGS[@]}" -s --supervised_data_dir "${SA[0]}" --results_folder "${CHECKPOINTS}/tfidf_similarity/supervised_results_test" -n
#mkdir "${CHECKPOINTS}/cosine_similarity"
#python test.py cosine_similarity "${CHECKPOINTS}/cosine_similarity" "${ARGS[@]}" -s --supervised_data_dir "${SA[0]}" --results_folder "${CHECKPOINTS}/cosine_similarity/supervised_results_test" -n
#mkdir "${CHECKPOINTS}/code_supervision_only_description_unfrozen"
MODEL_SPECIFIC_TRAIN_SUPERVISED_ARGS=( ${TSA[0]} ${SA[0]} ${TSA[1]} "${CHECKPOINTS}/code_supervision_only_description_unfrozen/supervised_results" )
#python train.py code_supervision_only_description_unfrozen --save_checkpoint_folder "${CHECKPOINTS}/code_supervision_only_description_unfrozen" "${ARGS[@]}" "${MODEL_SPECIFIC_TRAIN_SUPERVISED_ARGS[@]}"
#python train.py code_supervision_only_description_unfrozen --save_checkpoint_folder "${CHECKPOINTS}/code_supervision_only_description_unfrozen" "${ARGS[@]}" "${MODEL_SPECIFIC_TRAIN_SUPERVISED_ARGS[@]}" --load_checkpoint_folder "${CHECKPOINTS}/code_supervision_only_description_unfrozen"
#python test.py code_supervision_only_description_unfrozen "${CHECKPOINTS}/code_supervision_only_description_unfrozen" "${ARGS[@]}" -s --supervised_data_dir "${SA[0]}" --results_folder "${CHECKPOINTS}/code_supervision_only_description_unfrozen/supervised_results_test"
python test.py code_supervision_only_description_unfrozen "${CHECKPOINTS}/code_supervision_only_description_unfrozen" "${ARGS[@]}" --results_folder "${CHECKPOINTS}/code_supervision_only_description_unfrozen/results"
#mkdir "${CHECKPOINTS}/code_supervision_only_linearization_description_unfrozen"
MODEL_SPECIFIC_TRAIN_SUPERVISED_ARGS=( ${TSA[0]} ${SA[0]} ${TSA[1]} "${CHECKPOINTS}/code_supervision_only_linearization_description_unfrozen/supervised_results" )
#python train.py code_supervision_only_linearization_description_unfrozen --save_checkpoint_folder "${CHECKPOINTS}/code_supervision_only_linearization_description_unfrozen" "${ARGS[@]}" "${MODEL_SPECIFIC_TRAIN_SUPERVISED_ARGS[@]}"
#python train.py code_supervision_only_linearization_description_unfrozen --save_checkpoint_folder "${CHECKPOINTS}/code_supervision_only_linearization_description_unfrozen" "${ARGS[@]}" "${MODEL_SPECIFIC_TRAIN_SUPERVISED_ARGS[@]}" --load_checkpoint_folder "${CHECKPOINTS}/code_supervision_only_linearization_description_unfrozen"
#python test.py code_supervision_only_linearization_description_unfrozen "${CHECKPOINTS}/code_supervision_only_linearization_description_unfrozen" "${ARGS[@]}" -s --supervised_data_dir "${SA[0]}" --results_folder "${CHECKPOINTS}/code_supervision_only_linearization_description_unfrozen/supervised_results_test"
python test.py code_supervision_only_linearization_description_unfrozen "${CHECKPOINTS}/code_supervision_only_linearization_description_unfrozen" "${ARGS[@]}" --results_folder "${CHECKPOINTS}/code_supervision_only_linearization_description_unfrozen/results"
#mkdir "${CHECKPOINTS}/code_supervision_unfrozen"
MODEL_SPECIFIC_TRAIN_SUPERVISED_ARGS=( ${TSA[0]} ${SA[1]} ${TSA[1]} "${CHECKPOINTS}/code_supervision_unfrozen/supervised_results" )
#python train.py code_supervision_unfrozen --save_checkpoint_folder "${CHECKPOINTS}/code_supervision_unfrozen" "${ARGS[@]}" "${MODEL_SPECIFIC_TRAIN_SUPERVISED_ARGS[@]}"
#python train.py code_supervision_unfrozen --save_checkpoint_folder "${CHECKPOINTS}/code_supervision_unfrozen" "${ARGS[@]}" "${MODEL_SPECIFIC_TRAIN_SUPERVISED_ARGS[@]}" --load_checkpoint_folder "${CHECKPOINTS}/code_supervision_unfrozen"
#python test.py code_supervision_unfrozen "${CHECKPOINTS}/code_supervision_unfrozen" "${ARGS[@]}" -s --supervised_data_dir "${SA[1]}" --results_folder "${CHECKPOINTS}/code_supervision_unfrozen/supervised_results_test"
python test.py code_supervision_unfrozen "${CHECKPOINTS}/code_supervision_unfrozen" "${ARGS[@]}" --results_folder "${CHECKPOINTS}/code_supervision_unfrozen/results"
