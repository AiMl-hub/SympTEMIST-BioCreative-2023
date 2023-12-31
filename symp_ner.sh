#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

model_name=$1
dataset_name=$2
seed=$3
offline=$4
epochs=$5

output_dir=$SCRIPT_DIR/full_output/model-$model_name/dataset-$dataset_name/seed-$seed/epochs-$epochs
mkdir -p $output_dir

python $SCRIPT_DIR/ner/symp_run_ner.py \
--model_name_or_path $model_name \
--dataset_name $dataset_name \
--train_file $dataset_name/train \
--validation_file $dataset_name/validation \
--test_file $dataset_name/test \
--offline $offline \
--do_train \
--do_eval \
--per_device_train_batch_size 4 \
--gradient_accumulation_steps 4 \
--num_train_epochs $epochs \
--load_best_model_at_end \
--metric_for_best_model f1 \
--evaluation_strategy epoch \
--save_strategy epoch \
--overwrite_output_dir \
--seed $seed \
--logging_dir $output_dir/tb \
--output_dir $output_dir 2>&1 | tee $output_dir/train.log
