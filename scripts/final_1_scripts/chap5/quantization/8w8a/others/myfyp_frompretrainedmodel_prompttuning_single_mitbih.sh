#!/bin/bash

# environment variables
model_name=UniTS
d_model=128
base_name=myfyp_frompretrainedmodel_prompttuning_single_mitbih_lr0.01
exp_name=inference_tflite_${base_name}
wandb_mode=disabled
ckpt_path=/home/lps/UniTS/training_checkpoints/ALL_task_myfyp_frompretrainedmodel_prompttuning_single_mitbih_lr0.01_UniTS_All_ftM_dm128_el3_train_0/ptune_checkpoint.pth
random_port=1234
prj_name=${exp_name}_prj
task_data_config=data_provider/myfyp_single_mitbih.yaml
calib_data_path=/home/lps/UniTS/calib_data/save_calib_${base_name}_calib_data.npz
tflite_path=/home/lps/UniTS/final_tflite_models/conversion_${base_name}.tflite
quantization_type=8w8a
quantized_tflite_path=/home/lps/UniTS/final_tflite_models/conversion_${base_name}_quantized_${quantization_type}.tflite

python /home/lps/UniTS/tflite_code/quantizer_tflite.py \
  --float_tflite_model_path "$tflite_path" \
  --calib_data_path "$calib_data_path" \
  --output_model_path="$quantized_tflite_path" \
  --quantization_type "$quantization_type"