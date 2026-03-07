#!/bin/bash

# environment variables
model_name=UniTS
d_model=128
base_name=myfyp_frompretrainedmodel_prompttuning_single_ucihar_lr0.01
quantization_type=8w16a
exp_name=measure_perf_pi_${quantization_type}_${base_name}
wandb_mode=disabled
random_port=1234
prj_name=${exp_name}_prj
task_data_config=data_provider/myfyp_single_ucihar.yaml
tflite_path=final_tflite_models/conversion_${base_name}_quantized_${quantization_type}.tflite
output_folder=final_tflite_models/measure_perf_pi_results

mkdir -p ${output_folder}

python tflite_code/measure_latency_pi.py \
  --tflite_model_path "$tflite_path" | tee -a ${output_folder}/${exp_name}_inference_latency_log.txt