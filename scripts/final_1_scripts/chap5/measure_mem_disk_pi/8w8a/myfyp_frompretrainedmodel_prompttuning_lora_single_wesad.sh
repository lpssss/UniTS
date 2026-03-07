#!/bin/bash

# Set environment variables
model_name=UniTS
d_model=128
base_name=myfyp_frompretrainedmodel_prompttuning_lora_single_wesad_lr0.01_batch32
quantization_type=8w8a
exp_name=measure_mem_disk_pi_${quantization_type}_${base_name}
wandb_mode=disabled
random_port=1234
prj_name=${exp_name}_prj
task_data_config=data_provider/myfyp_single_wesad.yaml
tflite_path=final_tflite_models/conversion_${base_name}_quantized_${quantization_type}.tflite
output_folder=final_tflite_models/measure_mem_disk_pi_results

mkdir -p ${output_folder}

# measure disk space of the tflite model
disk_size_bytes=$(du -b "$tflite_path" | cut -f1)
echo "Disk size of the TFLite model: $disk_size_bytes bytes" | tee -a ${output_folder}/${exp_name}_memory_usage_log.txt
echo "Disk size of the TFLite model (MB): $(echo "scale=2; $disk_size_bytes / (1024 * 1024)" | bc)" | tee -a ${output_folder}/${exp_name}_memory_usage_log.txt

python tflite_code/measure_mem_pi.py \
  --tflite_path "$tflite_path" | tee -a ${output_folder}/${exp_name}_memory_usage_log.txt