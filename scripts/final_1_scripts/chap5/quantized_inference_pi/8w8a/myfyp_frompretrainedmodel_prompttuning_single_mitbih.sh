#!/bin/bash

# environment variables
model_name=UniTS
d_model=128
base_name=myfyp_frompretrainedmodel_prompttuning_single_mitbih_lr0.01
quantization_type=8w8a
exp_name=inference_tflite_pi_${quantization_type}_${base_name}
wandb_mode=disabled
ckpt_path=""
random_port=1234
prj_name=${exp_name}_prj
task_data_config=data_provider/myfyp_single_mitbih.yaml
calib_data_path=""
tflite_path=final_tflite_models/conversion_${base_name}_quantized_${quantization_type}.tflite

# Run the training script with torchrun
python run.py \
  --is_training 0 \
  --tflite_path "$tflite_path" \
  --model_id ${exp_name} \
  --model ${model_name} \
  --lradj prompt_tuning \
  --prompt_num 10 \
  --patch_len 16 \
  --stride 16 \
  --e_layers 3 \
  --d_model ${d_model} \
  --des "train" \
  --itr 1 \
  --learning_rate 0.01 \
  --weight_decay 0 \
  --train_epochs 0 \
  --prompt_tune_epoch 0 \
  --batch_size 64 \
  --acc_it 5 \
  --debug ${wandb_mode} \
  --project_name ${prj_name} \
  --clip_grad 100 \
  --pretrained_weight ${ckpt_path} \
  --task_data_config_path ${task_data_config}