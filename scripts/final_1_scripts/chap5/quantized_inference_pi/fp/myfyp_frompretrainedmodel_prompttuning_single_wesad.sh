#!/bin/bash

# Set environment variables
model_name=UniTS
d_model=128
base_name=myfyp_frompretrainedmodel_prompttuning_single_wesad_lr0.01_batch32
quantization_type=fp
exp_name=inference_tflite_pi_${base_name}_${quantization_type}
wandb_mode=disabled
ckpt_path=training_checkpoints/ALL_task_myfyp_frompretrainedmodel_prompttuning_single_wesad_lr0.01_batch32_UniTS_All_ftM_dm128_el3_train_0/ptune_checkpoint.pth
random_port=1234
prj_name=${exp_name}_prj
task_data_config=data_provider/myfyp_single_wesad.yaml
calib_data_path=calib_data/save_calib_${base_name}_calib_data.npz
tflite_path=final_tflite_models/conversion_${base_name}.tflite

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
  --prompt_tune_epoch 20 \
  --batch_size 32 \
  --acc_it 5 \
  --debug ${wandb_mode} \
  --project_name ${prj_name} \
  --clip_grad 100 \
  --pretrained_weight ${ckpt_path} \
  --task_data_config_path ${task_data_config}