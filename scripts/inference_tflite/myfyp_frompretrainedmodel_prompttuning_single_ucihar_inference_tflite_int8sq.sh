#!/bin/bash

# Set environment variables
model_name="UniTS"
d_model=128
exp_name="myfyp_frompretrainedmodel_x128_prompttuning_single_ucihar_inference_tflite_int8sq"
wandb_mode="disabled"
# Note: Ensure paths are correct for a Linux environment (use / instead of \)
ckpt_path="/home/lps/fyp//UniTS/checkpoints/ALL_task_myfyp_frompretrainedmodel_x128_prompttuning_single_ucihar_UniTS_All_ftM_dm128_el3_Exp_0/ptune_checkpoint.pth"
random_port=1234
prj_name="test_run_x128_prompt_tuning"
task_data_config="data_provider/myfyp_single_ucihar.yaml"
tflite_path="/home/lps/fyp/UniTS/tflite_models/myfyp_frompretrainedmodel_x128_prompttuning_single_ucihar_int8sq.tflite"

# Run the training script
python run.py \
  --is_training 0 \
  --tflite_path "$tflite_path" \
  --model_id "$exp_name" \
  --model "$model_name" \
  --lradj prompt_tuning \
  --prompt_num 10 \
  --patch_len 16 \
  --stride 16 \
  --e_layers 3 \
  --d_model "$d_model" \
  --des "Exp" \
  --itr 1 \
  --learning_rate 3e-3 \
  --weight_decay 0 \
  --train_epochs 0 \
  --prompt_tune_epoch 0 \
  --batch_size 32 \
  --acc_it 5 \
  --debug "$wandb_mode" \
  --project_name "$prj_name" \
  --clip_grad 100 \
  --pretrained_weight "$ckpt_path" \
  --task_data_config_path "$task_data_config"
