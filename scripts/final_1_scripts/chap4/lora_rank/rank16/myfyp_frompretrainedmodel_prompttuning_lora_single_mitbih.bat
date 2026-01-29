@echo off

REM Set environment variables
set model_name=UniTS
set d_model=128
set exp_name=myfyp_frompretrainedmodel_x128_prompttuning_lora_single_mitbih
set wandb_mode=disabled
set ckpt_path=units_x128_pretrain_checkpoint.pth
set random_port=1234
set prj_name=test_run_x128_prompt_tuning
set task_data_config=data_provider/myfyp_single_mitbih.yaml

REM Run the training script with torchrun
python run.py ^
  --is_training 1 ^
  --model_id %exp_name% ^
  --model %model_name% ^
  --lradj prompt_tuning ^
  --prompt_num 10 ^
  --patch_len 16 ^
  --stride 16 ^
  --e_layers 3 ^
  --d_model %d_model% ^
  --des "Exp" ^
  --itr 1 ^
  --learning_rate 3e-3 ^
  --weight_decay 0 ^
  --train_epochs 0 ^
  --prompt_tune_epoch 20 ^
  --batch_size 32 ^
  --acc_it 5 ^
  --debug %wandb_mode% ^
  --project_name %prj_name% ^
  --clip_grad 100 ^
  --pretrained_weight %ckpt_path% ^
  --task_data_config_path %task_data_config% ^
  --lora
  --lora_r 16
  --lora_alpha 32
