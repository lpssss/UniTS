@echo off

REM Set environment variables
set model_name=UniTS
set d_model=128
set exp_name=save_calib_myfyp_frompretrainedmodel_prompttuning_lora_single_mitbih_lr0.01_rank16
set wandb_mode=disabled
set ckpt_path=C:\Users\pslim\DataDrive\Documents\OneDrive\utm_master\research_methodology\fyp\UniTS\checkpoints\ALL_task_myfyp_frompretrainedmodel_prompttuning_lora_single_mitbih_lr0.01_rank16_UniTS_All_ftM_dm128_el3_train_0\ptune_checkpoint.pth
set random_port=1234
set prj_name=%exp_name%_prj
set task_data_config=data_provider/myfyp_single_mitbih.yaml
set calib_data_path=C:\Users\pslim\DataDrive\Documents\OneDrive\utm_master\research_methodology\fyp\UniTS\calib_data\%exp_name%_calib_data.npz

REM Run the training script with torchrun
python run.py ^
  --is_training 0 ^
  --save_calib_data ^
  --calib_data_path %calib_data_path% ^
  --model_id %exp_name% ^
  --model %model_name% ^
  --lradj prompt_tuning ^
  --prompt_num 10 ^
  --patch_len 16 ^
  --stride 16 ^
  --e_layers 3 ^
  --d_model %d_model% ^
  --des "train" ^
  --itr 1 ^
  --learning_rate 0.01 ^
  --weight_decay 0 ^
  --train_epochs 0 ^
  --prompt_tune_epoch 25 ^
  --batch_size 64 ^
  --acc_it 5 ^
  --debug %wandb_mode% ^
  --project_name %prj_name% ^
  --clip_grad 100 ^
  --pretrained_weight %ckpt_path% ^
  --task_data_config_path %task_data_config% ^
  --lora ^
  --lora_r 16 ^
  --lora_alpha 32
