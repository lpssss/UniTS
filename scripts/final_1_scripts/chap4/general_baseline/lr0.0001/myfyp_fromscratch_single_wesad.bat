@echo off

REM Set environment variables
set model_name=UniTS
set d_model=128
set exp_name=myfyp_fromscratch_single_wesad
set wandb_mode=disabled
set ckpt_path=units_x128_pretrain_checkpoint.pth
set random_port=1234
set prj_name=%exp_name%_prj
set task_data_config=data_provider/myfyp_single_wesad.yaml

REM Run the training script with torchrun
python run.py ^
  --is_training 1 ^
  --model_id %exp_name% ^
  --model %model_name% ^
  --lradj warmup_cosine ^
  --prompt_num 10 ^
  --patch_len 16 ^
  --stride 16 ^
  --e_layers 3 ^
  --d_model %d_model% ^
  --des "train" ^
  --learning_rate 1e-4 ^
  --weight_decay 5e-6 ^
  --train_epochs 60 ^
  --warmup_epochs 5 ^
  --batch_size 64 ^
  --acc_it 5 ^
  --debug %wandb_mode% ^
  --project_name %prj_name% ^
  --clip_grad 100 ^
  --task_data_config_path %task_data_config% ^
  --use_weighted_loss
