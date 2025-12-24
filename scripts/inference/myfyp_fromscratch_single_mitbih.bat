@echo off

REM Set environment variables
set model_name=UniTS
set d_model=128
set exp_name=myfyp_fromscratch_x128_supervised_single_mitbih_inference
set wandb_mode=disabled
set ckpt_path="C:\Users\LPS\OneDrive\utm_master\research_methodology\fyp\UniTS\checkpoints\ALL_task_myfyp_fromscratch_x128_supervised_single_mitbih_UniTS_All_ftM_dm128_el3_Exp_0\checkpoint.pth"
set random_port=1234
set prj_name=test_run_x128_supervised_targetds2
set task_data_config=data_provider/myfyp_single_mitbih.yaml

python run.py ^
  --is_training 0 ^
  --model_id %exp_name% ^
  --model %model_name% ^
  --lradj supervised ^
  --prompt_num 10 ^
  --patch_len 16 ^
  --stride 16 ^
  --e_layers 3 ^
  --d_model %d_model% ^
  --des "Exp" ^
  --learning_rate 1e-4 ^
  --weight_decay 5e-6 ^
  --train_epochs 20 ^
  --batch_size 32 ^
  --acc_it 32 ^
  --debug %wandb_mode% ^
  --project_name %prj_name% ^
  --pretrained_weight %ckpt_path% ^
  --clip_grad 100 ^
  --task_data_config_path %task_data_config%
