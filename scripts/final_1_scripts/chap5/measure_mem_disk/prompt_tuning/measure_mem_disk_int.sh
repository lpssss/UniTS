# Set environment variables
model_name="UniTS"
d_model=128
exp_name="myfyp_frompretrainedmodel_x128_prompttuning_single_ucihar_inference_tflite_dq_test_quantizer"
wandb_mode="disabled"
# Note: Ensure paths are correct for a Linux environment (use / instead of \)
ckpt_path="/home/lps/fyp//UniTS/checkpoints/ALL_task_myfyp_frompretrainedmodel_x128_prompttuning_single_ucihar_UniTS_All_ftM_dm128_el3_Exp_0/ptune_checkpoint.pth"
random_port=1234
prj_name="test_run_x128_prompt_tuning"
task_data_config="data_provider/myfyp_single_ucihar.yaml"
tflite_path="/home/lps/UniTS/example_models/test_model_quantized.tflite"

# measure disk space of the tflite model
disk_size_bytes=$(du -b "$tflite_path" | cut -f1)
echo "Disk size of the TFLite model: $disk_size_bytes bytes"

python -m memory_profiler /home/lps/UniTS/tflite_code/measure_latency.py --tflite_path $tflite_path --warm_up_runs 1 --measurement_runs 1