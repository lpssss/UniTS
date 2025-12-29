import torch
import ai_edge_torch
import numpy as np
import time  # Added for timing

# --- 1. Define the TFLite file path ---
# tflite_path = '/home/lps/fyp/UniTS/example_models/test_model_quantized.tflite'
tflite_path = '/home/lps/fyp/UniTS/example_models/test_model_dynamic_quantized.tflite'
# tflite_path = '/home/lps/fyp/UniTS/example_models/test_model_1.tflite'

# --- 2. Load the EdgeModel ---
try:
    edge_model_loaded = ai_edge_torch.load(tflite_path)
    print(f"Successfully loaded model from {tflite_path}")

    # get input details
    print(dir(edge_model_loaded))
    print(edge_model_loaded._interpreter_builder())
    print(edge_model_loaded._interpreter_builder().get_input_details())

    # get quantization details
    input_details = edge_model_loaded._interpreter_builder().get_input_details()
    print("Input details:", input_details)

    # loop through input details to find quantization parameters
    quantization_params = []
    input_dtypes = []
    for detail in input_details:
        print(f"Input tensor '{detail['name']}' quantization parameters: {detail['quantization']}")
        quantization_params.append(detail['quantization'])
        input_dtypes.append(detail['dtype'])

    print("Quantization parameters for all inputs:", quantization_params)
    print("Input data types for all inputs:", input_dtypes)

    input_tensors = [np.random.randn(*detail['shape']).astype(np.float32) for detail in input_details]

    # if quantized model, prepare quantized input
    for i, (q_param, dtype) in enumerate(zip(quantization_params, input_dtypes)):
        if dtype == np.float32 or dtype == np.float16:
            print("Model is not quantized.")
            input_tensors[i] = input_tensors[i].astype(dtype)
        else:
            scale, zero_point = q_param
            print(f"Model is quantized with scale: {scale}, zero_point: {zero_point}")

            input_tensors[i] = (input_tensors[i] / scale + zero_point).astype(dtype)



    assert False, 'stop here'

    # --- 3. Prepare the input data ---
    input_shape = (1, 1, 140, 128) 
    dummy_input = np.random.randn(*input_shape).astype(np.float32)

    # get int8 random input for quantized model

    # dummy_input = np.random.randint(-128, 127, size=input_shape).astype(np.int8)
    print("Dummy input shape:", dummy_input.shape)

    print(dummy_input)

    # --- 4. Warm-up Phase ---
    # Run the model a few times to initialize the interpreter/hardware
    print("Warming up...")
    for _ in range(10):
        _ = edge_model_loaded(dummy_input)

    # --- 5. Latency Measurement Loop ---
    num_runs = 100
    latencies = []

    print(f"Starting inference benchmark for {num_runs} iterations...")
    for i in range(num_runs):
        start_time = time.perf_counter()
        output = edge_model_loaded(dummy_input)
        end_time = time.perf_counter()
        
        # Calculate duration in milliseconds
        latency_ms = (end_time - start_time) * 1000
        latencies.append(latency_ms)

    # --- 6. Results Calculation ---
    avg_latency = np.mean(latencies)
    median_latency = np.median(latencies)
    std_dev = np.std(latencies)
    fps = 1000 / avg_latency

    print("-" * 30)
    print(f"Inference Results ({num_runs} runs):")
    print(f"  Average Latency: {avg_latency:.2f} ms")
    print(f"  Median Latency:  {median_latency:.2f} ms")
    print(f"  Std Deviation:   {std_dev:.2f} ms")
    print(f"  Throughput:      {fps:.2f} FPS")
    print("-" * 30)
    
    # Print output shape for verification
    print("Output shape:", output.shape)
    
except FileNotFoundError:
    print(f"Error: The file '{tflite_path}' was not found.")
except Exception as e:
    print(f"An error occurred: {e}")