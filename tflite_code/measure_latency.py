import torch
import ai_edge_torch
import numpy as np
import time  # Added for timing
from memory_profiler import profile
import argparse

# --- 1. Define the TFLite file path ---
# tflite_path = '/home/lps/fyp/UniTS/example_models/test_model_quantized.tflite'
tflite_path = '/home/lps/UniTS/example_models/test_model_dynamic_quantized.tflite'
# tflite_path = '/home/lps/fyp/UniTS/example_models/test_model_1.tflite'

# warm_up_runs = 1
# measurement_runs = 1
warm_up_runs = 10
measurement_runs = 100

import tracemalloc
memory_usage_data = []
def profile_memory(func):
    def wrapper(*args, **kwargs):
        tracemalloc.start()
        
        result = func(*args, **kwargs)
        
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        # Convert bytes to MB for readability
        peak_mb = peak / 1024 / 1024
        memory_usage_data.append(peak_mb)
            
        return result
    return wrapper

def main(args):
    tflite_path = args.tflite_path
    warm_up_runs = args.warm_up_runs
    measurement_runs = args.measurement_runs

    # --- 2. Load the EdgeModel ---
    try:
        def load_model():
            edge_model_loaded = ai_edge_torch.load(tflite_path)
            return edge_model_loaded

        edge_model_loaded = load_model()
        print(f"Successfully loaded model from {tflite_path}")

        # get input details
        # print(dir(edge_model_loaded))
        # print(edge_model_loaded._interpreter_builder())
        # print(edge_model_loaded._interpreter_builder().get_input_details())

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

        # --- 4. Warm-up Phase ---
        # Run the model a few times to initialize the interpreter/hardware
        print("Warming up...")
        for _ in range(warm_up_runs):
            _ = edge_model_loaded(*input_tensors)
        # --- 5. Latency Measurement Loop ---
        num_runs = measurement_runs
        latencies = []

        print(f"Starting inference benchmark for {num_runs} iterations...")
        for i in range(num_runs):
            start_time = time.perf_counter()
            output = edge_model_loaded(*input_tensors)
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
        print(f'Output length: {len(output)}')
        print("First Output shape:", output[0].shape)
        
    except FileNotFoundError:
        print(f"Error: The file '{tflite_path}' was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

@profile
def main_mem(args):
    tflite_path = args.tflite_path
    warm_up_runs = args.warm_up_runs
    measurement_runs = args.measurement_runs
    # --- 2. Load the EdgeModel ---
    try:
        @profile_memory
        def load_model():
            edge_model_loaded = ai_edge_torch.load(tflite_path)
            return edge_model_loaded

        edge_model_loaded = load_model()
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

        # --- 4. Warm-up Phase ---
        # Run the model a few times to initialize the interpreter/hardware
        print("Warming up...")
        for _ in range(warm_up_runs):
            _ = edge_model_loaded(*input_tensors)
        # --- 5. Latency Measurement Loop ---
        num_runs = measurement_runs
        latencies = []

        @profile_memory
        def run_inference():
            print(f"Starting inference benchmark for {num_runs} iterations...")
            for i in range(num_runs):
                start_time = time.perf_counter()
                output = edge_model_loaded(*input_tensors)
                end_time = time.perf_counter()
                
                # Calculate duration in milliseconds
                latency_ms = (end_time - start_time) * 1000
                latencies.append(latency_ms)

            return output
        
        output = run_inference()

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
        print(f'Total memory used: {sum(memory_usage_data):.2f} MB, {memory_usage_data}')
        
        # Print output shape for verification
        print(f'Output length: {len(output)}')
        print("First Output shape:", output[0].shape)
        
    except FileNotFoundError:
        print(f"Error: The file '{tflite_path}' was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Measure TFLite Model Latency")
    parser.add_argument('--tflite_path', type=str, default=tflite_path, help='Path to the TFLite model file')
    parser.add_argument('--warm_up_runs', type=int, default=warm_up_runs, help='Number of warm-up runs')
    parser.add_argument('--measurement_runs', type=int, default=measurement_runs, help='Number of measurement runs')
    args = parser.parse_args()

    tflite_path = args.tflite_path
    warm_up_runs = args.warm_up_runs
    measurement_runs = args.measurement_runs

    if measurement_runs == 1 and warm_up_runs == 1:
        main_mem(args)
    else:
        main(args)