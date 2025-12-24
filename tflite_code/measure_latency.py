import time
import numpy as np
import tflite_runtime.interpreter as tflite  # lighter than full TensorFlow

# Path to your TFLite model
MODEL_PATH = '/home/lps/fyp/UniTS/example_models/test_model_quantized.tflite'

# Number of warmup and timed runs
WARMUP_RUNS = 5
MEASURE_RUNS = 50

# Load TFLite model and allocate tensors
interpreter = tflite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Generate random input data (adjust shape and dtype as needed)
input_shape = input_details[0]['shape']
input_dtype = input_details[0]['dtype']
input_data = np.random.rand(*input_shape).astype(input_dtype)

# Warmup runs (stabilize caches, CPU freq scaling, etc.)
print("Warming up...")
for _ in range(WARMUP_RUNS):
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

# Measure inference latency
print(f"Measuring latency over {MEASURE_RUNS} runs...")
times = []
for i in range(MEASURE_RUNS):
    start = time.perf_counter()
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    end = time.perf_counter()
    times.append((end - start) * 1000)  # convert to ms

avg_latency = np.mean(times)
std_latency = np.std(times)
p50 = np.percentile(times, 50)
p90 = np.percentile(times, 90)
p99 = np.percentile(times, 99)

print("\n=== Inference Latency Results (ms) ===")
print(f"Average latency: {avg_latency:.2f} ms")
print(f"Std deviation : {std_latency:.2f} ms")
print(f"P50 latency   : {p50:.2f} ms")
print(f"P90 latency   : {p90:.2f} ms")
print(f"P99 latency   : {p99:.2f} ms")
