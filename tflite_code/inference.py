import torch
import ai_edge_torch
import numpy as np

# --- 1. Define the TFLite file path ---
tflite_path = ''
# tflite_path = '/home/lps/fyp/UniTS/test_model_1.tflite'
# NOTE: Make sure this file exists in your current directory, 
# or use the full path.

# --- 2. Load the EdgeModel from the TFLite file ---
try:
    edge_model_loaded = ai_edge_torch.load(tflite_path)
    print(f"Successfully loaded model from {tflite_path}")

    print(dir(edge_model_loaded))  # Print available methods and attributes

    # --- 3. Prepare the input data for inference ---
    # The input shape must match the shape the model was converted with 
    # (e.g., 1, 3, 224, 224 for a ResNet image model)
    # Using a dummy tensor for demonstration:
    input_shape = (1, 1, 140, 128) 
    # Convert to a NumPy array, as TFLite interpreters typically expect NumPy or bytes
    # dummy_input = np.random.randn(*input_shape).astype(np.float32)
    
    # get int8 random input for quantized model
    dummy_input = np.random.randint(-128, 127, size=input_shape).astype(np.int8)
    
    # --- 4. Run Inference ---
    # The loaded EdgeModel accepts the NumPy array directly
    output = edge_model_loaded(dummy_input)

    # --- 5. Process Output ---
    print("Inference successful.")
    # The output is a NumPy array (or a tuple of arrays)
    print("Output type:", type(output))
    print("Output shape:", output.shape)
    
except FileNotFoundError:
    print(f"Error: The file '{tflite_path}' was not found.")
    print("Please ensure you have run the conversion step and the file exists.")
except Exception as e:
    print(f"An error occurred during loading or inference: {e}")