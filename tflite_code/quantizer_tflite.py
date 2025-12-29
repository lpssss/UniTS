from ai_edge_quantizer import quantizer
from ai_edge_quantizer import recipe
from ai_edge_quantizer import qtyping
from ai_edge_quantizer.utils import tfl_flatbuffer_utils
import tensorflow as tf
import numpy as np
import json
import model_explorer

dynamic_quant_mnist_model_path = "/home/lps/UniTS/example_models/test_quantizer_8w16a_1.tflite"
# dynamic_quant_mnist_model_path = "/home/lps/UniTS/example_models/test_quantizer_dq.tflite"
model_path = '/home/lps/UniTS/tflite_models/myfyp_frompretrainedmodel_x128_prompttuning_single_ucihar.tflite'
calib_data_path = "/home/lps/UniTS/calib_data/myfyp_single_ucihar_calib_data.npz"

# read data from calib_data_path npz
calib_data = np.load(calib_data_path)

def prepare_calib_data(calib_data, input_signature):
    def representative_dataset():
        for i in range(calib_data['x'].shape[0]):
            data = calib_data['x'][i:i+1, ...]
            yield {input_signature[0]: data.astype(np.float32)}

    return representative_dataset

interpreter = tf.lite.Interpreter(model_path=model_path)   

# Print the signatures from the converted model
signatures = interpreter.get_signature_list()
print('Signature:', signatures)

# qt = quantizer.Quantizer(model_path, recipe.dynamic_wi8_afp32())
qt = quantizer.Quantizer(model_path, recipe.static_wi8_ai16())

# calibrate the model
calibration_results = qt.calibrate(
    {'serving_default': prepare_calib_data(calib_data, interpreter.get_signature_list()['serving_default']['inputs'])()},
)

quant_result = qt.quantize(calibration_result=calibration_results).export_model(dynamic_quant_mnist_model_path, overwrite=True)

quantized_model = tf.lite.Interpreter(model_path=dynamic_quant_mnist_model_path)
dummy_input = np.random.randint(0, 255, size=(1, 6, 19, 128)).astype(np.int16)

quantized_model.allocate_tensors()
input_details = quantized_model.get_input_details()
output_details = quantized_model.get_output_details()
quantized_model.set_tensor(input_details[0]['index'], dummy_input)
quantized_model.invoke()
output_data = quantized_model.get_tensor(output_details[0]['index'])
print("Output data:", output_data)


# save calibration results in json
print("Calibration Results:", calibration_results)

# Save
import pickle
with open('calibration_results.pkl', 'wb') as f:
    pickle.dump(calibration_results, f)



# if visualize_model:
#   model_explorer.visualize(dynamic_quant_mnist_model_path)