from ai_edge_quantizer import recipe
from ai_edge_quantizer import quantizer
import numpy as np
import model_explorer

dynamic_quant_mnist_model_path = "/home/lps/UniTS/example_models/test_quantizer_8w16a_1.tflite"
model_path = '/home/lps/UniTS/tflite_models/myfyp_frompretrainedmodel_x128_prompttuning_single_ucihar.tflite'
calib_data_path = "/home/lps/UniTS/calib_data/myfyp_single_ucihar_calib_data.npz"


# read data from calib_data_path npz
calib_data = np.load(calib_data_path)

qt = quantizer.Quantizer(model_path, recipe.static_wi8_ai16(), previous_quantized_model=dynamic_quant_mnist_model_path)

comparison_result = qt.validate(
    error_metrics='median_diff_ratio',
    use_xnnpack=False,
    num_threads=1,
).save('', 'dynamic')

model_explorer.visualize_from_config(
    model_explorer.config()
    .add_model_from_path(dynamic_quant_mnist_model_path)
    .add_node_data_from_path('dynamic_comparison_result_me_input.json')
)