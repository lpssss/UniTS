import torch.nn as nn
import torch
import tensorflow as tf
import numpy as np
import ai_edge_torch
import sys
sys.path.append("/home/lps/fyp/UniTS")
from models.UniTS import BasicBlock, CLSHead, Model


class TracedClassificationModel(nn.Module):
    def __init__(self, num_classes):
        super(TracedClassificationModel, self).__init__()
        self.block_num = 3
        self.n_heads = 8
        self.dropout = 0.1
        self.d_model = 128
        self.e_layers = 3
        self.prompt_num = 10

        # calculate seq len based on the input shape
        # TODO:
        self.seq_len = 128
        self.category_tokens = nn.Parameter(torch.randn(1, 1, self.prompt_num, self.d_model))

        self.blocks = nn.ModuleList(
            [BasicBlock(dim=self.d_model, num_heads=self.n_heads, qkv_bias=False, qk_norm=False,
                        mlp_ratio=8., proj_drop=self.dropout, attn_drop=0., drop_path=0.,
                        init_values=None, prefix_token_length=self.prompt_num) for l in range(self.e_layers)]
        )

        self.cls_head = CLSHead(self.d_model, head_dropout=self.dropout)


    def backbone(self, x, prefix_len, seq_len):
        attn_mask = None
        for block in self.blocks:
            x = block(x, prefix_seq_len=prefix_len +
                      seq_len, attn_mask=attn_mask)
        return x
    
    def forward(self, x):

        x = self.backbone(x, prefix_len=self.prompt_num, seq_len=self.seq_len)
        x = self.cls_head(x, self.category_tokens)
        return x
    

def trace_fp_model():
    model = TracedClassificationModel(num_classes=5)
    x = torch.randn(1, 1, 140, 128)

    edge_model = ai_edge_torch.convert(model.eval(), (x,))

    edge_model.export('test_model_1.tflite')

def trace_quantized_model():
    def representative_dataset():
        for _ in range(5):
            data = np.random.rand(1, 1, 140, 128)
            yield [data.astype(np.float32)]

    tfl_converter_flags={
        "optimizations": [tf.lite.Optimize.DEFAULT],
        "target_spec.supported_ops": [tf.lite.OpsSet.TFLITE_BUILTINS_INT8],
        "inference_input_type": tf.int8,
        "inference_output_type": tf.int8,
        "representative_dataset": representative_dataset
    }

    model = TracedClassificationModel(num_classes=5)
    x = torch.randn(1, 1, 140, 128)
    edge_model = ai_edge_torch.convert(model.eval(), (x,), _ai_edge_converter_flags=tfl_converter_flags)

    edge_model.export('test_model_quantized.tflite')

def trace_dynamic_quantized_model():
    tfl_converter_flags={
        "optimizations": [tf.lite.Optimize.DEFAULT],
    }

    model = TracedClassificationModel(num_classes=5)
    x = torch.randn(1, 1, 140, 128)
    edge_model = ai_edge_torch.convert(model.eval(), (x,), _ai_edge_converter_flags=tfl_converter_flags)

    edge_model.export('test_model_dynamic_quantized.tflite')

if __name__ == "__main__":
    # trace_fp_model()
    # trace_quantized_model()
    trace_dynamic_quantized_model()

# from ai_edge_torch.quantize.pt2e_quantizer import get_symmetric_quantization_config
# from ai_edge_torch.quantize.pt2e_quantizer import PT2EQuantizer
# from ai_edge_torch.quantize.quant_config import QuantConfig