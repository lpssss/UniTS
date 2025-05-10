import torch.nn as nn
import torch
from models.UniTS import BasicBlock, CLSHead

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
    

model = TracedClassificationModel(num_classes=5)
x = torch.randn(1, 1, 140, 128)

import ai_edge_torch
edge_model = ai_edge_torch.convert(model.eval(), (x,))

edge_model.export('test_model.tflite')