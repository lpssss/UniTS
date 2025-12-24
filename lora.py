# lora_fc_replace.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Replace existing nn.Linear with LoRALinear in-place
# ============================================================
def replace_fc_with_lora(
    model: nn.Module,
    fc_name: str,
    r: int = 8,
    lora_alpha: float = 1.0,
):
    """
    Replace model.<fc_name> (nn.Linear) with LoRALinear
    while preserving weight name and values.
    """
    print(f'Replacing {fc_name} with LoRALinear (r={r}, alpha={lora_alpha})')
    modules = dict(model.named_modules())
    assert fc_name in modules, f"{fc_name} not found in model"
    fc = modules[fc_name]
    assert isinstance(fc, nn.Linear), f"{fc_name} is not nn.Linear"

    # Create LoRA FC
    lora_fc = LoRALinear(
        in_features=fc.in_features,
        out_features=fc.out_features,
        r=r,
        lora_alpha=lora_alpha,
        bias=fc.bias is not None,
    )

    # Copy pretrained weights
    lora_fc.weight.data.copy_(fc.weight.data)
    if fc.bias is not None:
        lora_fc.bias.data.copy_(fc.bias.data)

    # Replace module in parent
    parent = model
    *path, name = fc_name.split(".")
    for p in path:
        parent = getattr(parent, p)
    setattr(parent, name, lora_fc)


# ============================================================
# Example model + verification
# ============================================================
if __name__ == "__main__":

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(512, 10)

        def forward(self, x):
            return self.fc(x)

    model = Model()

    print("Before replacement:")
    print(model)
    print()

    # Replace FC
    replace_fc_with_lora(model, "fc", r=4, lora_alpha=8)

    print("After replacement:")
    print(model)
    print()

    # --------------------------------------------------------
    # Verify state_dict keys
    # --------------------------------------------------------
    print("State dict keys:")
    for k in model.state_dict().keys():
        if "fc" in k:
            print(" ", k)

    # --------------------------------------------------------
    # Verify trainable parameters
    # --------------------------------------------------------
    print("\nTrainable parameters:")
    for name, param in model.named_parameters():
        print(f"{name:20s} requires_grad={param.requires_grad}")

    # --------------------------------------------------------
    # Forward test
    # --------------------------------------------------------
    x = torch.randn(2, 512)
    y = model(x)
    print("\nOutput shape:", y.shape)
