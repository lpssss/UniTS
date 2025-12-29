import torch

def compare_all_tensors(path_a, path_b, tolerance=1e-6):
    """
    Compares all corresponding tensors in two .pth files.
    """
    # Load state dicts
    data_a = torch.load(path_a, map_location='cpu')
    data_b = torch.load(path_b, map_location='cpu')
    
    # Extract state_dict if wrapped in a dictionary
    sd_a = data_a.get('state_dict', data_a)
    sd_b = data_b.get('state_dict', data_b)

    keys_a = set(sd_a.keys())
    keys_b = set(sd_b.keys())

    # Check for structural differences
    only_in_a = keys_a - keys_b
    only_in_b = keys_b - keys_a
    common_keys = keys_a & keys_b

    print(f"--- Model Structure Comparison ---")
    print(f"Tensors unique to File A: {len(only_in_a)}")
    print(f"Tensors unique to File B: {len(only_in_b)}")
    print(f"Unique to A: {only_in_a}")
    print(f"Unique to B: {only_in_b}")
    print(f"Common Tensors to compare: {len(common_keys)}\n")

    print(f"{'Layer Name':<50} | {'Status':<10} | {'Max Diff':<10}")
    print("-" * 75)

    identical_count = 0
    changed_count = 0

    for key in sorted(common_keys):
        tensor_a = sd_a[key]
        tensor_b = sd_b[key]

        if tensor_a.shape != tensor_b.shape:
            print(f"{key:<50} | SHAPE MISMATCH")
            continue

        # Calculate difference
        diff = torch.abs(tensor_a - tensor_b)
        max_diff = torch.max(diff).item()

        if max_diff <= tolerance:
            status = "IDENTICAL"
            identical_count += 1
        else:
            status = "CHANGED"
            changed_count += 1
            print(f"{key:<50} | {status:<10} | {max_diff:.2e}")

    print("-" * 75)
    print(f"Summary: {identical_count} identical, {changed_count} changed.")

# Usage
# compare_all_tensors('model_epoch_1.pth', 'model_epoch_2.pth')

if __name__ == "__main__":
    path_a = '/home/lps/fyp//UniTS/checkpoints/ALL_task_myfyp_frompretrainedmodel_x128_prompttuning_single_ucihar_UniTS_All_ftM_dm128_el3_Exp_0/ptune_checkpoint.pth'
    path_b = '/home/lps/fyp//UniTS/checkpoints/ALL_task_myfyp_frompretrainedmodel_x128_prompttuning_single_wesad_UniTS_All_ftM_dm128_el3_Exp_0/ptune_checkpoint.pth'
    compare_all_tensors(path_a, path_b)