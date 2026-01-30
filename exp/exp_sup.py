from data_provider.data_factory import data_provider
from utils.tools import adjust_learning_rate, cal_accuracy, adjustment
from utils.tools import NativeScalerWithGradNormCount as NativeScaler
from utils.metrics import metric
from utils.losses import mape_loss, mase_loss, smape_loss
from utils.dataloader import BalancedDataLoaderIterator
from utils.layer_decay import param_groups_lrd
from utils.ddp import get_world_size, is_main_process, gather_tensors_from_all_gpus

from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score


import torch
import torch.nn as nn
from torch import optim
import torch.distributed as dist

import os
import time
import warnings
import numpy as np
import yaml
import wandb
import sys
import copy

warnings.filterwarnings('ignore')


def apply_random_mask_for_imputation(x, patch_len, mask_rate):
    """
    Apply a random mask to the input tensor.

    Parameters:
    x (torch.Tensor): The input tensor with shape [B, T, N].
    patch_len (int): The length of each patch.
    mask_rate (float): The proportion of the tensor to be masked.

    Returns:
    torch.Tensor: The masked input tensor.
    torch.Tensor: The mask tensor.
    """
    B, T, N = x.shape
    num_keep = int((T // patch_len) * (1 - mask_rate))

    # Generate random noise and sort it
    noise = torch.rand(B, T // patch_len, device=x.device)
    ids_shuffle = torch.argsort(noise, dim=1)
    ids_restore = torch.argsort(ids_shuffle, dim=1)

    # Select indices to keep
    ids_keep = ids_shuffle[:, :num_keep]
    mask = torch.zeros([B, T], device=x.device)
    mask[:, :num_keep] = 1
    # unshuffle to get the binary mask
    mask = torch.gather(mask, dim=1, index=ids_restore)

    # Expand the mask to the original shape
    mask = mask.unsqueeze(-1).repeat(1, 1, patch_len).view(B, T)
    mask = mask.unsqueeze(-1).repeat(1, 1, N)

    # Apply the mask
    x_masked = x.masked_fill(mask == 0, 0)

    return x_masked, mask


def custom_print_decorator(func):
    def wrapper(*args, **kwargs):
        text = ' '.join(map(str, args))
        if 'file' not in kwargs or kwargs['file'] is None:
            sys.stdout.write(text + '\n')
        else:
            kwargs['file'].write(text + '\n')

        if 'folder' in kwargs and kwargs['folder']:
            with open(f'{kwargs["folder"]}/finetune_output.log', 'a') as log_file:
                log_file.write(text + '\n')
        if 'folder' in kwargs:
            del kwargs['folder']
        if 'file' in kwargs:
            del kwargs['file']

    return wrapper


# Replace print to save all print into log files
print = custom_print_decorator(print)


def read_task_data_config(config_path):
    with open(config_path, 'r') as config_file:
        config = yaml.load(config_file, Loader=yaml.FullLoader)
    task_dataset_config = config.get('task_dataset', {})
    return task_dataset_config


def get_task_data_config_list(task_data_config, default_batch_size=None):
    task_data_config_list = []

    for task_name, task_config in task_data_config.items():
        task_config['max_batch'] = default_batch_size
        task_data_config_list.append([task_name, task_config])

    return task_data_config_list


def change_config_list_pred_len(task_data_config_list, task_data_config, offset):
    print("Warning: change the forecasting len and remove the cls task!")
    new_task_data_config = copy.deepcopy(task_data_config)
    new_task_data_config_list = copy.deepcopy(task_data_config_list)
    for task_name, task_config in new_task_data_config.items():
        if task_config['task_name']=='long_term_forecast':
            new_task_data_config[task_name]['pred_len']+=offset
        else:
            del new_task_data_config[task_name]

    for each_config in new_task_data_config_list:
        if each_config[1]['task_name'] =='long_term_forecast':
            each_config[1]['pred_len']+=offset
        else:
            del each_config

    return new_task_data_config_list, new_task_data_config


def get_loss_by_name(loss_name, class_weights=None):
    if loss_name == 'MSE':
        return nn.MSELoss()
    elif loss_name == 'MAPE':
        return mape_loss()
    elif loss_name == 'MASE':
        return mase_loss()
    elif loss_name == 'SMAPE':
        return smape_loss()
    elif loss_name == 'CE':
        return nn.CrossEntropyLoss(weight=class_weights)
    else:
        print("no loss function found!")
        exit()


def init_and_merge_datasets(data_loader_list):
    dataloader = BalancedDataLoaderIterator(data_loader_list)
    train_steps = dataloader.__len__()
    return dataloader, train_steps


class Exp_All_Task(object):
    def __init__(self, args):
        super(Exp_All_Task, self).__init__()

        self.args = args
        self.ori_task_data_config = read_task_data_config(
            self.args.task_data_config_path)
        self.ori_task_data_config_list = get_task_data_config_list(
            self.ori_task_data_config, default_batch_size=self.args.batch_size)

        if self.args.zero_shot_forecasting_new_length is not None:
            print("Change the forecasting len!")
            self.task_data_config_list, self.task_data_config = change_config_list_pred_len(self.ori_task_data_config_list, self.ori_task_data_config, self.args.offset)
        else:
            self.task_data_config = self.ori_task_data_config
            self.task_data_config_list = self.ori_task_data_config_list
        # device_id = dist.get_rank() % torch.cuda.device_count()
        self.device_id = 0
        print("device id", self.device_id)
        self.model = self._build_model()
        self.start_training = False
        self.is_lora_initiated = False

    def _build_model(self, ddp=False):
        import importlib
        module = importlib.import_module("models."+self.args.model)
        model = module.CompositeModel(
            self.args, self.task_data_config_list).to(self.device_id)
        if ddp:
            model = nn.parallel.DistributedDataParallel(model, device_ids=[self.device_id],
                                                        find_unused_parameters=True, gradient_as_bucket_view=True, static_graph=False)
        return model

    def _get_data(self, flag, test_anomaly_detection=False):
        if self.args.zero_shot_forecasting_new_length is not None:
            _, max_offset_task_data_config = change_config_list_pred_len(self.ori_task_data_config_list, self.ori_task_data_config, self.args.max_offset)
            this_task_data_config = max_offset_task_data_config
        else:
            this_task_data_config = self.task_data_config
            
        data_set_list = []
        data_loader_list = []

        for task_data_name, task_config in this_task_data_config.items():
            if task_config['task_name'] == 'classification' and flag == 'val':
                # TODO strange that no val set is used for classification. Set to test set for val
                flag = 'test'
            if test_anomaly_detection and task_config['task_name'] == 'anomaly_detection':
                train_data_set, train_data_loader = data_provider(
                    self.args, task_config, flag='train', ddp=False)
                data_set, data_loader = data_provider(
                    self.args, task_config, flag, ddp=False)  # ddp false to avoid shuffle
                data_set_list.append([train_data_set, data_set])
                data_loader_list.append([train_data_loader, data_loader])
                print(task_data_name, len(data_set))
            else:
                data_set, data_loader = data_provider(
                    self.args, task_config, flag, ddp=False)
                data_set_list.append(data_set)
                data_loader_list.append(data_loader)
                print(f'Getting data: {task_data_name}, {len(data_set)}')
        return data_set_list, data_loader_list

    def _select_optimizer(self):
        eff_batch_size = self.args.batch_size * self.args.acc_it * get_world_size()
        real_learning_rate = self.args.learning_rate * eff_batch_size / 32
        self.real_learning_rate = real_learning_rate
        print(f'args learning rate: {self.args.learning_rate}')
        print("base lr: %.2e" % (self.args.learning_rate * 32 / eff_batch_size))
        print("actual lr: %.2e" % real_learning_rate)

        print("accumulate grad iterations: %d" % self.args.acc_it)
        print("args batch size: %d" % self.args.batch_size)
        print("effective batch size: %d" % eff_batch_size)
        if self.args.layer_decay is not None:
            print("layer decay: %.2f" % self.args.layer_decay)
            model_without_ddp = self.model.module
            param_groups = param_groups_lrd(model_without_ddp, self.args.weight_decay,
                                            no_weight_decay_list=[
                                                'prompts', 'mask_tokens', 'cls_tokens', 'category_tokens'],
                                            layer_decay=self.args.layer_decay
                                            )
            model_optim = optim.Adam(param_groups, lr=real_learning_rate)
        else:
            model_optim = optim.Adam(self.model.parameters(
            ), lr=real_learning_rate, weight_decay=self.args.weight_decay)
        return model_optim

    def _select_criterion(self, config_list):
        criterion_list = []
        for each_config in config_list:
            if 'loss' in each_config[1]:
                loss_name = each_config[1]['loss']
            else:
                if each_config[1]['task_name'] == 'long_term_forecast':
                    loss_name = 'MSE'
                elif each_config[1]['task_name'] == 'classification':
                    loss_name = 'CE'
                elif each_config[1]['task_name'] == 'imputation':
                    loss_name = 'MSE'
                elif each_config[1]['task_name'] == 'anomaly_detection':
                    loss_name = 'MSE'
                else:
                    print("this task has no loss now!", folder=self.path)
                    exit()
            criterion_list.append(get_loss_by_name(loss_name, class_weights=each_config[1].get('class_weights', None)))

        return criterion_list

    def choose_training_parts(self, prompt_tune=False, lora_tune=False):
        for name, param in self.model.named_parameters():
            if prompt_tune and lora_tune:
                if 'prompt_token' in name or 'mask_prompt' in name or 'cls_prompt' in name or 'mask_token' in name or 'cls_token' in name or 'category_token' in name:
                    param.requires_grad = True
                    print("trainable:", name)
                elif 'lora_' in name:
                    param.requires_grad = True
                    print("trainable:", name)
                else:
                    param.requires_grad = False
            elif prompt_tune:
                if 'prompt_token' in name or 'mask_prompt' in name or 'cls_prompt' in name or 'mask_token' in name or 'cls_token' in name or 'category_token' in name:
                    param.requires_grad = True
                    print("trainable:", name)
                else:
                    param.requires_grad = False
            elif lora_tune:
                # need to enable training for tokens as they are new parameters
                if 'prompt_token' in name or 'mask_prompt' in name or 'cls_prompt' in name or 'mask_token' in name or 'cls_token' in name or 'category_token' in name:
                    param.requires_grad = True
                    print("trainable:", name)
                elif 'lora_' in name:
                    param.requires_grad = True
                    print("trainable:", name)
                else:
                    param.requires_grad = False
            else:
                param.requires_grad = True

        if not prompt_tune and not lora_tune:
            print("all trainable.")

    def calculate_trainable_params(self, print_trainable=False):
        model_param = []
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                model_param.append(param.numel())
                if print_trainable:
                    print("In calc trainable:", name)
        model_total_params = sum(model_param)
        print("Trainable Parameters number for UniTS {} M".format(
            model_total_params/1e6), folder=self.path)
        
        return model_total_params

    def calculate_all_params(self, print_trainable=False):
        model_param = []
        for name, param in self.model.named_parameters():
            model_param.append(param.numel())
        model_total_params = sum(model_param)
        print("Trainable Parameters number for UniTS {} M".format(
            model_total_params/1e6), folder=self.path)
        
        return model_total_params

    

    def plot_loss_curve(self, losses, filename, title="Training Loss Over Time"):
        """
        Plots the loss curve from a list of loss values.
        """
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 5))
        plt.plot(losses, color='tab:red', linewidth=2, label='Loss')
        
        # Adding metadata to the plot
        plt.title(title, fontsize=14)
        plt.xlabel('Epochs/Iterations', fontsize=12)
        plt.ylabel('Loss Value', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()
        
        # Save the file before showing it
        # dpi=300 is standard for high-quality reports
        filepath = os.path.join(self.path, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Plot saved successfully as {filepath}")
        
        plt.close() # Close the plot to free up memory

    def get_class_weights_df(self, label_df):
        """
        label_df: A pandas Series or DataFrame column containing integer labels.
        Example: df['target_column']
        """
        # 1. Count occurrences of each class
        # sort_index() is crucial to ensure Class 0 is first, then Class 1, etc.
        counts = label_df.value_counts().sort_index()
        
        total_samples = len(label_df)
        num_classes = len(counts)
        
        # 2. Calculate Inverse Frequency: Total / (Classes * Count_per_Class)
        weights = total_samples / (num_classes * counts.values)
        
        # 3. Convert to Torch Tensor
        weights_tensor = torch.tensor(weights, dtype=torch.float)
        
        # 4. Print for verification
        for i, w in enumerate(weights):
            print(f"Class {i} (n={counts.values[i]}): Weight = {w:.4f}")
            
        return weights_tensor

    # ============================================================
    # Replace existing nn.Linear with LoRALinear in-place
    # ============================================================
    def replace_fc_with_lora(
        self,
        model: nn.Module,
        fc_name: str,
        r: int = 8,
        lora_alpha: float = 1.0,
    ):
        from models.UniTS import LoRALinear
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

        # set all weights in lora_fc to same device as fc weights
        lora_fc.to(fc.weight.device)

        # Replace module in parent
        parent = model
        *path, name = fc_name.split(".")
        for p in path:
            parent = getattr(parent, p)
        setattr(parent, name, lora_fc)

    def train(self, setting):
        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path) and is_main_process():
            os.makedirs(path)
        self.path = path

        train_loss_per_epoch_list = []
        test_loss_per_epoch_list = []

        initial_model_params = self.calculate_all_params()

        self.init_lora()

        final_model_params = self.calculate_all_params()

        # Load pretrained weights (Optional)
        if self.args.pretrained_weight is not None:
            if self.args.pretrained_weight == 'auto':
                pretrain_weight_path = os.path.join(
                    self.path, 'pretrain_checkpoint.pth')
            else:
                pretrain_weight_path = self.args.pretrained_weight
            print('loading pretrained model:',
                  pretrain_weight_path, folder=self.path)
            if 'pretrain_checkpoint.pth' in pretrain_weight_path:
                state_dict = torch.load(
                    pretrain_weight_path, map_location='cpu', weights_only=False)['student']
                ckpt = {}
                for k, v in state_dict.items():
                    if not ('cls_prompts' in k):
                        ckpt[k] = v
            else:
                ckpt = torch.load(pretrain_weight_path, map_location='cpu', weights_only=False)

            # remove module. prefix if present
            new_ckpt = {}
            for k, v in ckpt.items():
                if k.startswith('module.'):
                    new_ckpt[k[7:]] = v
                else:
                    new_ckpt[k] = v

            ckpt = new_ckpt
            
            # find intersection keys and missing keys
            intersection_keys = set(ckpt.keys()) & set(self.model.state_dict().keys())
            missing_keys = set(self.model.state_dict().keys()) - set(ckpt.keys())
            print("Intersection keys:", intersection_keys, folder=self.path)
            print("Missing keys:", missing_keys, folder=self.path)

            # # dump all model keys and ckpt keys for debugging in a json file
            # with open('model_keys.json', 'w') as f:
            #     import json
            #     model_keys = list(self.model.state_dict().keys())
            #     ckpt_keys = list(ckpt.keys())
            #     json.dump({'model_keys': model_keys, 'ckpt_keys': ckpt_keys}, f, indent=4)
            # assert False

            msg = self.model.load_state_dict(ckpt, strict=False)
            print(msg, folder=self.path)

        # Data
        _, train_loader_list = self._get_data(flag='train')
        # Since some datasets do not have val set, we use test set and report the performance of last epoch instead of the best epoch.
        test_data_list, test_loader_list = self._get_data(
            flag='test', test_anomaly_detection=True)
        data_loader_cycle, train_steps = init_and_merge_datasets(
            train_loader_list)

        # Model param check
        pytorch_total_params = sum(p.numel() for p in self.model.parameters())
        print("Parameters number for all {} M".format(
            pytorch_total_params/1e6), folder=self.path)
        model_param = []
        for name, param in self.model.named_parameters():
            if ('prompts' in name and 'prompt2forecat' not in name) or 'prompt_token' in name or \
                'mask_prompt' in name or 'cls_prompt' in name or 'mask_token' in name or 'cls_token' in name or 'category_token' in name:
                print('skip this:', name)
            else:
                model_param.append(param.numel())
        model_total_params = sum(model_param)
        print("Parameters number for UniTS {} M".format(
            model_total_params/1e6), folder=self.path)
        
        print("Choosing training parts...")
        self.choose_training_parts(lora_tune=self.args.lora)
        
        trainable_params = self.calculate_trainable_params(print_trainable=True)
        # exit(0)

        # print model params summary
        print('------------------- Model Parameters Summary ------------------', folder=self.path)
        print(f"Initial model params: {initial_model_params} | Final model params: {final_model_params} | Trainable params: {trainable_params}", folder=self.path)
        print(f'Lora parameters increase: {final_model_params - initial_model_params}, ratio {(final_model_params - initial_model_params) / initial_model_params:.2f} ', folder=self.path)
        print(f'Lora args enable: {self.args.lora}, r: {self.args.lora_r}, lora alpha: {self.args.lora_alpha}', folder=self.path)
        print(f'Trainable parameters percentage: {trainable_params / final_model_params * 100:.2f} %', folder=self.path)
        print('---------------------------------------------------------------', folder=self.path)

        # calculate class weights for classification tasks
        if self.args.use_weighted_loss:
            for task_id, each_config in enumerate(self.task_data_config_list):
                if each_config[1]['task_name'] == 'classification':
                    train_data_set = train_loader_list[task_id].dataset
                    class_weights = self.get_class_weights_df(train_data_set.labels_df)
                    each_config[1]['class_weights'] = class_weights.to(self.device_id)
                    print(f"Class weights for task {each_config[0]}: {each_config[1]['class_weights']}", folder=self.path)

        # Optimizer and Criterion
        model_optim = self._select_optimizer()
        criterion_list = self._select_criterion(self.task_data_config_list)
        scaler = NativeScaler(is_enabled=self.args.enable_mixed_precision_training)

        # Set up batch size for each task
        if self.args.memory_check:
            self.memory_check(data_loader_cycle, criterion_list, holdout_memory=1)
            torch.cuda.empty_cache()
        torch.cuda.synchronize()
        # dist.barrier()

        # profile training step
        if self.args.profile_training_step:
            self.profile_training_step(model_optim, data_loader_cycle, criterion_list, 1, train_steps, scaler)
            return self.model

        self.start_training = True

        for epoch in range(self.args.train_epochs+self.args.prompt_tune_epoch):
            adjust_learning_rate(model_optim, epoch,
                                 self.real_learning_rate, self.args)
            # Prompt learning
            if (epoch+1) <= self.args.prompt_tune_epoch:
                self.choose_training_parts(prompt_tune=True, lora_tune=self.args.lora)
            else:
                self.choose_training_parts(prompt_tune=False, lora_tune=self.args.lora)

            train_loss = self.train_one_epoch(
                model_optim, data_loader_cycle, criterion_list, epoch, train_steps, scaler)
            
            train_loss_per_epoch_list.append(train_loss)

            # we report the results of last epoch and not find the best epoch based on val set, since some datasets do not have val set
            avg_cls_acc, avg_forecast_mse, avg_forecast_mae, avg_loss = self.test(
                setting, load_pretrain=False, test_data_list=test_data_list, test_loader_list=test_loader_list)
            test_loss_per_epoch_list.append(avg_loss)

            # save ckpt
            if is_main_process():
                if self.args.prompt_tune_epoch >= 1:
                    torch.save(self.model.state_dict(),
                               os.path.join(path, 'ptune_checkpoint.pth'))
                else:
                    torch.save(self.model.state_dict(),
                               os.path.join(path, 'checkpoint.pth'))
            print(f'Loss for epoch {epoch}: {train_loss}', folder=self.path)

        if is_main_process():
            wandb.log({'Final_LF-mse': avg_forecast_mse,
                       'Final_LF-mae': avg_forecast_mae, 'Final_CLS-acc': avg_cls_acc})
            print("Final score: LF-mse: {}, LF-mae: {}, CLS-acc {}".format(avg_forecast_mse,
                                                                           avg_forecast_mae, avg_cls_acc), folder=self.path)
            # plot loss curve
            self.plot_loss_curve(train_loss_per_epoch_list, filename="train_loss_curve.png", title="Training Loss Over Epochs")
            self.plot_loss_curve(test_loss_per_epoch_list, filename="test_loss_curve.png", title="Test Loss Over Epochs")

        return self.model

    def train_one_epoch(self, model_optim, data_loader_cycle, criterion_list, epoch, train_steps, scaler):
        current_device = torch.cuda.current_device()
        train_loss_set = []
        acc_it = self.args.acc_it
        max_norm = self.args.clip_grad

        self.model.train()
        epoch_time = time.time()
        self.model.zero_grad(set_to_none=True)
        loss_sum = 0

        for i, (sample_init, task_id) in enumerate(data_loader_cycle):

            task_name = self.task_data_config_list[task_id][1]['task_name']
            small_batch_size = self.task_data_config_list[task_id][1]['max_batch']
            # print(f'small batch size: {small_batch_size} for task {task_name}')
            if small_batch_size != self.args.batch_size:
                sample_list = self.split_batch(
                    sample_init, small_batch_size, task_name)
                len_sample_list = len(sample_list)
            else:
                sample_list = [sample_init]
                len_sample_list = 1

            # print(f'len of sample list: {len_sample_list} for task {task_name}')

            for sample_idx in range(len_sample_list):
                sample = sample_list[sample_idx]
                if task_name == 'long_term_forecast':
                    loss = self.train_long_term_forecast(
                        self.model, sample, criterion_list[task_id], self.task_data_config_list[task_id][1], task_id)
                    loss_scale = 1.0
                elif task_name == 'classification':
                    loss = self.train_classification(
                        self.model, sample, criterion_list[task_id], self.task_data_config_list[task_id][1], task_id)
                    loss_scale = 1.0
                elif task_name == 'imputation':
                    loss = self.train_imputation(
                        self.model, sample, criterion_list[task_id], self.task_data_config_list[task_id][1], task_id)
                    loss_scale = 1.0
                elif task_name == 'anomaly_detection':
                    loss = self.train_anomaly_detection(
                        self.model, sample, criterion_list[task_id], self.task_data_config_list[task_id][1], task_id)
                    loss_scale = 1.0

                loss /= acc_it
                loss /= len_sample_list
                if sample_idx < len_sample_list-1:
                    norm_value = scaler(loss*loss_scale, model_optim, clip_grad=max_norm,
                                        parameters=self.model.parameters(), create_graph=False, update_grad=False)
            loss_display = loss.item()*len_sample_list*acc_it
            train_loss_set.append(loss_display)

            norm_value = scaler(loss*loss_scale, model_optim, clip_grad=max_norm,
                                parameters=self.model.parameters(), create_graph=False, update_grad=((i + 1) % acc_it == 0))

            if (i+1) % acc_it == 0:
                model_optim.zero_grad()
                for n, p in self.model.named_parameters():
                    if torch.isnan(p).any() or torch.isinf(p).any():
                        print("BAD PARAM:", n)
                        assert False
                
            torch.cuda.synchronize()

            loss_sum += loss_display
            loss_sum_display = loss_sum

            del sample_init
            del sample_list
            if torch.cuda.memory_reserved(current_device) > 30*1e9:
                torch.cuda.empty_cache()

            if is_main_process():
                wandb.log(
                    {'train_loss_'+self.task_data_config_list[task_id][0]: loss_display, 'norm_value': norm_value, "loss_sum": loss_sum_display/(i+1)})

            # if True:
            if (i + 1) % 20 == 0:
                if norm_value == None:
                    norm_value = -1
                if is_main_process():
                    print("\titers: {0}, epoch: {1} | norm: {2:.2f} | loss: {3:.7f} | current_loss: {4} |current task: {5}".format(
                        i + 1, epoch + 1, norm_value, loss_sum_display/(i+1), loss_display, task_name, folder=self.path))

        print("Epoch: {} cost time: {}".format(
            epoch + 1, time.time() - epoch_time), folder=self.path)
        train_loss = np.average(train_loss_set)
        torch.cuda.synchronize()
        # dist.barrier()

        return train_loss

    def train_long_term_forecast(self, model, this_batch, criterion, config, task_id):
        label_len = config['label_len']
        pred_len = config['pred_len']
        task_name = config['task_name']
        features = config['features']

        batch_x, batch_y, _, _ = this_batch

        batch_x = batch_x.float().to(self.device_id)
        batch_y = batch_y.float().to(self.device_id)

        dec_inp = None
        dec_inp = None
        batch_x_mark = None
        batch_y_mark = None

        with torch.amp.autocast(device_type="cuda", enabled=self.args.enable_mixed_precision_training):
            outputs = model(batch_x, batch_x_mark, dec_inp,
                            batch_y_mark, task_id=task_id, task_name=task_name)
            f_dim = -1 if features == 'MS' else 0
            outputs = outputs[:, -pred_len:, f_dim:]
            batch_y = batch_y[:, -pred_len:, f_dim:]
            loss = criterion(outputs, batch_y)

        return loss

    def train_classification(self, model, this_batch, criterion, config, task_id):
        task_name = config['task_name']

        batch_x, label, padding_mask = this_batch
        # if self.start_training:
        #     print(batch_x)
        # print(label)
        # print(padding_mask)

        batch_x = batch_x.float().to(self.device_id)
        padding_mask = padding_mask.float().to(self.device_id)
        label = label.to(self.device_id)
        assert torch.isfinite(batch_x).all(), "Inputs contain NaN/Inf"
        with torch.amp.autocast(device_type="cuda", enabled=self.args.enable_mixed_precision_training):
            outputs = model(batch_x, padding_mask, None,
                            None, task_id=task_id, task_name=task_name)
            # print(outputs)
            if outputs.shape[0] == label.shape[0]:
                loss = criterion(outputs, label.long().squeeze(-1))
            else:
                label = label.repeat(outputs.shape[0]//label.shape[0], 1)
                loss = criterion(outputs, label.long().squeeze(-1))

        # if self.start_training:
        #     print("classification loss debug:", loss.item())
        #     assert False
        if torch.isnan(loss):
            print("loss is nan!")
            print("outputs:", outputs)
            print("labels:", label)
            print(f'inputs:{batch_x}')
            assert False
        return loss

    def train_imputation(self, model, this_batch, criterion, config, task_id):
        task_name = config['task_name']
        features = config['features']

        batch_x, _, _, _ = this_batch
        batch_x = batch_x.float().to(self.device_id)

        # block-wise imputation
        inp, mask = apply_random_mask_for_imputation(
            batch_x, self.args.patch_len, self.args.mask_rate)

        with torch.amp.autocast(device_type="cuda", enabled=self.args.enable_mixed_precision_training):
            outputs = model(inp, None, None,
                            None, task_id=task_id, mask=mask, task_name=task_name)
        f_dim = -1 if features == 'MS' else 0
        outputs = outputs[:, :, f_dim:]
        loss = criterion(outputs[mask == 0], batch_x[mask == 0])

        return loss

    def train_anomaly_detection(self, model, this_batch, criterion, config, task_id):
        task_name = config['task_name']
        features = config['features']

        batch_x, _ = this_batch

        batch_x = batch_x.float().to(self.device_id)

        with torch.amp.autocast(device_type="cuda", enabled=self.args.enable_mixed_precision_training):
            outputs = model(batch_x, None, None,
                            None, task_id=task_id, task_name=task_name)
            f_dim = -1 if features == 'MS' else 0
            outputs = outputs[:, :, f_dim:]
            loss = criterion(outputs, batch_x)

        return loss
    
    def init_lora(self):
        # replace fc with lora fc
        replace_fc = self.args.lora
        lora_r = self.args.lora_r
        lora_alpha = self.args.lora_alpha
        if replace_fc and not self.is_lora_initiated:
            from models.UniTS import SeqAttBlock, MLPBlock, VarAttBlock
            print("Replace fc layers with LoRA layers...")
            target_modules = set()
            if 'attn' in self.args.lora_target_modules:
                target_modules.add(SeqAttBlock)
                target_modules.add(VarAttBlock)
            if 'mlp' in self.args.lora_target_modules:
                target_modules.add(MLPBlock)
            print("Target modules for LoRA:", target_modules, folder=self.path)
            for name, module in self.model.named_modules():
                if any(isinstance(module, t) for t in target_modules) and 'blocks.' in name:
                    for sub_name, sub_module in module.named_modules():
                        if isinstance(sub_module, nn.Linear):
                            full_name = name + '.' + sub_name
                            self.replace_fc_with_lora(
                                self.model, full_name, r=lora_r, lora_alpha=lora_alpha)
            self.is_lora_initiated = True


    def test(self, setting, load_pretrain=False, test_data_list=None, test_loader_list=None, tflite_path=None):
        self.path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(self.path) and is_main_process():
            os.makedirs(self.path)
        if test_data_list is None or test_loader_list is None:
            test_data_list, test_loader_list = self._get_data(
                flag='test', test_anomaly_detection=True)

        # replace fc with lora fc
        self.init_lora()                    

        if load_pretrain:
            if os.path.exists(self.args.pretrained_weight):
                pretrain_weight_path = self.args.pretrained_weight
                print('loading pretrained model:',
                      pretrain_weight_path, folder=self.path)
                if 'pretrain_checkpoint.pth' in pretrain_weight_path:
                    state_dict = torch.load(
                        pretrain_weight_path, map_location='cpu', weights_only=False)['student']
                    ckpt = {}
                    for k, v in state_dict.items():
                        if not ('cls_prompts' in k):
                            ckpt[k] = v
                else:
                    ckpt = torch.load(pretrain_weight_path, map_location='cpu', weights_only=False)
                msg = self.model.load_state_dict(ckpt, strict=False)
                print(msg)
            else:
                print("no ckpt found!")
                exit()

        # load tflite model if provided
        if tflite_path is not None:
            print(f'Detected tflite model path: {tflite_path}, loading tflite model for classification task evaluation.')
            import ai_edge_torch
            self.tflite_model = ai_edge_torch.load(tflite_path)

            # get quantization details
            input_details = self.tflite_model._interpreter_builder().get_input_details()
            print("Input details:", input_details)

            # loop through input details to find quantization parameters
            self.quantization_params = []
            self.input_dtypes = []
            for detail in input_details:
                print(f"Input tensor '{detail['name']}' quantization parameters: {detail['quantization']}")
                self.quantization_params.append(detail['quantization'])
                self.input_dtypes.append(detail['dtype'])
            print("Quantization parameters for all inputs:", self.quantization_params)
            print("Input data types for all inputs:", self.input_dtypes)

            # get output details
            output_details = self.tflite_model._interpreter_builder().get_output_details()
            print("Output details:", output_details)

            self.output_quantization_params = []
            self.output_dtypes = []
            for detail in output_details:
                print(f"Output tensor '{detail['name']}' quantization parameters: {detail['quantization']}")
                self.output_quantization_params.append(detail['quantization'])
                self.output_dtypes.append(detail['dtype'])
            print("Quantization parameters for all outputs:", self.output_quantization_params)
            print("Output data types for all outputs:", self.output_dtypes)

        total_dict = {}
        avg_classification_acc = []
        avg_long_term_forecast_mse = []
        avg_long_term_forecast_mae = []
        avg_imputation_mse = []
        avg_imputation_mae = []
        avg_anomaly_f_score = []
        for task_id, (test_data, test_loader) in enumerate(zip(test_data_list, test_loader_list)):
            task_name = self.task_data_config_list[task_id][1]['task_name']
            data_task_name = self.task_data_config_list[task_id][0]
            if task_name == 'long_term_forecast':
                if self.args.zero_shot_forecasting_new_length=='unify':
                    mse, mae = self.test_long_term_forecast_offset_unify(
                        setting, test_data, test_loader, data_task_name, task_id)
                else:
                    mse, mae = self.test_long_term_forecast(
                        setting, test_data, test_loader, data_task_name, task_id)
                data_task_name = self.task_data_config_list[task_id][0]
                total_dict[data_task_name] = {'mse': mse, 'mae': mae}
                if is_main_process():
                    wandb.log({'eval_LF-mse_'+data_task_name: mse})
                    wandb.log({'eval_LF-mae_'+data_task_name: mae})
                avg_long_term_forecast_mse.append(mse)
                avg_long_term_forecast_mae.append(mae)
            elif task_name == 'classification':
                if tflite_path is not None:
                    acc = self.test_classification_tflite(
                        setting, test_data, test_loader, data_task_name, task_id)
                else:
                    criterion = self._select_criterion([self.task_data_config_list[task_id]])[0]
                    acc = self.test_classification(
                        setting, test_data, test_loader, data_task_name, task_id, criterion)
                total_dict[data_task_name] = {'acc': acc}
                if is_main_process():
                    wandb.log({'eval_CLS-acc_'+data_task_name: acc})
                avg_classification_acc.append(acc)
            elif task_name == 'imputation':
                mse, mae = self.test_imputation(
                    setting, test_data, test_loader, data_task_name, task_id)
                total_dict[data_task_name] = {'mse': mse, 'mae': mae}
                if is_main_process():
                    wandb.log({'eval_Imputation-mse_'+data_task_name: mse})
                    wandb.log({'eval_Imputation-mae_'+data_task_name: mae})
                avg_imputation_mse.append(mse)
                avg_imputation_mae.append(mae)
            elif task_name == 'anomaly_detection':
                f_score = self.test_anomaly_detection(
                    setting, test_data, test_loader, data_task_name, task_id)
                total_dict[data_task_name] = {'f_score': f_score}
                if is_main_process():
                    wandb.log({'eval_Anomaly-f_score_' +
                              data_task_name: f_score})
                avg_anomaly_f_score.append(f_score)

        avg_long_term_forecast_mse = np.average(avg_long_term_forecast_mse)
        avg_long_term_forecast_mae = np.average(avg_long_term_forecast_mae)

        avg_classification_acc = np.average(avg_classification_acc)

        avg_imputation_mse = np.average(avg_imputation_mse)
        avg_imputation_mae = np.average(avg_imputation_mae)

        avg_anomaly_f_score = np.average(avg_anomaly_f_score)

        if is_main_process():
            wandb.log({'avg_eval_LF-mse': avg_long_term_forecast_mse, 'avg_eval_LF-mae': avg_long_term_forecast_mae,
                       'avg_eval_CLS-acc': avg_classification_acc,
                       'avg_eval_IMP-mse': avg_imputation_mse, 'avg_eval_IMP-mae': avg_imputation_mae,
                       'avg_eval_Anomaly-f_score': avg_anomaly_f_score})
            print("Avg score: LF-mse: {}, LF-mae: {}, CLS-acc {}, IMP-mse: {}, IMP-mae: {}, Ano-F: {}".format(avg_long_term_forecast_mse,
                                                                                                              avg_long_term_forecast_mae, avg_classification_acc, avg_imputation_mse, avg_imputation_mae, avg_anomaly_f_score), folder=self.path)
            print(total_dict, folder=self.path)
        return avg_classification_acc, avg_long_term_forecast_mse, avg_long_term_forecast_mae

    def test_long_term_forecast(self, setting, test_data, test_loader, data_task_name, task_id):
        config = self.task_data_config_list[task_id][1]
        label_len = config['label_len']
        pred_len = config['pred_len']
        features = config['features']

        preds = []
        trues = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, _, _) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device_id)
                batch_y = batch_y.float().to(self.device_id)

                dec_inp = None
                dec_inp = None
                batch_x_mark = None
                batch_y_mark = None

                with torch.amp.autocast(device_type="cuda", enabled=self.args.enable_mixed_precision_training):
                    outputs = self.model(
                        batch_x, batch_x_mark, dec_inp, batch_y_mark, task_id=task_id, task_name='long_term_forecast')

                f_dim = -1 if features == 'MS' else 0
                outputs = outputs[:, -pred_len:, f_dim:]
                batch_y = batch_y[:, -pred_len:, f_dim:]

                outputs = outputs.detach().cpu()
                batch_y = batch_y.detach().cpu()
                if test_data.scale and self.args.inverse:
                    outputs = test_data.inverse_transform(outputs)
                    batch_y = test_data.inverse_transform(batch_y)

                pred = outputs
                true = batch_y

                preds.append(pred)
                trues.append(true)
                del batch_x
                del batch_y

        preds = gather_tensors_from_all_gpus(preds, self.device_id)
        trues = gather_tensors_from_all_gpus(trues, self.device_id)
        preds = np.array(preds)
        trues = np.array(trues)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('data_task_name: {} mse:{}, mae:{}'.format(
            data_task_name, mse, mae), folder=self.path)
        torch.cuda.empty_cache()
        return mse, mae

    def test_classification(self, setting, test_data, test_loader, data_task_name, task_id, criterion):
        preds = []
        trues = []
        losses = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, label, padding_mask) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device_id)
                padding_mask = padding_mask.float().to(self.device_id)
                label = label.to(self.device_id)

                outputs = self.model(
                    batch_x, padding_mask, None, None, task_id=task_id, task_name='classification')
                
                loss = criterion(outputs, label.long().squeeze(-1))
                losses.append(loss.item())
                outputs = torch.nn.functional.softmax(outputs)

                predictions = torch.argmax(outputs, dim=1)
                preds.append(predictions.detach())
                trues.append(label)

        # preds = gather_tensors_from_all_gpus(
        #     preds, self.device_id, to_numpy=False)
        # trues = gather_tensors_from_all_gpus(
        #     trues, self.device_id, to_numpy=False)
        preds = torch.cat(preds, 0)
        trues = torch.cat(trues, 0)

        predictions = preds.cpu().numpy()
        trues = trues.flatten().cpu().numpy()
        accuracy = cal_accuracy(predictions, trues)
        avg_loss = sum(losses) / len(losses) if losses else 0

        print('data_task_name: {} accuracy:{}, avg_loss:{}'.format(
            data_task_name, accuracy, avg_loss), folder=self.path)
        
        print("In-depth classification analysis for task: {}".format(data_task_name), folder=self.path)
        self.in_depth_classification_analysis(trues, predictions)

        del predictions
        del trues
        torch.cuda.empty_cache()

        return accuracy, avg_loss
    
    def in_depth_classification_analysis(self, labels, predictions):
        from sklearn.metrics import (
            classification_report,
            confusion_matrix,
            accuracy_score,
            precision_recall_fscore_support
        )
        all_labels = np.array(labels)
        all_preds = np.array(predictions)
        # Basic accuracy
        acc = accuracy_score(all_labels, all_preds)

        # Per-class metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average=None
        )

        # Confusion matrix
        cm = confusion_matrix(all_labels, all_preds)

        # Text summary
        report = classification_report(all_labels, all_preds)

        # log to folder
        print("Overall Accuracy: {:.4f}".format(acc), folder=self.path)
        print("Per-class Precision: ", precision, folder=self.path)
        print("Per-class Recall: ", recall, folder=self.path)
        print("Per-class F1-score: ", f1, folder=self.path)
        print("Confusion Matrix:\n", cm, folder=self.path)
        print("Classification Report:\n", report, folder=self.path)

    def test_imputation(self, setting, test_data, test_loader, data_task_name, task_id):
        preds = []
        trues = []
        masks = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, _, batch_x_mark, _) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device_id)
                batch_x_mark = batch_x_mark.float().to(self.device_id)

                # block-wise imputation
                inp, mask = apply_random_mask_for_imputation(
                    batch_x, self.args.patch_len, self.args.mask_rate)

                # imputation
                outputs = self.model(
                    inp, batch_x_mark, None, None, task_id=task_id, mask=mask, task_name='imputation')

                # eval
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, :, f_dim:]
                pred = outputs.detach().cpu()
                true = batch_x.detach().cpu()
                preds.append(pred)
                trues.append(true)
                masks.append(mask.detach().cpu())

        preds = gather_tensors_from_all_gpus(preds, self.device_id)
        trues = gather_tensors_from_all_gpus(trues, self.device_id)
        masks = gather_tensors_from_all_gpus(masks, self.device_id)
        preds = np.array(preds)
        trues = np.array(trues)
        masks = np.array(masks)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        masks = masks.reshape(-1, trues.shape[-2], trues.shape[-1])

        mae, mse, rmse, mape, mspe = metric(
            preds[masks == 0], trues[masks == 0])
        print('data_task_name: {} mse:{}, mae:{}'.format(
            data_task_name, mse, mae), folder=self.path)
        torch.cuda.empty_cache()
        return mse, mae

    def test_anomaly_detection(self, setting, test_data, test_loader_set, data_task_name, task_id):
        train_loader, test_loader = test_loader_set
        attens_energy = []
        anomaly_criterion = nn.MSELoss(reduce=False)

        self.model.eval()
        # (1) stastic on the train set
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(train_loader):
                batch_x = batch_x.float().to(self.device_id)
                # reconstruction
                outputs = self.model(
                    batch_x, None, None, None, task_id=task_id, task_name='anomaly_detection')
                # criterion
                score = torch.mean(anomaly_criterion(batch_x, outputs), dim=-1)
                score = score.detach().cpu()
                attens_energy.append(score)

        attens_energy = gather_tensors_from_all_gpus(
            attens_energy, self.device_id, to_numpy=True)
        train_energy = np.concatenate(attens_energy, axis=0).reshape(-1)

        # (2) find the threshold
        attens_energy = []
        test_labels = []
        for i, (batch_x, batch_y) in enumerate(test_loader):
            batch_x = batch_x.float().to(self.device_id)
            # reconstruction
            outputs = self.model(batch_x, None, None, None,
                                 task_id=task_id, task_name='anomaly_detection')
            # criterion
            score = torch.mean(anomaly_criterion(batch_x, outputs), dim=-1)
            score = score.detach().cpu()
            attens_energy.append(score)
            test_labels.append(batch_y)

        attens_energy = gather_tensors_from_all_gpus(
            attens_energy, self.device_id, to_numpy=True)
        test_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        combined_energy = np.concatenate([train_energy, test_energy], axis=0)
        threshold = np.percentile(
            combined_energy, 100 - self.args.anomaly_ratio)
        print("Threshold :", threshold)

        # (3) evaluation on the test set
        pred = (test_energy > threshold).astype(int)
        test_labels = np.concatenate(test_labels, axis=0).reshape(-1)
        test_labels = np.array(test_labels)
        gt = test_labels.astype(int)

        print("pred:   ", pred.shape)
        print("gt:     ", gt.shape)

        # (4) detection adjustment
        gt, pred = adjustment(gt, pred)

        pred = np.array(pred)
        gt = np.array(gt)
        print("pred: ", pred.shape)
        print("gt:   ", gt.shape)
        accuracy = accuracy_score(gt, pred)
        precision, recall, f_score, support = precision_recall_fscore_support(
            gt, pred, average='binary')
        print("Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f} ".format(
            accuracy, precision,
            recall, f_score))

        return f_score

    def test_long_term_forecast_offset_unify(self, setting, test_data, test_loader, data_task_name, task_id):
        config = self.task_data_config_list[task_id][1]
        pred_len = config['pred_len']
        features = config['features']
        max_pred_len = pred_len-self.args.offset+self.args.max_offset

        preds = []
        trues = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, _, _) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device_id)
                batch_y = batch_y.float().to(self.device_id)
                batch_y = batch_y[:,-max_pred_len:][:,:pred_len]

                dec_inp = None
                batch_x_mark = None
                batch_y_mark = None

                with torch.amp.autocast(device_type="cuda", enabled=self.args.enable_mixed_precision_training):
                    outputs = self.model(
                        batch_x, batch_x_mark, dec_inp, batch_y_mark, task_id=task_id, task_name='long_term_forecast')

                f_dim = -1 if features == 'MS' else 0
                outputs = outputs[:, -pred_len:, f_dim:]

                outputs = outputs.detach().cpu()
                batch_y = batch_y.detach().cpu()
                if test_data.scale and self.args.inverse:
                    outputs = test_data.inverse_transform(outputs)
                    batch_y = test_data.inverse_transform(batch_y)

                pred = outputs
                true = batch_y

                preds.append(pred)
                trues.append(true)

        preds = gather_tensors_from_all_gpus(preds, self.device_id)
        trues = gather_tensors_from_all_gpus(trues, self.device_id)
        preds = np.array(preds)
        trues = np.array(trues)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('data_task_name: {} mse:{}, mae:{}'.format(
            data_task_name, mse, mae), folder=self.path)
        torch.cuda.empty_cache()
        return mse, mae

    def split_batch(self, batch, small_batch_size, task_name):
        def split_tensor(tensor, size):
            return [tensor[i:min(i + size, tensor.size(0))] for i in range(0, tensor.size(0), size)]
        if task_name == 'classification':
            batch_x, label, padding_mask = batch
            split_batch_x = split_tensor(batch_x, small_batch_size)
            split_label = split_tensor(label, small_batch_size)
            split_padding_mask = split_tensor(padding_mask, small_batch_size)
            return list(zip(split_batch_x, split_label, split_padding_mask))
        elif task_name == 'long_term_forecast' or task_name == 'imputation':
            batch_x, batch_y, batch_x_mark, batch_y_mark = batch
            split_batch_x = split_tensor(batch_x, small_batch_size)
            split_batch_y = split_tensor(batch_y, small_batch_size)
            split_batch_x_mark = split_tensor(batch_x_mark, small_batch_size)
            split_batch_y_mark = split_tensor(batch_y_mark, small_batch_size)
            return list(zip(split_batch_x, split_batch_y, split_batch_x_mark, split_batch_y_mark))
        elif task_name == 'anomaly_detection':
            batch_x, batch_y = batch
            split_batch_x = split_tensor(batch_x, small_batch_size)
            split_batch_y = split_tensor(batch_y, small_batch_size)
            return list(zip(split_batch_x, split_batch_y))

    def memory_check(self, data_loader_cycle, criterion_list, holdout_memory=3):
        """
        Checks the memory usage of the model by gradually increasing the batch size until it reaches the maximum batch size that can be supported without running out of memory.

        Args:
            data_loader_cycle (DataLoaderCycle): The data loader cycle object.
            holdout_memory (int): The amount of memory (in GB) to hold out for other operations.

        Returns:
            None
        """
        num_elements = holdout_memory * 1024 * 1024 * 1024 // 4
        extra_mem = torch.empty(
            num_elements, dtype=torch.float32, device=self.device_id)

        model_tmp = self._build_model(ddp=False)
        model_tmp.train()
        model_tmp.zero_grad(set_to_none=True)

        for data_loader_id in range(data_loader_cycle.num_dataloaders):
            batch_size = 1  # Initial batch size
            max_batch_size = 0  # Record the maximum batch size before OOM
            torch.cuda.synchronize()
            model_tmp.zero_grad(set_to_none=True)
            while True:
                try:
                    sample, task_id = data_loader_cycle.generate_fake_samples_for_batch(
                        data_loader_id, batch_size)  # 2 makes the memory larger
                    task_name = self.task_data_config_list[task_id][1]['task_name']
                    # Try running the function with the current batch size
                    print(task_id, task_name,
                          sample[0].shape, "max batch size", max_batch_size)
                    if task_name == 'long_term_forecast':
                        loss = self.train_long_term_forecast(
                            model_tmp, sample, criterion_list[task_id], self.task_data_config_list[task_id][1], task_id)
                    elif task_name == 'classification':
                        loss = self.train_classification(
                            model_tmp, sample, criterion_list[task_id], self.task_data_config_list[task_id][1], task_id)
                    elif task_name == 'imputation':
                        loss = self.train_imputation(
                            model_tmp, sample, criterion_list[task_id], self.task_data_config_list[task_id][1], task_id)
                    elif task_name == 'anomaly_detection':
                        loss = self.train_anomaly_detection(
                            model_tmp, sample, criterion_list[task_id], self.task_data_config_list[task_id][1], task_id)
                    loss = loss * 0.0
                    loss.backward()
                    max_batch_size = batch_size  # Update the maximum batch size
                    batch_size *= 2  # Increase the batch size

                    if max_batch_size >= self.args.batch_size:
                        print("Can support default batch size:",
                              self.args.batch_size, max_batch_size)
                        self.task_data_config_list[task_id][1]['max_batch'] = max_batch_size
                        self.task_data_config_list[task_id][1]['checkpointing'] = False
                        break

                except Exception as e:
                    task_name = self.task_data_config_list[task_id][1]['task_name']
                    print(task_id,  "max batch size:", max_batch_size)
                    # If any exception occurs, break the loop
                    self.task_data_config_list[task_id][1]['max_batch'] = max_batch_size
                    del model_tmp
                    model_tmp = self._build_model(ddp=False)
                    print(f"An exception occurred: {e}")
                    break
        print(self.task_data_config_list)
        del model_tmp
        del extra_mem
        torch.cuda.empty_cache()
        return

    def convert_model_to_tflite(self, setting, load_pretrain=False, test_data_list=None, test_loader_list=None,
                                calib_data_path=None, tflite_path=None):
        self.path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(self.path) and is_main_process():
            os.makedirs(self.path)
        if test_data_list is None or test_loader_list is None:
            test_data_list, test_loader_list = self._get_data(
                flag='test', test_anomaly_detection=True)
            
        assert len(test_data_list) == 1
        print(test_data_list)

        assert calib_data_path is not None, "Please provide calibration data path for TFLite conversion."
        assert tflite_path is not None, "Please provide tflite model save path."

        # replace fc with lora fc
        self.init_lora()
        
        # assert False
        if load_pretrain:
            if os.path.exists(self.args.pretrained_weight):
                pretrain_weight_path = self.args.pretrained_weight
                print('loading pretrained model:',
                      pretrain_weight_path, folder=self.path)
                if 'pretrain_checkpoint.pth' in pretrain_weight_path:
                    state_dict = torch.load(
                        pretrain_weight_path, map_location='cpu', weights_only=False)['student']
                    ckpt = {}
                    for k, v in state_dict.items():
                        if not ('cls_prompts' in k):
                            ckpt[k] = v
                else:
                    ckpt = torch.load(pretrain_weight_path, map_location='cpu', weights_only=False)
                msg = self.model.load_state_dict(ckpt, strict=False)
                print(msg)
            else:
                print("no ckpt found!")
                exit()

        total_dict = {}
        avg_classification_acc = []
        avg_long_term_forecast_mse = []
        avg_long_term_forecast_mae = []
        avg_imputation_mse = []
        avg_imputation_mae = []
        avg_anomaly_f_score = []
        for task_id, (test_data, test_loader) in enumerate(zip(test_data_list, test_loader_list)):
            task_name = self.task_data_config_list[task_id][1]['task_name']
            data_task_name = self.task_data_config_list[task_id][0]
            assert task_name == 'classification'

            self.model.set_dataset(self.task_data_config_list[task_id][1]['dataset'])
            self.model.enable_tracing_mode()

            def trace_fp_model(model, x, tflite_path):
                import ai_edge_torch

                print(f'Sample input shape: {x.shape}')
                edge_model = ai_edge_torch.convert(model.eval(), (x,))

                edge_model.export(tflite_path)

            def trace_fp16_model(model, x, tflite_path):
                import ai_edge_torch
                import tensorflow as tf
                tfl_converter_flags={
                    "optimizations": [tf.lite.Optimize.DEFAULT],
                    "target_spec": {"supported_ops": [tf.float16]},
                }

                print(f'Sample input shape: {x.shape}')
                edge_model = ai_edge_torch.convert(model.eval(), (x,), _ai_edge_converter_flags=tfl_converter_flags)

                edge_model.export(tflite_path)

            def trace_dynamic_quantized_model(model, x, tflite_path):
                import ai_edge_torch
                import tensorflow as tf
                tfl_converter_flags={
                    "optimizations": [tf.lite.Optimize.DEFAULT],
                }

                edge_model = ai_edge_torch.convert(model.eval(), (x,), _ai_edge_converter_flags=tfl_converter_flags)

                edge_model.export(tflite_path)

            def trace_quantized_model(model, x, tflite_path, calib_data):
                import ai_edge_torch
                import tensorflow as tf
                def representative_dataset():
                    for i in range(calib_data['x'].shape[0]):
                        data = calib_data['x'][i:i+1, ...]
                        yield [data.astype(np.float32)]

                tfl_converter_flags={
                    "optimizations": [tf.lite.Optimize.DEFAULT],
                    "target_spec.supported_ops": [tf.lite.OpsSet.TFLITE_BUILTINS_INT8],
                    "inference_input_type": tf.int8,
                    "inference_output_type": tf.int8,
                    "representative_dataset": representative_dataset
                }

                edge_model = ai_edge_torch.convert(model.eval(), (x,), _ai_edge_converter_flags=tfl_converter_flags)

                edge_model.export(tflite_path)

            # read data from calib_data_path npz
            calib_data = np.load(calib_data_path)
            sample_inputs = calib_data['x']
            sample_inputs = torch.from_numpy(sample_inputs).float().to(self.device_id)

            # check pytorch model device
            # Assuming your model is named 'model'
            if self.args.convert_to_tflite_dtype == 'int8':
                print("Converting to INT8 quantized TFLite model...")
                trace_quantized_model(self.model.to('cpu'), sample_inputs[:1, ...].to('cpu'), tflite_path, calib_data)
            elif self.args.convert_to_tflite_dtype == 'dq':
                print("Converting to DQ quantized TFLite model...")
                trace_dynamic_quantized_model(self.model.to('cpu'), sample_inputs[:1, ...].to('cpu'), tflite_path)
            elif self.args.convert_to_tflite_dtype == 'fp32':
                print("Converting to FLOAT32 TFLite model...")
                trace_fp_model(self.model.to('cpu'), sample_inputs[:1, ...].to('cpu'), tflite_path)
            elif self.args.convert_to_tflite_dtype == 'fp16':
                print("Converting to FLOAT16 TFLite model...")
                trace_fp16_model(self.model.to('cpu'), sample_inputs[:1, ...].to('cpu'), tflite_path)
            else:
                raise ValueError("Unsupported TFLite data type. Supported types are: int8, dq, fp32.")           

        self.model.disable_tracing_mode()

    def save_calib_data(self, setting, load_pretrain=False, test_data_list=None, test_loader_list=None,
                                calib_data_path=None):
        self.path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(self.path) and is_main_process():
            os.makedirs(self.path)
        if test_data_list is None or test_loader_list is None:
            # to use train data for calibration, and for shuffle
            test_data_list, test_loader_list = self._get_data(
                flag='train', test_anomaly_detection=True)
            
        assert len(test_data_list) == 1
        print(test_data_list)

        assert calib_data_path is not None, "Please provide calibration data path for TFLite conversion."

        self.init_lora()

        if load_pretrain:
            if os.path.exists(self.args.pretrained_weight):
                pretrain_weight_path = self.args.pretrained_weight
                print('loading pretrained model:',
                      pretrain_weight_path, folder=self.path)
                if 'pretrain_checkpoint.pth' in pretrain_weight_path:
                    state_dict = torch.load(
                        pretrain_weight_path, map_location='cpu', weights_only=False)['student']
                    ckpt = {}
                    for k, v in state_dict.items():
                        if not ('cls_prompts' in k):
                            ckpt[k] = v
                else:
                    ckpt = torch.load(pretrain_weight_path, map_location='cpu', weights_only=False)
                msg = self.model.load_state_dict(ckpt, strict=False)
                print(msg)
            else:
                print("no ckpt found!")
                exit()

        total_dict = {}
        avg_classification_acc = []
        avg_long_term_forecast_mse = []
        avg_long_term_forecast_mae = []
        avg_imputation_mse = []
        avg_imputation_mae = []
        avg_anomaly_f_score = []
        for task_id, (test_data, test_loader) in enumerate(zip(test_data_list, test_loader_list)):
            task_name = self.task_data_config_list[task_id][1]['task_name']
            data_task_name = self.task_data_config_list[task_id][0]
            assert task_name == 'classification'

            self.model.save_calibration_npz(
                test_loader, task_id, save_path=calib_data_path
            )

    def test_classification_tflite(self, setting, test_data, test_loader, data_task_name, task_id):
        preds = []
        trues = []
        self.model.eval()
        self.model.to('cpu')
        print(f'Check self.model device: {next(self.model.parameters()).device}')
        with torch.no_grad():
            for i, (batch_x, label, padding_mask) in enumerate(test_loader):
                batch_x = batch_x.float().to('cpu')
                padding_mask = padding_mask.float().to('cpu')
                label = label.to('cpu')

                # run preprocess using pytorch model
                outputs = self.model.preprocess_classification(
                    batch_x, None,task_id=task_id)
                
                # run inference using tflite model
                tflite_input = outputs['x'].detach().cpu().numpy()

                assert tflite_input.dtype == np.float32, "TFLite model input dtype must be float32"

                input_tensors = [tflite_input]

                # if quantized model, prepare quantized input
                for i, (q_param, dtype) in enumerate(zip(self.quantization_params, self.input_dtypes)):
                    if dtype == np.float32 or dtype == np.float16:
                        # print("Model is not quantized.")
                        input_tensors[i] = input_tensors[i].astype(dtype)
                    else:
                        scale, zero_point = q_param
                        # print(f"input is quantized with scale: {scale}, zero_point: {zero_point}")

                        input_tensors[i] = (input_tensors[i] / scale + zero_point).astype(dtype)

                # print(f'TFLite input shape: {tflite_input.shape}')
                outputs = self.tflite_model(*input_tensors)
                outputs_tensors = [outputs] if not isinstance(outputs, (list, tuple)) else outputs

                assert len(outputs_tensors) == 1, "TFLite model should have only one output for classification task."
                assert len(self.output_quantization_params) == 1, "Output quantization params should have only one entry for classification task."
                # dequantize output if needed
                outputs = outputs_tensors[0]
                for i, (q_param, dtype) in enumerate(zip(self.output_quantization_params, self.output_dtypes)):
                    if dtype == np.float32 or dtype == np.float16:
                        # print("Model output is not quantized.")
                        outputs = outputs.astype(np.float32)
                    else:
                        scale, zero_point = q_param
                        # print(f"output is quantized with scale: {scale}, zero_point: {zero_point}")

                        outputs = (outputs.astype(np.float32) - zero_point) * scale
                
                outputs = torch.from_numpy(outputs).to('cpu')
                outputs = torch.nn.functional.softmax(outputs)

                predictions = torch.argmax(outputs, dim=1)
                preds.append(predictions.detach())
                trues.append(label)

        # preds = gather_tensors_from_all_gpus(
        #     preds, self.device_id, to_numpy=False)
        # trues = gather_tensors_from_all_gpus(
        #     trues, self.device_id, to_numpy=False)
        preds = torch.cat(preds, 0)
        trues = torch.cat(trues, 0)

        predictions = preds.cpu().numpy()
        trues = trues.flatten().cpu().numpy()
        accuracy = cal_accuracy(predictions, trues)

        print('data_task_name: {} accuracy:{}'.format(
            data_task_name, accuracy), folder=self.path)
        
        print("In-depth classification analysis for task: {}".format(data_task_name), folder=self.path)
        self.in_depth_classification_analysis(trues, predictions)

        del predictions
        del trues
        torch.cuda.empty_cache()

        return accuracy
    
    def profile_training_step(self, model_optim, data_loader_cycle, criterion_list, epoch, train_steps, scaler):
        """
        Measures peak memory usage and training throughput (samples/sec).
        """
        
        # 1. Warm-up
        # We run a few steps to initialize CUDA kernels and memory allocators
        print("Starting warm-up...")

        self.train_one_epoch(model_optim, data_loader_cycle, criterion_list, epoch, train_steps, scaler)

        # 2. Reset Memory Stats
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        # 3. Benchmark Loop
        print("Starting benchmark...")
        # print(f'len(data_loader_cycle): {data_loader_cycle.calc_total_samples()}')
        start_time = time.perf_counter()
        
        # Track a fixed number of batches for consistent measurement
        dataloader_loop_count = 2
        total_samples = data_loader_cycle.calc_total_samples() * dataloader_loop_count
        
        while True:
            self.train_one_epoch(model_optim, data_loader_cycle, criterion_list, epoch, train_steps, scaler)

            dataloader_loop_count -= 1
            if dataloader_loop_count <= 0:
                break

        # Synchronize to ensure all GPU work is finished before stopping the clock
        torch.cuda.synchronize()
        end_time = time.perf_counter()

        # 4. Calculations
        duration = end_time - start_time
        throughput = total_samples / duration
        peak_mem_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)

        print("-" * 30, folder=self.path)
        print(f"Peak Memory: {peak_mem_gb:.3f} GB", folder=self.path)
        print(f"Throughput:  {throughput:.2f} samples/sec", folder=self.path)
        print("-" * 30, folder=self.path)
        
        return {"peak_memory_gb": peak_mem_gb, "throughput": throughput}