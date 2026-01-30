import csv
import os
import math
import random

import yaml
import torch
import wandb
from dotenv import load_dotenv
from types import SimpleNamespace
from codecarbon import EmissionsTracker
from torch.utils.data import DataLoader
import torchvision.transforms.v2 as v2
import torchvision.transforms as T

from src.data.datasets.mvtec import ImageLabelDataset, RandomNoiseAugmentation

load_dotenv()

WANDB_KEY = os.getenv("WANDB_KEY")

def write_results_to_csv(config_name, hw_config, class_name, samples_per_defect, train_keep_ratio, results, train_dataset_size, csv_path="results.csv"):
    """
    Write model results to a CSV file.
    
    Args:
        config_name (str): Name of the model configuration
        class_name (str): Name of the class being evaluated
        samples_per_defect (int): Number of samples per defect
        train_keep_ratio (float): Ratio of training data to keep
        results (tuple): Tuple containing (mean_dict, std_dict) from mean_and_std_of_dicts
        csv_path (str): Path to the CSV file to write results
    """
    # Get mean and std dictionaries
    mean_dict, std_dict = results[class_name]
    
    # Prepare row data
    row_data = {
        'config_name': config_name,
        'hw_config': hw_config,
        'class_name': class_name,
        'samples_per_defect': samples_per_defect,
        'train_keep_ratio': train_keep_ratio,
        'train_dataset_size': train_dataset_size
    }
    
    # Add each metric with its mean and std
    for metric in mean_dict.keys():
        row_data[f'{metric}_mean'] = mean_dict[metric]
        row_data[f'{metric}_std'] = std_dict[metric]
    
    # Check if file exists
    file_exists = os.path.isfile(csv_path)
    
    # If file exists, check if this config-class combination already exists
    if file_exists:
        with open(csv_path, 'r', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if (row['config_name'] == config_name and 
                    row['class_name'] == class_name and 
                    float(row['samples_per_defect']) == samples_per_defect and 
                    float(row['train_keep_ratio']) == train_keep_ratio and
                    row['hw_config'] == hw_config and
                    int(row['train_dataset_size']) == train_dataset_size):
                    print(f"Results for config '{config_name}', class '{class_name}', samples_per_defect {samples_per_defect}, and train_keep_ratio {train_keep_ratio}, and hw_config {hw_config} already exist. Skipping...")
                    return
    
    # Write to CSV
    with open(csv_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=row_data.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row_data)

def get_hardware_config(hw_config_path):
    tracker = EmissionsTracker()
    tracker = tracker._conf
    env_cfg = SimpleNamespace(**tracker)

    if hasattr(env_cfg, "gpu_model"):
        config_data = {
            "cpu_model": env_cfg.cpu_model,
            "cpu_count": env_cfg.cpu_count,
            "gpu_model": env_cfg.gpu_model,
            "gpu_count": env_cfg.gpu_count,
            "ram_total_size_gb": env_cfg.ram_total_size,
            "os": env_cfg.os,
            "python_version": env_cfg.python_version
        }
        config_name = f"{env_cfg.cpu_model}_{env_cfg.gpu_model}_{env_cfg.ram_total_size}GB"
    else:
        config_data = {
            "cpu_model": env_cfg.cpu_model,
            "cpu_count": env_cfg.cpu_count,
            "ram_total_size_gb": env_cfg.ram_total_size,
            "os": env_cfg.os,
            "python_version": env_cfg.python_version
        }
        config_name = f"{env_cfg.cpu_model}_{env_cfg.ram_total_size}GB"

    config_file = f"{config_name}.yaml"
    config_path = os.path.join(hw_config_path, config_file)

    if not os.path.exists(config_path):    
        with open(config_path, "w") as f:
            yaml.dump(config_data, f, default_flow_style=False)
        print(f"Config created at: {config_path}")
    else:
        print(f"Config already exists at: {config_path}")
    
    return tracker, config_name

def create_datasets(fold, model_transform, training_cfg):
    train_transform = v2.Compose([
        training_cfg.augmentation_transform,
        model_transform
    ])
    noisy_transform = RandomNoiseAugmentation(model_transform)

    train_dataset = ImageLabelDataset(fold[0], transform=train_transform)
    val_dataset = ImageLabelDataset(fold[1], transform=model_transform, return_original_image_with_transform=True)
    noisy_val_dataset = ImageLabelDataset(fold[1], transform=noisy_transform, return_original_image_with_transform=True)
    return train_dataset, val_dataset, noisy_val_dataset

def create_loaders(train_dataset, val_dataset, noisy_val_dataset, cfg):
    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False)
    noisy_val_loader = DataLoader(noisy_val_dataset, batch_size=cfg.batch_size, shuffle=False)
    return train_loader, val_loader, noisy_val_loader

from collections import defaultdict

def mean_and_std_of_dicts(dict_list):
    sums = defaultdict(float)
    counts = defaultdict(int)
    squared_diffs = defaultdict(float)

    # First pass: compute means
    for d in dict_list:
        for key, value in d.items():
            if isinstance(value, torch.Tensor):
                value = value.item()
            sums[key] += value
            counts[key] += 1

    means = {key: sums[key] / counts[key] for key in sums}

    # Second pass: compute squared differences from the mean
    for d in dict_list:
        for key, value in d.items():
            squared_diffs[key] += (value - means[key]) ** 2

    stds = {
        key: math.sqrt(squared_diffs[key] / counts[key])
        for key in squared_diffs
    }

    return means, stds

def prettify_mean_dict(results, decimals=4):
    
    # Round values
    rounded_mean = {k: round(v, decimals) for k, v in results[0].items()}
    rounded_std = {k: round(v, decimals) for k, v in results[1].items()}
    
    # Build pretty string
    lines = [f"{k:25s}: {v:.{decimals}f}, {rounded_std[k]:.{decimals}f}" for k, v in sorted(rounded_mean.items())]
    return "\n".join(lines)

def init_wandb(train_class, samples_per_defect, train_keep_ratio, training_cfg, model_cfg):
    run = None
    if training_cfg.use_wandb:
        wandb.login(key=WANDB_KEY)
        
        run = wandb.init(
            project=training_cfg.wandb_project,
            group=f"{model_cfg['backbone']['backbone_name']}_{model_cfg['model']['params']['head']}_{model_cfg['backbone']['params']['resolution']}@{train_class}",
            config=training_cfg.__dict__() | model_cfg | {"train_class": train_class, "samples_per_defect": samples_per_defect, "train_keep_ratio": train_keep_ratio}
        )
        model_artifact = wandb.Artifact(model_cfg["model"]["model_name"], 
                                type="model",
                                description="Base training",
                                metadata=training_cfg.__dict__() | model_cfg | {"train_class": train_class, "samples_per_defect": samples_per_defect, "train_keep_ratio": train_keep_ratio}
                                )
    return run, model_artifact