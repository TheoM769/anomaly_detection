#!/usr/bin/env python3
"""
Memory Bank Inference Script

This script provides inference capabilities for memory bank models.
Inspired by run_memory_bank.py but adapted for the new memory bank architecture.
"""

import argparse
import os
import yaml
import json
import time
import csv
from pathlib import Path
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import precision_recall_curve, auc, average_precision_score, roc_auc_score, roc_curve

# Import local packages
from src.models.adapters.models.memory_bank import MemoryBankModel
from src.models.adapters.vision_backbones import *
from src.models.core.services import MODEL_REGISTRY
from src.benchmark.memory_bank.utils import *


def parse_args():
    parser = argparse.ArgumentParser(description="Memory Bank Model Inference")
    
    # Model configuration
    parser.add_argument("--model_config", type=str, default="anomaly_dino",
                        help="Name of the model config file (without .yaml extension)")
    
    parser.add_argument("--model_config_path", type=str, default="/Users/theo.moreau/Documents/futur/src/benchmark/memory_bank/configs/models/",
                        help="Path to model configuration directory")
    
    parser.add_argument("--dataset", type=str, default="mvtec_ad",
                        help="Dataset name")
    
    parser.add_argument("--data_root", type=str, default="/Users/theo.moreau/Documents/futur/datasets/mvtec_anomaly_detection/",
                        help="Path to dataset root directory (overrides config if provided)")
    
    parser.add_argument("--object_names", nargs='+', type=str, default=None,
                        help="List of object names to evaluate (overrides config if provided)")
    
    parser.add_argument("--device", type=str, default="mps",
                        help="Device to use for model inference (e.g., 'cuda:0', 'cpu', 'mps')")
    
    parser.add_argument("--shots", type=int, default=1,
                        help="Number of shots to use for inference")
    
    parser.add_argument("--informed_preprocessing", action="store_true",
                        help="Use informed preprocessing for patch feature extraction (overrides config if provided)")
    
    parser.add_argument("--use_patch_features", action="store_true",
                        help="Use patch features for anomaly detection (overrides config if provided)")
    
    parser.add_argument("--processing_config_path", type=str, default=None,
                        help="Path to processing configuration directory (overrides config if provided)")
    
    parser.add_argument("--seeds", nargs='+', type=int, default=[157],
                        help="List of seeds for multiple runs")
    
    parser.add_argument("--mask_ref_images", action="store_true",
                        help="Mask reference images for feature extraction (overrides config if provided)")
    
    # Inference configuration
    parser.add_argument("--knn_metric", type=str, default="L2_normalized",
                        choices=["L2", "L2_normalized"],
                        help="Distance metric for k-NN search")
    parser.add_argument("--knn_neighbors", type=int, default=1,
                        help="Number of neighbors for k-NN search")
    parser.add_argument("--scoring_type", type=str, default="mean_top1p",
                        choices=["mean_top1p_adjusted", "mean_top1p", "max", "max_adjusted"],
                        help="Scoring method for anomaly detection")
    parser.add_argument("--faiss_on_cpu", action="store_true",
                        help="Force FAISS k-NN search to use CPU (overrides automatic device detection)")
    
    # Output configuration
    parser.add_argument("--output_dir", type=str, default="./results/",
                        help="Output directory for results")
    parser.add_argument("--model_name", type=str, default=None,
                        help="Model name for folder structure (extracted from config if not provided)")
    parser.add_argument("--resolution", type=int, default=None,
                        help="Resolution for folder structure (extracted from config if not provided)")
    parser.add_argument("--save_anomaly_maps", action="store_true",
                        help="Save anomaly maps as images")
    parser.add_argument("--save_tsne_plots", action="store_true",
                        help="Save t-SNE visualizations")
    
    # Metrics configuration
    parser.add_argument("--metrics", nargs='+', type=str, 
                        default=["f1", "precision", "recall", "AP", "f1_max", "precision@0.95", "precision@0.90"],
                        help="List of metrics to compute")
    
    # Cost weights for optimal threshold computation (simpler format)
    parser.add_argument("--compute_optimal_threshold", action="store_true",
                        help="Compute optimal threshold based on cost weights")
    parser.add_argument("--cost_tp", type=float, default=0.0,
                        help="Cost/reward for True Positives (correctly detected anomalies)")
    parser.add_argument("--cost_tn", type=float, default=0.02,
                        help="Cost/reward for True Negatives (correctly identified normal samples)")
    parser.add_argument("--cost_fp", type=float, default=-0.02,
                        help="Cost/penalty for False Positives (false alarms)")
    parser.add_argument("--cost_fn", type=float, default=-0.20,
                        help="Cost/penalty for False Negatives (missed anomalies)")
    
    # Evaluation configuration
    parser.add_argument("--warmup_iters", type=int, default=10,
                        help="Number of warmup iterations for timing")
    parser.add_argument("--max_samples_per_object", type=int, default=None,
                        help="Maximum number of test samples per object (for faster evaluation)")
    
    # Visualization
    parser.add_argument("--plot_examples", action="store_true",
                        help="Plot example anomaly detections")
    parser.add_argument("--num_examples", type=int, default=5,
                        help="Number of examples to plot per object")
    
    args = parser.parse_args()
    return args


def override_model_config(model_cfg, args):
    """Override model configuration with command line arguments."""
    if "model" not in model_cfg or "params" not in model_cfg["model"]:
        raise ValueError("Invalid model configuration: missing 'model.params' section")
    
    model_params = model_cfg["model"]["params"]
    
    # Override model parameters with command line arguments
    if hasattr(args, 'dataset') and args.dataset:
        model_params["dataset_name"] = args.dataset
        print(f"Overriding dataset_name: {args.dataset}")
    
    if hasattr(args, 'data_root') and args.data_root:
        model_params["data_root"] = args.data_root
        print(f"Overriding data_root: {args.data_root}")
    
    if hasattr(args, 'object_names') and args.object_names:
        model_params["object_names"] = args.object_names
        print(f"Overriding object_names: {args.object_names}")
    
    if hasattr(args, 'device') and args.device:
        model_params["device"] = args.device
        print(f"Overriding device: {args.device}")
    
    if hasattr(args, 'shots') and args.shots is not None:
        model_params["shots"] = args.shots
        print(f"Overriding shots: {args.shots}")
    
    if hasattr(args, 'informed_preprocessing') and args.informed_preprocessing is not None:
        model_params["informed_preprocessing"] = args.informed_preprocessing
        print(f"Overriding informed_preprocessing: {args.informed_preprocessing}")
    
    if hasattr(args, 'use_patch_features') and args.use_patch_features is not None:
        model_params["use_patch_features"] = args.use_patch_features
        print(f"Overriding use_patch_features: {args.use_patch_features}")
    
    if hasattr(args, 'processing_config_path') and args.processing_config_path:
        model_params["processing_config_path"] = args.processing_config_path
        print(f"Overriding processing_config_path: {args.processing_config_path}")
    
    if hasattr(args, 'seed') and args.seed:
        model_params["seed"] = args.seed
        print(f"Overriding seeds: {args.seed}")

    if hasattr(args, 'mask_ref_images') and args.mask_ref_images is not None:
        model_params["mask_ref_images"] = args.mask_ref_images
        print(f"Overriding mask_ref_images: {args.mask_ref_images}")
    
    return model_cfg


def load_model_with_overrides(model_config_path, model_config_name, args):
    """Load memory bank model from configuration with command line argument overrides."""
    config_file = os.path.join(model_config_path, f"{model_config_name}.yaml")
    
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Model config file not found: {config_file}")
    
    with open(config_file, "r") as f:
        model_cfg = yaml.safe_load(f)
    
    # Override model configuration with command line arguments
    model_cfg = override_model_config(model_cfg, args)
    
    print(f"Loading model from config: {config_file}")
    print("Applied command line overrides to model configuration")
    model = MODEL_REGISTRY.build_model(model_cfg)
    
    return model, model_cfg


def run_warmup(model, warmup_iters=10, device="cpu"):
    """Run warmup iterations for timing accuracy."""
    print(f"Running {warmup_iters} warmup iterations...")
    
    # Get a sample image from the first object's test set
    object_name = model.object_names[0]
    sample_data = next(iter(model.test_dataset[object_name]))
    sample_image = sample_data[0]
    
    # Build kNN index (use CPU for warmup to avoid GPU memory issues)
    knn_index = build_knn_index(model.memory_bank, faiss_on_cpu=True)
    
    for _ in tqdm(range(warmup_iters), desc="Warmup"):
        _ = model.predict(sample_image, knn_index)


def compute_metrics_for_object(ground_truth, anomaly_scores, cost_weights=None):
    """Compute evaluation metrics for a single object."""
    # Convert to torch tensors and ensure correct types
    scores_tensor = torch.tensor(anomaly_scores, dtype=torch.float32)
    labels_tensor = torch.tensor(ground_truth, dtype=torch.float32).round().long()
    
    # Initialize results dictionary
    results = {}
    
    ap = average_precision_score(labels_tensor, scores_tensor)
    # Take the AP score for the positive class (anomaly class)
    results['AP'] = ap

    auroc = roc_auc_score(labels_tensor, scores_tensor)
    results['AUROC'] = auroc
    
    # Compute optimal cost if weights provided
    if cost_weights:
        # For cost computation, use score directly
        # Try 100 different thresholds between min and max score
        thresholds = torch.linspace(scores_tensor.min().item(), scores_tensor.max().item(), 100)
        max_gain = float('-inf')
        optimal_threshold = None
        all_gains = []
        
        for threshold in thresholds:
            predictions = (scores_tensor >= threshold).long()
            tp = torch.sum((predictions == 1) & (labels_tensor == 1)).item()
            tn = torch.sum((predictions == 0) & (labels_tensor == 0)).item()
            fp = torch.sum((predictions == 1) & (labels_tensor == 0)).item()
            fn = torch.sum((predictions == 0) & (labels_tensor == 1)).item()
            
            total_samples = len(labels_tensor)
            total_gain = (cost_weights['tp'] * tp + 
                         cost_weights['tn'] * tn + 
                         cost_weights['fp'] * fp + 
                         cost_weights['fn'] * fn)
            expected_gain = total_gain / total_samples
            all_gains.append(expected_gain)
            
            if expected_gain > max_gain:
                max_gain = expected_gain
                optimal_threshold = threshold.item()
        
        results.update({
            'optimal_threshold': optimal_threshold,
            'expected_gain': max_gain,
            'all_thresholds': thresholds.tolist(),
            'all_gains': all_gains
        })
    
    return results


def make_serializable(obj):
    """Convert numpy arrays and other non-serializable objects to Python native types."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
        np.int16, np.int32, np.int64, np.uint8,
        np.uint16, np.uint32, np.uint64)):
        return int(obj)
    elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [make_serializable(x) for x in obj]
    return obj


def main():
    """Main function for memory bank inference."""
    args = parse_args()
    
    print("=== Memory Bank Inference ===")
    print(f"Model config: {args.model_config}")
    print(f"Dataset: {args.dataset}")
    print(f"Device: {args.device}")
    print(f"Shots: {args.shots}")
    print(f"Seeds: {args.seeds}")
    print("="*50)
    
    # Load model configuration first to get model_name and resolution if not provided
    temp_model, temp_model_cfg = load_model_with_overrides(
        args.model_config_path, 
        args.model_config, 
        args
    )
    
    # Extract model information from config if not provided via args
    model_name = args.model_name or temp_model_cfg.get("backbone", {}).get("params", {}).get("model_name", "unknown")
    resolution = args.resolution or temp_model_cfg.get("backbone", {}).get("params", {}).get("resolution", "unknown")
    shots = args.shots
    
    print(f"Model: {model_name}, Resolution: {resolution}, Shots: {shots}")
    
    # Define cost weights if using optimal cost computation
    cost_weights = {
        'tp': args.cost_tp,
        'tn': args.cost_tn,
        'fp': args.cost_fp,
        'fn': args.cost_fn
    } if args.compute_optimal_threshold else None
    
    # Store results for all runs
    all_runs_results = {}
    all_inference_times = []  # Track all inference times for mean calculation
    
    # Run multiple seeds
    for seed in args.seeds:
        print(f"\n=== Running with seed {seed} ===")
        
        # Load model with command line argument overrides and current seed
        args.seed = seed  # Temporarily set single seed for model loading
        model, model_cfg = temp_model, temp_model_cfg  # Use already loaded model for first seed
        if seed != args.seeds[0]:  # Only reload for subsequent seeds
            model, model_cfg = load_model_with_overrides(
                args.model_config_path, 
                args.model_config, 
                args
            )
        
        print(f"Loaded model for objects: {model.object_names}")
        print(f'Seed: {seed}')
        print(f"Memory bank size: {model.memory_bank.shape}")
        print(f"Device: {model.device}")
        
        # Run warmup if specified
        if args.warmup_iters > 0:
            run_warmup(model, args.warmup_iters, args.device)
        
        # Build kNN index for inference
        print("Building k-NN index...")
        knn_index = build_knn_index(
            model.memory_bank, 
            faiss_on_cpu=args.faiss_on_cpu,
            knn_metric=args.knn_metric
        )
        
        # Run inference for each object
        for object_name in model.object_names:
            if object_name not in all_runs_results:
                all_runs_results[object_name] = {
                    'AP_values': [],
                    'AUROC_values': [],
                    'expected_gain_values': [] if cost_weights else None,
                    'optimal_threshold_values': [] if cost_weights else None
                }
            
            print(f"\nProcessing object: {object_name}")
            
            # Create single output directory with all parameters
            masking_type = model.preprocessing_configs[object_name]["masking_type"] if model.preprocessing_configs else None
            folder_name = f"{model_name}_{resolution}_shots={shots}_{masking_type}_{object_name}"
            object_output_dir = os.path.join(args.output_dir, folder_name)
            os.makedirs(object_output_dir, exist_ok=True)
            print(f"Output directory: {object_output_dir}")
            
            test_dataset = model.test_dataset[object_name]
            if args.max_samples_per_object:
                # Limit number of samples for faster evaluation
                test_dataset = list(test_dataset)[:args.max_samples_per_object]
            
            object_results = {
                'ground_truth': [],
                'anomaly_scores': [],
                'inference_times': [],
                'true_masks': [],
                'anomaly_maps': [],  # Store anomaly maps for consistent color scaling
                'images': []  # Store images for plotting
            }
            
            # First pass: Process all test samples and collect anomaly maps
            for i, (image, mask, label) in enumerate(tqdm(test_dataset, desc=f"Processing {object_name}")):
                # Run inference
                distances, _, inf_time, anomaly_score, anomaly_map = model.predict(
                    image, 
                    knn_index,
                    knn_metric=args.knn_metric,
                    knn_neighbors=args.knn_neighbors,
                    scoring_type=args.scoring_type,
                    masking_type=masking_type,
                    object_name=object_name
                )

                #print(anomaly_score, label)
                
                # Store results
                object_results['ground_truth'].append(label)
                object_results['anomaly_scores'].append(anomaly_score)
                object_results['inference_times'].append(inf_time)
                object_results['true_masks'].append(mask)
                object_results['anomaly_maps'].append(anomaly_map)
                object_results['images'].append(image)
                all_inference_times.append(inf_time)  # Track for overall mean
            
            # Calculate global min and max for consistent color scaling
            if args.save_anomaly_maps and object_results['anomaly_maps']:
                all_maps = np.array(object_results['anomaly_maps'])
                global_vmin = 0  # Fixed minimum at 0
                global_vmax = np.max(all_maps)  # Maximum across all images
                print(f"Color scale for {object_name}: vmin=0, vmax={global_vmax:.4f}")
                
                # Second pass: Save anomaly maps with consistent color scaling
                os.makedirs(os.path.join(object_output_dir, "anomaly_maps"), exist_ok=True)
                
                # Select all defect images + 5 random good images
                defect_indices = [i for i, label in enumerate(object_results['ground_truth']) if label == 1]
                good_indices = [i for i, label in enumerate(object_results['ground_truth']) if label == 0]
                
                # Select 5 random good images (or all if less than 5)
                num_good_to_select = min(5, len(good_indices))
                selected_good_indices = np.random.choice(good_indices, num_good_to_select, replace=False) if num_good_to_select > 0 else []
                
                # Combine all defect indices with selected good indices
                indices_to_plot = defect_indices + list(selected_good_indices)
                
                print(f"Plotting {len(defect_indices)} defect images and {len(selected_good_indices)} good images")
                
                for idx in indices_to_plot:
                    image = object_results['images'][idx]
                    anomaly_map = object_results['anomaly_maps'][idx] 
                    label = object_results['ground_truth'][idx]
                    anomaly_score = object_results['anomaly_scores'][idx]
                    
                    map_save_path = os.path.join(object_output_dir, "anomaly_maps", f"{object_name}_seed{seed}_anomaly_map_{idx:04d}.png")
                    
                    # Create a figure with three subplots
                    fig = plt.figure(figsize=(18, 6))
                    gs = plt.GridSpec(1, 3, figure=fig)
                    ax1 = fig.add_subplot(gs[0, 0])
                    ax2 = fig.add_subplot(gs[0, 1])
                    ax3 = fig.add_subplot(gs[0, 2])
                    
                    # Plot original image
                    ax1.imshow(image)
                    ax1.set_title('Good Image' if label == 0 else 'Defect Image')
                    ax1.axis('off')
                    
                    # Plot original image with anomaly map overlay (fixed color scale)
                    ax2.imshow(image)
                    im = ax2.imshow(anomaly_map, cmap='hot', alpha=0.5, vmin=global_vmin, vmax=global_vmax)
                    ax2.set_title(f'Anomaly Map (Score: {anomaly_score:.3f})')
                    ax2.axis('off')
                    
                    # Add colorbar to anomaly map
                    plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
                    
                    # Plot histogram of distances with global scale reference
                    flattened_distances = anomaly_map.flatten()
                    ax3.hist(flattened_distances, bins=50, color='blue', alpha=0.7, range=(global_vmin, global_vmax))
                    ax3.axvline(x=anomaly_score, color='red', linestyle='--', label=f'Score: {anomaly_score:.3f}')
                    ax3.axvline(x=global_vmax, color='orange', linestyle=':', label=f'Global Max: {global_vmax:.3f}')
                    ax3.set_title('Distance Distribution')
                    ax3.set_xlabel('Distance')
                    ax3.set_ylabel('Frequency')
                    ax3.set_xlim(global_vmin, global_vmax)
                    ax3.legend()
                    
                    # Adjust layout and save
                    plt.tight_layout()
                    plt.savefig(map_save_path)
                    plt.close()
                    
            # Compute metrics for this object
            metrics = compute_metrics_for_object(
                object_results['ground_truth'],
                object_results['anomaly_scores'],
                cost_weights
            )
            
            # Save threshold vs gains plot if cost weights provided
            if cost_weights:
                os.makedirs(os.path.join(object_output_dir, "threshold_plots"), exist_ok=True)
                plot_path = os.path.join(object_output_dir, "threshold_plots", f"{object_name}_seed{seed}_thresholds.png")
                
                plt.figure(figsize=(10, 6))
                plt.plot(metrics['all_thresholds'], [g * 100000 for g in metrics['all_gains']])
                plt.axvline(x=metrics['optimal_threshold'], color='red', linestyle='--', 
                          label=f'Optimal threshold: {metrics["optimal_threshold"]:.3f}')
                plt.xlabel('Threshold')
                plt.ylabel('Expected Gain')
                plt.title(f'Threshold vs Expected Gain for {object_name}')
                plt.legend()
                plt.grid(True)
                plt.savefig(plot_path)
                plt.close()
            
            # Save PR curve
            os.makedirs(os.path.join(object_output_dir, "PR_curve"), exist_ok=True)
            pr_plot_path = os.path.join(object_output_dir, "PR_curve", f"{object_name}_seed{seed}_pr_curve.png")
            
            # Compute precision-recall curve
            precision, recall, pr_thresholds = precision_recall_curve(
                object_results['ground_truth'], 
                object_results['anomaly_scores']
            )
            
            # Plot PR curve
            plt.figure(figsize=(10, 6))
            plt.plot(recall, precision, 'b-', linewidth=2, label=f'PR Curve (AP = {metrics["AP"]:.3f})')
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title(f'Precision-Recall Curve for {object_name}')
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            
            # Add some key points on the curve if cost weights are provided
            if cost_weights and 'optimal_threshold' in metrics:
                # Find the point on PR curve closest to optimal threshold
                optimal_thresh = metrics['optimal_threshold']
                
                # Get predictions at optimal threshold
                optimal_predictions = np.array(object_results['anomaly_scores']) >= optimal_thresh
                optimal_precision = np.sum((optimal_predictions == 1) & (np.array(object_results['ground_truth']) == 1)) / np.sum(optimal_predictions == 1) if np.sum(optimal_predictions == 1) > 0 else 0
                optimal_recall = np.sum((optimal_predictions == 1) & (np.array(object_results['ground_truth']) == 1)) / np.sum(np.array(object_results['ground_truth']) == 1) if np.sum(np.array(object_results['ground_truth']) == 1) > 0 else 0
                
                plt.plot(optimal_recall, optimal_precision, 'ro', markersize=8, 
                        label=f'Optimal Point (R={optimal_recall:.3f}, P={optimal_precision:.3f})')
                plt.legend()
            
            plt.tight_layout()
            plt.savefig(pr_plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            # Save ROC curve
            os.makedirs(os.path.join(object_output_dir, "ROC_curve"), exist_ok=True)
            roc_plot_path = os.path.join(object_output_dir, "ROC_curve", f"{object_name}_seed{seed}_roc_curve.png")
            
            # Compute ROC curve
            fpr, tpr, roc_thresholds = roc_curve(
                object_results['ground_truth'], 
                object_results['anomaly_scores']
            )
            
            # Plot ROC curve
            plt.figure(figsize=(10, 6))
            plt.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC Curve (AUC = {metrics["AUROC"]:.3f})')
            plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC Curve for {object_name}')
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            
            # Add some key points on the curve if cost weights are provided
            if cost_weights and 'optimal_threshold' in metrics:
                # Find the point on ROC curve closest to optimal threshold
                optimal_thresh = metrics['optimal_threshold']
                
                # Get predictions at optimal threshold
                optimal_predictions = np.array(object_results['anomaly_scores']) >= optimal_thresh
                
                # Calculate TPR and FPR at optimal threshold
                tp = np.sum((optimal_predictions == 1) & (np.array(object_results['ground_truth']) == 1))
                fp = np.sum((optimal_predictions == 1) & (np.array(object_results['ground_truth']) == 0))
                tn = np.sum((optimal_predictions == 0) & (np.array(object_results['ground_truth']) == 0))
                fn = np.sum((optimal_predictions == 0) & (np.array(object_results['ground_truth']) == 1))
                
                optimal_tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
                optimal_fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
                
                plt.plot(optimal_fpr, optimal_tpr, 'ro', markersize=8, 
                        label=f'Optimal Point (FPR={optimal_fpr:.3f}, TPR={optimal_tpr:.3f})')
                plt.legend()
            
            plt.tight_layout()
            plt.savefig(roc_plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            # Store metrics for this run
            all_runs_results[object_name]['AP_values'].append(metrics['AP'])
            all_runs_results[object_name]['AUROC_values'].append(metrics['AUROC'])
            if cost_weights:
                all_runs_results[object_name]['expected_gain_values'].append(metrics['expected_gain'])
                all_runs_results[object_name]['optimal_threshold_values'].append(metrics['optimal_threshold'])
    
    # Compute and print statistics across all runs
    print("\n=== Final Statistics Across All Runs ===")
    
    # Prepare structured results
    structured_results = {
        "metrics_by_class": {},
        "overall_metrics": {}
    }
    
    # Collect AP values for mAP calculation
    all_ap_values = []
    
    for object_name in all_runs_results:
        print(f"\nObject: {object_name}")
        
        ap_mean = np.mean(all_runs_results[object_name]['AP_values'])
        ap_std = np.std(all_runs_results[object_name]['AP_values'])
        auroc_mean = np.mean(all_runs_results[object_name]['AUROC_values'])
        auroc_std = np.std(all_runs_results[object_name]['AUROC_values'])
        
        print(f"AP: {ap_mean:.4f} ± {ap_std:.4f}")
        print(f"AUROC: {auroc_mean:.4f} ± {auroc_std:.4f}")
        
        # Store class metrics
        class_metrics = {
            "AP": {
                "mean": float(ap_mean),
                "std": float(ap_std)
            },
            "AUROC": {
                "mean": float(auroc_mean),
                "std": float(auroc_std)
            }
        }
        
        all_ap_values.extend(all_runs_results[object_name]['AP_values'])
        
        if cost_weights:
            gain_mean = np.mean(all_runs_results[object_name]['expected_gain_values'])
            gain_std = np.std(all_runs_results[object_name]['expected_gain_values'])
            threshold_mean = np.mean(all_runs_results[object_name]['optimal_threshold_values'])
            threshold_std = np.std(all_runs_results[object_name]['optimal_threshold_values'])
            print(f"Expected Gain: {gain_mean:.4f} ± {gain_std:.4f}")
            print(f"Optimal Threshold: {threshold_mean:.4f} ± {threshold_std:.4f}")
            
            class_metrics["Expected_Gain"] = {
                "mean": float(gain_mean),
                "std": float(gain_std)
            }
            class_metrics["Optimal_Threshold"] = {
                "mean": float(threshold_mean),
                "std": float(threshold_std)
            }
        
        structured_results["metrics_by_class"][object_name] = class_metrics
    
    # Calculate overall metrics
    mAP = np.mean([np.mean(all_runs_results[obj]['AP_values']) for obj in all_runs_results])
    mAP_std = np.std([np.mean(all_runs_results[obj]['AP_values']) for obj in all_runs_results])
    mean_inference_time = np.mean(all_inference_times)
    
    structured_results["overall_metrics"] = {
        "mAP": float(mAP),
        "mAP_std": float(mAP_std),
        "mean_inference_time_ms": float(mean_inference_time * 1000)  # Convert to ms
    }
    
    print(f"\n=== Overall Metrics ===")
    print(f"mAP: {mAP:.4f} ± {mAP_std:.4f}")
    print(f"Mean Inference Time: {mean_inference_time*1000:.2f} ms")
    
    # Save results for each class individually
    for object_name in all_runs_results:
        masking_type = model.preprocessing_configs[object_name]["masking_type"] if model.preprocessing_configs else None    
        folder_name = f"{model_name}_{resolution}_shots={shots}_{masking_type}_{object_name}"
        object_output_dir = os.path.join(args.output_dir, folder_name)
        
        # Create class-specific results
        class_results = {
            "metrics_by_class": {
                object_name: structured_results["metrics_by_class"][object_name]
            },
            "overall_metrics": {
                "AP": structured_results["metrics_by_class"][object_name]["AP"]["mean"],
                "AUROC": structured_results["metrics_by_class"][object_name]["AUROC"]["mean"],
                "mean_inference_time_ms": structured_results["overall_metrics"]["mean_inference_time_ms"]
            }
        }
        
        # Save class-specific results
        results_file = os.path.join(object_output_dir, "evaluation_results.json")
        with open(results_file, 'w') as f:
            json.dump(class_results, f, indent=2)
        print(f"Results for {object_name} saved to: {results_file}")
        
        # Save class-specific run parameters
        run_params = {
            "model_configuration": {
                "model_config": args.model_config,
                "model_config_path": args.model_config_path,
                "dataset": args.dataset,
                "data_root": args.data_root,
                "object_names": [object_name],  # Only this object
                "device": args.device,
                "shots": args.shots,
                "masking_type": masking_type,
                "use_patch_features": args.use_patch_features,
                "processing_config_path": args.processing_config_path,
                "mask_ref_images": args.mask_ref_images,
                "model_name": model_name,
                "resolution": resolution,
                "shots": shots
            },
            "inference_configuration": {
                "knn_metric": args.knn_metric,
                "knn_neighbors": args.knn_neighbors,
                "scoring_type": args.scoring_type,
                "faiss_on_cpu": args.faiss_on_cpu
            },
            "evaluation_configuration": {
                "seeds": args.seeds,
                "warmup_iters": args.warmup_iters,
                "max_samples_per_object": args.max_samples_per_object,
                "metrics": args.metrics
            },
            "cost_configuration": {
                "compute_optimal_threshold": args.compute_optimal_threshold,
                "cost_tp": args.cost_tp,
                "cost_tn": args.cost_tn,
                "cost_fp": args.cost_fp,
                "cost_fn": args.cost_fn
            } if args.compute_optimal_threshold else None,
            "output_configuration": {
                "output_dir": object_output_dir,
                "save_anomaly_maps": args.save_anomaly_maps,
                "save_tsne_plots": args.save_tsne_plots,
                "plot_examples": args.plot_examples,
                "num_examples": args.num_examples
            }
        }
        
        params_file = os.path.join(object_output_dir, "run_parameters.yaml")
        with open(params_file, 'w') as f:
            yaml.dump(run_params, f, default_flow_style=False, indent=2)


if __name__ == "__main__":
    main()
