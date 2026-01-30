import os
import json
import shutil
from pathlib import Path
import numpy as np
from PIL import Image
import random
import datetime
import argparse
from collections import defaultdict
from dotenv import load_dotenv

load_dotenv()
IMAGE_PATH = os.getenv("IMAGE_PATH")

def create_coco_structure(output_path='mvtec_coco'):
    """Create the COCO directory structure."""
    os.makedirs(f'{output_path}/annotations', exist_ok=True)
    os.makedirs(f'{output_path}/train', exist_ok=True)
    os.makedirs(f'{output_path}/val', exist_ok=True)

def get_bbox_from_mask(mask_img):
    """Extract bounding box from a binary mask image."""
    # Convert to numpy array if it's a PIL Image
    if isinstance(mask_img, Image.Image):
        mask = np.array(mask_img)
    else:
        mask = mask_img
        
    # Find non-zero pixels
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    
    if not np.any(rows) or not np.any(cols):
        # Empty mask, return None
        return None, 0
        
    # Find the bounding box
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    
    # COCO format is [x, y, width, height]
    bbox = [int(cmin), int(rmin), int(cmax - cmin + 1), int(rmax - rmin + 1)]
    area = int(bbox[2] * bbox[3])
    
    return bbox, area

def collect_dataset_samples(input_base_path, selected_categories=None, selected_defect_types=None):
    """Collect all samples from the dataset with their metadata"""
    all_samples = []
    all_defect_types = defaultdict(set)
    
    # Get all categories
    all_categories = [d for d in os.listdir(input_base_path) 
                     if os.path.isdir(os.path.join(input_base_path, d)) and 
                     not d.startswith('.') and d not in ['.cache']]
    
    # Filter categories if specified
    categories = all_categories
    if selected_categories and selected_categories != ['all']:
        categories = [c for c in all_categories if c in selected_categories]
        if not categories:
            raise ValueError(f"No valid categories found. Available categories: {all_categories}")
        print(f"Selected categories: {categories}")
    
    # First collect all defect types across categories
    for category in categories:
        category_path = os.path.join(input_base_path, category)
        test_dir = os.path.join(category_path, 'test')
        
        # Check if the category has a test directory
        if os.path.exists(test_dir):
            defect_types = [d for d in os.listdir(test_dir) 
                           if os.path.isdir(os.path.join(test_dir, d)) and d != 'good']
            for defect_type in defect_types:
                all_defect_types[category].add(defect_type)
    
    # Filter defect types if specified
    if selected_defect_types and selected_defect_types != ['all']:
        filtered_defect_types = {}
        for category, defects in all_defect_types.items():
            filtered_defect_types[category] = {d for d in defects if d in selected_defect_types}
        all_defect_types = filtered_defect_types
    
    # Create category mapping for COCO format
    category_info = [
        {
            "id": 0,
            "name": "good",
            "supercategory": "none"
        },
        {
            "id": 1,
            "name": "defect",
            "supercategory": "none"
        }
    ]
    
    # Collect all samples
    for category_name in categories:
        category_path = os.path.join(input_base_path, category_name)
        
        # Process normal training samples
        good_train_dir = os.path.join(category_path, 'train', 'good')
        if os.path.exists(good_train_dir):
            for img_path in sorted(os.listdir(good_train_dir)):
                if img_path.endswith('.png'):
                    sample = {
                        'category': category_name,
                        'type': 'good',
                        'phase': 'train',
                        'img_path': os.path.join(good_train_dir, img_path),
                        'has_defect': False
                    }
                    all_samples.append(sample)
        
        # Process test images (defects and normal)
        test_dir = os.path.join(category_path, 'test')
        if os.path.exists(test_dir):
            # Process normal test samples
            good_test_dir = os.path.join(test_dir, 'good')
            if os.path.exists(good_test_dir):
                for img_path in sorted(os.listdir(good_test_dir)):
                    if img_path.endswith('.png'):
                        sample = {
                            'category': category_name,
                            'type': 'good',
                            'phase': 'test',
                            'img_path': os.path.join(good_test_dir, img_path),
                            'has_defect': False
                        }
                        all_samples.append(sample)
            
            # Process defect test samples
            category_defect_types = all_defect_types.get(category_name, set())
            for defect_type in category_defect_types:
                defect_dir = os.path.join(test_dir, defect_type)
                ground_truth_dir = os.path.join(category_path, 'ground_truth', defect_type)
                
                if os.path.exists(defect_dir):
                    for img_path in sorted(os.listdir(defect_dir)):
                        if img_path.endswith('.png'):
                            mask_path = os.path.join(ground_truth_dir, img_path.replace('.png', '_mask.png'))
                            if os.path.exists(mask_path):
                                sample = {
                                    'category': category_name,
                                    'type': defect_type,
                                    'phase': 'test',
                                    'img_path': os.path.join(defect_dir, img_path),
                                    'mask_path': mask_path,
                                    'has_defect': True
                                }
                                all_samples.append(sample)
    
    return all_samples, category_info

def process_mvtec_dataset(max_examples_per_defect=None, input_path='mvtec_anomaly_detection', output_path='mvtec_coco', 
                         selected_categories=None, selected_defect_types=None, 
                         mode='split', train_ratio=0.8, n_folds=5, seed=42, stratify=False):
    """Process MVTec dataset and convert to COCO format"""
    mvtec_path = Path(input_path)
    coco_path = Path(output_path)
    
    # Collect all samples with metadata
    print(f"Collecting samples from {input_path}...")
    all_samples, categories = collect_dataset_samples(
        input_path, 
        selected_categories=selected_categories,
        selected_defect_types=selected_defect_types
    )
    print(f"Found {len(all_samples)} samples total")
    
    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    
    # Process based on mode
    if mode == 'cv':
        process_cross_validation(
            all_samples,
            output_path,
            categories,
            n_folds=n_folds,
            seed=seed,
            stratify_by_category=stratify,
            max_examples_per_defect=max_examples_per_defect
        )
    else:  # single split mode
        process_custom_split(
            all_samples,
            output_path,
            categories,
            train_ratio=train_ratio,
            seed=seed,
            stratify_by_category=stratify,
            max_examples_per_defect=max_examples_per_defect
        )

def process_cross_validation(all_samples, output_path, categories, n_folds=5, seed=42, stratify_by_category=True, max_examples_per_defect=None):
    """Process dataset with cross-validation splits"""
    np.random.seed(seed)
    random.seed(seed)
    
    # Import here to avoid dependency if not using CV mode
    from sklearn.model_selection import KFold, StratifiedKFold
    
    # Prepare data for stratification if needed
    if stratify_by_category:
        # Stratify by category and defect status
        stratify_labels = [f"{s['category']}_{int(s['has_defect'])}" for s in all_samples]
        kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        splits = list(kf.split(all_samples, stratify_labels))
    else:
        # Simple k-fold without stratification
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
        splits = list(kf.split(all_samples))
    
    # Process each fold
    for fold, (train_idx, val_idx) in enumerate(splits):
        print(f"Processing fold {fold+1}/{n_folds}")
        
        # Create fold directory
        fold_dir = os.path.join(output_path, f"fold_{fold+1}")
        os.makedirs(os.path.join(fold_dir, 'train'), exist_ok=True)
        os.makedirs(os.path.join(fold_dir, 'val'), exist_ok=True)
        os.makedirs(os.path.join(fold_dir, 'annotations'), exist_ok=True)
        
        # If max examples per defect is set, limit the training samples
        if max_examples_per_defect is not None:
            train_samples = [all_samples[i] for i in train_idx]
            limited_train_samples = limit_defect_samples(train_samples, max_examples_per_defect)
            # Create COCO dataset with limited samples
            create_coco_dataset(fold_dir, limited_train_samples, [all_samples[i] for i in val_idx], categories)
        else:
            # Create COCO dataset with all samples
            create_coco_dataset(fold_dir, [all_samples[i] for i in train_idx], [all_samples[i] for i in val_idx], categories)
        
        print(f"Saved fold {fold+1} to {fold_dir}")

def process_custom_split(all_samples, output_path, categories, train_ratio=0.8, seed=42, stratify_by_category=True, max_examples_per_defect=None):
    """Process dataset with a single custom train/val split"""
    np.random.seed(seed)
    random.seed(seed)
    
    # Create directory structure
    os.makedirs(os.path.join(output_path, 'train'), exist_ok=True)
    os.makedirs(os.path.join(output_path, 'val'), exist_ok=True)
    os.makedirs(os.path.join(output_path, 'annotations'), exist_ok=True)
    
    if stratify_by_category:
        # Group samples by category and has_defect status
        groups = {}
        for sample in all_samples:
            key = f"{sample['category']}_{int(sample['has_defect'])}"
            if key not in groups:
                groups[key] = []
            groups[key].append(sample)
        
        # Split each group and collect samples
        train_samples = []
        val_samples = []
        
        for key, samples in groups.items():
            random.shuffle(samples)
            split_idx = int(len(samples) * train_ratio)
            train_samples.extend(samples[:split_idx])
            val_samples.extend(samples[split_idx:])
    else:
        # Simple random split
        random.shuffle(all_samples)
        split_idx = int(len(all_samples) * train_ratio)
        train_samples = all_samples[:split_idx]
        val_samples = all_samples[split_idx:]
    
    # If max examples per defect is set, limit the training samples
    if max_examples_per_defect is not None:
        train_samples = limit_defect_samples(train_samples, max_examples_per_defect)
    
    # Create COCO dataset
    create_coco_dataset(output_path, train_samples, val_samples, categories)
    
    print(f"Dataset saved to {output_path}")

def limit_defect_samples(samples, max_examples_per_defect):
    """Limit the number of examples per defect type in the samples"""
    # Group samples by category and defect type
    defect_groups = defaultdict(list)
    normal_samples = []
    
    for sample in samples:
        if sample['has_defect']:
            key = f"{sample['category']}_{sample['type']}"
            defect_groups[key].append(sample)
        else:
            normal_samples.append(sample)
    
    # Limit each defect group to max_examples_per_defect
    limited_defect_samples = []
    for key, defect_samples in defect_groups.items():
        random.shuffle(defect_samples)
        limited_defect_samples.extend(defect_samples[:max_examples_per_defect])
    
    # Combine with normal samples
    return normal_samples + limited_defect_samples

def create_coco_dataset(output_dir, train_samples, val_samples, categories):
    """Create COCO dataset from the given samples"""
    # Initialize COCO format data structures
    train_data = {
        "info": {
            "description": "MVTec Anomaly Detection Dataset in COCO Format",
            "url": "https://www.mvtec.com/company/research/datasets/mvtec-ad",
            "version": "1.0",
            "year": 2023,
            "contributor": "Converted from MVTec AD",
            "date_created": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        },
        "licenses": [
            {
                "id": 1,
                "name": "CC BY-NC-SA 4.0",
                "url": "https://creativecommons.org/licenses/by-nc-sa/4.0/"
            }
        ],
        "images": [],
        "annotations": [],
        "categories": categories
    }
    
    val_data = {
        "info": train_data["info"].copy(),
        "licenses": train_data["licenses"].copy(),
        "images": [],
        "annotations": [],
        "categories": categories
    }
    
    # Process training samples
    image_id = 0
    annotation_id = 1
    
    for sample in train_samples:
        # Process image
        img = Image.open(sample['img_path'])
        width, height = img.size
        
        # Create new filename
        new_filename = f"{image_id:012d}.jpg"
        new_path = os.path.join(output_dir, 'train', new_filename)
        
        # Copy and convert image
        img.convert('RGB').save(new_path, 'JPEG')
        
        # Add image info
        image_info = {
            "id": image_id,
            "file_name": new_filename,
            "width": width,
            "height": height,
            "license": 1,
            "date_captured": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        train_data["images"].append(image_info)
        
        # If it's a defect, add annotation
        if sample.get('has_defect', False):
            bbox, area = get_bbox_from_mask(Image.open(sample['mask_path']))
            if bbox is not None:
                annotation = {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": 1,  # 1 for defect
                    "bbox": bbox,
                    "area": area,
                    "iscrowd": 0,
                    "segmentation": []  # No segmentation data
                }
                train_data["annotations"].append(annotation)
                annotation_id += 1
        
        image_id += 1
    
    # Process validation samples
    val_image_id = 0
    
    for sample in val_samples:
        # Process image
        img = Image.open(sample['img_path'])
        width, height = img.size
        
        # Create new filename
        new_filename = f"{val_image_id:012d}.jpg"
        new_path = os.path.join(output_dir, 'val', new_filename)
        
        # Copy and convert image
        img.convert('RGB').save(new_path, 'JPEG')
        
        # Add image info
        image_info = {
            "id": val_image_id,
            "file_name": new_filename,
            "width": width,
            "height": height,
            "license": 1,
            "date_captured": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        val_data["images"].append(image_info)
        
        # If it's a defect, add annotation
        if sample.get('has_defect', False):
            bbox, area = get_bbox_from_mask(Image.open(sample['mask_path']))
            if bbox is not None:
                annotation = {
                    "id": annotation_id,
                    "image_id": val_image_id,
                    "category_id": 1,  # 1 for defect
                    "bbox": bbox,
                    "area": area,
                    "iscrowd": 0,
                    "segmentation": []  # No segmentation data
                }
                val_data["annotations"].append(annotation)
                annotation_id += 1
        
        val_image_id += 1
    
    # Save COCO format JSON files
    with open(os.path.join(output_dir, 'annotations', 'instances_train.json'), 'w') as f:
        json.dump(train_data, f, indent=2)
    
    with open(os.path.join(output_dir, 'annotations', 'instances_val.json'), 'w') as f:
        json.dump(val_data, f, indent=2)

def main():
    parser = argparse.ArgumentParser(description='Convert MVTec dataset to COCO format with cross-validation')
    parser.add_argument('--input', type=str, default=f'{IMAGE_PATH}/mvtec_anomaly_detection', 
                        help='Path to the MVTec dataset')
    parser.add_argument('--output', type=str, default=f'{IMAGE_PATH}/mvtec_coco', 
                        help='Output directory for COCO dataset')
    parser.add_argument('--mode', type=str, choices=['cv', 'split'], default='split',
                        help='Mode: cross-validation (cv) or single split (split)')
    parser.add_argument('--folds', type=int, default=5, 
                        help='Number of folds for cross-validation')
    parser.add_argument('--train_ratio', type=float, default=0.8, 
                        help='Training ratio for single split mode')
    parser.add_argument('--seed', type=int, default=42, 
                        help='Random seed for reproducibility')
    parser.add_argument('--stratify', action='store_true', 
                        help='Stratify by category and defect status')
    parser.add_argument('--categories', nargs='+', default=['all'],
                        help='Select specific object categories (e.g., --categories bottle screw). Use "all" for all categories.')
    parser.add_argument('--defect_types', nargs='+', default=['all'],
                        help='Select specific defect types (e.g., --defect_types contamination broken_large). Use "all" for all defect types.')
    parser.add_argument('--max_examples_per_defect', type=int, default=None,
                        help='Maximum number of examples per defect type to include in the training set')
    
    args = parser.parse_args()
    args.output += f"-{args.categories}"
    
    # Create COCO structure
    create_coco_structure(args.output)
    
    # Process MVTec dataset
    process_mvtec_dataset(
        max_examples_per_defect=args.max_examples_per_defect,
        input_path=args.input,
        output_path=args.output,
        selected_categories=args.categories,
        selected_defect_types=args.defect_types,
        mode=args.mode,
        train_ratio=args.train_ratio,
        n_folds=args.folds,
        seed=args.seed,
        stratify=args.stratify
    )
    
    print("Conversion completed successfully!")
    print(f"Dataset saved to {os.path.abspath(args.output)}")

if __name__ == "__main__":
    main() 