import os

import cv2
import numpy as np
import random
import torch
import yaml
import time

from src.benchmark.memory_bank.utils import compute_distances, dists2map
from src.models.core.ports.backbone import Backbone
from src.models.core.ports.model import Model
from src.models.core.services import MODEL_REGISTRY
from src.data.splits.mvtec import MVTech_Unsupervised_Split
from src.data.datasets.mvtec import ImageDataset, ImageMaskLabelDataset
from ultralytics import FastSAM

@MODEL_REGISTRY.register_model()
class MemoryBankModel(Model):
    def __init__(self, 
                 backbone: Backbone, 
                 data_root: str,
                 dataset_name: str,
                 object_names: list[str], 
                 device: str, 
                 shots: int=-1, 
                 informed_preprocessing: bool=False,
                 processing_config_path: str=None,
                 seed: int=1,
                 use_patch_features: bool=True,
                 mask_ref_images: bool=False,
                 ):
        """
        Args:
            backbone: The backbone model to use.
            data_root: The root directory of the dataset.
            dataset_name: The name of the dataset (used to load the preprocessing configs).
            object_name: The name of the object to use.
            device: The device to use.
            shots: The number of examples to use to populate the memory bank.
            informed_preprocessing: Whether to use preprocessing configs for each object.
            use_patch_features: Whether to use patch features or image features.
            multi_object: Whether to use multiple objects in the memory bank.
        """
        self.model = backbone
        self.data_root = data_root
        self.dataset_name = dataset_name
        self.object_names = object_names
        self.device = device
        self.shots = shots
        self.informed_preprocessing = informed_preprocessing
        self.processing_config_path = processing_config_path
        self.seed = seed
        self.use_patch_features = use_patch_features
        self.mask_ref_images = mask_ref_images
        self.grid_size = self.model.grid_size if hasattr(self.model, 'grid_size') and self.use_patch_features else None

        self.model.to(self.device)
        self.model.eval()

        self.memory_bank = []
        self.reference_samples = {} # All training samples per object
        self.test_samples = {} # All test samples per object

        self.reference_dataset = {} # Dataset of reference examples per object
        self.test_dataset = {} # Dataset of test examples per object

        # For visualization
        self.reference_images = []
        self.reference_masks = []

        self._load_dataset()
        self._pick_reference_examples()
        self._load_preprocessing_configs()
        self.build_memory_bank()


    def _load_preprocessing_configs(self):
        if self.informed_preprocessing:
            with open(os.path.join(self.processing_config_path, self.dataset_name + ".yaml"), "r") as f:
                self.preprocessing_configs = yaml.safe_load(f)["objects"]
        else:
            self.preprocessing_configs = None

    def _load_dataset(self):
        for object_name in self.object_names:
            self.split = MVTech_Unsupervised_Split(self.data_root, object_name, defect_ratio=0.05, seed=self.seed)
            self.reference_samples[object_name] = self.split.train
            self.test_samples[object_name] = self.split.val
            self.test_dataset[object_name] = ImageMaskLabelDataset(self.test_samples[object_name])

    def _pick_reference_examples(self):
        # Validate shots parameter
        invalid_shots_number = self.shots == 0 or self.shots < -1
        if invalid_shots_number:
            raise ValueError(f"Shots must be a positive integer or -1, got {self.shots}")

        # Process each object
        for object_name in self.object_names:
            num_reference_samples = len(self.reference_samples[object_name])
            use_all_samples = self.shots == -1
            more_shots_than_samples = self.shots > num_reference_samples

            # Case 1: Use all samples
            if use_all_samples:
                print(f"Using all samples for object {object_name}")
                reference_examples = [
                    sample[0]
                    for sample in self.reference_samples[object_name]
                ]
                self.reference_dataset[object_name] = ImageDataset(reference_examples)
                continue

            # Case 2: Validate and use specified number of shots
            if more_shots_than_samples:
                raise ValueError(
                    f"Shots ({self.shots}) cannot exceed number of reference samples "
                    f"({num_reference_samples}) for object {object_name}"
                )
            
            print(f"Using {self.shots} shots for object {object_name}")

            # Select shots using seed-based circular sampling
            start_idx = (self.seed * self.shots) % num_reference_samples
            selected_indices = [
                (start_idx + i) % num_reference_samples 
                for i in range(self.shots)
            ]
            reference_examples = [
                self.reference_samples[object_name][i][0] 
                for i in selected_indices
            ]
            self.reference_dataset[object_name] = ImageDataset(reference_examples)

    def predict(self, query_image, knn_index, knn_metric="L2_normalized", knn_neighbors=1, masking_type=None, scoring_type="mean_top1p", object_name=None):
        processed_query_image = self.model.prepare_image(query_image).to(self.device)
        
        if self.use_patch_features:
            start_time = time.time()
            mask, masked_features = self.extract_reference_patch_features(processed_query_image, masking_type=masking_type, object_name=object_name, base_image=query_image)
            masked_features = np.ascontiguousarray(masked_features, dtype=np.float32)
            
            
            distances = compute_distances(knn_index, masked_features, knn_metric, knn_neighbors)

            output_distances = np.zeros_like(mask, dtype=float)
            output_distances[mask] = distances.squeeze()
            d_masked = output_distances.reshape(self.grid_size)
            d_masked[~mask.reshape(self.grid_size)] = 0.0


            if scoring_type == "mean_top1p":
                if int(len(distances) * 0.01) == 0:
                    anomaly_score = np.max(output_distances) if len(output_distances) > 0 else 0
                else:
                    anomaly_score = np.mean(sorted(output_distances, reverse = True)[:int(len(output_distances) * 0.01)])
            elif scoring_type == "mean_top1p_adjusted":
                    # Calculate base anomaly score from top 1% distances
                    top_distances = sorted(output_distances, reverse=True)[:int(len(output_distances) * 0.01)]
                    anomaly_score = np.mean(top_distances)
                    
                    # Calculate softmax of all distances first
                    softmax_distances = np.exp(output_distances) / np.sum(np.exp(output_distances))
                    
                    # Get indices of top 1% distances
                    top_indices = np.argsort(output_distances)[-int(len(output_distances) * 0.01):]
                    
                    # Get corresponding softmax values for those indices
                    corresponding_softmax = softmax_distances[top_indices]
                    
                    # Weight each distance by its corresponding softmax value
                    weighted_distances = output_distances[top_indices] * (1-corresponding_softmax)
                    
                    # Final anomaly score is mean of weighted distances
                    anomaly_score = np.mean(weighted_distances)
            elif scoring_type == "max":
                # Find most anomalous test patch (m_test*) by taking argmax of distances
                most_anomalous_patch_idx = np.argmax(distances.flatten())
                m_test = masked_features[most_anomalous_patch_idx:most_anomalous_patch_idx+1]
                
                # Find closest memory bank patch (m*) to most anomalous test patch
                distances_to_memory, _ = knn_index.search(m_test, k=1)
                min_distance = distances_to_memory[0][0]

                # Use this distance as anomaly score
                anomaly_score = min_distance
            elif scoring_type == "max_adjusted":
                # Find most anomalous test patch (m_test*) by taking argmax of distances
                most_anomalous_patch_idx = np.argmax(distances.flatten())
                m_test = masked_features[most_anomalous_patch_idx:most_anomalous_patch_idx+1]
                
                # Find closest memory bank patch (m*) to most anomalous test patch
                distances_to_memory, _ = knn_index.search(m_test, k=10)  # Returns shape (1,5) since m_test has shape (1,D)
                distances_to_memory = distances_to_memory.squeeze()  # Convert to shape (5,) for subsequent operations
                min_distances = np.exp(distances_to_memory) / np.sum(np.exp(distances_to_memory))
                
                # Use this distance as anomaly score
                anomaly_score = (1 - min_distances[0]) * distances_to_memory[0]
            else:
                raise ValueError(f"Scoring type {scoring_type} not supported")
            
            anomaly_map = dists2map(d_masked, query_image.size)
            inf_time = time.time() - start_time
        else:
            start_time = time.time()
            # Use image features instead of patch features
            image_features = self.extract_reference_image_features(processed_query_image)
            image_features = np.ascontiguousarray(image_features, dtype=np.float32)

            distances = compute_distances(knn_index, image_features, knn_metric, knn_neighbors)

            # For image features, we have a single distance value
            anomaly_score = distances[0] if distances.size == 1 else np.mean(distances)
            
            # Create a uniform anomaly map since we don't have spatial information with image features
            anomaly_map = np.full(query_image.size[::-1], anomaly_score)  # Use reverse for height, width
            masked_features = image_features
            inf_time = time.time() - start_time
        return output_distances, masked_features, inf_time, anomaly_score, anomaly_map

    def build_memory_bank(self):
        # Initialize FastSAM model once if needed
        masking_types = [self.preprocessing_configs[obj].get("masking_type") for obj in self.object_names]
        if any(masking_type == "fast_sam" or masking_type == "fast_sam_dilated" for masking_type in masking_types):
            self.sam = FastSAM("FastSAM-x.pt").to(self.device)
        else:
            self.sam = None

        for object_name in self.object_names:
            for img_ref in self.reference_dataset[object_name]:
                if self.preprocessing_configs:
                    rotation = self.preprocessing_configs[object_name]["rotation"]
                    imgs = self.rotation_augmentation(img_ref) if rotation else [img_ref]
                    for img in imgs:
                        image_tensor = self.model.prepare_image(img).to(self.device)
                        if self.use_patch_features:
                            masking_type = self.preprocessing_configs[object_name]["masking_type"] if self.mask_ref_images else None
                            mask, features_ref_i = self.extract_reference_patch_features(image_tensor, masking_type, object_name, base_image=img)
                            self.memory_bank.append(features_ref_i)
                            self.reference_masks.append(mask)
                        else:
                            features_ref_i = self.extract_reference_image_features(image_tensor)
                            self.memory_bank.append(features_ref_i)
                            self.reference_masks.append(np.ones((1, 1)))  # No masking for image features
                        self.reference_images.append(img)
                else:
                    image_tensor = self.model.prepare_image(img_ref).to(self.device)
                    if self.use_patch_features:
                        mask, features_ref_i = self.extract_reference_patch_features(image_tensor, object_name, base_image=img_ref)
                        self.memory_bank.append(features_ref_i)
                        self.reference_masks.append(np.ones((self.grid_size[0] * self.grid_size[1], 1)))
                    else:
                        features_ref_i = self.extract_reference_image_features(image_tensor)
                        self.memory_bank.append(features_ref_i)
                        self.reference_masks.append(np.ones((1, 1)))  # No masking for image features
                    self.reference_images.append(img_ref)

        self.memory_bank = np.ascontiguousarray(np.concatenate(self.memory_bank, axis=0), dtype=np.float32)

    def extract_reference_patch_features(self, processed_image, masking_type=None, object_name=None, base_image=None):
        features_ref_i = self.model.extract_patch_features(processed_image, strip_cls_token=True).squeeze()
        if len(features_ref_i.shape) > 2:
            features_ref_i = features_ref_i.reshape(features_ref_i.shape[0], -1).T

        if masking_type == "zero_shot":
            mask = self.model.compute_background_mask(features_ref_i.detach().cpu().numpy(), threshold=12)
            return mask, features_ref_i[mask].detach().cpu().numpy()
        elif masking_type == "fast_sam":
            results = self.sam(base_image, texts=f"a {object_name}", imgsz=480, verbose=False)
            mask = results[0].masks.data[0]
            mask_resized = cv2.resize(mask.cpu().numpy().astype(np.uint8), (self.grid_size[1], self.grid_size[0])).astype(bool).flatten()
            return mask_resized, features_ref_i[mask_resized].detach().cpu().numpy()
        elif masking_type == "fast_sam_dilated":
            results = self.sam(base_image, texts=f"a {object_name}", imgsz=480, verbose=False)
            mask = results[0].masks.data[0]
            mask_resized = cv2.resize(mask.cpu().numpy().astype(np.uint8), (self.grid_size[1], self.grid_size[0]))
            kernel = np.ones((3,3), np.uint8)
            mask_dilated = cv2.dilate(mask_resized, kernel, iterations=1).astype(bool).flatten()
            features_ref_i = features_ref_i.detach().cpu().numpy()
            return mask_dilated, features_ref_i[mask_dilated]
        else:
            return np.ones(features_ref_i.shape[0], dtype=bool), features_ref_i.detach().cpu().numpy()
        
    def extract_reference_image_features(self, processed_image):
        """Extract image-level features instead of patch features."""
        features_ref_i = self.model.extract_image_features(processed_image).squeeze()
        # Ensure features are 2D for consistency with memory bank structure
        if len(features_ref_i.shape) == 1:
            features_ref_i = features_ref_i.unsqueeze(0)
        return features_ref_i.detach().cpu().numpy()
    
    def rotation_augmentation(self, img_ref, angles = [0, 45, 90, 135, 180, 225, 270, 315]):
        imgs = []
        for angle in angles:
            imgs.append(self._rotate_image(img_ref, angle))
        return imgs
    
    def _rotate_image(self, image_np, angle):
        image_center = (image_np.shape[1] / 2, image_np.shape[0] / 2)
        rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
        result = cv2.warpAffine(image_np, rot_mat, (image_np.shape[1], image_np.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        return result
        