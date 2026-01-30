from typing import Optional, Union
import random
import glob
import os
import time

import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.neighbors import NearestNeighbors
from PIL import Image
import numpy as np

from core.ports.model import Model
from core.ports.backbone import Backbone
from adapters.datasets.image_dataset import ImageDataset
from adapters.types.memory_bank_result import MemoryBankResult

seed = 42
random.seed(seed)

class MemoryBankModelAdapter(Model):
    def __init__(self, backbone: Backbone, root_directory: str, memory_bank_path: Optional[str] = None):
        self.backbone = backbone
        self.root_directory = root_directory
        self.memory_bank_path = memory_bank_path

        self.samples = None
        self.global_memory_bank = None
        self.memory_bank = None
        self.knn = None
        self.anomaly_threshold = None
        self.in_bank_split = None

        self._load_weights()
        self._load_samples_path_from_root_directory()
        self._build_memory_bank()

    def _load_weights(self):
        self.backbone.load_weights()

    def _load_samples_path_from_root_directory(self):
        directory_tree = glob.glob(os.path.join(self.root_directory, '**', '*'), recursive=True)
        all_files = [file for file in directory_tree if os.path.isfile(file)]
        self.samples = all_files

    def _build_memory_bank(self, batch_size=32, shuffle=False, num_workers=0):
        if self.memory_bank_path is None:
            return self._build_memory_bank_from_scratch(batch_size, shuffle, num_workers)
        else:
            return self._load_memory_bank_from_path()
        
    def _build_memory_bank_from_scratch(self, batch_size, shuffle, num_workers):
        selected_samples_dataset = ImageDataset(self.samples)
        memory_bank = self.extract_all_features(selected_samples_dataset, batch_size, shuffle, num_workers)
        self.global_memory_bank = memory_bank
    
    def _load_memory_bank_from_path(self):
        try:
            return torch.load(self.memory_bank_path)
        except FileNotFoundError:
            raise FileNotFoundError(f"Memory bank file not found at {self.memory_bank_path}")
    
    def _select_memory_bank_samples(self, in_bank_split: float = 1):
        if in_bank_split == 1:
            self.memory_bank = self.global_memory_bank
        else:
            train_length = len(self.samples)
            train_split = int(train_length * in_bank_split)
            train_indices = random.sample(range(train_length), train_split)
            self.memory_bank = self.global_memory_bank[train_indices]
            
    def _compute_threshold(self, memory_bank: torch.Tensor):

        distance_list, _ = self.knn.kneighbors(memory_bank)
        distances_tensor = torch.tensor(distance_list[:, 1]).squeeze()
        threshold = torch.quantile(distances_tensor, 0.95)
        self.anomaly_threshold = threshold.item()
    
    def _build_knn(self):
        self.knn = NearestNeighbors(n_neighbors=2, algorithm="brute")
        self.knn.fit(self.memory_bank)
        self._compute_threshold(self.memory_bank)
    
    def extract_all_features(self, dataset, batch_size=32, shuffle=True, num_workers=0):
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    
        image_features = []
        for batch in tqdm(train_loader, desc="Processing batches"):
            batch_features = self.backbone.image_feature_extraction(batch)
            image_features.append(batch_features)

        memory_bank = torch.cat(image_features, dim=0)
        self.global_memory_bank = memory_bank
        return memory_bank
    
    def predict(self, pixel_values: Union[str, Image.Image, np.ndarray, torch.Tensor], in_bank_split: float = 1):
        if isinstance(pixel_values, str):
            image = Image.open(pixel_values).convert('RGB')
            image_features = self.backbone.image_feature_extraction(image)
        else:
            image_features = self.backbone.image_feature_extraction(pixel_values)
        
        if self.in_bank_split != in_bank_split:
            self._select_memory_bank_samples(in_bank_split)
            self._build_knn()
            self.in_bank_split = in_bank_split

        neighbors_distances, _ = self.knn.kneighbors(image_features)
        closest_neighbor_distance = torch.tensor(neighbors_distances[:, 0]).squeeze()
        anomaly_scores = (closest_neighbor_distance > self.anomaly_threshold).int()
        return MemoryBankResult(anomaly_scores)
    
    def predict_with_inference_time(self, image_data, in_bank_split: float = 1):
        start_time = time.time()
        prediction = self.predict(image_data, in_bank_split)
        end_time = time.time()
        inference_time = end_time - start_time
        return prediction, inference_time
    
    def get_memory_bank(self):
        return self.memory_bank 
    
    def get_global_memory_bank(self):
        return self.global_memory_bank
    
    def save_memory_bank(self, path):
        torch.save(self.global_memory_bank, path)
