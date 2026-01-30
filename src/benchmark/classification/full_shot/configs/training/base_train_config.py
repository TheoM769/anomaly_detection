import glob
import os
import re
import random

import torch
from dotenv import load_dotenv
from torchvision.transforms import v2

from src.benchmark.classification.full_shot.training_utils.metrics import ClassificationMetrics

load_dotenv()
IMAGE_PATH = os.getenv("IMAGE_PATH")

class TrainConfig:

    def __init__(
            self,
            dataset_name : str,
            classname : str,
            dist_adjust : bool,
            multiclass : bool,
            loss_weights : list[float] = None,
            train_split: float = 0.5,
            val_split: float = 0.8,
            num_epochs: int = 10,
            batch_size: int = 16,
            weight_decay: float = 0.0,
            lr : float = 1e-4,
            horizontal_flip: bool = 0,
            vertical_flip: bool = 0,
            rotation: tuple = (-180, 180),
            brightness: float = 0,
            contrast: float = 0,
            saturation: float = 0,
            hue: float = 0,
            train_metrics = ["AP"],
            val_metrics = ["AP", "optimal_cost"],
            criterion : str = "CE",
            use_wandb: bool = True,
            wandb_project: str = "Benchmark",
            cost_weights: dict = {"tp": 0.0, "tn": 0.02, "fp": -0.02, "fn": -0.2}
    ):
        
        # Data config
        self.dataset_name = dataset_name
        self.classname = classname
        self.dist_adjust = dist_adjust
        self.multiclass = multiclass

        # Hyperparameters
        self.train_split = train_split
        self.val_split = val_split
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.lr = lr
        self.loss_weights = loss_weights

        # Augmentation config
        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip
        self.rotation = rotation
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
        self.cost_weights = cost_weights

        # Training config
        self.train_metrics = train_metrics
        self.val_metrics = val_metrics
        self.criterion = criterion
        self.augmentation_transform = v2.Compose([
            v2.RandomHorizontalFlip(p=self.horizontal_flip),
            v2.RandomVerticalFlip(p=self.vertical_flip),
            v2.RandomRotation(degrees=self.rotation),
            v2.ColorJitter(brightness=self.brightness, contrast=self.contrast, saturation=self.saturation, hue=self.hue)
        ])

        torch.manual_seed(42)
        random.seed(42)

        # Monitoring config
        self.use_wandb = use_wandb
        self.wandb_project = wandb_project

        self.nb_classes = self.get_nb_classes() if self.multiclass else 2
        if not self.loss_weights:
            self.loss_weights = [1.0] * self.nb_classes
        self.train_metrics = ClassificationMetrics(self.nb_classes, self.train_metrics, cost_weights=self.cost_weights)
        self.val_metrics = ClassificationMetrics(self.nb_classes, self.val_metrics, cost_weights=self.cost_weights)

    def get_nb_classes(self):
        all_files = []
        for c in self.classname:
            all_files.extend(glob.glob(os.path.join(IMAGE_PATH, self.dataset_name, c, '*/*')))
        pattern = re.compile(r'good|ground_truth')
        filtered_files = [file for file in all_files if not pattern.search(file)]
        last_subfolders = [os.path.basename(path) for path in filtered_files]
        return len(last_subfolders) + 1
    
    def __dict__(self):
        return {
            "dataset_name": self.dataset_name,
            "classname": self.classname,
            "dist_adjust": self.dist_adjust,
            "multiclass": self.multiclass,
            "train_split": self.train_split,
            "val_split": self.val_split,
            "num_epochs": self.num_epochs,
            "batch_size": self.batch_size,
            "weight_decay": self.weight_decay,
            "lr": self.lr,
            "train_metrics": self.train_metrics,
            "val_metrics": self.val_metrics,
            "criterion": self.criterion,
            "loss_weights": self.loss_weights,
            "use_wandb": self.use_wandb,
            "wandb_project": self.wandb_project,
            "image_path": IMAGE_PATH,
            "nb_classes": self.nb_classes,
            "horizontal_flip": self.horizontal_flip,
            "vertical_flip": self.vertical_flip,
            "rotation": self.rotation,
            "brightness": self.brightness,
            "contrast": self.contrast,
            "saturation": self.saturation,
            "hue": self.hue
        }

    def __str__(self):
        return f"{self.__dict__()}"
    
    def __repr__(self):
        return self.__str__()