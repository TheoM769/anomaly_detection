import os
import glob
import re
import math
from collections import Counter
from abc import ABC, abstractmethod

from PIL import Image
import numpy as np
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt


class MVTech_SP_split(ABC):
    def __init__(self, dataset_path, classname, train_split, multiclass=False, samples_per_defect=None, total_folds=4, train_keep_ratio=1.0):
        """
        Args:
            dataset_path (string): Path to the dataset directory
            classname (string): Name of the class/dataset
            train_split (float): Proportion of data to use for training
            multiclass (bool): Whether to use multiclass classification
            samples_per_defect (int): Number of samples per defect class
            total_folds (int): Total number of folds for cross-validation
            train_keep_ratio (float): Proportion of training samples to keep in each fold (between 0 and 1)
        
        Returns: A train set with good and defect samples, a validation set with good and defect samples, and a test set with good and defect samples.
        """
        self.dataset_path = dataset_path
        self.classname = classname
        self.root_dir = os.path.join(dataset_path, classname)
        self.train_split = train_split
        self.defect_classes = self.defect_classes()
        self.multiclass = multiclass
        self.class_mapping = {}
        self.samples_per_defect = samples_per_defect
        self.total_folds = total_folds
        self.train_keep_ratio = train_keep_ratio
        
        self.no_defect_samples = glob.glob(os.path.join(self.root_dir, 'train/good/*.png')) + glob.glob(os.path.join(self.root_dir, 'train/good/*.jpg'))
        self.nb_no_defect_samples = len(self.no_defect_samples)
        self.nb_defect_samples = 0
        self.defect_samples = {}

        self.train = []
        self.val = []
        self.test = []

        self.create_samples()
        self.supervised_train_test()
    
    def create_samples(self):
      self.class_mapping["0"] = "No defect"

      for i, defect_class in enumerate(self.defect_classes):
        image_path = os.path.join(self.root_dir, "test/", defect_class)
        mask_path = os.path.join(self.root_dir, "ground_truth/", defect_class)

        images = glob.glob(os.path.join(image_path, '*.png')) + glob.glob(os.path.join(image_path, '*.jpg'))
        images.sort()

        masks = glob.glob(os.path.join(mask_path, '*.png')) + glob.glob(os.path.join(mask_path, '*.jpg'))
        masks.sort()

        self.defect_samples[str(i+1)] = []
        if self.multiclass:
            for image, mask in zip(images, masks):
              self.defect_samples[str(i+1)].append((image, mask, i+1))

            self.class_mapping[str(i+1)] = defect_class
        else:
          for image, mask in zip(images, masks):
            self.defect_samples[str(i+1)].append((image, mask, 1))
            self.class_mapping["1"] = "Defect"
        self.nb_defect_samples += len(self.defect_samples[str(i+1)])

    def defect_classes(self):
      all_files = glob.glob(os.path.join(self.root_dir, '*/*'))
      pattern = re.compile(r'good|ground_truth')
      filtered_files = [file for file in all_files if not pattern.search(file)]
      last_subfolders = [os.path.basename(path) for path in filtered_files]
      return last_subfolders

    def supervised_train_test(self):
        """
        Creates k deterministic folds with balanced distributions of good and defect samples.
        Each fold maintains the same proportion of samples across classes.
        Returns a list of k folds, where each fold is a tuple of (train, val) splits.
        """
        folds = []
        
        # Shuffle and split no defect samples
        no_defect_samples = self.no_defect_samples.copy()
        np.random.seed(42)  # For reproducibility
        np.random.shuffle(no_defect_samples)
        
        # Create k folds for no defect samples
        no_defect_folds = []
        total_samples = len(no_defect_samples)
        base_fold_size = total_samples // self.total_folds
        remainder = total_samples % self.total_folds
        
        current_idx = 0
        for i in range(self.total_folds):
            # Add one extra sample to earlier folds if there's a remainder
            current_fold_size = base_fold_size + (1 if i < remainder else 0)
            
            val_fold = no_defect_samples[current_idx:current_idx + current_fold_size]
            train_fold = no_defect_samples[:current_idx] + no_defect_samples[current_idx + current_fold_size:]
            
            # Apply train_keep_ratio only to good samples
            if self.train_keep_ratio < 1.0:
                keep_size = int(len(train_fold) * self.train_keep_ratio)
                train_fold = train_fold[:keep_size]
            
            no_defect_folds.append((train_fold, val_fold))
            current_idx += current_fold_size
        
        # Handle defect samples for each class
        defect_folds = []
        for i, defect_class in enumerate(self.defect_classes):
            samples = self.defect_samples[str(i+1)]
            np.random.shuffle(samples)
            
            # Create k folds for defect samples
            class_folds = []
            total_samples = len(samples)
            
            # Calculate validation size (5% of total samples)
            val_size = max(1, int(total_samples * 0.05))  # At least 1 sample for validation
            train_size = total_samples - val_size
            
            for j in range(self.total_folds):
                # For each fold, take 5% of samples for validation
                val_fold = samples[:val_size]
                train_fold = samples[val_size:]
                
                # Apply samples_per_defect limit to training set if specified
                if self.samples_per_defect is not None:
                    train_fold = train_fold[:self.samples_per_defect]
                
                class_folds.append((train_fold, val_fold))
            defect_folds.append(class_folds)
        
        # Combine folds
        for fold_idx in range(self.total_folds):
            train_fold = []
            val_fold = []
            
            # Add no defect samples for this fold
            train_fold.extend(no_defect_folds[fold_idx][0])
            val_fold.extend(no_defect_folds[fold_idx][1])
            
            # Add defect samples for this fold
            for class_idx in range(len(self.defect_classes)):
                train_fold.extend(defect_folds[class_idx][fold_idx][0])
                val_fold.extend(defect_folds[class_idx][fold_idx][1])
            
            # Shuffle the folds
            np.random.shuffle(train_fold)
            np.random.shuffle(val_fold)
            
            folds.append((train_fold, val_fold))
            
            # Log distribution for verification
            if self.multiclass:
                train_classes = Counter(item[2] for item in train_fold)
                val_classes = Counter(item[2] for item in val_fold)
                print(f"\nFold {fold_idx + 1} distribution:")
                print("Training set class distribution:", train_classes)
                print("Validation set class distribution:", val_classes)
                print(f"Defect ratio in validation: {sum(1 for x in val_fold if x[2] != 0) / len(val_fold):.3f}")
        
        # Use the first fold for the default train/val/test split
        self.train = folds[0][0]  # First fold's training set
        self.val = folds[0][1]    # First fold's validation set
        
        # For test set, use the validation set from the second fold
        if len(folds) > 1:
            self.test = folds[1][1]
        else:
            # If there's only one fold, split the validation set
            val_size = len(self.val)
            split_idx = val_size // 2
            self.test = self.val[split_idx:]
            self.val = self.val[:split_idx]
        
        return folds
    
    def plot_dist(self, fold=None):
        """
        Plot the distribution of samples in a specific fold or in the default train/val/test splits.
        
        Args:
            fold (tuple, optional): A tuple containing (train_fold, val_fold) to plot. If None, plots the default splits.
        """
        plt.figure(figsize=(12, 5))
        
        if fold is not None:
            train_fold, val_fold = fold
            fig, ax = plt.subplots(1, 2, figsize=(10, 5), sharey=True)
            datasets = [train_fold, val_fold]
            titles = ["Train Fold", "Val Fold"]
        else:
            fig, ax = plt.subplots(1, 3, figsize=(12, 5), sharey=True)
            datasets = [self.train, self.val, self.test]
            titles = ["Train", "Val", "Test"]
        
        global_dist = []
        for i, (dataset_part, title) in enumerate(zip(datasets, titles)):
            labels = []
            for item in dataset_part:
                if isinstance(item, tuple):
                    if len(item) == 3:  # (image_path, mask_path, label)
                        labels.append(item[2])
                    elif len(item) == 2:  # (image_path, label)
                        labels.append(item[1])
                else:
                    labels.append(0)  # "good" images are labeled as 0
            
            # Count the occurrences of each label
            label_counts = Counter(labels)
            print(f"\nDistribution for {title}:")
            print(f"Label counts: {dict(label_counts)}")

            # Plotting the stacked bar chart
            labels_list = sorted(list(label_counts.keys()))
            counts_list = [label_counts[label] for label in labels_list]
            global_dist.append(counts_list)

            # Use a colormap that's visually distinct
            colors = plt.cm.get_cmap('tab10', len(labels_list))
            color_list = [colors(i) for i in range(len(labels_list))]

            ax[i].bar(labels_list, counts_list, color=color_list)
            ax[i].set_xlabel('Labels')
            ax[i].set_ylabel('Frequency')
            ax[i].set_title(title)

            # Create legend with proper class names
            patches = []
            for label in labels_list:
                class_name = self.class_mapping.get(str(label), f"Class {label}")
                patches.append(mpatches.Patch(color=color_list[labels_list.index(label)], 
                                           label=class_name))
            
            # Add legend to the last subplot only to avoid duplication
            if i == len(datasets) - 1:
                plt.legend(handles=patches, title="Classes", bbox_to_anchor=(1.05, 1), 
                          loc='upper left')

        plt.tight_layout()
        plt.show()
        plt.close()  # Close the figure to free memory
        return global_dist
    
    def return_splits(self):
        return self.train, self.val, self.test


class MVTech_Unsupervised_Split():
    def __init__(self, dataset_path, classname, seed=42, defect_ratio=0.05):
        """
        Args:
            dataset_path (string): Path to the dataset directory
            classname (string): Name of the class/dataset
            seed (int): Random seed for deterministic splits
            defect_ratio (float): Proportion of defects in validation set (0.0 to 1.0).
                                 E.g., 0.05 means 5% defects, 95% good samples.
        
        Returns: A train set with remaining good samples, a validation set with good and defect samples.
                 Validation set contains exactly one example per defect type, with good examples 
                 representing (1-defect_ratio) of the validation dataset.
        """
        self.dataset_path = dataset_path
        self.classname = classname
        self.root_dir = os.path.join(dataset_path, classname)
        self.seed = seed
        self.defect_ratio = defect_ratio
        self.defect_classes = self.defect_classes()
        self.class_mapping = {"0": "good", "1": "defect"}
        
        # Set random seed for deterministic behavior
        np.random.seed(self.seed)
        
        # Get all good samples from train directory
        self.train_good_samples = glob.glob(os.path.join(self.root_dir, 'train/good/*.png')) + \
                                  glob.glob(os.path.join(self.root_dir, 'train/good/*.jpg'))
        
        # Shuffle good samples deterministically
        np.random.shuffle(self.train_good_samples)
        
        # Get all test defect samples
        self.test_defect_samples = []
        
        self.train = []
        self.val = []
        
        self.create_samples()
        
    def defect_classes(self):
        """Get all defect class names from the test directory"""
        test_dir = os.path.join(self.root_dir, 'test')
        if not os.path.exists(test_dir):
            return []
        
        all_subdirs = [d for d in os.listdir(test_dir) 
                       if os.path.isdir(os.path.join(test_dir, d)) and d != 'good']
        return all_subdirs
    
    def create_samples(self):
        """Create train and validation splits for unsupervised learning"""
        
        # Collect all defect samples first (exactly one per defect type)
        defect_samples_by_class = {}
        selected_defect_samples = []
        
        for defect_class in self.defect_classes:
            mask_path = os.path.join(self.root_dir, "ground_truth/", defect_class)
            defect_path = os.path.join(self.root_dir, 'test', defect_class)
            defect_samples = glob.glob(os.path.join(defect_path, '*.png')) + \
                           glob.glob(os.path.join(defect_path, '*.jpg'))
            defect_samples.sort()
            defect_masks = glob.glob(os.path.join(mask_path, '*.png')) + \
                           glob.glob(os.path.join(mask_path, '*.jpg'))
            defect_masks.sort()
            
            class_samples = list(zip(defect_samples, defect_masks))
            defect_samples_by_class[defect_class] = class_samples
            
            # Take exactly one sample per defect class (randomly selected based on seed)
            if class_samples:
                # Shuffle the class samples using the seed to make selection deterministic
                np.random.shuffle(class_samples)
                selected_defect_samples.append(class_samples[0])
        
        num_defect_samples = len(selected_defect_samples)
        
        # Calculate number of good samples needed based on defect_ratio
        # If defect_ratio is the proportion of defects in validation set, then:
        # defect_ratio = num_defect_samples / (num_defect_samples + num_good_samples)
        # Solving for num_good_samples:
        # defect_ratio * (num_defect_samples + num_good_samples) = num_defect_samples
        # defect_ratio * num_defect_samples + defect_ratio * num_good_samples = num_defect_samples
        # defect_ratio * num_good_samples = num_defect_samples * (1 - defect_ratio)
        # num_good_samples = num_defect_samples * (1 - defect_ratio) / defect_ratio
        
        if self.defect_ratio <= 0 or self.defect_ratio >= 1:
            raise ValueError(f"defect_ratio must be between 0 and 1, got {self.defect_ratio}")
        
        num_good_samples_needed = int(num_defect_samples * (1 - self.defect_ratio) / self.defect_ratio)
        
        # Check if we have enough good samples
        if len(self.train_good_samples) < num_good_samples_needed:
            raise ValueError(f"Not enough good samples available. Need {num_good_samples_needed}, "
                           f"but only have {len(self.train_good_samples)} good samples.")
        
        # Select good samples for validation (first num_good_samples_needed from shuffled list)
        val_good_samples = self.train_good_samples[:num_good_samples_needed]
        
        # Remaining good samples go to training set
        train_good_samples = self.train_good_samples[num_good_samples_needed:]
        
        # Create training set with remaining good samples
        for sample_path in train_good_samples:
            self.train.append((sample_path, 0))  # (image_path, label)
        
        # Create validation set with selected good samples
        for sample_path in val_good_samples:
            self.val.append((sample_path, 0))
        
        # Add selected defect samples to validation set
        for sample_path, mask_path in selected_defect_samples:
            self.val.append((sample_path, mask_path, 1))  # All defects labeled as 1 for binary classification
            self.test_defect_samples.append((sample_path, mask_path))
        
        # Shuffle validation set
        np.random.shuffle(self.val)
        
        # Calculate actual ratios
        actual_defect_ratio = len(self.test_defect_samples) / len(self.val) if len(self.val) > 0 else 0
        actual_good_ratio = len(val_good_samples) / len(self.val) if len(self.val) > 0 else 0
        
        print(f"Training samples (good): {len(self.train)}")
        print(f"Validation samples (good): {len(val_good_samples)}")
        print(f"Validation samples (defect): {len(self.test_defect_samples)} (1 per defect type)")
        print(f"Total validation samples: {len(self.val)}")
        print(f"Actual good ratio in validation: {actual_good_ratio:.3f} (target: {1-self.defect_ratio:.3f})")
        print(f"Actual defect ratio in validation: {actual_defect_ratio:.3f} (target: {self.defect_ratio:.3f})")
        print(f"Defect types: {self.defect_classes}")
    
    def plot_dist(self):
        """Plot the distribution of samples in train and validation sets"""
        
        fig, ax = plt.subplots(1, 2, figsize=(10, 5), sharey=True)
        datasets = [self.train, self.val]
        titles = ["Train", "Validation"]
        
        global_dist = []
        for i, (dataset_part, title) in enumerate(zip(datasets, titles)):
            labels = [item[1] for item in dataset_part]  # Extract labels
            
            # Count the occurrences of each label
            label_counts = Counter(labels)
            
            # Plotting the bar chart
            labels_list = list(label_counts.keys())
            counts_list = list(label_counts.values())
            global_dist.append(counts_list)
            
            colors = ['green' if label == 0 else 'red' for label in labels_list]
            
            ax[i].bar([self.class_mapping[str(label)] for label in labels_list], 
                     counts_list, color=colors)
            ax[i].set_xlabel('Classes')
            ax[i].set_ylabel('Frequency')
            ax[i].set_title(title)
        
        # Create legend
        good_patch = mpatches.Patch(color='green', label='Good')
        defect_patch = mpatches.Patch(color='red', label='Defect')
        plt.legend(handles=[good_patch, defect_patch], title="Classes")
        
        plt.tight_layout()
        plt.show()
        return global_dist
    
    def return_splits(self):
        """Return the train and validation splits"""
        return self.train, self.val
    
    def get_stats(self):
        """Return statistics about the dataset splits"""
        train_stats = Counter([item[1] for item in self.train])
        val_stats = Counter([item[1] for item in self.val])
        
        stats = {
            'train': {
                'total': len(self.train),
                'good': train_stats.get(0, 0),
                'defect': train_stats.get(1, 0)
            },
            'val': {
                'total': len(self.val),
                'good': val_stats.get(0, 0),
                'defect': val_stats.get(1, 0)
            }
        }
        
        return stats