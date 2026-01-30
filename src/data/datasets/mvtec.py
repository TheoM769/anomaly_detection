from torch.utils.data import Dataset
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision.transforms.v2 as v2
import torchvision.transforms as T
import random
from torchvision import transforms

# Use CPU for random operations - much faster than MPS transfers
device = torch.device("cpu")

class PilSaltPepperNoise:
    """Optimized salt and pepper noise augmentation for PIL images."""
    
    def __init__(self, noise_prob=0.01):
        self.noise_prob = noise_prob
    
    def __call__(self, img):
        """Apply salt and pepper noise to PIL image using CPU operations."""
        # Work directly with numpy - much faster than tensor operations
        img_array = np.array(img)
        
        # Generate random noise mask on CPU
        noise_mask = np.random.random(img_array.shape[:2])
        
        # Apply salt noise (white pixels)
        salt_mask = noise_mask < self.noise_prob / 2
        img_array[salt_mask] = 255
        
        # Apply pepper noise (black pixels)  
        pepper_mask = (noise_mask >= self.noise_prob / 2) & (noise_mask < self.noise_prob)
        img_array[pepper_mask] = 0
        
        return Image.fromarray(img_array)

class PilGaussianNoise:
    """Optimized Gaussian noise augmentation for PIL images."""
    
    def __init__(self, mean=0, std=0.1):
        self.mean = mean
        self.std = std
    
    def __call__(self, img):
        """Apply Gaussian noise to PIL image using CPU numpy operations."""
        # Work directly with numpy - much faster
        img_array = np.array(img, dtype=np.float32)
        
        # Generate Gaussian noise on CPU
        noise = np.random.normal(self.mean, self.std * 255, img_array.shape)
        
        # Add noise to image
        noisy_img = img_array + noise
        
        # Clip values to valid range [0, 255] and convert back to uint8
        noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)
        
        return Image.fromarray(noisy_img)

class PilCutMix:
    """Optimized CutMix augmentation for PIL images."""
    
    def __init__(self, max_area_ratio=0.05, avoid_center=True, center_margin_ratio=0.25):
        self.max_area_ratio = max_area_ratio
        self.avoid_center = avoid_center
        self.center_margin_ratio = center_margin_ratio
    
    def __call__(self, img):
        """Apply cutmix to PIL image using CPU numpy operations."""
        img_array = np.array(img)
        H, W = img_array.shape[:2]
        
        # Calculate maximum cut size using CPU operations
        max_area = self.max_area_ratio * H * W
        max_side = int(np.sqrt(max_area))
        
        # Random cut dimensions using CPU
        cut_w = np.random.randint(max_side//2, max_side+1)
        cut_h = np.random.randint(max_side//2, max_side+1)
        
        if self.avoid_center:
            # Define center region to avoid
            center_x, center_y = W // 2, H // 2
            margin_x = int(W * self.center_margin_ratio / 2)
            margin_y = int(H * self.center_margin_ratio / 2)
            
            # Generate coordinates avoiding center region - max 10 attempts
            for _ in range(10):
                cx = np.random.randint(0, W)
                cy = np.random.randint(0, H)
                
                # Check if box overlaps with center region
                if not (center_x - margin_x <= cx <= center_x + margin_x and 
                       center_y - margin_y <= cy <= center_y + margin_y):
                    break
        else:
            cx = np.random.randint(0, W)
            cy = np.random.randint(0, H)
        
        # Calculate bounding box coordinates
        bbx1 = max(0, cx - cut_w // 2)
        bby1 = max(0, cy - cut_h // 2)
        bbx2 = min(W, cx + cut_w // 2)
        bby2 = min(H, cy + cut_h // 2)
        
        # Create random patch using numpy
        if len(img_array.shape) == 3:  # RGB image
            random_patch = np.random.randint(0, 256, (bby2-bby1, bbx2-bbx1, img_array.shape[2]), dtype=np.uint8)
        else:  # Grayscale image
            random_patch = np.random.randint(0, 256, (bby2-bby1, bbx2-bbx1), dtype=np.uint8)
        
        # Apply cutmix
        img_copy = img_array.copy()
        img_copy[bby1:bby2, bbx1:bbx2] = random_patch
        
        return Image.fromarray(img_copy)

class ImageDataset(Dataset):
    """
    Dataset to store images.
    """
    def __init__(self, samples, transform=None):
        """
        Args:
            samples (list): List of image paths.
            transform (callable, optional): Optional transform to be applied.
        """
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            images = []
            for i in range(idx.start, idx.stop, idx.step if idx.step else 1):
                image = self.__getitem__(i)
                images.append(image)
            return np.array(images)
        else:
            img_path = self.samples[idx]
            image = Image.open(img_path).convert('RGB')
            image = np.array(image)

        return image


class RandomNoiseAugmentation(torch.nn.Module):
    def __init__(self, model_transform):
        super().__init__()

        self.cutmix = T.Lambda(lambda x: self._cutmix(x))
        self.model_transform = model_transform

        # Reduced kernel sizes for much better performance
        self.transforms = [
            v2.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
            v2.GaussianBlur(kernel_size=(15, 15), sigma=(2, 5)),  # Reduced from 31x31
            v2.GaussianBlur(kernel_size=(25, 25), sigma=(5, 10)),  # Reduced from 51x51
            PilSaltPepperNoise(noise_prob=0.20),
            PilGaussianNoise(mean=0, std=0.2),
            PilCutMix()
        ]

    def _cutmix(self, img):
        C, H, W = img.shape
        
        # Keep tensor on CPU for better performance with small operations
        
        # Maximum size is 5% of image area - use numpy for speed
        max_area = 0.05 * H * W
        max_side = int(np.sqrt(max_area))
        
        # Random size up to max_side - use numpy
        cut_w = np.random.randint(max_side//2, max_side+1)
        cut_h = np.random.randint(max_side//2, max_side+1)

        # Define center region to avoid (25% of image around center)
        center_x, center_y = W // 2, H // 2
        margin_x, margin_y = W // 4, H // 4
        
        # Generate coordinates avoiding center region - limit to 10 attempts
        for _ in range(10):
            cx = np.random.randint(0, W)
            cy = np.random.randint(0, H)
            
            # Check if box overlaps with center region
            if not (center_x - margin_x <= cx <= center_x + margin_x and 
                   center_y - margin_y <= cy <= center_y + margin_y):
                break

        bbx1 = max(0, cx - cut_w // 2)
        bby1 = max(0, cy - cut_h // 2)  
        bbx2 = min(W, cx + cut_w // 2)
        bby2 = min(H, cy + cut_h // 2)

        # Create random patch scaled between 0 and 1
        random_patch = torch.rand(C, bby2-bby1, bbx2-bbx1)
        
        # Apply cutmix
        img_copy = img.clone()
        img_copy[:, bby1:bby2, bbx1:bbx2] = random_patch
        
        return img_copy

    def _get_available_transforms(self, label):
        """Get available transforms based on label. Cutmix is only applied to normal samples (label=0)."""
        if label == 0:
            return self.transforms
        else:
            return [t for t in self.transforms if t != self.cutmix]
    
    def _is_visual_transform(self, transform):
        """Check if transform is a visual augmentation that should be applied before normalization."""
        return isinstance(transform, (v2.ColorJitter, v2.GaussianBlur, PilSaltPepperNoise))
    
    def _get_transforms(self, transform, img):
        """Create transform compositions for visual augmentations (ColorJitter, GaussianBlur)."""
        if isinstance(self.model_transform, transforms.Compose):
            # Model transform is a Compose object - we can access its transforms
            half_transform = v2.Compose([
                #*self.model_transform.transforms[:-2],  # All transforms except normalization
                transform,
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                #self.model_transform.transforms[-2],
            ])
            full_transform = v2.Compose([
                transform,
                #self.model_transform.transforms[-1],  # Add normalization back
                self.model_transform,
            ])
        else:
            # Model transform is a single transform object - extract properties manually
            half_transform = v2.Compose([
                # v2.Resize(self.model_transform.resize_size, interpolation=self.model_transform.interpolation),
                # v2.CenterCrop(self.model_transform.crop_size),
                transform,
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
            ])
            full_transform = v2.Compose([
                transform,
                self.model_transform,
            ])
        
        return half_transform(img), full_transform(img)

    def forward(self, img, label):
        """
        Apply random augmentation to the input image.
        
        Args:
            img: Input image
            label: Image label (0 for normal, non-zero for anomaly)
            
        Returns:
            tuple: (half_transformed_image, full_transformed_image)
        """
        # Select available transforms based on label
        available_transforms = self._get_available_transforms(label)
        transform = random.choice(available_transforms)
        
        return self._get_transforms(transform, img)
    
class ImageLabelDataset(Dataset):
    """
    Dataset to store images labels pairs.
    """
    def __init__(self, samples, transform=None, read_with_masks=True, return_original_image_with_transform=False):
        """
        Args:
            root_dir (string): Path to either 'train' or 'test' directory
            transform (callable, optional): Optional transform to be applied
                on a sample.
            read_with_masks (bool, optional): Whether to read the masks from samples, without returning it in the output.
        """
        self.samples = samples
        self.read_with_masks = read_with_masks
        self.transform = transform
        self.return_original_image_with_transform = return_original_image_with_transform
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if isinstance(self.samples[idx], tuple):
            if self.read_with_masks: 
                img_path, _, label = self.samples[idx]
                image = Image.open(img_path).convert('RGB')
            else:
                img_path, label = self.samples[idx]
                image = Image.open(img_path).convert('RGB')
        else:
            image = Image.open(self.samples[idx]).convert('RGB')
            label = 0

        if self.transform:
            if isinstance(self.transform, RandomNoiseAugmentation):
                image, transformed_image = self.transform(image, label)
            else:
                t = v2.Compose([
                    v2.ToImage(),
                    v2.ToDtype(torch.float32, scale=True),
                ])
                transformed_image = self.transform(image)
                image = t(image)

            if self.return_original_image_with_transform:
                return image, transformed_image, label
            else:
                return transformed_image, label
        return image, label

class ImageMaskLabelDataset(Dataset):
    """
    Dataset to store images, masks and labels.
    """
    def __init__(self, samples, transform=None):
        """
        Args:
            root_dir (string): Path to either 'train' or 'test' directory
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if len(self.samples[idx]) == 3:
            img_path, mask_path, label = self.samples[idx]
            image = Image.open(img_path).convert('RGB')
            mask = plt.imread(mask_path)
            mask = np.expand_dims(mask, axis=0)
        else:
            img_path, label = self.samples[idx]
            image = Image.open(img_path).convert('RGB')
            mask = np.ones((1, image.size[0], image.size[1]))

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        return image, mask, label