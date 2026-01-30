from abc import ABC, abstractmethod
from typing import Tuple, Union, List

from PIL import Image
import torch.nn as nn
import torch
import numpy as np
import cv2
from sklearn.decomposition import PCA

class Backbone(ABC, nn.Module):
    def __init__(self, model_name, resolution):
        super(Backbone, self).__init__()
        self.resolution = resolution
        self.model_name = model_name
        self.device = "cpu"
        self.grid_size = None

    def load_model(self):
        pass

    def eval(self):
        self.model.eval()
        return self
    
    def to(self, device):
        self.device = device
        self.model.to(device)
        return self

class VisionBackbone(Backbone):

    def __init__(self, model_name, resolution=224):
        super(VisionBackbone, self).__init__(model_name, resolution)
        self.model, self.transform = self.load_model()

    def prepare_image(self, img):
        if isinstance(img, str):
            img = Image.open(img).convert("RGB")
            image_tensor = self.transform(img).unsqueeze(0)
        elif isinstance(img, Image.Image):
            image_tensor = self.transform(img).unsqueeze(0)
        elif isinstance(img, list):
            if isinstance(img[0], str):
                image_tensor = torch.stack([self.transform(Image.open(image).convert("RGB")) for image in img])
            elif isinstance(img[0], Image.Image):
                image_tensor = torch.stack([self.transform(image) for image in img])
            else:
                raise ValueError(f"Unsupported image type: {type(img[0])}")
        elif isinstance(img, np.ndarray):
            if img.ndim == 3:
                image_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
            elif img.ndim == 4:
                image_tensor = torch.from_numpy(img).permute(0, 3, 1, 2)
            else:
                raise ValueError(f"Unsupported image type: {type(img)}")
        
        return image_tensor.to(self.device)
    
    def extract_image_features(self, img):
        pass

class PatchBackbone(VisionBackbone):

    def __init__(self, model_name, resolution=224):
        super(PatchBackbone, self).__init__(model_name, resolution)
        self.grid_size = self.get_grid_size()
    
    def extract_patch_features(self, img_tensor: torch.tensor, masking: str) -> torch.tensor:
        pass

    # def get_embedding_visualization(self, tokens, grid_size = (14,14), resized_mask=None, normalize=True):
    #     pass

    def compute_background_mask(self, img_features, threshold = 10, kernel_size = 2, border = 0.2):
        # Kernel size for morphological operations should be odd
        pca = PCA(n_components=1, svd_solver='randomized')
        first_pc = pca.fit_transform(img_features.astype(np.float32))
        mask = first_pc > threshold
        # test whether the center crop of the images is kept (adaptive masking), adapt if your objects of interest are not centered!
        m = mask.reshape(self.grid_size)[int(self.grid_size[0] * border):int(self.grid_size[0] * (1-border)), int(self.grid_size[1] * border):int(self.grid_size[1] * (1-border))]
        if m.sum() <=  m.size * 0.35:
            mask = - first_pc > threshold
        # postprocess mask, fill small holes in the mask, enlarge slightly
        mask = cv2.dilate(mask.astype(np.uint8), np.ones((kernel_size, kernel_size), np.uint8)).astype(bool)
        mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((kernel_size, kernel_size), np.uint8)).astype(bool)
        return mask.squeeze()

class VLBackbone(Backbone):

    def __init__(self, model_name, resolution=224):
        super(VLBackbone, self).__init__(model_name, resolution)
        self.model = self.load_model()

    def prepare_image(self, img):
        pass

    def prepare_text(self, text):
        pass

    def extract_image_features(self, img_tensor):
        pass

    def extract_patch_features(self, img):
        pass

    def extract_text_features(self, text_tensor):
        pass

    def extract_features(self, img_tensor, text_tensor):
        pass