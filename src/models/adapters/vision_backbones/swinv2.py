import torch
import torchvision.models as models
import numpy as np
from PIL import Image
from sklearn.decomposition import PCA
import torch.nn as nn

from src.models.core.ports.backbone import PatchBackbone
from src.models.core.services import MODEL_REGISTRY

@MODEL_REGISTRY.register_backbone()
class SwinBackbone(PatchBackbone):
    def __init__(self, model_name, resolution=224):
        super().__init__(model_name, resolution)
        
    def load_model(self):
        if self.model_name == "swin_v2_t":
            model = models.swin_v2_t(weights = models.Swin_V2_T_Weights.DEFAULT)
            transform = models.Swin_V2_T_Weights.DEFAULT.transforms()
        elif self.model_name == "swin_v2_s":
            model = models.swin_v2_s(weights = models.Swin_V2_S_Weights.DEFAULT)
            transform = models.Swin_V2_S_Weights.DEFAULT.transforms()
        elif self.model_name == "swin_v2_b":
            model = models.swin_v2_b(weights = models.Swin_V2_B_Weights.DEFAULT)
            transform = models.Swin_V2_B_Weights.DEFAULT.transforms()
        else:
            raise ValueError(f"Unknown Swin model name: {self.model_name}")
        
        transform.crop_size = [self.resolution]
        if self.resolution <= 224:
            transform.resize_size = [self.resolution+2]
        elif self.resolution <= 384:
            transform.resize_size = [self.resolution + 4]
        else:
            transform.resize_size = [self.resolution + 8]
        
        self.hidden_size = model.head.in_features
        self.patch_size = self.resolution // 32
        model.head = nn.Identity()  # Output will be [batch_size, hidden_size]

        return model, transform
    
    def get_grid_size(self):
        """Compute grid size based on resolution where patch size = resolution / 32"""
        grid_size = self.resolution // self.patch_size
        return (grid_size, grid_size)
    
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
                img = Image.fromarray(img)
                image_tensor = self.transform(img).unsqueeze(0)
            elif img.ndim == 4:
                img = [Image.fromarray(i) for i in img]
                image_tensor = torch.stack([self.transform(img) for img in img])
            else:
                raise ValueError(f"Unsupported image type: {type(img)}")
        else:
            raise ValueError(f"Unsupported image type: {type(img)}")

        return image_tensor.to(self.device)
    
    def extract_image_features(self, img_tensor):
        with torch.inference_mode():
            image_batch = img_tensor.unsqueeze(0) if img_tensor.ndim == 3 else img_tensor
            return self.model(image_batch)
    
    def extract_patch_features(self, img_tensor, strip_cls_token=False):
        with torch.inference_mode():
            image_batch = img_tensor.unsqueeze(0) if img_tensor.ndim == 3 else img_tensor
            x1 = self.model.features(image_batch)
            x2 = self.model.norm(x1)
            return x2
    
    def forward(self, img_tensor):
        image_batch = img_tensor.unsqueeze(0) if img_tensor.ndim == 3 else img_tensor
        return self.model(image_batch)