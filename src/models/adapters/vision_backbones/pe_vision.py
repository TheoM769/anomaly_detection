from PIL import Image
import numpy as np
import torch
from sklearn.decomposition import PCA
import cv2
from types import MethodType
import torch.nn as nn

from src.models.core.ports.backbone import PatchBackbone
import src.models.adapters.vision_backbones.perception_models_2.core.vision_encoder.pe as pe
import src.models.adapters.vision_backbones.perception_models_2.core.vision_encoder.transforms as transforms
from src.models.core.services import MODEL_REGISTRY

@MODEL_REGISTRY.register_backbone()
class PEVisionBackbone(PatchBackbone):
    def __init__(self, model_name, resolution=224):
        super().__init__(model_name, resolution)

    def load_model(self):
        if self.model_name == 'PE-Spatial-G14-448': # Not used for classification, or use a feature pooler (no CLS token for this model)
            model = pe.VisionTransformer.from_config(self.model_name, pretrained=True)
            self.grid_size = (34,34)
            self.patch_size = 14
        elif self.model_name == 'PE-Core-G14-448':
            model = pe.CLIP.from_config("PE-Core-G14-448", pretrained=True)
            self.grid_size = (34,34)
            self.patch_size = 14
        elif self.model_name == 'PE-Core-L14-336':
            model = pe.CLIP.from_config("PE-Core-L14-336", pretrained=True)
            self.grid_size = (24,24)
            self.patch_size = 14
        elif self.model_name == 'PE-Core-B16-224':
            model = pe.CLIP.from_config("PE-Core-B16-224", pretrained=True)
            self.grid_size = (14,14)
            self.patch_size = 16
        else:
            raise ValueError(f"Unknown PE Vision model name: {self.model_name}")
        
        transform = transforms.get_image_transform(model.image_size)
        self.hidden_size = model.output_dim
        return model, transform
    
    def get_grid_size(self):
        cropped_width = cropped_height =  self.resolution - self.resolution % self.patch_size # Crop image to dimensions that are a multiple of the patch size
        return (cropped_height // self.patch_size, cropped_width // self.patch_size)
    
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
        
        # Crop image to dimensions that are a multiple of the patch size
        height, width = image_tensor.shape[2:] # B x C x H x W
        cropped_width, cropped_height = width - width % self.patch_size, height - height % self.patch_size
        image_tensor = image_tensor[:, :cropped_height, :cropped_width]

        return image_tensor.to(self.device)
    
    def extract_image_features(self, image_tensor):
        with torch.inference_mode():
            return self(image_tensor).cpu()
    
    def extract_patch_features(self, image_tensor, strip_cls_token=False):
        with torch.inference_mode():
            image_batch = image_tensor.unsqueeze(0) if image_tensor.ndim == 3 else image_tensor
            # if self.half_precision:
            #     image_batch = image_batch.half()
            tokens = self.model.visual.forward_features(image_batch).cpu()
        return tokens[:, 1:, :] if strip_cls_token else tokens
    
    def forward(self, img):
        image_batch = img.unsqueeze(0) if img.ndim == 3 else img
        # if self.half_precision:
        #     image_batch = image_batch.half()
        tokens = self.model(image_batch)[0]
        return tokens