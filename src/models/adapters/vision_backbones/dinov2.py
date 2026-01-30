import cv2
import torch
import torchvision.models as models
# import clip
from PIL import Image
from torchvision import transforms
from sklearn.decomposition import PCA
import numpy as np

from src.models.core.ports.backbone import PatchBackbone
from src.models.core.services import MODEL_REGISTRY

# DINOv2 Wrapper
@MODEL_REGISTRY.register_backbone()
class DINOv2Backbone(PatchBackbone):
    def __init__(self, model_name, resolution=224, half_precision=False):
        self.half_precision = half_precision
        super().__init__(model_name, resolution)

    def load_model(self):
        if self.model_name in ["dinov2_vits14", "dinov2_vitb14", "dinov2_vitl14", "dinov2_vitg14"]:
            model = torch.hub.load('facebookresearch/dinov2', self.model_name)
        else:
            raise ValueError(f"Unknown DINOv2 model name: {self.model_name}")
        
        self.hidden_size = model.embed_dim
        self.patch_size = model.patch_size
        self.smaller_edge_size = self.resolution - self.resolution % self.patch_size
        
        # Set transform for DINOv2
        transform = transforms.Compose([
            transforms.Resize(size=self.smaller_edge_size, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), # imagenet defaults
            ])
        
        return model, transform
    
    def get_grid_size(self):
        cropped_width = cropped_height =  self.resolution - self.resolution % self.model.patch_size # Crop image to dimensions that are a multiple of the patch size
        return (cropped_height // self.model.patch_size, cropped_width // self.model.patch_size)
    
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
    
    def extract_patch_features(self, image_tensor, strip_cls_token=True):
        with torch.inference_mode():
            image_batch = image_tensor.unsqueeze(0) if image_tensor.ndim == 3 else image_tensor
            if self.half_precision:
                image_batch = image_batch.half()
            tokens = self.model.get_intermediate_layers(image_batch, return_class_token=True)
            if strip_cls_token:
                tokens = tokens[0][0]
            else:
                tokens = torch.cat([tokens[0][1].unsqueeze(0), tokens[0][0]], dim=1)
        return tokens
        
    def extract_image_features(self, image_tensor):
        with torch.inference_mode():
            return self(image_tensor).cpu()
    
    def forward(self, image_tensor):
        image_batch = image_tensor.unsqueeze(0) if image_tensor.ndim == 3 else image_tensor
        if self.half_precision:
            image_batch = image_batch.half()
        tokens = self.model.get_intermediate_layers(image_batch, return_class_token=True)[0][1]
        return tokens