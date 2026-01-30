import torch
import torchvision.models as models
import numpy as np
from PIL import Image
from sklearn.decomposition import PCA
import torch.nn as nn
from torchvision.models.vision_transformer import interpolate_embeddings

from src.models.core.ports.backbone import PatchBackbone
from src.models.core.services import MODEL_REGISTRY

@MODEL_REGISTRY.register_backbone()
class ViTBackbone(PatchBackbone):
    def __init__(self, model_name, resolution=224):
        super().__init__(model_name, resolution)
        
    def load_model(self):
        if self.model_name == "vit_b_16":
            model = models.vit_b_16(image_size=self.resolution)
            model_state_dict = models.ViT_B_16_Weights.IMAGENET1K_V1.get_state_dict()
            transform = models.ViT_B_16_Weights.IMAGENET1K_V1.transforms()
            patch_size = 16
        elif self.model_name == "vit_b_32":
            model = models.vit_b_32(image_size=self.resolution)
            model_state_dict = models.ViT_B_32_Weights.IMAGENET1K_V1.get_state_dict()
            transform = models.ViT_B_32_Weights.DEFAULT.transforms()
            patch_size = 32
        elif self.model_name == "vit_l_16":
            model = models.vit_l_16(image_size=self.resolution)
            model_state_dict = models.ViT_L_16_Weights.IMAGENET1K_V1.get_state_dict()
            transform = models.ViT_L_16_Weights.DEFAULT.transforms()
            patch_size = 16
        elif self.model_name == "vit_l_32":
            model = models.vit_l_32(image_size=self.resolution)
            model_state_dict = models.ViT_L_32_Weights.IMAGENET1K_V1.get_state_dict()
            transform = models.ViT_L_32_Weights.DEFAULT.transforms()
            patch_size = 32
        else:
            raise ValueError(f"Unknown ViT model name: {self.model_name}")
        
        if self.resolution % patch_size != 0:
            raise ValueError(f"Resolution {self.resolution} must be divisible by patch size {patch_size}")
        
        grid_dim = self.resolution // patch_size
        self.grid_size = (grid_dim, grid_dim)
        
        adapted_model_state_dict = interpolate_embeddings(self.resolution, patch_size, model_state_dict)
        model.load_state_dict(adapted_model_state_dict)
        transform.crop_size = [self.resolution]
        transform.resize_size = [self.resolution]

        self.hidden_size = model.hidden_dim
        self.patch_size = model.patch_size
        model.heads = nn.Identity()  # Output will be [batch_size, hidden_size]

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
    def extract_image_features(self, img_tensor):
        with torch.inference_mode():
            return self(img_tensor).cpu()
    
    def extract_patch_features(self, img_tensor, strip_cls_token=False):
        with torch.no_grad():
            image_batch = img_tensor.unsqueeze(0) if img_tensor.ndim == 3 else img_tensor
            patches = self.model._process_input(image_batch)
            class_token = self.model.class_token.expand(patches.size(0), -1, -1)
            patches = torch.cat((class_token, patches), dim=1)
            patch_features = self.model.encoder(patches).cpu()
            return patch_features[:, 1:, :] if strip_cls_token else patch_features

    def forward(self, img):
        image_batch = img.unsqueeze(0) if img.ndim == 3 else img
        image_features = self.model(image_batch)
        return image_features