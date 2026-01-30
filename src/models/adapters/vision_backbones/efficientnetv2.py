import torch
import torchvision.models as models
import numpy as np
from PIL import Image
from sklearn.decomposition import PCA
import torch.nn as nn

from src.models.core.ports.backbone import VisionBackbone
from src.models.core.services import MODEL_REGISTRY

@MODEL_REGISTRY.register_backbone()
class EfficientNetV2Backbone(VisionBackbone):
    def __init__(self, model_name, resolution=224):
        super().__init__(model_name, resolution)
        
    def load_model(self):
        if self.model_name == "efficientnet_v2_s":
            model = models.efficientnet_v2_s(weights = models.EfficientNet_V2_S_Weights.DEFAULT)
            transform = models.EfficientNet_V2_S_Weights.DEFAULT.transforms()
        elif self.model_name == "efficientnet_v2_m":
            model = models.efficientnet_v2_m(weights = models.EfficientNet_V2_M_Weights.DEFAULT)
            transform = models.EfficientNet_V2_M_Weights.DEFAULT.transforms()
        elif self.model_name == "efficientnet_v2_l":
            model = models.efficientnet_v2_l(weights = models.EfficientNet_V2_L_Weights.DEFAULT)
            transform = models.EfficientNet_V2_L_Weights.DEFAULT.transforms()
        else:
            raise ValueError(f"Unknown EfficientNetV2 model name: {self.model_name}")
        
        transform.crop_size = [self.resolution]
        transform.resize_size = [self.resolution]
        
        self.hidden_size = model.features[-1][0].out_channels
        
        model.classifier = nn.Identity()

        return model, transform
    
    def extract_image_features(self, img_tensor):
        with torch.inference_mode():
            return self(img_tensor)

    def forward(self, img_tensor):
        return self.model(img_tensor)

    def compute_background_mask(self, img_features, grid_size, threshold = 10, masking_type = False):
        raise NotImplementedError("EfficientNetV2 does not support background masking")