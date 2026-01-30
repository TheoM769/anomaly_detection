from typing import Tuple

import torch
import torch.nn as nn
from torchvision import models
from PIL import Image
import numpy as np

from src.models.core.services import MODEL_REGISTRY
from src.models.core.ports.backbone import PatchBackbone

class HierarchicalResNet50Backbone(nn.Module):
    """Internal model class that implements the forward method"""
    def __init__(self, resnet):
        super(HierarchicalResNet50Backbone, self).__init__()
        
        # Group all model components in a ModuleDict
        self.components = nn.ModuleDict({
            # Feature extraction layers
            'layer0': nn.Sequential(
                resnet.conv1,
                resnet.bn1,
                resnet.relu,
                resnet.maxpool
            ),
            'layer1': resnet.layer1,  # features de bas niveau
            'layer2': resnet.layer2,  # features de niveau intermédiaire
            'layer3': resnet.layer3,  # features de haut niveau
            'layer4': resnet.layer4,  # features de très haut niveau
            
            # Weights for each channel
            'attention1': nn.Sequential(
                nn.Conv2d(256, 1, kernel_size=1),
                nn.Sigmoid()
            ),
            'attention2': nn.Sequential(
                nn.Conv2d(512, 1, kernel_size=1),
                nn.Sigmoid()
            ),
            'attention3': nn.Sequential(
                nn.Conv2d(1024, 1, kernel_size=1),
                nn.Sigmoid()
            ),
            'attention4': nn.Sequential(
                nn.Conv2d(2048, 1, kernel_size=1),
                nn.Sigmoid()
            ),
            
            # Adaptive pooling layers (reduce to 1 scalar per feature channel --> B x C vector per layer)
            'adaptive_pool1': nn.AdaptiveAvgPool2d((1, 1)),
            'adaptive_pool2': nn.AdaptiveAvgPool2d((1, 1)),
            'adaptive_pool3': nn.AdaptiveAvgPool2d((1, 1)),
            'adaptive_pool4': nn.AdaptiveAvgPool2d((1, 1)),
        })
    
    def forward_features(self, x):
        x0 = self.components['layer0'](x)
        
        x1 = self.components['layer1'](x0)
        att1 = self.components['attention1'](x1)
        x1_att = x1 * att1  # Weight features before pooling
        #x1_pool = self.components['adaptive_pool1'](x1_att).flatten(1)
        
        x2 = self.components['layer2'](x1)
        att2 = self.components['attention2'](x2)
        x2_att = x2 * att2
        #x2_pool = self.components['adaptive_pool2'](x2_att).flatten(1)
        
        x3 = self.components['layer3'](x2)
        att3 = self.components['attention3'](x3)
        x3_att = x3 * att3
        #x3_pool = self.components['adaptive_pool3'](x3_att).flatten(1)
        
        x4 = self.components['layer4'](x3)
        att4 = self.components['attention4'](x4)
        x4_att = x4 * att4
        #x4_pool = self.components['adaptive_pool4'](x4_att).flatten(1)

        return {
            "layer1": x1_att,
            "layer2": x2_att,
            "layer3": x3_att,
            "layer4": x4_att
        }
    
    def forward(self, x):
        
        x0 = self.components['layer0'](x)

        x1 = self.components['layer1'](x0)
        att1 = self.components['attention1'](x1)
        x1_att = x1 * att1  # Weight features before pooling
        x1_pool = self.components['adaptive_pool1'](x1_att).flatten(1)
        
        x2 = self.components['layer2'](x1)
        att2 = self.components['attention2'](x2)
        x2_att = x2 * att2
        x2_pool = self.components['adaptive_pool2'](x2_att).flatten(1)
        
        x3 = self.components['layer3'](x2)
        att3 = self.components['attention3'](x3)
        x3_att = x3 * att3
        x3_pool = self.components['adaptive_pool3'](x3_att).flatten(1)
        
        x4 = self.components['layer4'](x3)
        att4 = self.components['attention4'](x4)
        x4_att = x4 * att4
        x4_pool = self.components['adaptive_pool4'](x4_att).flatten(1)
        # Concaténation des features de différents niveaux
        features_concat = torch.cat((x1_pool, x2_pool, x3_pool, x4_pool), dim=1)
        
        # Return features (classifier would be added separately if needed)
        return features_concat

@MODEL_REGISTRY.register_backbone()
class HierarchicalResNet(PatchBackbone):
    def __init__(self, model_name, resolution=224):
        super(HierarchicalResNet, self).__init__(model_name, resolution)
        
    def load_model(self):
        if self.model_name == "resnet50":
            resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            transform = models.ResNet50_Weights.DEFAULT.transforms()
        elif self.model_name == "wide_resnet50":
            resnet = models.wide_resnet50_2(weights=models.Wide_ResNet50_2_Weights.DEFAULT)
            transform = models.Wide_ResNet50_2_Weights.DEFAULT.transforms()
        else:
            raise ValueError(f"Unknown ResNet model name: {self.model_name}")
        
        # Create the model with proper forward method
        transform.crop_size = [self.resolution]
        transform.resize_size = [self.resolution]
        model = HierarchicalResNet50Backbone(resnet)
        self.hidden_size = 256 + 512 + 1024 + 2048

        return model, transform
    
    def get_grid_size(self):
        if self.resolution % 8 != 0:
            raise ValueError(f"Resolution {self.resolution} is not divisible by 8")
        return (self.resolution // 8, self.resolution // 8)
    
    def forward(self, x):
        return self.model(x)
    
    def extract_image_features(self, img):
        with torch.inference_mode():
            return self(img).cpu()
    
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
    
    def _fit_embedding_dim(self, concat_features, embed_dim):
        # Permute to move channel dimension to the end: (b, c, w, h) -> (b, w, h, c)
        concat_features = concat_features.permute(0, 2, 3, 1)
        b, w, h, c = concat_features.shape
        flattened = concat_features.view(b * w * h, c)
        
        # Apply adaptive pooling along the channel dimension
        adaptive_pool = nn.AdaptiveAvgPool1d(embed_dim)
        pooled = adaptive_pool(flattened.unsqueeze(0)).squeeze()  # (b*w*h, embed_dim)

        # Reshape back to spatial dimensions: (b*w*h, embed_dim) -> (b, w, h, embed_dim)
        pooled_features = pooled.view(b, w, h, embed_dim)
        
        # Permute back to standard format: (b, w, h, embed_dim) -> (b, embed_dim, w, h)
        pooled_features = pooled_features.permute(0, 3, 1, 2)
        return pooled_features
    
    def extract_patch_features(self, img, strip_cls_token=False, layer_to_extract_from=["layer2", "layer3"], patch_size=3, patch_stride=1, embed_dim=256):
        features = self.model.forward_features(img)
        mid_features = {key: value for key, value in features.items() if key in layer_to_extract_from}

        if isinstance(patch_size, int):
            calculated_padding = (patch_size - 1) // 2
        elif isinstance(patch_size, tuple) and len(patch_size) == 2:
            pad_h = (patch_size[0] - 1) // 2
            pad_w = (patch_size[1] - 1) // 2
            calculated_padding = (pad_h, pad_w)
        else:
            raise ValueError(
                f"patch_size must be an int or a tuple of two ints. Got {type(patch_size)}"
            )

        pooling_layer = nn.AvgPool2d(kernel_size=patch_size, stride=patch_stride, padding=calculated_padding)
        features_list = []
        max_resolution = mid_features["layer2"].shape[2]
        for _, value in mid_features.items():
            value = pooling_layer(value)
            if value.shape[2] < max_resolution:
                rescaled_value = nn.functional.interpolate(value, size=(max_resolution, max_resolution), mode="bilinear", align_corners=False)
                features_list.append(rescaled_value)
            else:
                features_list.append(value)
        concat_features = torch.cat(features_list, dim=1)
        pooled_features = self._fit_embedding_dim(concat_features, embed_dim)
        return pooled_features
    
    def get_model_parameters(self):
        return self.model.parameters()