from PIL import Image
import numpy as np
import torch
from sklearn.decomposition import PCA
import cv2
from types import MethodType
import torch.nn as nn
from typing import Tuple

from src.models.core.ports.backbone import VLBackbone
import src.models.adapters.vision_backbones.perception_models_2.core.vision_encoder.pe as pe
import src.models.adapters.vision_backbones.perception_models_2.core.vision_encoder.transforms as transforms
from src.models.core.services import MODEL_REGISTRY

@MODEL_REGISTRY.register_backbone()
class PEVLBackbone(VLBackbone):
    def __init__(self, model_name):
        super(PEVLBackbone, self).__init__(model_name)
        self.model.eval()

    def load_model(self):
        if self.model_name == 'PE-Core-G14-448':
            model = pe.CLIP.from_config("PE-Core-G14-448", pretrained=True)
        elif self.model_name == 'PE-Core-L14-336': # Not used for classification, or use a feature pooler (no CLS token for this model)
            model = pe.CLIP.from_config("PE-Core-L14-336", pretrained=True)
        elif self.model_name == 'PE-Core-B16-224':
            model = pe.CLIP.from_config("PE-Core-B16-224", pretrained=True)

        self.image_transform = transforms.get_image_transform(model.image_size)
        self.tokenizer = transforms.get_text_tokenizer(model.context_length)
        self.hidden_size = model.output_dim

        return model
    
    def prepare_text(self, text):
        if isinstance(text, str):
            text_inputs = self.tokenizer([text])
        else:
            text_inputs = self.tokenizer(text)
        return text_inputs.to(self.device)
    
    def prepare_image(self, image):
        image_processed = self.image_transform(image)
        return image_processed.to(self.device)
    
    def prepare_text_image(self, text, image):
        image_processed = self.prepare_image(image)
        text_processed = self.prepare_text(text)
        return image_processed, text_processed
    
    def extract_image_features(self, image_processed):
        image_batch = image_processed.unsqueeze(0) if image_processed.ndim == 3 else image_processed
        with torch.inference_mode():
            return self.model.encode_image(image_batch).cpu()
    
    def extract_patch_features(self, image_processed, strip_cls_token=True):
        with torch.inference_mode():
            image_batch = image_processed.unsqueeze(0) if image_processed.ndim == 3 else image_processed
            # if self.half_precision:
            #     image_batch = image_batch.half()
            tokens = self.model.visual.forward_features(image_batch).cpu()
        return tokens[:, 1:, :] if strip_cls_token else tokens

    def extract_text_features(self, text_processed):
        with torch.inference_mode():
            return self.model.encode_text(text_processed).cpu()
    
    def extract_features(self, image_processed, text_processed):
        """
        features_processed: Tuple containing (image_processed, text_processed)
        """
        image_batch = image_processed.unsqueeze(0) if image_processed.ndim == 3 else image_processed
        with torch.inference_mode():
            image_features, text_features, logit_scale = self(image_batch, text_processed)
            return image_features.cpu(), text_features.cpu(), logit_scale.cpu()
    
    def forward(self, image_processed, text_processed):
        """
        features_processed: Tuple containing (image_processed, text_processed)
        """
        return self.model(image_processed, text_processed)