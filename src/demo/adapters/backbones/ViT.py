import torch

from transformers import ViTModel, ViTImageProcessor
from core.ports.vision_backbone import VisionBackbone

class ViTBackbone(VisionBackbone):
    def __init__(self, checkpoint_name="google/vit-base-patch16-224"):
        self.checkpoint_name = checkpoint_name
        super().__init__()

    def load_weights(self):
        self.processor = ViTImageProcessor.from_pretrained(self.checkpoint_name)
        self.model = ViTModel.from_pretrained(self.checkpoint_name, ignore_mismatched_sizes=True)

        self.hidden_size = self.model.config.hidden_size
        self.patch_size = self.model.config.patch_size
        self.num_layers = self.model.config.num_hidden_layers

    def eval(self):
        self.model.eval()

    def image_feature_extraction(self, image):
        with torch.no_grad():
            inputs = self.processor(image, return_tensors="pt")
            outputs = self.model(**inputs)
            cls_token = outputs.last_hidden_state[:, 0, :] # Model trained on cls token, not on pooler output
            return cls_token
    
    def patch_feature_extraction(self, image):
        with torch.no_grad():   
            inputs = self.processor(image, return_tensors="pt")
            outputs = self.model(**inputs)
            return outputs.last_hidden_state
    
    def feature_map_extraction(self, image):
        with torch.no_grad():
            inputs = self.processor(image, return_tensors="pt")
            outputs = self.model(**inputs, output_hidden_states=True)
            return outputs.hidden_states