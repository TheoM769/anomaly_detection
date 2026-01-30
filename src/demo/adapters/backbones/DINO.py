import torch

from transformers import AutoModel, AutoImageProcessor
from core.ports.vision_backbone import VisionBackbone

class DinoBackbone(VisionBackbone):
    def __init__(self, checkpoint_name="facebook/dinov2-base"):
        self.checkpoint_name = checkpoint_name
        super().__init__()

    def load_weights(self):
        self.processor = AutoImageProcessor.from_pretrained(self.checkpoint_name)
        self.model = AutoModel.from_pretrained(self.checkpoint_name)

        self.hidden_size = self.model.config.hidden_size
        self.patch_size = self.model.config.patch_size
        self.num_layers = self.model.config.num_hidden_layers

    def eval(self):
        self.model.eval()

    def image_feature_extraction(self, image):
        self.model.eval()
        with torch.no_grad():
            inputs = self.processor(image, return_tensors="pt")
            outputs = self.model(**inputs)
            return outputs.pooler_output
    
    def patch_feature_extraction(self, image):
        self.model.eval()
        with torch.no_grad():   
            inputs = self.processor(image, return_tensors="pt")
            outputs = self.model(**inputs)
            return outputs.last_hidden_state
    
    def feature_map_extraction(self, image):
        self.model.eval()
        with torch.no_grad():
            inputs = self.processor(image, return_tensors="pt")
            outputs = self.model(**inputs, output_hidden_states=True)
            return outputs.hidden_states
