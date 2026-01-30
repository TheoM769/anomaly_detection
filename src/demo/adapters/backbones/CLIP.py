import torch
from transformers import AutoProcessor, CLIPModel

from core.ports.vision_langage_backbone import VisionLangageBackbone

class CLIPBackbone(VisionLangageBackbone):
    def __init__(self, checkpoint_name="openai/clip-vit-base-patch32"):
        self.checkpoint_name = checkpoint_name
        super().__init__()

    def load_weights(self):
        self.processor = AutoProcessor.from_pretrained(self.checkpoint_name)
        self.model = CLIPModel.from_pretrained(self.checkpoint_name)

    def eval(self):
        self.model.eval()

    def image_feature_extraction(self, image):
        self.model.eval()
        with torch.no_grad():
            inputs = self.processor(images=image, return_tensors="pt")
            image_features = self.model.get_image_features(**inputs)
            return image_features
    
    def text_feature_extraction(self, text):
        self.model.eval()
        with torch.no_grad():
            if isinstance(text, str):
                text_inputs = self.processor([text], padding=True, return_tensors="pt")
            else:
                text_inputs = self.processor(text, padding=True, return_tensors="pt")
            text_features = self.model.get_text_features(**text_inputs)
            return text_features
