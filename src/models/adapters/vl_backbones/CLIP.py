import torch
from transformers import AutoProcessor, CLIPModel
import numpy as np
from PIL import Image

from src.models.core.ports.backbone import VLBackbone

class CLIPBackbone(VLBackbone):
    def __init__(self, model_name):
        super().__init__(model_name)
        self.model.eval()

    def load_model(self):
        if self.model_name == "vitb32":
            self.transform = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
            model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        elif self.model_name == "vitl14":
            self.transform = AutoProcessor.from_pretrained("openai/clip-vit-large-patch14")
            model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        else:
            raise ValueError(f"Model name {self.model_name} not supported")
        return model

    def prepare_image(self, image):
        if isinstance(image, str):
            image = [Image.open(image).convert("RGB")]
        elif isinstance(image, Image.Image):
            image = [image]
        elif isinstance(image, list):
            if isinstance(image[0], str):
                image = [Image.open(img).convert("RGB") for img in image]
        processed_image = self.processor(images=image, return_tensors="pt")
        return processed_image.to(self.device)
    
    def prepare_text(self, text):
        if isinstance(text, str):
            text_inputs = self.processor([text], padding=True, return_tensors="pt")
        else:
            text_inputs = self.processor(text, padding=True, return_tensors="pt")
        return text_inputs.to(self.device)
    
    def prepare_text_image(self, text, image):
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        if isinstance(text, str):
            text = self.processor([text], padding=True, return_tensors="pt")
        else:
            text = self.processor(text, padding=True, return_tensors="pt")
        inputs = self.processor(images=image, text=text, return_tensors="pt")
        return inputs.to(self.device)
    
    def extract_image_features(self, image_processed):
        self.model.eval()
        with torch.no_grad():
            image_features = self.model.get_image_features(**image_processed)
            return image_features.cpu()
    
    def extract_text_features(self, text_processed):
        self.model.eval()
        with torch.no_grad():
            text_features = self.model.get_text_features(**text_processed)
            return text_features.cpu()
    
    def extract_patch_features(self, image_processed):
        pass
    
    def extract_features(self, features_processed):
        self.model.eval()
        with torch.no_grad():
            features = self.model.get_image_text_embeds(**features_processed)
            return features.image_embeds.cpu(), features.text_embeds.cpu()