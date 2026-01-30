from typing import Union, List
import time

import torch
import numpy as np
from PIL import Image

from core.ports.model import Model
from core.ports.backbone import Backbone
from adapters.types.classification_result import ClassificationResult

class CLIPClsModelAdapter(Model):
    def __init__(self, backbone: Backbone):
        self.clip_backbone = backbone

        self._load_weights()

    def _load_weights(self):
        self.clip_backbone.load_weights()
    
    def _eval(self):
        self.clip_backbone.eval()

    def forward(self, text: List[str], image: Union[np.array, Image.Image]) -> torch.Tensor:
        image_features = self.clip_backbone.image_feature_extraction(image)
        text_features = self.clip_backbone.text_feature_extraction(text)

        similarity = (image_features @ text_features.T)
        cosine_similarity = similarity / (
            torch.norm(image_features, dim=-1).unsqueeze(0).T @ torch.norm(text_features, dim=-1).unsqueeze(0)
        )
        return cosine_similarity
    
    def inference(self, text: List[str], image: Union[np.array, Image.Image]) -> tuple[torch.Tensor, torch.Tensor]:
        self._eval()
        with torch.no_grad():
            logits = self.forward(text, image)
            scores = torch.nn.functional.softmax(logits, dim=1)
            score, predicted_class = torch.max(scores, dim=1)
            return predicted_class, score
        
    def predict(self, text: List[str], image: Union[np.array, Image.Image]) -> ClassificationResult:
        prediction, score = self.inference(text, image)
        return ClassificationResult(text[prediction.item()], score.item())
        
    def predict_with_inference_time(self, text: List[str], image: Union[np.array, Image.Image]) -> tuple[ClassificationResult, float]:
        """
        Run inference and measure the time it takes.
        
        Args:
            text: List of text prompts to compare against the image
            image: The image to classify
            
        Returns:
            A tuple containing the classification result and the inference time in seconds
        """
        start_time = time.time()
        result = self.predict(text, image)
        inference_time = time.time() - start_time
        return result, inference_time