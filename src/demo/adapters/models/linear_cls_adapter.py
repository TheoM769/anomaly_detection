import torch
import numpy as np
import json
import time

from core.ports.model import Model
from core.ports.backbone import Backbone
from adapters.types.classification_result import ClassificationResult

class LinearClsModelAdapter(Model):
    def __init__(self, backbone: Backbone, checkpoint: str, num_classes: int, dataset_name: str):
        self.backbone = backbone
        self.classifier = None
        self.checkpoint = checkpoint
        self.num_classes = num_classes
        self.dataset_name = dataset_name
        self.id2labels = None

        self._load_weights()
        self._get_labels_mapping()

    def _load_weights(self):
        self.backbone.load_weights()
        self.classifier = torch.nn.Linear(self.backbone.hidden_size, self.num_classes)
        state_dict = torch.load(self.checkpoint, weights_only=True, map_location=torch.device('cpu'))
        
        new_state_dict = {}
        for key, value in state_dict.items():
            if key == 'classifier.bias':
                new_key = 'bias'
                new_state_dict[new_key] = value
            elif key == 'classifier.weight':
                new_key = 'weight'
                new_state_dict[new_key] = value
            
        self.classifier.load_state_dict(new_state_dict)
    
    def _eval(self):
        self.backbone.eval()
        self.classifier.eval()

    def _get_labels_mapping(self):
        with open("infrastructure/config/labels_mapping.json", "r") as f:
            labels_mapping = json.load(f)
            str_key_id2labels = labels_mapping[self.dataset_name]["id2labels"]
            int_keys_id2labels = {int(k): v for k, v in str_key_id2labels.items()}
        self.id2labels = int_keys_id2labels
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        image_feature = self.backbone.image_feature_extraction(x)
        logits = self.classifier(image_feature)
        return logits
    
    def inference(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self._eval()
        with torch.no_grad():
            logits = self.forward(x)
            scores = torch.nn.functional.softmax(logits, dim=1)
            score, predicted_class = torch.max(scores, dim=1)
            return predicted_class,score
    
    def predict(self, image: np.array) -> ClassificationResult:
        prediction, score = self.inference(image)
        return ClassificationResult(self.id2labels[prediction.item()], score.item())
    
    def predict_with_inference_time(self, image: np.array):
        start_time = time.time()
        prediction = self.predict(image)
        end_time = time.time()
        inference_time = end_time - start_time
        return prediction, inference_time