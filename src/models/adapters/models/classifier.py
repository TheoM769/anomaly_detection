import torch
import torch.nn as nn

from src.models.core.ports.backbone import Backbone
from src.models.core.services import MODEL_REGISTRY
from src.models.core.ports.model import Model

@MODEL_REGISTRY.register_model()
class Classifier(nn.Module, Model):
    def __init__(self, backbone: Backbone, num_classes: int = 2, fine_tune_backbone: bool = False, head: str = "linear", dropout: float = 0.5):
        super(Classifier, self).__init__()
        self.num_classes = num_classes
        self.backbone = backbone
        self.fine_tune_backbone = fine_tune_backbone
        self.head = head
        self.dropout = dropout
        self.transform = self.backbone.transform

        self.model = self._get_model()
    
    def _create_classifier(self):
        if self.head == "linear":
            classifier = nn.Linear(self.backbone.hidden_size, self.num_classes)
        elif self.head == "MLP":
            classifier = nn.Sequential(
                nn.Linear(self.backbone.hidden_size, 512),
                nn.ReLU(),
                nn.Dropout(self.dropout),
                nn.Linear(512, self.num_classes)
            )
        else:
            raise ValueError(f"Unknown head type: {self.head}")
    
        return classifier
    
    def _get_model(self):

        backbone = self.backbone
        classifier = self._create_classifier()

        model = nn.Sequential(
            backbone,
            classifier
        )

        if self.fine_tune_backbone:
            for param in self.get_backbone_parameters_from_model(model):
                param.requires_grad = True
        else:
            for param in self.get_backbone_parameters_from_model(model):
                param.requires_grad = False

        for param in self.get_classifier_parameters_from_model(model):
            param.requires_grad = True
        
        return model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        with torch.inference_mode():
            return self.model(x).cpu()
    
    def get_backbone_parameters_from_model(self, model: nn.Module):
        return model[:-1].parameters()
    
    def get_classifier_parameters_from_model(self, model: nn.Module):
        return model[-1].parameters()
    
    def get_backbone_parameters(self):
        return self.model[:-1].parameters()
    
    def get_classifier_parameters(self):
        return self.model[-1].parameters()
    
    def __repr__(self):
        return repr(self.model)