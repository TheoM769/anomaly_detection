import torch
from src.models.core.services import MODEL_REGISTRY
from src.models.core.ports.model import Model
from src.models.core.ports.backbone import VLBackbone
from typing import Union, List
from PIL import Image

@MODEL_REGISTRY.register_model()
class ZeroShotClassifier(Model):
    def __init__(self, backbone: VLBackbone, use_patch_features=False):
        super(ZeroShotClassifier, self).__init__()
        self.model = backbone
        self.use_patch_features = use_patch_features
    
    def compute_class_probs(self, text: Union[str, List[str]], image: Union[str, Image.Image]):
        if self.use_patch_features:
            return self._compute_class_probs_with_image_features(text, image)
        else:
            pass
    
    def _compute_class_probs_with_image_features(self, text: Union[str, List[str]], image: Union[str, Image.Image]):
        features_processed = self.model.prepare_text_image(text, image)
        extracted_output = self.model.extract_features(features_processed)

        if isinstance(extracted_output, tuple) and len(extracted_output) == 3:
            image_features, text_features, logit_scale = extracted_output
            probs = (logit_scale * image_features @ text_features.T).softmax(dim=-1)
        elif isinstance(extracted_output, tuple) and len(extracted_output) == 2:
            image_features, text_features = extracted_output
            probs = (image_features @ text_features.T).softmax(dim=-1)
        else:
            raise ValueError(f"Unexpected output from backbone.extract_features: {extracted_output}")

        return probs

    def _compute_class_probs_with_patch_features(self, text: Union[str, List[str]], image: Union[str, Image.Image], masking: str =None):
        image_processed = self.model.prepare_image(image)
        text_processed = self.model.prepare_text(text)
        pass

    def classify_binary(self, text: str, image: Union[str, Image.Image], threshold: float = 0.5):
        if isinstance(text, list) and len(text) > 1:
            raise ValueError("The model can only discriminate from a single class. Use classify_multiclass instead.")
        probs = self.compute_class_probs(text, image)
        return (probs[:, 0] > threshold).float()
    
    def classify_multiclass(self, text: Union[str, List[str]], image: Union[str, Image.Image]):
        probs = self.compute_class_probs(text, image)
        return probs.argmax(dim=-1).item()
    
    def __repr__(self):
        return f"ZeroShotClassifier(model={self.model})"