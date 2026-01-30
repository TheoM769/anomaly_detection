from abc import abstractmethod
from typing import Union, Tuple, List

from PIL import Image
import numpy as np
import torch

from core.ports.backbone import Backbone

class VisionLangageBackbone(Backbone):
    def __init__(self):
        self.processor = None
        self.model = None

    @abstractmethod
    def image_feature_extraction(self, image: Union[Image.Image, np.ndarray]) -> torch.Tensor:
        pass

    @abstractmethod
    def text_feature_extraction(self, text: Union[str, List[str]]) -> torch.Tensor:
        pass
