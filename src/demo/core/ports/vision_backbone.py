from abc import abstractmethod
from typing import Union, Tuple

from PIL import Image
import numpy as np
import torch

from core.ports.backbone import Backbone

class VisionBackbone(Backbone):
    def __init__(self):
        self.processor = None
        self.model = None

        self.hidden_size = None
        self.patch_size = None
        self.num_layers = None

    @abstractmethod
    def image_feature_extraction(self, image: Union[Image.Image, np.ndarray]) -> torch.Tensor:
        pass

    @abstractmethod
    def patch_feature_extraction(self, image: Union[Image.Image, np.ndarray]) -> torch.Tensor:
        pass

    @abstractmethod
    def feature_map_extraction(self, image: Union[Image.Image, np.ndarray]) -> Tuple[torch.Tensor]:
        pass