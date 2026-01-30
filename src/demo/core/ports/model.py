from abc import ABC, abstractmethod
import time

import numpy as np

from core.ports.model_prediction import ModelPrediction

class Model(ABC):
    @abstractmethod
    def predict(self, image_data: np.array) -> ModelPrediction:
        pass
    
    @abstractmethod
    def predict_with_inference_time(self, image_data: np.array) -> tuple[ModelPrediction, float]:
        pass
    
    def release_resources(self):
        # Libération explicite des ressources
        if hasattr(self, 'model'):
            del self.model
            # Si votre framework a des fonctions spécifiques pour libérer la mémoire GPU
            # comme torch.cuda.empty_cache(), appelez-les ici
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            import gc
            gc.collect()
