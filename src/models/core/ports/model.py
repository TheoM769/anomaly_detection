from abc import ABC, abstractmethod

class Model(ABC):
    def __init__(self):
        self.device = "cpu"

    def release_resources(self):
        if hasattr(self, 'model'):
            del self.model
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            import gc
            gc.collect()
    
    def to(self, device):
        self.model.to(device)
        self.device = device
        return self
    
    def eval(self):
        self.model.eval()
        return self