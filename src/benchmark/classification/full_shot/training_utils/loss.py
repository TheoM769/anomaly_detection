import typing

import torch.nn as nn
import torch

class Loss():
    
    def __init__(self, selected_loss):
        self.selected_loss = selected_loss
    
    def compute_loss(self, pred, labels):
        pass

        
class ClassificationLoss(Loss):
    
    # CE -> Cross-Entropy
    def __init__(self, selected_loss: str, weights: list[float], device):
        super().__init__(selected_loss)
        self.weights = torch.tensor(weights).to(device)
        
        if self.selected_loss == "CE":
            self.compute_loss = nn.CrossEntropyLoss(weight=self.weights)
        else:
            print("Select a valid loss")
    
    