import torch

from core.ports.model_prediction import ModelPrediction

class MemoryBankResult(ModelPrediction):
    
    def __init__(self, predicted_class: torch.Tensor):
        self.predicted_class = predicted_class
        self.class_mapping = {
            0: "normal",
            1: "defect"
        }
        self.human_readable_predicted_class = None
        self.make_predictions_human_readable()

    def make_predictions_human_readable(self):
        if self.predicted_class.ndim > 0:
            self.human_readable_predicted_class = [self.class_mapping[predicted_class.item()] for predicted_class in self.predicted_class]
        else:
            self.human_readable_predicted_class = self.class_mapping[self.predicted_class.item()]

    def __str__(self):
        return f"Predicted class: {self.human_readable_predicted_class}"

    def __repr__(self):
        return f"MemoryBankResult(predicted_class={self.human_readable_predicted_class})"
    
    def __getitem__(self, index):
        if self.predicted_class.ndim > 0:
            return self.human_readable_predicted_class[index]
        else:
            if index == 0:
                return self.human_readable_predicted_class
            else:
                raise IndexError("MemoryBankResult only has one item")
    
    def __len__(self):
        return len(self.human_readable_predicted_class)