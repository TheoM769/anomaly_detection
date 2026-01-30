from core.ports.model_prediction import ModelPrediction

class ClassificationResult(ModelPrediction):
    
    def __init__(self, predicted_class: str, score: float):
        self.predicted_class = predicted_class
        self.score = score

    def __str__(self):
        return f"Predicted class: {self.predicted_class}, Score: {self.score:.4f}"

    def __repr__(self):
        return f"ClassificationResult(predicted_class={self.predicted_class}, score={self.score:.4f})"