import torch
from transformers import Owlv2Processor, Owlv2ForObjectDetection
from PIL import Image, ImageDraw, ImageFont

class Owlv2Backbone():
    def __init__(self, checkpoint_name="google/owlv2-large-patch14"):
        self.checkpoint_name = checkpoint_name
        self.processor = None
        self.model = None

    def load_weights(self):
        self.processor = Owlv2Processor.from_pretrained(self.checkpoint_name)
        self.model = Owlv2ForObjectDetection.from_pretrained(self.checkpoint_name)

    def eval(self):
        self.model.eval()

    def detect_objects(self, texts, image, threshold=0.1, topk=3):
        inputs = self.processor(text=texts, images=image, return_tensors="pt", padding=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
        # Target image sizes (height, width) to rescale box predictions [batch_size, 2]
        target_sizes = torch.Tensor([image.size[::-1]])
        # Convert outputs (bounding boxes and class logits) to Pascal VOC Format (xmin, ymin, xmax, ymax)
        results = self.processor.post_process_object_detection(outputs=outputs, target_sizes=target_sizes, threshold=threshold)
        indices = torch.argsort(results[0]["scores"], dim=0, descending=True)[:topk]
        new_results = {
            "labels": results[0]["labels"][indices],
            "boxes": results[0]["boxes"][indices],
            "scores": results[0]["scores"][indices]
        }
        return new_results

    def visualize_results(self, texts, results, image):
        draw = ImageDraw.Draw(image)
        # Create a font object with larger size
        try:
            font = ImageFont.truetype("Arial", 20)  # Try to use Arial with size 20
        except:
            font = ImageFont.load_default()  # Fallback to default font if Arial is not available

        for box, label, score in zip(results["boxes"], results["labels"], results["scores"]):
            x1, y1, x2, y2 = tuple(box) # Pascal VOC Format (xmin, ymin, xmax, ymax)
            draw.rectangle((x1, y1, x2, y2), outline="red", width=3)
            text = f"{texts[label]} ({score:.2f})"
            draw.text((x1, y1), text, fill="black", font=font)

        return image