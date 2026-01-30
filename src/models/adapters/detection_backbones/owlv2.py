from PIL import Image, ImageDraw
import torch
import torch.nn as nn
from transformers import Owlv2Processor, Owlv2ForObjectDetection

class Owlv2(nn.Module):
    def __init__(self):
        super(Owlv2, self).__init__()
        self.processor = Owlv2Processor.from_pretrained("google/owlv2-large-patch14")
        self.model = Owlv2ForObjectDetection.from_pretrained("google/owlv2-large-patch14")

    def forward(self, texts, image):
        inputs = self.processor(text=texts, images=image, return_tensors="pt", padding=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
        # Target image sizes (height, width) to rescale box predictions [batch_size, 2]
        target_sizes = torch.Tensor([image.size[::-1]])
        # Convert outputs (bounding boxes and class logits) to Pascal VOC Format (xmin, ymin, xmax, ymax)
        results = self.processor.post_process_object_detection(outputs=outputs, target_sizes=target_sizes, threshold=0.1)
        return results

    def visualize_results(self, texts,results, image):
        draw = ImageDraw.Draw(image)

        for box, label in zip(results[0]["boxes"], results[0]["labels"]):
            x1, y1, x2, y2 = tuple(box) # Pascal VOC Format (xmin, ymin, xmax, ymax)
            draw.rectangle((x1, y1, x2, y2), outline="red", width=3)
            draw.text((x1, y1), texts[0][label], fill="green")

        return image