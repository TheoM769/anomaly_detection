import unittest
import os
import typing

from dotenv import load_dotenv
import numpy as np
from PIL import Image
import torch

from adapters.backbones.CLIP import CLIPBackbone

load_dotenv()
IMAGE_PATH = os.getenv("IMAGE_PATH")

class TestCLIP(unittest.TestCase):

    def setUp(self):
        self.backbone = CLIPBackbone()
        self.backbone.load_weights()
        self.image = Image.open(IMAGE_PATH + "/bottle/train/good/000.png")
        self.np_image = np.array(self.image)
        self.prompts = ["a good bottle", "a bad bottle"]

        self.final_hidden_size = self.backbone.model.config.projection_dim
        self.model_image_size = (self.backbone.processor.image_processor.crop_size["height"], self.backbone.processor.image_processor.crop_size["width"])
        
    def test_processor(self):
        text_inputs = self.backbone.processor(self.prompts, return_tensors="pt")
        image_inputs = self.backbone.processor(images=self.image, return_tensors="pt")
        
        self.assertEqual(text_inputs["input_ids"].shape, (len(self.prompts), 5))
        self.assertEqual(image_inputs["pixel_values"].shape, (1, 3, self.model_image_size[0], self.model_image_size[1]))
    
    def test_image_feature_extraction(self):
        image_features_from_pil = self.backbone.image_feature_extraction(self.image)
        image_features_from_numpy = self.backbone.image_feature_extraction(self.np_image)

        self.assertEqual(image_features_from_pil.shape, (1, self.final_hidden_size))
        self.assertEqual(image_features_from_numpy.shape, (1, self.final_hidden_size))
        self.assertTrue(torch.allclose(image_features_from_pil, image_features_from_numpy))
    
    def test_text_feature_extraction(self):
        text_features_from_list = self.backbone.text_feature_extraction(self.prompts)
        text_features_from_str = self.backbone.text_feature_extraction(self.prompts[0])

        self.assertEqual(text_features_from_list.shape, (len(self.prompts), self.final_hidden_size))
        self.assertEqual(text_features_from_str.shape, (1, self.final_hidden_size))

if __name__ == "__main__":
    unittest.main()