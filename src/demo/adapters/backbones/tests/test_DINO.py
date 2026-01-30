import unittest
import os
import typing

from dotenv import load_dotenv
import numpy as np
from PIL import Image
import torch

from adapters.backbones.DINO import DinoBackbone

load_dotenv()
IMAGE_PATH = os.getenv("IMAGE_PATH")

class TestDINO(unittest.TestCase):

    def setUp(self):
        self.model = DinoBackbone()
        self.model.load_weights()
        self.image = Image.open(IMAGE_PATH + "/bottle/train/good/000.png")
        self.np_image = np.array(self.image)

        self.image2 = Image.open(IMAGE_PATH + "/bottle/train/good/001.png")
        self.np_image2 = np.array(self.image2)
        self.image_batch1 = [self.np_image, self.np_image2]
        self.image_batch2 = [self.np_image2, self.np_image]

        self.model_image_size = (self.model.processor.crop_size["height"], self.model.processor.crop_size["width"])
        self.model_num_patches = (self.model_image_size[0] * self.model_image_size[1]) // (self.model.patch_size ** 2)
        
    def test_processor(self):
        inputs = self.model.processor(self.image, return_tensors="pt")
        self.assertEqual(inputs["pixel_values"].shape, (1, 3, self.model_image_size[0], self.model_image_size[1]))
    
    def test_numpy_features_equal_to_PIL_features(self):
        inputs = self.model.processor(self.image, return_tensors="pt")
        inputs2 = self.model.processor(self.np_image, return_tensors="pt")

        self.assertTrue(torch.allclose(inputs["pixel_values"], inputs2["pixel_values"]))

    def test_image_feature_extraction(self):
        image_features = self.model.image_feature_extraction(self.image)
        np_image_features = self.model.image_feature_extraction(self.np_image)

        batch_image_features1 = self.model.image_feature_extraction(self.image_batch1)
        batch_image_features2 = self.model.image_feature_extraction(self.image_batch2)

        self.assertEqual(image_features.shape, (1, self.model.hidden_size))
        self.assertEqual(np_image_features.shape, (1, self.model.hidden_size))
        self.assertTrue(torch.allclose(image_features, np_image_features))

        self.assertEqual(batch_image_features1.shape, (2, self.model.hidden_size))
        self.assertEqual(batch_image_features2.shape, (2, self.model.hidden_size))
        self.assertTrue(torch.allclose(batch_image_features1[0], batch_image_features2[1]))
        self.assertTrue(torch.allclose(batch_image_features1[1], batch_image_features2[0]))
    
    def test_patch_feature_extraction(self):
        patch_features = self.model.patch_feature_extraction(self.image)
        np_patch_features = self.model.patch_feature_extraction(self.np_image)

        self.assertEqual(patch_features.shape, (1, self.model_num_patches + 1, self.model.hidden_size))
        self.assertEqual(np_patch_features.shape, (1, self.model_num_patches + 1, self.model.hidden_size))
        self.assertTrue(torch.allclose(patch_features, np_patch_features))
    
    def test_feature_map_extraction(self):
        feature_map = self.model.feature_map_extraction(self.image)
        np_feature_map = self.model.feature_map_extraction(self.np_image)

        self.assertEqual(len(feature_map), self.model.num_layers + 1)
        self.assertEqual(feature_map[0].shape, (1, self.model_num_patches + 1, self.model.hidden_size))
        self.assertEqual(np_feature_map[0].shape, (1, self.model_num_patches + 1, self.model.hidden_size))
        self.assertTrue(torch.allclose(feature_map[0], np_feature_map[0]))

if __name__ == "__main__":
    unittest.main()