import unittest
import os
import typing

from dotenv import load_dotenv
import numpy as np
from PIL import Image
import torch

from adapters.backbones.owlv2 import Owlv2Backbone

load_dotenv()
IMAGE_PATH = os.getenv("IMAGE_PATH")

class TestOwlv2(unittest.TestCase):

    def setUp(self):
        self.model = Owlv2Backbone()
        self.model.load_weights()
        self.image = Image.open(IMAGE_PATH + "/bottle/train/good/000.png")
        self.np_image = np.array(self.image)

        self.image2 = Image.open(IMAGE_PATH + "/bottle/train/good/001.png")
        self.np_image2 = np.array(self.image2)
        self.image_batch1 = [self.np_image, self.np_image2]
        self.image_batch2 = [self.np_image2, self.np_image]
        
    def test_detect_objects(self):
        results = self.model.detect_objects([["a bottle", "a hazelnut"]], self.image, topk=2)
        self.assertIsInstance(results[0]["labels"], torch.Tensor)
        self.assertIsInstance(results[0]["boxes"], torch.Tensor)
        self.assertIsInstance(results[0]["scores"], torch.Tensor)
        self.assertLessEqual(len(results[0]["labels"]), 2)

        results = self.model.detect_objects([["a bottle", "a hazelnut"]], self.image, topk=4)
        self.assertLessEqual(len(results[0]["labels"]), 4)

if __name__ == "__main__":
    unittest.main()