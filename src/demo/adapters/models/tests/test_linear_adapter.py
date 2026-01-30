import unittest
import os

import numpy as np
from PIL import Image
from dotenv import load_dotenv

from adapters.models.linear_cls_adapter import LinearClsModelAdapter
from adapters.backbones.DINO import DinoBackbone
from adapters.types.classification_result import ClassificationResult

load_dotenv()
IMAGE_PATH = os.getenv("IMAGE_PATH")

class TestLinearClsAdapter(unittest.TestCase):

    def setUp(self):
        backbone = DinoBackbone()
        checkpoint = "/Users/theo.moreau/Documents/futur-1/infrastructure/checkpoints/MVTech_AD/MVTech_DINO.pth"
        dataset_name = "mvtech_ad"
        self.model = LinearClsModelAdapter(backbone, checkpoint, 15, dataset_name)
        
        self.image = Image.open(IMAGE_PATH + "/hazelnut/train/good/004.png")
        self.np_image = np.array(self.image)
        
    def test_predict(self):
        result = self.model.predict(self.np_image)
        self.assertIsInstance(result, ClassificationResult)
        print(result)

if __name__ == "__main__":
    unittest.main()