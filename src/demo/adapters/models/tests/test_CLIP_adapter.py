import unittest
import os

import numpy as np
from PIL import Image
from dotenv import load_dotenv

from adapters.models.CLIP_cls_adapter import CLIPClsModelAdapter
from adapters.backbones.CLIP import CLIPBackbone
from adapters.types.classification_result import ClassificationResult

load_dotenv()
IMAGE_PATH = os.getenv("IMAGE_PATH")

class TestCLIPAdapter(unittest.TestCase):

    def setUp(self):
        backbone = CLIPBackbone()
        self.model = CLIPClsModelAdapter(backbone)
        
        self.image = Image.open(IMAGE_PATH + "/hazelnut/train/good/004.png")
        self.np_image = np.array(self.image)
        
    def test_predict(self):
        result = self.model.predict(["a good hazelnut", "a bad hazelnut"], self.np_image)
        self.assertIsInstance(result, ClassificationResult)
        print(result)

if __name__ == "__main__":
    unittest.main()