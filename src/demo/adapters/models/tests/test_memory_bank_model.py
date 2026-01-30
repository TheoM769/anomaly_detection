import unittest
import os

import torch
import numpy as np
from PIL import Image
from dotenv import load_dotenv

from adapters.backbones.DINO import DinoBackbone
from adapters.models.memory_bank_model import MemoryBankModelAdapter
from adapters.types.memory_bank_result import MemoryBankResult
load_dotenv()
IMAGE_PATH = os.getenv("IMAGE_PATH")

class TestLinearClsAdapter(unittest.TestCase):

    def setUp(self):
        self.backbone = DinoBackbone()
        self.model = MemoryBankModelAdapter(self.backbone, os.path.join(IMAGE_PATH, "hazelnut/train"))

        self.model._select_memory_bank_samples(0.5)
        self.train_size = 391
        self.test_images_path = [os.path.join(IMAGE_PATH, "hazelnut/test/hole/003.png"), os.path.join(IMAGE_PATH, "hazelnut/test/hole/004.png")]
        self.test_PIL_images = [Image.open(image_path) for image_path in self.test_images_path]
        self.test_np_images = [np.array(image) for image in self.test_PIL_images]
        
    def test_predict(self):
        self.assertIsInstance(self.model.predict(self.test_images_path[0], 1), MemoryBankResult)
        self.assertIsInstance(self.model.predict(self.test_PIL_images[0], 1), MemoryBankResult)
        self.assertIsInstance(self.model.predict(self.test_np_images[0], 1), MemoryBankResult)
        self.assertIsInstance(self.model.predict(self.test_np_images, 1), MemoryBankResult)
        self.assertEqual(self.model.predict(self.test_images_path[0], 1)[0], self.model.predict(self.test_np_images[0], 1)[0])

        self.assertIsInstance(self.model.get_memory_bank(), torch.Tensor)
        self.assertEqual(self.model.get_memory_bank().shape, (self.train_size, 768))


if __name__ == "__main__":
    unittest.main()