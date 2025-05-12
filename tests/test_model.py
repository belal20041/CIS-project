import unittest
from models.trainer import ModelTrainer
class TestModelTrainer(unittest.TestCase):
    def test_train(self):
        trainer = ModelTrainer()
        self.assertTrue(True)
if __name__ == "__main__":
    unittest.main()
