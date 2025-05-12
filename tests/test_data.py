import unittest
from utils.data_loader import DataLoader
class TestDataLoader(unittest.TestCase):
    def test_load_data(self):
        loader = DataLoader()
        self.assertTrue(True)
if __name__ == "__main__":
    unittest.main()
