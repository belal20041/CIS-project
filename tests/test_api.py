import unittest
from fastapi.testclient import TestClient
from deployment.api import app
class TestAPI(unittest.TestCase):
    def test_read_root(self):
        client = TestClient(app)
        response = client.get("/")
        self.assertEqual(response.status_code, 200)
if __name__ == "__main__":
    unittest.main()
