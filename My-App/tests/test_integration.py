import unittest
from fastapi.testclient import TestClient
from app.backend import app
import os


class TestIntegration(unittest.TestCase):

    def setUp(self):
        self.client = TestClient(app)

    def test_root_endpoint(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("message", response.json())
        self.assertEqual(response.json()["message"], "Welcome to the Emotion Detection API")

    def test_process_video_endpoint(self):
        # Mock a video processing request
        response = self.client.post("/process-video/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("message", response.json())
        self.assertEqual(response.json()["message"], "Video processing started")

    def test_export_csv_endpoint(self):
        response = self.client.get("/export-csv/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("path", response.json())
        
        # Verify the file exists
        path = response.json()["path"]
        self.assertTrue(os.path.exists(path))
