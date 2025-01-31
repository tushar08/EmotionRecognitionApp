import unittest
from unittest.mock import patch
from app.emotion_recognition import process_video_feed, export_emotions_to_csv
import os


class TestEmotionRecognition(unittest.TestCase):

    @patch("app.emotion_recognition.emotion_model")
    def test_process_video_feed(self, mock_emotion_model):
        # Mock the response from the emotion model
        mock_emotion_model.return_value = [{"label": "Happy", "score": 0.98}]
        frame = b"fake_binary_frame"
        
        # Call the function
        result = process_video_feed(frame)
        
        # Check results
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["label"], "Happy")
        self.assertGreater(result[0]["score"], 0.9)

    def test_export_emotions_to_csv(self):
        # Call the function
        path = export_emotions_to_csv()
        
        # Check if the file exists
        self.assertTrue(os.path.exists(path))
        
        # Validate content
        with open(path, "r") as f:
            content = f.readlines()
        
        self.assertGreater(len(content), 1)  # Ensure header + data rows
        self.assertIn("Emotion,Count", content[0])
