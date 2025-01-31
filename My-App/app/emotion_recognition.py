import cv2
import numpy as np
from transformers import AutoModel
from transformers import pipeline
import os



# emotion_model = AutoModel.from_pretrained("motheecreator/vit-Facial-Expression-Recognition", token=access_token)
emotion_model = AutoModel.from_pretrained("motheecreator/vit-Facial-Expression-Recognition")
# emotion_model = pipeline("image-classification", model)

def process_video_feed(frame):
    # Decode the frame and process it for emotion detection
    np_frame = np.frombuffer(frame, np.uint8)
    image = cv2.imdecode(np_frame, cv2.IMREAD_COLOR)
    results = emotion_model(image)
    return results

def export_emotions_to_csv():
    # Export emotion statistics to a CSV
    path = "app/exports/emotion_stats.csv"
    with open(path, "w") as f:
        f.write("Emotion,Count\n")
        # Add dummy data for now
        f.write("Happy,10\nSad,5\n")
    return path
