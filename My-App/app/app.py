# import cv2
# import torch
# import numpy as np
# import face_recognition
# import sounddevice as sd
# import librosa
# import json
# from transformers import AutoImageProcessor, AutoModelForImageClassification
# from collections import defaultdict
# from datetime import datetime
# import pandas as pd
# import streamlit as st
# from fastapi import FastAPI, WebSocket
# from fastapi.middleware.cors import CORSMiddleware
# import uvicorn

# # FastAPI setup
# app = FastAPI()
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Emotion detection model
# model_name = "nateraw/vit-base-beans"
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = AutoModelForImageClassification.from_pretrained(model_name).to(device)
# processor = AutoImageProcessor.from_pretrained(model_name)

# # Data structures for tracking emotions
# person_emotions = defaultdict(lambda: defaultdict(int))
# aggregate_emotions = defaultdict(int)
# person_encodings = {}
# audio_emotions = defaultdict(int)

# # Helper functions
# def preprocess_image(image):
#     """Preprocess an image for the model."""
#     inputs = processor(images=image, return_tensors="pt").to(device)
#     return inputs

# def predict_emotion(image):
#     """Predict emotion for a given cropped face image."""
#     inputs = preprocess_image(image)
#     outputs = model(**inputs)
#     probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
#     predicted_class = torch.argmax(probs).item()
#     emotion = model.config.id2label[predicted_class]
#     return emotion, probs.squeeze().tolist()

# def update_statistics(person_id, emotion, emotion_probs):
#     """Update emotion stats for a person."""
#     person_emotions[person_id][emotion] += 1
#     aggregate_emotions[emotion] += 1

# def export_to_csv():
#     """Export statistics to a CSV file."""
#     stats = {"person_id": [], "emotion": [], "count": []}
#     for pid, emotions in person_emotions.items():
#         for emotion, count in emotions.items():
#             stats["person_id"].append(pid)
#             stats["emotion"].append(emotion)
#             stats["count"].append(count)
#     df = pd.DataFrame(stats)
#     df.to_csv("emotion_stats.csv", index=False)

# def recognize_audio_emotion(audio, sr):
#     """Basic voice emotion recognition."""
#     # Dummy example: Integrate an ML/DL model for voice emotion recognition.
#     if np.mean(audio) > 0.1:
#         return "Happy"
#     return "Neutral"

# def get_face_id(face_encoding):
#     """Match or create a new ID for a detected face."""
#     for pid, encoding in person_encodings.items():
#         if face_recognition.compare_faces([encoding], face_encoding)[0]:
#             return pid
#     new_id = len(person_encodings) + 1
#     person_encodings[new_id] = face_encoding
#     return new_id

# # WebSocket endpoint
# @app.websocket("/emotion-feed")
# async def emotion_feed(websocket: WebSocket):
#     await websocket.accept()
#     cap = cv2.VideoCapture(0)
#     start_time = datetime.now()
    
#     while (datetime.now() - start_time).seconds < 60:  # 1 minute runtime
#         ret, frame = cap.read()
#         if not ret:
#             break

#         # Detect faces
#         rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         face_locations = face_recognition.face_locations(rgb_frame)
#         face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

#         for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
#             face_image = rgb_frame[top:bottom, left:right]
#             resized_face = cv2.resize(face_image, (224, 224))

#             emotion, emotion_probs = predict_emotion(resized_face)
#             person_id = get_face_id(face_encoding)

#             update_statistics(person_id, emotion, emotion_probs)

#             # Draw bounding box and label
#             cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
#             cv2.putText(frame, f"Person {person_id}: {emotion}", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

#         # Send video frame to frontend
#         _, buffer = cv2.imencode(".jpg", frame)
#         await websocket.send_bytes(buffer.tobytes())

#     cap.release()
#     await websocket.close()

# # Real-time voice emotion detection
# @app.post("/audio-emotion")
# async def audio_emotion_endpoint():
#     duration = 5  # seconds
#     fs = 16000  # sample rate
#     audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype="float32")
#     sd.wait()
#     emotion = recognize_audio_emotion(audio, fs)
#     audio_emotions[emotion] += 1
#     return {"emotion": emotion, "audio_emotions": dict(audio_emotions)}

# # Streamlit app
# st.title("Emotion Detection App")
# run_video = st.checkbox("Run Video Feed", value=True)
# if run_video:
#     st.write("Running...")
#     uvicorn.run(app, host="127.0.0.1", port=8000)
# st.button("Export to CSV", on_click=export_to_csv)