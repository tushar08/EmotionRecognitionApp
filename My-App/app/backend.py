from fastapi import FastAPI, WebSocket
from fastapi.staticfiles import StaticFiles
from app.emotion_recognition import process_video_feed, export_emotions_to_csv
from app.utils import generate_statistics

app = FastAPI()

# app.mount("/static", StaticFiles(directory="app/static"), name="static")

@app.get("/")
async def root():
    return {"message": "Welcome to the Emotion Detection API"}

@app.post("/process-video/")
async def process_video():
    # Implement logic for processing video here
    return {"message": "Video processing started"}

@app.websocket("/emotion-feed")
async def emotion_feed(websocket: WebSocket):
    await websocket.accept()
    try:
        async for frame in websocket.iter_bytes():
            result = process_video_feed(frame)
            await websocket.send_json(result)
    except Exception as e:
        print(f"Error in WebSocket connection: {e}")
    finally:
        await websocket.close()

@app.get("/export-csv/")
async def export_csv():
    path = export_emotions_to_csv()
    return {"message": "CSV exported", "path": path}
