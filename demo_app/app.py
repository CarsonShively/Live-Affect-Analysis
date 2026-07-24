from contextlib import asynccontextmanager
from live_affect_analysis.process_frame import ProcessFrame
from fastapi import FastAPI, Request, UploadFile, File
import numpy as np
import cv2
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path

@asynccontextmanager
async def lifespan(app):
    app.state.processor = ProcessFrame()
    yield
    
app = FastAPI(lifespan=lifespan)

frontend_path = Path(__file__).resolve().parents[0] / "frontend"

app.mount("/static", StaticFiles(directory=frontend_path), name="static")

@app.get("/")
async def home():
    return FileResponse(frontend_path / "index.html")

@app.post("/predict")
async def predict(request: Request, file: UploadFile = File(...)):
    buffer = await file.read()
    
    if not buffer:
        return {"sucess": False, "details": "empty image file"}
    
    array = np.frombuffer(buffer, dtype=np.uint8)
    frame = cv2.imdecode(array, cv2.IMREAD_COLOR)
    
    if frame is None:
        return {"sucess": False, "details": "frame decode fail"}
    
    result = request.app.state.processor.inference(frame)
    
    return result