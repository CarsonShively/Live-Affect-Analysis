from contextlib import asynccontextmanager
from live_affect_analysis.demo_process_frame import ProcessFrame
from fastapi import FastAPI, Request, UploadFile, File
import numpy as np
import cv2

@asynccontextmanager
async def lifespan(app):
    app.state.processor = ProcessFrame()
    yield
    
app = FastAPI(lifespan=lifespan)

@app.post("/predict")
async def predict(request: Request, file: UploadFile = File(...)):
    buffer = await file.read()
    array = np.frombuffer(buffer, dtype=np.uint8)
    frame = cv2.imdecode(array, cv2.IMREAD_COLOR)
    
    result = request.app.state.processor.inference(frame)
    
    return result