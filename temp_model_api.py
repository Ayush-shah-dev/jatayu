from fastapi import FastAPI, UploadFile, File
import torch
import cv2
import numpy as np
from io import BytesIO
from ultralytics import YOLO

# Load YOLO model
model = YOLO("harras.pt")

app = FastAPI()

@app.post("/detect/")
async def detect_objects(file: UploadFile = File(...)):
    # Read image
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Perform detection
    results = model(img)

    # Extract detections
    detections = []
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box
            conf = float(box.conf[0])  # Confidence
            cls = int(box.cls[0])  # Class index
            label = model.names[cls]  # Class label
            detections.append({"label": label, "confidence": conf, "bbox": [x1, y1, x2, y2]})

    return {"detections": detections}

