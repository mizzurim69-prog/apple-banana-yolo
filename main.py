
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
from PIL import Image, UnidentifiedImageError
import io

app = FastAPI(title="Apple-Banana Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model = YOLO("best.pt")


@app.get("/")
def home():
    return {"message": "Apple-Banana Detection API is running"}


@app.get("/health")
def health():
    return {"status": "ok", "model": "YOLOv8n"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        content = await file.read()
        image = Image.open(io.BytesIO(content)).convert("RGB")
    except (UnidentifiedImageError, OSError):
        raise HTTPException(
            status_code=400,
            detail="Please upload a valid image file",
        )

    results = model(image, verbose=False)[0]
    detections = []

    for box in results.boxes:
        detections.append({
            "class": results.names[int(box.cls.item())],
            "confidence": round(float(box.conf.item()), 3),
            "box": [
                round(float(value), 2)
                for value in box.xyxy[0].tolist()
            ],
        })

    return {"detections": detections}
