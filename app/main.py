from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from ultralytics import YOLO
from PIL import Image
import io

app = FastAPI()

model = YOLO("app/model/best.pt")

@app.get("/")
async def root():
    return {"message": "Server is running. Use /predict for POST image prediction."}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        results = model(image)

        probs = results[0].probs  

        if probs is None:
            return JSONResponse(status_code=400, content={"error": "No classification probabilities found."})

        class_id = int(probs.top1)  
        confidence = float(probs.data[class_id])
        class_name = results[0].names[class_id]

        return {
            "class_id": class_id,
            "class_name": class_name,
            "confidence": confidence
        }

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

