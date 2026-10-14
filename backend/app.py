
import os
import tempfile

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from inference_sdk import InferenceHTTPClient

app = FastAPI(title="CivicRoute AI Backend")

# Allows your local website to communicate with this backend.
# Restrict this to your actual website origin before deployment.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

API_KEY = os.getenv("ROBOFLOW_API_KEY")

if not API_KEY:
    raise RuntimeError(
        "ROBOFLOW_API_KEY is missing. Set it before starting the server."
    )

client = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key=API_KEY,
)

MODEL_ID = "rdd2022-zw12e/2"


@app.get("/")
def home():
    return {"message": "CivicRoute AI backend is running"}


@app.post("/api/detect")
async def detect_road_damage(image: UploadFile = File(...)):
    if image.content_type not in ("image/jpeg", "image/png", "image/webp"):
        raise HTTPException(
            status_code=400,
            detail="Please upload a JPG, PNG, or WebP image."
        )

    contents = await image.read()

    if not contents:
        raise HTTPException(status_code=400, detail="The uploaded image is empty.")

    if len(contents) > 10 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Image must be 10 MB or smaller.")

    suffix = {
        "image/jpeg": ".jpg",
        "image/png": ".png",
        "image/webp": ".webp",
    }[image.content_type]

    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp:
            temp.write(contents)
            temp_path = temp.name

        result = client.infer(temp_path, model_id=MODEL_ID)

        predictions = []

        for p in result.get("predictions", []):
            if p.get("class", "").upper() != "D40":
                continue

            width = float(p.get("width", 0))
            height = float(p.get("height", 0))
            confidence = float(p.get("confidence", 0))
            area = width * height

            # Preliminary image-area estimate, not real-world severity.
            if area < 3000:
                severity = "LOW"
            elif area < 10000:
                severity = "MODERATE"
            else:
                severity = "HIGH"

            predictions.append({
                "class": p.get("class", "D40"),
                "confidence": confidence,
                "x": p.get("x"),
                "y": p.get("y"),
                "width": width,
                "height": height,
                "area_pixels": area,
                "estimated_severity": severity,
            })

        return {
            "success": True,
            "model": MODEL_ID,
            "total_potholes": len(predictions),
            "predictions": predictions,
        }

    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Roboflow inference failed: {str(exc)}"
        )

    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
