from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from config import CLASSES, IMG_SIZE
from inference import get_model, predict_from_bytes

MOBILE_PAGE = Path(__file__).resolve().parent / "mobile.html"

app = FastAPI(
    title="Intel Image Classification API",
    description="API deployment model CNN untuk klasifikasi gambar (buildings, forest, mountain).",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictionResponse(BaseModel):
    label: str
    confidence: float
    all_scores: dict[str, float]


@app.on_event("startup")
def load_model_on_startup():
    get_model()


@app.get("/mobile", include_in_schema=False)
def mobile_page():
    return FileResponse(MOBILE_PAGE, media_type="text/html")


@app.get("/")
def root():
    return {
        "message": "Intel Image Classification API",
        "endpoints": {
            "health": "/health",
            "classes": "/classes",
            "predict": "/predict (POST, multipart/form-data, field: file)",
            "mobile_web": "/mobile",
            "docs": "/docs",
        },
    }


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": True}


@app.get("/classes")
def classes():
    return {"classes": CLASSES, "image_size": IMG_SIZE}


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File harus berupa gambar (jpg, png, jpeg).")

    try:
        contents = await file.read()
        result = predict_from_bytes(contents)
        return result
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediksi gagal: {exc}") from exc
