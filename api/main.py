"""HTTP API сегментации полипов.

GET  /health           - 200 только после загрузки и прогрева модели, иначе 503
POST /predict          - JSON: маска (PNG base64), площадь, bbox, уверенность, наложение
POST /predict/mask     - PNG с бинарной маской (удобно смотреть результат без jq/python)
POST /predict/overlay  - PNG с маской поверх снимка
Swagger UI: /docs
"""

import io
import logging
import os
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse, RedirectResponse, Response
from PIL import Image, UnidentifiedImageError

from api.predictor import Predictor, b64
from api.schemas import ErrorResponse, HealthResponse, PredictResponse

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("polyp-api")

MODEL_PATH = os.environ.get("MODEL_PATH", "artifacts/final/model.pt")
MAX_UPLOAD_MB = float(os.environ.get("MAX_UPLOAD_MB", "15"))
MAX_PIXELS = int(os.environ.get("MAX_PIXELS", str(40_000_000)))
MIN_SIDE = 32

# Защита PIL от «декомпрессионных бомб»: файл маленький, а картинка гигантская
Image.MAX_IMAGE_PIXELS = MAX_PIXELS

state = {"predictor": None, "error": "модель ещё загружается"}

ERRORS = {
    400: {"model": ErrorResponse, "description": "файл пустой, повреждён или не является изображением"},
    413: {"model": ErrorResponse, "description": "файл или изображение слишком большие"},
    422: {"description": "нет поля file или неверные параметры запроса"},
    503: {"model": ErrorResponse, "description": "модель не загружена"},
}


@asynccontextmanager
async def lifespan(app):
    threads = os.environ.get("TORCH_THREADS")
    if threads:
        torch.set_num_threads(int(threads))
    try:
        state["predictor"] = Predictor(MODEL_PATH)
        state["error"] = None
        logger.info("Модель загружена: %s", state["predictor"].model_info)
    except Exception as exc:  # noqa: BLE001 - сервис остаётся живым и сообщает причину через /health
        state["error"] = f"{type(exc).__name__}: {exc}"
        logger.exception("Не удалось загрузить модель из %s", MODEL_PATH)
    yield


app = FastAPI(
    title="Polyp segmentation API",
    description="Сегментация полипов на эндоскопических снимках (U-Net, Kvasir-SEG)",
    version="1.0.0",
    lifespan=lifespan,
)


def get_predictor():
    if state["predictor"] is None:
        raise HTTPException(status_code=503, detail=f"Модель не готова: {state['error']}")
    return state["predictor"]


def read_image(file):
    """Читает и проверяет загруженный файл. Любая проблема -> 4xx с описанием."""
    limit = int(MAX_UPLOAD_MB * 2**20)
    data = file.file.read(limit + 1)
    if not data:
        raise HTTPException(status_code=400, detail="Пустой файл")
    if len(data) > limit:
        raise HTTPException(status_code=413, detail=f"Файл больше {MAX_UPLOAD_MB:g} МБ")
    try:
        with Image.open(io.BytesIO(data)) as probe:
            probe.verify()
        image = Image.open(io.BytesIO(data))
        image.load()
    except Image.DecompressionBombError:
        raise HTTPException(status_code=413, detail=f"Изображение больше {MAX_PIXELS} пикселей") from None
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError):
        raise HTTPException(
            status_code=400, detail="Файл не является корректным изображением (ожидается JPEG, PNG, BMP, TIFF или WebP)"
        ) from None
    width, height = image.size
    if width * height > MAX_PIXELS:
        raise HTTPException(status_code=413, detail=f"Изображение больше {MAX_PIXELS} пикселей")
    if min(width, height) < MIN_SIDE:
        raise HTTPException(status_code=400, detail=f"Изображение слишком маленькое: нужно не меньше {MIN_SIDE}x{MIN_SIDE}")
    return image.convert("RGB")


THRESHOLD_QUERY = Query(None, ge=0.0, le=1.0, description="порог вероятности; по умолчанию - из чекпоинта (0.5)")


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")


@app.get("/health", response_model=HealthResponse, responses={503: ERRORS[503]})
def health():
    predictor = state["predictor"]
    if predictor is None:
        return JSONResponse(status_code=503, content={"status": "unavailable", "detail": state["error"]})
    return {"status": "ok", "model": predictor.model_info}


@app.post("/predict", response_model=PredictResponse, responses=ERRORS)
def predict(
    file: UploadFile = File(..., description="эндоскопический снимок"),
    threshold: float = THRESHOLD_QUERY,
    include_overlay: bool = Query(True, description="вернуть наложение маски на снимок"),
):
    predictor = get_predictor()
    image = read_image(file)
    result = predictor.predict(image, threshold)
    mask = result.pop("mask")
    return {
        "filename": file.filename,
        "width": image.size[0],
        "height": image.size[1],
        **result,
        "mask_png_base64": b64(predictor.mask_png(mask)),
        "overlay_png_base64": b64(predictor.overlay_png(image, mask)) if include_overlay else None,
        "model": predictor.model_info,
    }


@app.post("/predict/mask", response_class=Response, responses={200: {"content": {"image/png": {}}}, **ERRORS})
def predict_mask(file: UploadFile = File(...), threshold: float = THRESHOLD_QUERY):
    predictor = get_predictor()
    result = predictor.predict(read_image(file), threshold)
    return Response(content=predictor.mask_png(result["mask"]), media_type="image/png")


@app.post("/predict/overlay", response_class=Response, responses={200: {"content": {"image/png": {}}}, **ERRORS})
def predict_overlay(file: UploadFile = File(...), threshold: float = THRESHOLD_QUERY):
    predictor = get_predictor()
    image = read_image(file)
    result = predictor.predict(image, threshold)
    return Response(content=predictor.overlay_png(image, result["mask"]), media_type="image/png")
