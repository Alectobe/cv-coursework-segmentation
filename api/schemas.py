"""Схемы ответов API (отображаются в Swagger UI на /docs)."""

from typing import Optional

from pydantic import BaseModel, Field


class ModelInfo(BaseModel):
    experiment: str = Field(description="имя эксперимента, из которого взяты веса")
    epoch: Optional[int] = Field(None, description="эпоха лучшего val Dice")
    val_dice: Optional[float] = Field(None, description="mDice на валидации для этих весов")
    image_size: int = Field(description="размер входа модели, px")
    default_threshold: float
    clearml_task_id: Optional[str] = None
    clearml_task_url: Optional[str] = None
    checkpoint_sha256: str


class HealthResponse(BaseModel):
    status: str = Field(description="ok - модель загружена и готова к запросам")
    model: ModelInfo


class ErrorResponse(BaseModel):
    detail: str


class PredictResponse(BaseModel):
    filename: Optional[str]
    width: int = Field(description="ширина исходного снимка, px")
    height: int = Field(description="высота исходного снимка, px")
    threshold: float = Field(description="порог вероятности, по которому построена маска")
    polyp_found: bool = Field(description="есть ли в маске хотя бы один пиксель полипа")
    polyp_area_ratio: float = Field(description="доля площади кадра, занятая маской полипа")
    bbox: Optional[list[int]] = Field(None, description="[x_min, y_min, x_max, y_max] маски полипа, px")
    mean_confidence: float = Field(description="средняя вероятность полипа внутри маски (0, если маска пустая)")
    max_probability: float = Field(description="максимальная вероятность полипа по кадру")
    mask_png_base64: str = Field(description="бинарная маска (0/255) в исходном разрешении, PNG в base64")
    overlay_png_base64: Optional[str] = Field(None, description="маска поверх снимка, PNG в base64")
    inference_ms: float
    model: ModelInfo
