from adapters.matryoshka import MatryoshkaAdapter
from adapters.yoloe import YoloeAdapter
from core import BaseDetectorAdapter
from service.config import settings

_REGISTRY: dict[str, BaseDetectorAdapter] = {
    "pseudo": MatryoshkaAdapter(),
    "yoloe": YoloeAdapter(
        weights=settings.yoloe_weights,
        conf=settings.yoloe_conf,
        imgsz=settings.yoloe_imgsz,
        device=settings.yoloe_device,
    ),
}


def get_adapter(model_name: str) -> BaseDetectorAdapter:
    adapter = _REGISTRY.get(model_name)
    if adapter is None:
        available = ", ".join(_REGISTRY.keys())
        raise KeyError(
            f"Модель '{model_name}' не найдена. Доступные модели: {available}"
        )
    return adapter


def list_available_models() -> list[str]:
    return list(_REGISTRY.keys())
