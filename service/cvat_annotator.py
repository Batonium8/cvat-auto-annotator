from collections.abc import Callable
from dataclasses import dataclass
import logging

import PIL.Image
from cvat_sdk import make_client, models

from core import DetectionContext, Detector, TextPromptDetector
from service.config import settings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TaskMeta:
    """Базовые метаданные задачи CVAT для pre-flight проверки."""

    task_id: int
    labels: tuple[str, ...]
    total_frames: int


@dataclass(frozen=True)
class AnnotationResult:
    """Сводка по итогам авторазметки задачи."""

    task_id: int
    model: str
    frames_processed: int
    shapes_created: int
    labels: tuple[str, ...] = ()


def get_task_meta(task_id: int) -> TaskMeta:
    """Проверяет существование задачи в CVAT и возвращает её классы и число кадров."""
    with make_client(
        host=settings.cvat_host,
        credentials=(settings.cvat_user, settings.cvat_pass),
    ) as client:
        task = client.tasks.retrieve(task_id)
        labels = tuple(label.name for label in task.get_labels())
        return TaskMeta(task_id=task_id, labels=labels, total_frames=task.size)


def annotate_task(
    task_id: int,
    adapter: Detector,
    prompts: dict[str, str] | None = None,
    clear_existing: bool = False,
    on_progress: Callable[[int, int, int], None] | None = None,
) -> AnnotationResult:
    """
    Забирает кадры и классы задачи из CVAT, прогоняет через переданный адаптер
    и сохраняет полученные прямоугольники обратно в задачу.
    """
    prompts_map = prompts or {}

    with make_client(
        host=settings.cvat_host,
        credentials=(settings.cvat_user, settings.cvat_pass),
    ) as client:
        task = client.tasks.retrieve(task_id)

        task_labels = task.get_labels()
        label_name_to_id = {label.name: label.id for label in task_labels}
        label_names = list(label_name_to_id.keys())

        if not label_names or task.size == 0:
            logger.warning(
                "Задача #%d пропущена: классов=%d, кадров=%d",
                task_id,
                len(label_names),
                task.size,
            )
            return AnnotationResult(
                task_id=task_id,
                model=adapter.name,
                frames_processed=0,
                shapes_created=0,
                labels=tuple(label_names),
            )

        logger.info(
            "Старт разметки задачи #%d: модель='%s', кадров=%d, clear_existing=%s, классы=%s",
            task_id,
            adapter.name,
            task.size,
            clear_existing,
            label_names,
        )

        # Явно прогреваем модель и кодируем текстовые промпты один раз до обхода кадров
        adapter.load()
        if isinstance(adapter, TextPromptDetector):
            warmup_ctx = DetectionContext(
                width=0,
                height=0,
                labels=label_names,
                prompts=prompts_map,
            )
            expanded_prompts, _ = warmup_ctx.expand_prompts()
            if expanded_prompts:
                adapter.set_prompts(expanded_prompts)

        shapes: list[models.LabeledShapeRequest] = []

        for frame_idx in range(task.size):
            frame_stream = task.get_frame(frame_idx, quality="original")
            with PIL.Image.open(frame_stream) as raw_img:
                image = raw_img.convert("RGB")

            width, height = image.size
            context = DetectionContext(
                width=width,
                height=height,
                labels=label_names,
                prompts=prompts_map,
            )

            boxes = adapter.predict(image, context)
            for box in boxes:
                label_id = label_name_to_id.get(box.label)
                if label_id is None:
                    continue

                shapes.append(
                    models.LabeledShapeRequest(
                        frame=frame_idx,
                        label_id=label_id,
                        type="rectangle",
                        points=[box.xtl, box.ytl, box.xbr, box.ybr],
                    )
                )

            processed = frame_idx + 1
            if on_progress is not None:
                on_progress(processed, task.size, len(shapes))

            if processed % 5 == 0 or processed == task.size:
                logger.info(
                    "Задача #%d [%s]: обработано %d/%d кадров (рамок: %d)",
                    task_id,
                    adapter.name,
                    processed,
                    task.size,
                    len(shapes),
                )

        if clear_existing:
            task.set_annotations(models.LabeledDataRequest(shapes=shapes))
        elif shapes:
            task.update_annotations(models.PatchedLabeledDataRequest(shapes=shapes))

        return AnnotationResult(
            task_id=task_id,
            model=adapter.name,
            frames_processed=task.size,
            shapes_created=len(shapes),
            labels=tuple(label_names),
        )
