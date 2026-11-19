from dataclasses import dataclass
import PIL.Image
from cvat_sdk import make_client, models

from adapters import get_adapter
from core import DetectionContext
from service.config import settings


@dataclass(frozen=True)
class AnnotationResult:
    """Сводка по итогам авторазметки задачи (уйдёт в JSON-ответ FastAPI)."""
    task_id: int
    model: str
    frames_processed: int
    shapes_created: int
    labels: tuple[str, ...] = ()


def annotate_task(
    task_id: int,
    model_name: str = "pseudo",
    prompts: dict[str, str] | None = None,
    clear_existing: bool = True,
) -> AnnotationResult:
    """
    Забирает кадры и классы задачи из CVAT, прогоняет через выбранный адаптер
    и сохраняет полученные прямоугольники обратно в задачу.
    """
    adapter = get_adapter(model_name)
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
            return AnnotationResult(
                task_id=task_id,
                model=adapter.name,
                frames_processed=0,
                shapes_created=0,
                labels=tuple(label_names),
            )

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
