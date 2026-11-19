import PIL.Image

from core import BaseDetectorAdapter, BBox, DetectionContext


class MatryoshkaAdapter(BaseDetectorAdapter):
    """
    Адаптер Этапа 1: генерирует вложенные по центру прямоугольники ('матрёшку')
    для каждого класса задачи с учетом размеров конкретного изображения.
    """

    @property
    def name(self) -> str:
        return "pseudo"

    def predict(
        self, image: PIL.Image.Image, context: DetectionContext
    ) -> list[BBox]:
        width, height = image.size
        labels = context.labels
        n = len(labels)
        if n == 0:
            return []

        cx = width / 2.0
        cy = height / 2.0
        max_scale = 0.80
        min_scale = 0.25

        boxes: list[BBox] = []
        for idx, label in enumerate(labels):
            if n == 1:
                scale = 0.60
            else:
                scale = max_scale - idx * ((max_scale - min_scale) / (n - 1))

            box_w = width * scale
            box_h = height * scale

            boxes.append(
                BBox(
                    label=label,
                    xtl=round(cx - box_w / 2.0, 2),
                    ytl=round(cy - box_h / 2.0, 2),
                    xbr=round(cx + box_w / 2.0, 2),
                    ybr=round(cy + box_h / 2.0, 2),
                    confidence=1.0,
                )
            )
        return boxes
