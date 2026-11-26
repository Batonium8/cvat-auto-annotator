import logging
import threading

from cvat_sdk.api_client.exceptions import ApiException, NotFoundException
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from urllib3.exceptions import MaxRetryError

from adapters import DartAdapter, MatryoshkaAdapter, YoloeAdapter
from core import BaseDetectorAdapter, ModelRegistry
from service.config import settings
from service.cvat_annotator import annotate_task, get_task_meta

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def _build_registry() -> ModelRegistry:
    registry = ModelRegistry()
    registry.register(MatryoshkaAdapter())
    registry.register(
        YoloeAdapter(
            weights=settings.yoloe_weights,
            conf=settings.yoloe_conf,
            imgsz=settings.yoloe_imgsz,
            device=settings.yoloe_device,
        )
    )
    registry.register(
        DartAdapter(
            checkpoint_path=settings.dart_checkpoint,
            conf=settings.dart_conf,
            nms=settings.dart_nms,
            imgsz=settings.dart_imgsz,
            device=settings.dart_device,
        )
    )
    return registry

model_registry = _build_registry()

app = FastAPI(
    title="CVAT Auto-Annotator API",
    version="0.2.1",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8080",
        "http://127.0.0.1:8080",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class HealthResponse(BaseModel):
    status: str


class ModelsResponse(BaseModel):
    models: list[str]


class AnnotateRequest(BaseModel):
    task_id: int = Field(..., gt=0, description="ID задачи в CVAT")
    model: str = Field(default="pseudo", description="Имя адаптера из реестра")
    prompts: dict[str, str] | None = Field(
        default=None,
        description="Опциональный маппинг классов CVAT в текстовые промпты",
    )
    clear_existing: bool = Field(
        default=False,
        description="Очищать существующие аннотации перед записью новых",
    )


class AnnotateResponse(BaseModel):
    status: str
    task_id: int
    model: str
    frames_processed: int | None = None
    shapes_created: int | None = None
    labels: list[str] = Field(default_factory=list)
    message: str


class TaskStatusResponse(BaseModel):
    task_id: int
    status: str  # running | completed | failed
    model: str
    frames_processed: int = 0
    total_frames: int = 0
    shapes_created: int = 0
    labels: list[str] = Field(default_factory=list)
    message: str = ""
    error: str | None = None


_task_statuses: dict[int, TaskStatusResponse] = {}
_tasks_lock = threading.Lock()


def _run_annotation_job(
    task_id: int,
    adapter: BaseDetectorAdapter,
    prompts: dict[str, str] | None,
    clear_existing: bool,
) -> None:
    """Фоновая обёртка (строго sync def, чтобы FastAPI гонял её в threadpool)."""

    def _on_progress(processed: int, total: int, shapes_count: int) -> None:
        with _tasks_lock:
            current = _task_statuses.get(task_id)
            if current is not None and current.status == "running":
                _task_statuses[task_id] = current.model_copy(
                    update={
                        "frames_processed": processed,
                        "total_frames": total,
                        "shapes_created": shapes_count,
                        "message": (
                            f"Обработано кадров: {processed}/{total} "
                            f"(рамок: {shapes_count})"
                        ),
                    }
                )

    try:
        result = annotate_task(
            task_id=task_id,
            adapter=adapter,
            prompts=prompts,
            clear_existing=clear_existing,
            on_progress=_on_progress,
        )
        with _tasks_lock:
            _task_statuses[task_id] = TaskStatusResponse(
                task_id=result.task_id,
                status="completed",
                model=result.model,
                frames_processed=result.frames_processed,
                total_frames=result.frames_processed,
                shapes_created=result.shapes_created,
                labels=list(result.labels),
                message=(
                    f"Завершено [{result.model}]: кадров {result.frames_processed}, "
                    f"рамок {result.shapes_created}"
                ),
            )
        logger.info(
            "Задача #%d завершена [%s]: кадров=%d, рамок=%d",
            result.task_id,
            result.model,
            result.frames_processed,
            result.shapes_created,
        )
    except Exception as exc:
        logger.exception(
            "Ошибка при фоновой разметке задачи #%d (модель '%s')",
            task_id,
            adapter.name,
        )
        with _tasks_lock:
            current = _task_statuses.get(task_id)
            if current is not None:
                _task_statuses[task_id] = current.model_copy(
                    update={
                        "status": "failed",
                        "error": str(exc),
                        "message": f"Ошибка авторазметки: {exc}",
                    }
                )
            else:
                _task_statuses[task_id] = TaskStatusResponse(
                    task_id=task_id,
                    status="failed",
                    model=adapter.name,
                    error=str(exc),
                    message=f"Ошибка авторазметки: {exc}",
                )
    finally:
        with _tasks_lock:
            current = _task_statuses.get(task_id)
            if current is not None and current.status == "running":
                _task_statuses[task_id] = current.model_copy(
                    update={
                        "status": "failed",
                        "error": "Фоновая задача была прервана",
                        "message": "Фоновая задача была прервана",
                    }
                )


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.get("/models", response_model=ModelsResponse)
def get_models() -> ModelsResponse:
    return ModelsResponse(models=model_registry.list_models())


@app.get("/status/{task_id}", response_model=TaskStatusResponse)
def get_task_status(task_id: int) -> TaskStatusResponse:
    with _tasks_lock:
        status = _task_statuses.get(task_id)
    if status is None:
        raise HTTPException(
            status_code=404,
            detail=f"Для задачи #{task_id} ещё нет истории запусков в текущей сессии",
        )
    return status


@app.post("/annotate", response_model=AnnotateResponse, status_code=202)
def annotate(
    req: AnnotateRequest, background_tasks: BackgroundTasks
) -> AnnotateResponse:
    try:
        adapter = model_registry.get(req.model)
    except KeyError as exc:
        detail = exc.args[0] if exc.args else str(exc)
        raise HTTPException(status_code=400, detail=detail) from exc

    try:
        meta = get_task_meta(req.task_id)
    except NotFoundException as exc:
        raise HTTPException(
            status_code=404,
            detail=f"Задача #{req.task_id} не найдена в CVAT",
        ) from exc
    except (ApiException, MaxRetryError) as exc:
        raise HTTPException(
            status_code=502,
            detail=f"CVAT API недоступен: {exc}",
        ) from exc

    with _tasks_lock:
        current = _task_statuses.get(req.task_id)
        if current is not None and current.status == "running":
            raise HTTPException(
                status_code=409,
                detail=f"Задача #{req.task_id} уже находится в процессе авторазметки",
            )
        _task_statuses[req.task_id] = TaskStatusResponse(
            task_id=req.task_id,
            status="running",
            model=adapter.name,
            frames_processed=0,
            total_frames=meta.total_frames,
            shapes_created=0,
            labels=list(meta.labels),
            message=f"Задача #{req.task_id} запущена (модель: {adapter.name})...",
        )

    try:
        background_tasks.add_task(
            _run_annotation_job,
            req.task_id,
            adapter,
            req.prompts,
            req.clear_existing,
        )
    except Exception:
        with _tasks_lock:
            _task_statuses.pop(req.task_id, None)
        raise

    return AnnotateResponse(
        status="started",
        task_id=req.task_id,
        model=adapter.name,
        labels=list(meta.labels),
        message=(
            f"Задача #{req.task_id} поставлена в очередь (модель: {adapter.name}). "
            "Статус доступен по GET /status/{task_id}."
        ),
    )
