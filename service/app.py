import logging
import threading

from cvat_sdk.api_client.exceptions import ApiException, NotFoundException
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from adapters import get_adapter, list_available_models
from service.cvat_annotator import annotate_task, get_task_labels

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

_active_tasks: set[int] = set()
_active_tasks_lock = threading.Lock()

app = FastAPI(
    title="CVAT Auto-Annotator API",
    version="0.2.0",
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
        default=True,
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


def _run_annotation_job(
    task_id: int,
    model_name: str,
    prompts: dict[str, str] | None,
    clear_existing: bool,
) -> None:
    """Фоновая обёртка (строго sync def, чтобы FastAPI гонял её в threadpool)."""
    try:
        result = annotate_task(
            task_id=task_id,
            model_name=model_name,
            prompts=prompts,
            clear_existing=clear_existing,
        )
        logger.info(
            "Задача #%d завершена [%s]: кадров=%d, рамок=%d",
            result.task_id,
            result.model,
            result.frames_processed,
            result.shapes_created,
        )
    except Exception:
        logger.exception(
            "Ошибка при фоновой разметке задачи #%d (модель '%s')",
            task_id,
            model_name,
        )
    finally:
        with _active_tasks_lock:
            _active_tasks.discard(task_id)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.get("/models", response_model=ModelsResponse)
def get_models() -> ModelsResponse:
    return ModelsResponse(models=list_available_models())


@app.post("/annotate", response_model=AnnotateResponse, status_code=202)
def annotate(
    req: AnnotateRequest, background_tasks: BackgroundTasks
) -> AnnotateResponse:
    try:
        adapter = get_adapter(req.model)
    except KeyError as exc:
        detail = exc.args[0] if exc.args else str(exc)
        raise HTTPException(status_code=400, detail=detail) from exc

    try:
        labels = get_task_labels(req.task_id)
    except NotFoundException as exc:
        raise HTTPException(
            status_code=404,
            detail=f"Задача #{req.task_id} не найдена в CVAT",
        ) from exc
    except ApiException as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Ошибка CVAT API ({exc.status}): {exc.reason}",
        ) from exc

    with _active_tasks_lock:
        if req.task_id in _active_tasks:
            raise HTTPException(
                status_code=409,
                detail=f"Задача #{req.task_id} уже находится в процессе авторазметки",
            )
        _active_tasks.add(req.task_id)

    try:
        background_tasks.add_task(
            _run_annotation_job,
            req.task_id,
            adapter.name,
            req.prompts,
            req.clear_existing,
        )
    except Exception:
        with _active_tasks_lock:
            _active_tasks.discard(req.task_id)
        raise

    return AnnotateResponse(
        status="started",
        task_id=req.task_id,
        model=adapter.name,
        labels=labels,
        message=(
            f"Задача #{req.task_id} поставлена в очередь (модель: {adapter.name}). "
            "Обновите задание через несколько секунд."
        ),
    )
