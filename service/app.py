from cvat_sdk.api_client.exceptions import ApiException, NotFoundException
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from adapters import list_available_models
from service.cvat_annotator import annotate_task

app = FastAPI(
    title="CVAT Auto-Annotator API",
    version="0.1.0",
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
    frames_processed: int
    shapes_created: int
    labels: list[str]
    message: str


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.get("/models", response_model=ModelsResponse)
def get_models() -> ModelsResponse:
    return ModelsResponse(models=list_available_models())


@app.post("/annotate", response_model=AnnotateResponse)
def annotate(req: AnnotateRequest) -> AnnotateResponse:
    try:
        result = annotate_task(
            task_id=req.task_id,
            model_name=req.model,
            prompts=req.prompts,
            clear_existing=req.clear_existing,
        )
    except KeyError as exc:
        detail = exc.args[0] if exc.args else str(exc)
        raise HTTPException(status_code=400, detail=detail) from exc
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

    return AnnotateResponse(
        status="success",
        task_id=result.task_id,
        model=result.model,
        frames_processed=result.frames_processed,
        shapes_created=result.shapes_created,
        labels=list(result.labels),
        message=(
            f"Обработано кадров: {result.frames_processed}, "
            f"создано рамок: {result.shapes_created}"
        ),
    )
