from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    cvat_host: str = "http://localhost:8080"
    cvat_user: str = "admin"
    cvat_pass: str = "admin"
    service_port: int = 8000

    # Настройки YOLOE (Этап 2)
    yoloe_weights: str = "yoloe-11s-seg.pt"
    yoloe_conf: float = 0.25
    yoloe_imgsz: int = 640
    yoloe_device: str | None = None  # None -> автовыбор ("cuda", если доступна, иначе "cpu")
    
    dart_checkpoint: str | None = None
    dart_conf: float = 0.3
    dart_nms: float = 0.5
    dart_imgsz: int = 1008
    dart_device: str = "cpu"    
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
