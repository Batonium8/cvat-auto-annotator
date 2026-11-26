from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    cvat_host: str = "http://localhost:8080"
    cvat_user: str = "admin"
    cvat_pass: str = "admin"
    service_port: int = 8000

    # Опциональная предзагрузка моделей при старте (через запятую, напр. "yoloe")
    preload_models: str = ""

    # Настройки YOLOE
    yoloe_weights: str = "yoloe-11s-seg.pt"
    yoloe_conf: float = 0.25
    yoloe_imgsz: int = 640
    yoloe_device: str | None = None  # None или "auto" -> cuda если доступна, иначе cpu

    # Настройки DART (SAM3)
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

    def preload_model_list(self) -> list[str]:
        return [item.strip() for item in self.preload_models.split(",") if item.strip()]


settings = Settings()
