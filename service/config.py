from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    cvat_host: str = "http://localhost:8080"
    cvat_user: str = "admin"
    cvat_pass: str = "admin"
    service_port: int = 8000

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


settings = Settings()
