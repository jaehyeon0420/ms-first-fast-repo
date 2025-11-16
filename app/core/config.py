from pydantic_settings import BaseSettings
from pathlib import Path

# 프로젝트 최상단 경로
BASE_DIR = Path(__file__).resolve().parents[2]  # app/core/config.py 기준으로 상위 2단계

class Settings(BaseSettings):
    CUSTOM_VISION_KEY: str
    CUSTOM_VISION_ENDPOINT: str
    CUSTOM_VISION_PROJECT_ID : str
    CUSTOM_VISION_MODEL_NAME : str
    
    
    class Config:
        env_file = BASE_DIR / ".env"  # .env 절대 경로 지정

settings = Settings()