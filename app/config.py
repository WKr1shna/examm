import os
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent

EXACT_ABSTENTION_MESSAGE = (
    "I couldn't find this information in the provided PDF. "
    "I can only answer questions based on the contents of the Grand Bistro Culinary Recipe Manual."
)

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=str(BASE_DIR / ".env"), extra="ignore")

    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")
    PDF_PATH: str = str(BASE_DIR / "grand_bistro_culinary_recipe_manual_50p.pdf")
    CHROMA_PERSIST_DIR: str = str(BASE_DIR / "chroma_db")
    EMBEDDING_MODEL: str = "models/gemini-embedding-001"
    LLM_MODEL: str = "gemini-2.0-flash"
    FALLBACK_MODELS: list = ["gemini-2.0-flash", "gemini-1.5-flash"]
    TOP_K: int = 5
    RRF_K: int = 60
    FASTAPI_PORT: int = 8000
    STREAMLIT_PORT: int = 8501
    ABSTENTION_MESSAGE: str = EXACT_ABSTENTION_MESSAGE

settings = Settings()

if not os.environ.get("GEMINI_API_KEY") and settings.GEMINI_API_KEY:
    os.environ["GEMINI_API_KEY"] = settings.GEMINI_API_KEY
if not os.environ.get("GOOGLE_API_KEY") and settings.GOOGLE_API_KEY:
    os.environ["GOOGLE_API_KEY"] = settings.GOOGLE_API_KEY
