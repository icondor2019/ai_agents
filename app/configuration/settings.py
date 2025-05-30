import os
from dotenv import load_dotenv
from pydantic_settings import BaseSettings

load_dotenv('.env')


class Settings(BaseSettings):
    # Gemini
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY")

    # Google sheets
    GSHEET_CRED_PATH: str = os.getenv("GSHEET_CRED_PATH")
    GSHEET_KEY: str = os.getenv("GSHEET_KEY")
    GSHEET_SHEET_BASE: str = 'base'

    # OpenAi
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY")

    # Single Store db
    SINGLE_STORE_USER: str = os.getenv("SINGLE_STORE_USER")
    SINGLE_STORE_PASSWORD: str = os.getenv("SINGLE_STORE_PASSWORD")
    SINGLE_STORE_HOST: str = os.getenv("SINGLE_STORE_HOST")
    SINGLE_STORE_PORT: str = os.getenv("SINGLE_STORE_PORT")
    SINGLE_STORE_DB: str = os.getenv("SINGLE_STORE_DB")
    SINGLE_STORE_CA: str = os.getenv("SINGLE_STORE_CA")

    # Telegram bot
    TELEGRAM_BOT_TOKEN: str = os.getenv("TELEGRAM_BOT_TOKEN")
    TELEGRAM_BOT_USER: str = os.getenv("TELEGRAM_BOT_USER")

    # huggingface
    HUGGINGFACEHUB_API_TOKEN: str = os.getenv("HUGGINGFACEHUB_API_TOKEN")

    # FastAPI docs
    OPEN_API_PATH: str = "/swagger-ui"
    OPEN_API_JSON: str = "/openapi.json"

    # Groq
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY")

    class Config():
        case_sensitive = True


settings = Settings()
