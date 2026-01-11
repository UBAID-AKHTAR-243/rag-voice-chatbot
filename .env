from pydantic_settings import BaseSettings
from pydantic import Field

class Settings(BaseSettings):
    hf_api_token: str = Field(..., description="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")
    embedding_model: str = "paraphrase-multilingual-MiniLM-L12-v2"
    llm_model_name: str = "meta-llama/Llama-3.2-1B-Instruct"
    persist_directory: str = "./vector_store"
    chunk_size: int = 500
    chunk_overlap: int = 50

    class Config:
        env_file = ".env"

settings = Settings()
