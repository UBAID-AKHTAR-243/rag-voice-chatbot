from pydantic import BaseSettings
from typing import List
import os

class Settings(BaseSettings):
    app_name: str = "RAG Voice Chatbot"
    data_dir: str = os.getenv("DATA_DIR", "data")
    docs_dir: str = os.getenv("DOCS_DIR", "data/documents")
    index_dir: str = os.getenv("INDEX_DIR", "data/embeddings")
    index_file: str = os.getenv("INDEX_FILE", "data/embeddings/faiss.index")
    meta_file: str = os.getenv("META_FILE", "data/embeddings/meta.jsonl")
    voice_sample: str = os.getenv("VOICE_SAMPLE", "data/voice/sample.wav")
    max_upload_size_mb: int = int(os.getenv("MAX_UPLOAD_MB", "50"))

    allowed_doc_types: List[str] = [
        "text/plain",
        "text/markdown",
        "application/pdf",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "application/octet-stream",
    ]
    allowed_audio_types: List[str] = ["audio/wav", "audio/x-wav", "audio/mpeg", "audio/mp3", "audio/ogg"]

    # Models
    stt_model_name: str = os.getenv("STT_MODEL", "base")
    llama_model_id: str = os.getenv("LLAMA_MODEL", "meta-llama/Llama-3.2-3B-Instruct")
    embedding_model_id: str = os.getenv("EMBED_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

    # Generation Controls
    max_ctx_docs: int = int(os.getenv("MAX_CTX_DOCS", "5"))
    max_gen_tokens: int = int(os.getenv("MAX_GEN_TOKENS", "1024"))
    temperature: float = float(os.getenv("TEMP", "0.2"))
    language_default: str = os.getenv("LANG_DEFAULT", "en")

    # Security / Ops
    enable_rate_limit: bool = os.getenv("RATE_LIMIT", "true").lower() == "true"
    rate_limit_rps: float = float(os.getenv("RATE_LIMIT_RPS", "3.0"))
    cors_origins: List[str] = os.getenv("CORS_ORIGINS", "*").split(",")

settings = Settings()
