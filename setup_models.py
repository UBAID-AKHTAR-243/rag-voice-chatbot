# setup_models.py
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from TTS.utils.manage import ModelManager
from TTS.utils.downloader import download_model
import whisper
from config import settings

def download_all():
    print("Downloading embedding model...")
    # This will be cached by sentence-transformers
    from sentence_transformers import SentenceTransformer
    SentenceTransformer(settings.embedding_model)

    print("Downloading LLM tokenizer and model...")
    AutoTokenizer.from_pretrained(settings.llm_model_name, token=settings.hf_api_token)
    AutoModelForCausalLM.from_pretrained(settings.llm_model_name, token=settings.hf_api_token)

    print("Downloading Whisper model...")
    whisper.load_model("base")  # Or "small", "medium" based on your needs

    print("Downloading Coqui XTTS v2...")
    manager = ModelManager()
    model_path, config_path, model_item = download_model(manager, "tts_models/multilingual/multi-dataset/xtts_v2")

    print("All models downloaded and ready!")

if __name__ == "__main__":
    download_all()
