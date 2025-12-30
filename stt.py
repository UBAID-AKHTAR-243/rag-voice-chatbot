import os
import tempfile
import torch
import whisper
import langcodes
import logging
from fastapi import UploadFile

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_NAME = os.getenv("STT_MODEL", "base")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
whisper_model = whisper.load_model(MODEL_NAME, device=DEVICE)

async def transcribe(audio_file: UploadFile) -> dict:
    """
    Transcribe an uploaded audio file to text and detect language.
    """
    try:
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(await audio_file.read())
            tmp_path = tmp.name

        # Run Whisper transcription
        result = whisper_model.transcribe(tmp_path)

        # Detect language
        language_code = result.get("language", "unknown")
        try:
            language_name = langcodes.Language.get(language_code).language_name()
        except Exception:
            language_name = "Unknown"

        return {
            "text": result.get("text", "").strip(),
            "language_code": language_code,
            "language_name": language_name
        }

    except Exception as e:
        logger.error(f"STT transcription failed: {e}")
        return {
            "text": "",
            "language_code": "error",
            "language_name": "Error",
            "error": str(e)
        }

    finally:
        if "tmp_path" in locals() and os.path.exists(tmp_path):
            os.remove(tmp_path)
