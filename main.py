import os
import logging
from typing import List
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.utils import ensure_dirs, safe_filename, save_upload, rate_limit_check
from app.vectorstore import VectorStore, chunk_text
from app.stt import STTService
from app.tts import TTSService
from app.llm import LLMService

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s :: %(message)s")
logger = logging.getLogger("rag-voice-chatbot")

# Prepare dirs
ensure_dirs(settings.data_dir, settings.docs_dir, settings.index_dir, Path(settings.voice_sample).parent)

# App
app = FastAPI(title=settings.app_name)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins if settings.cors_origins != ["*"] else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Services
DEVICE = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"
vector_store = VectorStore(settings.index_file, settings.meta_file, settings.embedding_model_id, device=DEVICE)
stt_service = STTService(settings.stt_model_name)
tts_service = TTSService(settings.voice_sample, settings.language_default)
llm_service = LLMService(settings.llama_model_id, settings.max_gen_tokens, settings.temperature)

HTML_MENU = """
<!DOCTYPE html><html><head><meta charset="utf-8"><title>RAG Voice Chatbot</title>
<style>body{font-family:system-ui,sans-serif;margin:2rem}section{margin-bottom:2rem;padding:1rem;border:1px solid #ddd;border-radius:8px}
h2{margin-top:0}label{display:block;margin:.5rem 0}button{padding:.5rem 1rem}input[type=text]{width:100%;padding:.5rem}</style></head>
<body><h1>RAG Voice Chatbot</h1>
<section><h2>Upload knowledge files (vector store)</h2>
<form action="/ingest" method="post" enctype="multipart/form-data"><label>Files (txt, md, pdf, docx):
<input type="file" name="files" multiple /></label><button type="submit">Upload & Ingest</button></form></section>
<section><h2>Upload voice sample (sample.wav) for cloning</h2>
<form action="/set-voice" method="post" enctype="multipart/form-data"><label>Voice WAV:
<input type="file" name="voice" accept=".wav,audio/wav" /></label><button type="submit">Upload</button></form></section>
<section><h2>Chat with text</h2><form action="/chat" method="post">
<label>Your question: <input type="text" name="query" /></label>
<label>Reply language code (e.g., en, ur): <input type="text" name="language" value="en" /></label>
<label>Return audio? (true/false): <input type="text" name="return_audio" value="false" /></label>
<button type="submit">Ask</button></form></section>
<section><h2>Chat with voice</h2><form action="/chat-voice" method="post" enctype="multipart/form-data">
<label>Audio file: <input type="file" name="audio" accept="audio/*" /></label>
<label>Reply language code (e.g., en, ur): <input type="text" name="language" value="en" /></label>
<label>Return audio? (true/false): <input type="text" name="return_audio" value="true" /></label>
<button type="submit">Ask</button></form></section></body></html>
"""

@app.get("/", response_class=HTMLResponse)
async def menu():
    return HTML_MENU

@app.post("/set-voice")
async def set_voice(request: Request, voice: UploadFile = File(...)):
    rate_limit_check(request, settings.rate_limit_rps, settings.enable_rate_limit)
    if voice.content_type not in settings.allowed_audio_types:
        raise HTTPException(status_code=400, detail=f"Unsupported audio type: {voice.content_type}")
    dest = settings.voice_sample
    Path(dest).parent.mkdir(parents=True, exist_ok=True)
    save_upload(voice, dest, settings.max_upload_size_mb)
    return {"message": "Voice sample uploaded", "path": dest}

@app.post("/ingest")
async def ingest(request: Request, files: List[UploadFile] = File(...)):
    rate_limit_check(request, settings.rate_limit_rps, settings.enable_rate_limit)
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")
    added = 0
    for uf in files:
        if uf.content_type not in settings.allowed_doc_types and not uf.filename.lower().endswith((".txt", ".md", ".pdf", ".docx")):
            continue
        fname = safe_filename(uf.filename)
        dest_path = os.path.join(settings.docs_dir, fname)
        save_upload(uf, dest_path, settings.max_upload_size_mb)

        # Parse content
        text = extract_text(dest_path, uf.content_type)
        if not text.strip():
            continue
        chunks = chunk_text(text)
        sources = [fname] * len(chunks)
        if chunks:
            vector_store.add_texts(chunks, sources)
            added += len(chunks)
    return {"message": "Ingestion complete", "chunks_added": added}

def extract_text(path: str, mime: str) -> str:
    # Minimal parsers; extend as needed
    try:
        if mime in ("text/plain", "text/markdown") or path.lower().endswith((".txt", ".md")):
            return Path(path).read_text(encoding="utf-8", errors="ignore")
        if mime == "application/pdf" or path.lower().endswith(".pdf"):
            try:
                import fitz
            except Exception:
                raise HTTPException(status_code=400, detail="PDF support not available. Install PyMuPDF.")
            text = []
            with fitz.open(path) as doc:
                for page in doc:
                    text.append(page.get_text())
            return "\n".join(text)
        if mime == "application/vnd.openxmlformats-officedocument.wordprocessingml.document" or path.lower().endswith(".docx"):
            try:
                import docx
            except Exception:
                raise HTTPException(status_code=400, detail="DOCX support not available. Install python-docx.")
            d = docx.Document(path)
            return "\n".join(p.text for p in d.paragraphs)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Extract text error: {e}")
    # Fallback
    try:
        return Path(path).read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""

@app.post("/stt")
async def stt_endpoint(request: Request, audio: UploadFile = File(...)):
    rate_limit_check(request, settings.rate_limit_rps, settings.enable_rate_limit)
    if audio.content_type not in settings.allowed_audio_types:
        raise HTTPException(status_code=400, detail=f"Unsupported audio type: {audio.content_type}")
    return await stt_service.transcribe(audio)

@app.post("/chat")
async def chat_endpoint(
    request: Request,
    query: str = Form(...),
    language: str = Form(settings.language_default),
    return_audio: str = Form("false"),
):
    rate_limit_check(request, settings.rate_limit_rps, settings.enable_rate_limit)
    if not query.strip():
        raise HTTPException(status_code=400, detail="Empty query")
    docs = vector_store.search(query, k=settings.max_ctx_docs)
    prompt = llm_service.build_prompt(query, docs)
    answer = llm_service.generate(prompt)

    resp = {"text": answer, "context_docs": docs}
    if return_audio.lower() == "true":
        out_path = f"data/voice/tts_{safe_filename(os.urandom(8).hex())}.wav"
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        tts_service.synthesize(answer, out_path, language=language)
        resp["audio_path"] = out_path
    return resp

@app.post("/chat-voice")
async def chat_voice_endpoint(
    request: Request,
    audio: UploadFile = File(...),
    language: str = Form(settings.language_default),
    return_audio: str = Form("true"),
):
    rate_limit_check(request, settings.rate_limit_rps, settings.enable_rate_limit)
    if audio.content_type not in settings.allowed_audio_types:
        raise HTTPException(status_code=400, detail=f"Unsupported audio type: {audio.content_type}")

    stt_res = await stt_service.transcribe(audio)
    user_text = stt_res.get("text", "")
    if not user_text:
        raise HTTPException(status_code=400, detail="STT produced empty text")

    docs = vector_store.search(user_text, k=settings.max_ctx_docs)
    prompt = llm_service.build_prompt(user_text, docs)
    answer = llm_service.generate(prompt)

    resp = {"user_text": user_text, "text": answer, "context_docs": docs}
    if return_audio.lower() == "true":
        out_path = f"data/voice/tts_{safe_filename(os.urandom(8).hex())}.wav"
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        tts_service.synthesize(answer, out_path, language=language)
        resp["audio_path"] = out_path
    return resp

@app.get("/audio/{fname}")
async def get_audio(fname: str):
    path = os.path.join("data/voice", safe_filename(fname))
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Audio not found")
    return FileResponse(path, media_type="audio/wav")

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse({"error": exc.detail}, status_code=exc.status_code)

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse({"error": "Internal server error"}, status_code=500)

@app.on_event("startup")
async def startup():
    logger.info("Chatbot API starting up...")

@app.on_event("shutdown")
async def shutdown():
    logger.info("Chatbot API shutting down...")
