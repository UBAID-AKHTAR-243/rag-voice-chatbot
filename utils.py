import os
import re
import time
import logging
from pathlib import Path
from fastapi import UploadFile, HTTPException, Request
from typing import Dict

logger = logging.getLogger(__name__)

def ensure_dirs(*dirs):
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)

def safe_filename(name: str) -> str:
    name = re.sub(r"[^A-Za-z0-9_\-\.]", "_", name or "")
    return name[:128] if name else "file"

def save_upload(upload: UploadFile, dest_path: str, max_mb: int):
    data = upload.file.read()
    if len(data) > max_mb * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Uploaded file too large")
    with open(dest_path, "wb") as f:
        f.write(data)
    upload.file.seek(0)

# Simple in-memory token-bucket per IP
_client_buckets: Dict[str, Dict[str, float]] = {}

def rate_limit_check(request: Request, rps: float, enabled: bool):
    if not enabled:
        return
    ip = request.client.host if request.client else "unknown"
    now = time.time()
    bucket = _client_buckets.get(ip, {"tokens": rps, "last": now})
    elapsed = now - bucket["last"]
    bucket["tokens"] = min(rps, bucket["tokens"] + elapsed * rps)
    bucket["last"] = now
    if bucket["tokens"] < 1.0:
        raise HTTPException(status_code=429, detail="Too many requests. Slow down.")
    bucket["tokens"] -= 1.0
    _client_buckets[ip] = bucket
