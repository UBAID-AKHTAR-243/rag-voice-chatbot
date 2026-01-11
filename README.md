# rag-voice-chatbot
rag + LLM +voice

# RAG Voice Chatbot

- STT: Whisper
- LLM: Llama 3.2 (Hugging Face)
- TTS: Coqui XTTS v2
- Embeddings: paraphrase-multilingual-MiniLM-L12-v2
- Vector store: FAISS

## Run locally

```bash
docker-compose up --build



# 1. Clone and configure
git clone <repo-url>
cd rag-voice-chatbot
cp .env.example .env  # Create an example file first
# Edit .env with your HF token

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download models (can take time, bandwidth)
python setup_models.py

# 4. Add your documents to ./data folder
mkdir data
cp your_document.pdf ./data/

# 5. Ingest documents into the vector database
python ingest_documents.py

# 6. Run the API server
uvicorn main:app --reload --host 0.0.0.0 --port 8000
