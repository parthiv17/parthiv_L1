# Local-RAG-with-Ollama

Small local RAG demo that:
- Scrapes Wikipedia (scripted)
- Chunks documents and creates embeddings (Jina API or local Ollama fallback)
- Simple Streamlit UI for retrieval & debugging

---

## Prerequisites

- Python 3.9+ (64-bit)
- PowerShell / Terminal on Windows
- (Optional) Ollama installed and running locally if you want local embeddings
- Jina API access if using Jina embeddings endpoint

Install Python dependencies:
```powershell
python -m pip install --upgrade pip setuptools wheel
python -m pip install python-dotenv requests numpy pandas openpyxl streamlit
# Optional (for Ollama/local embeddings)
python -m pip install langchain_ollama ollama
```

---

## Important files

- `4_scraping_wikipedia_jina.py` — scrape keywords.xlsx -> datasets/*.txt
- `5_chunking_embedding_ingestion_jina.py` — chunk texts and call Jina embeddings API (recommended)
- `6_chatboat_jina.py` — Streamlit UI using local embeddings + Jina for query embedding
- `datasets/` — produced .txt, pending files, embeddings.jsonl, embeddings.npy, ids.json, debug files

---

## .env (example)

Create a `.env` in repo root or edit existing:

```text
# filepath: c:\Users\ParthivShah\Source\AI_L1_HandsOn\Local-RAG-with-Ollama\.env
DATASET_STORAGE_FOLDER=datasets
SNAPSHOT_STORAGE_FILE=snapshot.txt

# Jina API (if using Jina)
JINA_API_KEY=your_jina_api_key_here
JINA_API_ENDPOINT=https://api.jina.ai/v1/embeddings
# Use a supported Jina tag; recommended: jina-embeddings-v4
JINA_EMBEDDING_MODEL=jina-embeddings-v4
JINA_BATCH_SIZE=8

# Ollama local model (if embedding locally)
EMBEDDING_MODEL=mxbai-embed-large

# Optional UI settings
DATABASE_LOCATION=chroma_db
COLLECTION_NAME=local_collection
```

Notes:
- If Jina returns 422, set `JINA_EMBEDDING_MODEL` to one of the allowed tags (e.g. `jina-embeddings-v4`).
- If using Ollama locally, ensure the model is pulled: `ollama pull <model>` and `.env` EMBEDDING_MODEL matches a local model.

---

## Typical workflow (Windows PowerShell)

1. Scrape Wikipedia (creates `.txt` files under datasets/):
```powershell
python .\4_scraping_wikipedia_jina.py
```

2. Produce embeddings via Jina:
```powershell
python .\5_chunking_embedding_ingestion_jina.py
```
- On success you'll have `datasets/embeddings.jsonl` and `datasets/embeddings.npy` + `datasets/ids.json`.
- If script writes `datasets/pending_embeddings.jsonl`, you can embed locally using Ollama (see next step).

3. Launch Streamlit UI:
```powershell
streamlit run .\6_chatboat_jina.py
```
- Open browser at the Streamlit URL shown in terminal.
- Sidebar shows diagnostics (embeddings count, debug info).

---

## Troubleshooting

- Blank Streamlit UI or nothing loads:
  - Check terminal where `streamlit run` is running for Python exceptions.
  - Verify `datasets/embeddings.jsonl` or `embeddings.npy + ids.json` exist.

- Jina 422 Unprocessable Entity:
  - Error indicates the model tag in `.env` is not accepted by Jina.
  - Update `.env` to a supported tag, e.g. `JINA_EMBEDDING_MODEL=jina-embeddings-v4`.
  - The ingestion script attempts to parse the 422 and retry with a suggested tag.

- chromadb / Chroma DLL error (ImportError: chromadb_rust_bindings):
  - The project has been moved to use Jina/local-Numpy retriever to avoid chromadb native dependency.
  - If you still need chromadb, ensure:
    - 64-bit Python
    - Microsoft Visual C++ Redistributable (x64) installed: https://aka.ms/vs/17/release/vc_redist.x64.exe
    - Reinstall `chromadb` in your environment.

- Ollama embedding fails:
  - Check local models: `ollama models`
  - Pull model: `ollama pull <model>`
  - Ensure `EMBEDDING_MODEL` in `.env` matches a local model name.

---

## Developer notes

- Embeddings are stored as newline JSONL (`datasets/embeddings.jsonl`) and as `embeddings.npy` + `ids.json` for fast retrieval.
- The Streamlit app uses a local cosine similarity retriever (numpy) and the Jina API for query embeddings (or Ollama when you choose the tools script).
- If ingestion fails, `datasets/jina_debug_failed_request.json` will contain debugging details.

---

If you want, I can:
- Generate a minimal `requirements.txt` from the environment.
- Add a CONTRIBUTING or short quickstart script to automate the three-step flow.
```# filepath: c:\Users\ParthivShah\Source\AI_L1_HandsOn\Local-RAG-with-Ollama\README.md
# Local-RAG-with-Ollama

Small local RAG demo that:
- Scrapes Wikipedia (scripted)
- Chunks documents and creates embeddings (Jina API or local Ollama fallback)
- Simple Streamlit UI for retrieval & debugging

---

## Prerequisites

- Python 3.9+ (64-bit)
- PowerShell / Terminal on Windows
- (Optional) Ollama installed and running locally if you want local embeddings
- Jina API access if using Jina embeddings endpoint

Install Python dependencies:
```powershell
python -m pip install --upgrade pip setuptools wheel
python -m pip install python-dotenv requests numpy pandas openpyxl streamlit
# Optional (for Ollama/local embeddings)
python -m pip install langchain_ollama ollama
```

---

## Important files

- `4_scraping_wikipedia_jina.py` — scrape keywords.xlsx -> datasets/*.txt
- `5_chunking_embedding_ingestion_jina.py` — chunk texts and call Jina embeddings API (recommended)
- `6_chatboat_jina.py` — Streamlit UI using local embeddings + Jina for query embedding
- `datasets/` — produced .txt, pending files, embeddings.jsonl, embeddings.npy, ids.json, debug files

---

## .env (example)

Create a `.env` in repo root or edit existing:

```text
# filepath: c:\Users\ParthivShah\Source\AI_L1_HandsOn\Local-RAG-with-Ollama\.env
DATASET_STORAGE_FOLDER=datasets
SNAPSHOT_STORAGE_FILE=snapshot.txt

# Jina API (if using Jina)
JINA_API_KEY=your_jina_api_key_here
JINA_API_ENDPOINT=https://api.jina.ai/v1/embeddings
# Use a supported Jina tag; recommended: jina-embeddings-v4
JINA_EMBEDDING_MODEL=jina-embeddings-v4
JINA_BATCH_SIZE=8

# Ollama local model (if embedding locally)
EMBEDDING_MODEL=mxbai-embed-large

# Optional UI settings
DATABASE_LOCATION=chroma_db
COLLECTION_NAME=local_collection
```

Notes:
- If Jina returns 422, set `JINA_EMBEDDING_MODEL` to one of the allowed tags (e.g. `jina-embeddings-v4`).
- If using Ollama locally, ensure the model is pulled: `ollama pull <model>` and `.env` EMBEDDING_MODEL matches a local model.

---

## Typical workflow (Windows PowerShell)

1. Scrape Wikipedia (creates `.txt` files under datasets/):
```powershell
python .\4_scraping_wikipedia_jina.py
```

2. Produce embeddings via Jina:
```powershell
python .\5_chunking_embedding_ingestion_jina.py
```
- On success you'll have `datasets/embeddings.jsonl` and `datasets/embeddings.npy` + `datasets/ids.json`.
- If script writes `datasets/pending_embeddings.jsonl`, you can embed locally using Ollama (see next step).

3. Launch Streamlit UI:
```powershell
streamlit run .\6_chatboat_jina.py
```
- Open browser at the Streamlit URL shown in terminal.
- Sidebar shows diagnostics (embeddings count, debug info).

---

## Troubleshooting

- Blank Streamlit UI or nothing loads:
  - Check terminal where `streamlit run` is running for Python exceptions.
  - Verify `datasets/embeddings.jsonl` or `embeddings.npy + ids.json` exist.

- Jina 422 Unprocessable Entity:
  - Error indicates the model tag in `.env` is not accepted by Jina.
  - Update `.env` to a supported tag, e.g. `JINA_EMBEDDING_MODEL=jina-embeddings-v4`.
  - The ingestion script attempts to parse the 422 and retry with a suggested tag.

- chromadb / Chroma DLL error (ImportError: chromadb_rust_bindings):
  - The project has been moved to use Jina/local-Numpy retriever to avoid chromadb native dependency.
  - If you still need chromadb, ensure:
    - 64-bit Python
    - Microsoft Visual C++ Redistributable (x64) installed: https://aka.ms/vs/17/release/vc_redist.x64.exe
    - Reinstall `chromadb` in your environment.

- Ollama embedding fails:
  - Check local models: `ollama models`
  - Pull model: `ollama pull <model>`
  - Ensure `EMBEDDING_MODEL` in `.env` matches a local model name.

---

## Developer notes

- Embeddings are stored as newline JSONL (`datasets/embeddings.jsonl`) and as `embeddings.npy` + `ids.json` for fast retrieval.
- The Streamlit app uses a local cosine similarity retriever (numpy) and the Jina API for query embeddings (or Ollama when you choose the tools script).
- If ingestion fails, `datasets/jina_debug_failed_request.json` will contain debugging details.

---

If you want, I can:
- Generate a minimal `requirements.txt` from the environment.
- Add a CONTRIBUTING or short quickstart script to automate the three-step flow.