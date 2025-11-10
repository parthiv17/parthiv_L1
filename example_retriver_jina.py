# ...existing code...
#!/usr/bin/env python3
import os
import json
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np
import requests
from dotenv import load_dotenv

# load env
ROOT = Path(__file__).parent
load_dotenv(ROOT / ".env")

DATASET_FOLDER = Path(os.getenv("DATASET_STORAGE_FOLDER", "datasets"))
EMBEDDINGS_JSONL = DATASET_FOLDER / "embeddings.jsonl"
EMBEDDINGS_NPY = DATASET_FOLDER / "embeddings.npy"
IDS_JSON = DATASET_FOLDER / "ids.json"

# Jina config for embedding the query
JINA_API_KEY = os.getenv("JINA_API_KEY")
JINA_API_ENDPOINT = os.getenv("JINA_API_ENDPOINT", "https://api.jina.ai/v1/embeddings")
JINA_EMBEDDING_MODEL = os.getenv("JINA_EMBEDDING_MODEL", os.getenv("EMBEDDING_MODEL", "mxbai-embed-large"))

def _parse_jina_response(resp_json, batch_len: int):
    if isinstance(resp_json, dict):
        if "embeddings" in resp_json and isinstance(resp_json["embeddings"], list):
            return resp_json["embeddings"]
        if "data" in resp_json and isinstance(resp_json["data"], list):
            embeds = []
            for item in resp_json["data"]:
                if isinstance(item, dict) and "embedding" in item:
                    embeds.append(item["embedding"])
            if len(embeds) == batch_len:
                return embeds
        if "results" in resp_json and isinstance(resp_json["results"], list):
            embeds = []
            for item in resp_json["results"]:
                if isinstance(item, dict) and "embedding" in item:
                    embeds.append(item["embedding"])
            if len(embeds) == batch_len:
                return embeds
    if isinstance(resp_json, list) and len(resp_json) == batch_len and isinstance(resp_json[0], list):
        return resp_json
    return None

def embed_query_via_jina(text: str) -> Optional[List[float]]:
    if not JINA_API_KEY:
        print("No JINA_API_KEY set in .env; cannot embed query.")
        return None
    headers = {"Authorization": f"Bearer {JINA_API_KEY}", "Content-Type": "application/json", "Accept": "application/json"}
    payload_variants = [
        {"model": JINA_EMBEDDING_MODEL, "input": [text]},
        {"model": JINA_EMBEDDING_MODEL, "inputs": [text]},
        {"model": JINA_EMBEDDING_MODEL, "texts": [text]},
        {"model": JINA_EMBEDDING_MODEL, "input": [{"text": text}]},
    ]
    last_resp = None
    for payload in payload_variants:
        try:
            r = requests.post(JINA_API_ENDPOINT, headers=headers, json=payload, timeout=30)
            last_resp = (r.status_code, r.text)
            if r.status_code >= 400:
                continue
            data = r.json()
            embeds = _parse_jina_response(data, 1)
            if embeds:
                return embeds[0]
        except Exception:
            continue
    print("Failed to embed query via Jina. Last response:", last_resp)
    return None

def load_embeddings():
    # prefer numpy files for speed if present
    if EMBEDDINGS_NPY.exists() and IDS_JSON.exists():
        arr = np.load(EMBEDDINGS_NPY)
        with IDS_JSON.open("r", encoding="utf-8") as f:
            ids = json.load(f)
        # try to recover texts/metadata from embeddings.jsonl if exists, else set placeholders
        texts = {}
        if EMBEDDINGS_JSONL.exists():
            with EMBEDDINGS_JSONL.open("r", encoding="utf-8") as f:
                for line in f:
                    try:
                        obj = json.loads(line)
                        texts[obj["id"]] = {"text": obj.get("text", ""), "metadata": obj.get("metadata", {})}
                    except Exception:
                        continue
        docs = []
        for i, _id in enumerate(ids):
            md = texts.get(_id, {"text": "", "metadata": {}})
            docs.append({"id": _id, "text": md["text"], "metadata": md["metadata"], "embedding": arr[i]})
        return docs
    # fallback: read embeddings.jsonl
    if EMBEDDINGS_JSONL.exists():
        docs = []
        embeddings = []
        with EMBEDDINGS_JSONL.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    emb = np.array(obj["embedding"], dtype=np.float32)
                    docs.append({"id": obj.get("id"), "text": obj.get("text", ""), "metadata": obj.get("metadata", {}), "embedding": emb})
                except Exception:
                    continue
        return docs
    raise FileNotFoundError("No embeddings found. Run chunking/ingestion to produce datasets/embeddings.jsonl or embeddings.npy + ids.json")

def cosine_sim(a: np.ndarray, b: np.ndarray):
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom > 0 else 0.0

def retrieve(query: str, top_k: int = 5):
    docs = load_embeddings()
    if not docs:
        print("No docs loaded.")
        return
    q_emb = embed_query_via_jina(query)
    if q_emb is None:
        print("Cannot embed query; aborting retrieval.")
        return
    q_vec = np.array(q_emb, dtype=np.float32)
    scores = []
    for d in docs:
        s = cosine_sim(q_vec, np.array(d["embedding"], dtype=np.float32))
        scores.append((s, d))
    scores.sort(key=lambda x: x[0], reverse=True)
    results = scores[:top_k]
    for rank, (score, doc) in enumerate(results, start=1):
        print(f"Rank {rank} | score: {score:.4f} | id: {doc['id']}")
        txt = (doc["text"] or "")[:800].replace("\n", " ")
        print(f"  text preview: {txt}")
        print(f"  metadata: {doc.get('metadata')}\n")

if __name__ == "__main__":
    import sys
    if len(sys.argv) >= 2:
        q = " ".join(sys.argv[1:])
    else:
        q = input("Query: ").strip()
    retrieve(q, top_k=5)
# ...existing code...