#!/usr/bin/env python3
import os
import re
import json
from pathlib import Path
from datetime import datetime
from typing import List, Optional

import requests
import numpy as np
import streamlit as st
from dotenv import load_dotenv

# --- load env & paths ---
ROOT = Path(__file__).parent
load_dotenv(ROOT / ".env")

DATASET_FOLDER = Path(os.getenv("DATASET_STORAGE_FOLDER", "datasets"))
EMBEDDINGS_JSONL = DATASET_FOLDER / "embeddings.jsonl"
EMBEDDINGS_NPY = DATASET_FOLDER / "embeddings.npy"
IDS_JSON = DATASET_FOLDER / "ids.json"
JINA_DEBUG = DATASET_FOLDER / "jina_debug_failed_request.json"
PENDING_JSONL = DATASET_FOLDER / "pending_embeddings.jsonl"

JINA_API_KEY = os.getenv("JINA_API_KEY")
JINA_API_ENDPOINT = os.getenv("JINA_API_ENDPOINT", "https://api.jina.ai/v1/embeddings")
JINA_EMBEDDING_MODEL = os.getenv("JINA_EMBEDDING_MODEL", os.getenv("EMBEDDING_MODEL", "mxbai-embed-large"))
BATCH_SIZE = int(os.getenv("JINA_BATCH_SIZE", "8"))

# --- utilities ---
def ensure_folder(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def _parse_jina_response(resp_json, batch_len: int) -> Optional[List]:
    if isinstance(resp_json, dict):
        if "embeddings" in resp_json and isinstance(resp_json["embeddings"], list):
            return resp_json["embeddings"]
        if "data" in resp_json and isinstance(resp_json["data"], list):
            embeddings = []
            for item in resp_json["data"]:
                if isinstance(item, dict) and "embedding" in item:
                    embeddings.append(item["embedding"])
            if len(embeddings) == batch_len:
                return embeddings
        if "results" in resp_json and isinstance(resp_json["results"], list):
            embeddings = []
            for item in resp_json["results"]:
                if isinstance(item, dict) and "embedding" in item:
                    embeddings.append(item["embedding"])
            if len(embeddings) == batch_len:
                return embeddings
    if isinstance(resp_json, list) and len(resp_json) == batch_len and isinstance(resp_json[0], list):
        return resp_json
    return None

def _extract_allowed_tags_from_error(resp_json) -> list:
    try:
        text = ""
        if isinstance(resp_json, dict):
            text = resp_json.get("detail", "") or ""
            errs = resp_json.get("errors")
            if not text and isinstance(errs, list) and errs:
                text = errs[0].get("message", "") or ""
        if not text:
            text = json.dumps(resp_json)
        matches = re.findall(r"'([a-zA-Z0-9\-_\.]+)'", text)
        candidates = [m for m in matches if ("jina-" in m or "embeddings" in m or "code" in m or "clip" in m or "colbert" in m)]
        return candidates
    except Exception:
        return []

def embed_query_via_jina(text: str) -> Optional[List[float]]:
    if not JINA_API_KEY:
        return None
    headers = {"Authorization": f"Bearer {JINA_API_KEY}", "Content-Type": "application/json", "Accept": "application/json"}
    variants = [
        {"model": JINA_EMBEDDING_MODEL, "input": [text]},
        {"model": JINA_EMBEDDING_MODEL, "inputs": [text]},
        {"model": JINA_EMBEDDING_MODEL, "data": [text]},
        {"model": JINA_EMBEDDING_MODEL, "texts": [text]},
        {"model": JINA_EMBEDDING_MODEL, "input": [{"text": text}]},
    ]
    last_status = None
    last_resp_text = None
    attempted_override = None

    for payload in variants:
        try:
            r = requests.post(JINA_API_ENDPOINT, headers=headers, json=payload, timeout=30)
            last_status = r.status_code
            last_resp_text = r.text
            if r.status_code == 422:
                try:
                    resp_json = r.json()
                    allowed = _extract_allowed_tags_from_error(resp_json)
                    if allowed:
                        suggested = allowed[0]
                        if suggested != payload.get("model"):
                            new_payload = dict(payload)
                            new_payload["model"] = suggested
                            attempted_override = suggested
                            r2 = requests.post(JINA_API_ENDPOINT, headers=headers, json=new_payload, timeout=30)
                            last_status = r2.status_code
                            last_resp_text = r2.text
                            if r2.status_code >= 400:
                                continue
                            data2 = r2.json()
                            emb2 = _parse_jina_response(data2, 1)
                            if emb2:
                                st.sidebar.info(f"Used suggested Jina tag '{suggested}'. Set JINA_EMBEDDING_MODEL in .env to persist.")
                                return emb2[0]
                            continue
                except Exception:
                    pass
                continue
            if r.status_code >= 400:
                continue
            data = r.json()
            emb = _parse_jina_response(data, 1)
            if emb:
                if attempted_override:
                    st.sidebar.info(f"Used runtime override '{attempted_override}'. Consider updating .env")
                return emb[0]
        except Exception:
            continue

    # write debug file
    debug = {"endpoint": JINA_API_ENDPOINT, "model": JINA_EMBEDDING_MODEL, "last_status": last_status, "last_response": last_resp_text}
    ensure_folder(DATASET_FOLDER)
    with JINA_DEBUG.open("w", encoding="utf-8") as df:
        json.dump(debug, df, ensure_ascii=False, indent=2)
    return None

def load_local_embeddings():
    # prefer numpy index
    if EMBEDDINGS_NPY.exists() and IDS_JSON.exists():
        arr = np.load(EMBEDDINGS_NPY)
        with IDS_JSON.open("r", encoding="utf-8") as f:
            ids = json.load(f)
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
    # fallback to jsonl
    if EMBEDDINGS_JSONL.exists():
        docs = []
        with EMBEDDINGS_JSONL.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    docs.append({"id": obj.get("id"), "text": obj.get("text", ""), "metadata": obj.get("metadata", {}), "embedding": np.array(obj["embedding"], dtype=np.float32)})
                except Exception:
                    continue
        return docs
    return []

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom > 0 else 0.0

class LocalRetriever:
    def __init__(self):
        self.docs = load_local_embeddings()
        for d in self.docs:
            if not isinstance(d["embedding"], np.ndarray):
                d["embedding"] = np.array(d["embedding"], dtype=np.float32)

    def get_relevant_documents(self, query: str, k: int = 5):
        q_emb = embed_query_via_jina(query)
        if q_emb is None:
            return []
        qv = np.array(q_emb, dtype=np.float32)
        scored = []
        for d in self.docs:
            s = cosine_sim(qv, d["embedding"])
            scored.append((s, d))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [d for _, d in scored[:k]]

# --- Streamlit UI ---
st.set_page_config(page_title="Local RAG (Jina)", layout="wide")
st.title("Local RAG Chatbot (Jina embeddings)")

with st.sidebar:
    st.header("Diagnostics")
    st.write(f"DATASET_FOLDER: {DATASET_FOLDER}")
    n = 0
    try:
        docs_preview = load_local_embeddings()
        n = len(docs_preview)
    except Exception as e:
        st.error(f"Failed to load embeddings: {e}")
    st.write(f"Embeddings loaded: {n}")
    st.write(f"JINA_EMBEDDING_MODEL: {JINA_EMBEDDING_MODEL}")
    if not JINA_API_KEY:
        st.warning("No JINA_API_KEY set in .env; query embedding will fail.")
    if JINA_DEBUG.exists():
        st.subheader("Last Jina debug")
        try:
            st.code(JINA_DEBUG.read_text(encoding="utf-8")[:2000])
        except Exception:
            st.write("Could not read debug file.")
    if PENDING_JSONL.exists():
        st.warning("pending_embeddings.jsonl present — ingestion previously failed.")

st.markdown("Enter query below. If no results, check 'Diagnostics' in sidebar and the terminal.")

# instantiate retriever once
try:
    retriever = LocalRetriever()
except Exception as e:
    st.error(f"Failed to prepare retriever: {e}")
    retriever = None

query = st.text_input("Query", value="", placeholder="Ask about your documents...")
if st.button("Search") or (query and st.session_state.get("auto_search", False)):
    if not query.strip():
        st.info("Type a query first.")
    else:
        if retriever is None or not retriever.docs:
            st.error("No embeddings available. Run ingestion (2/5_chunking scripts) to produce embeddings.jsonl or embeddings.npy + ids.json.")
        else:
            with st.spinner("Embedding query and retrieving..."):
                try:
                    res = retriever.get_relevant_documents(query, k=5)
                except Exception as e:
                    st.error(f"Retrieval failed: {e}")
                    res = []
            if not res:
                st.info("No results. Check Jina settings and debug file in datasets/")
            else:
                for i, doc in enumerate(res, start=1):
                    st.subheader(f"Result {i} — id: {doc['id']}")
                    preview = (doc.get("text") or "")[:4000].replace("\n", " ")
                    st.write(preview)
                    st.json(doc.get("metadata", {}))

# small footer tips
with st.expander("Tips"):
    st.write("- Ensure datasets/embeddings.jsonl or embeddings.npy + ids.json exist.")
    st.write("- If ingestion failed, inspect datasets/jina_debug_failed_request.json and datasets/pending_embeddings.jsonl.")
    st.write("- To create embeddings locally with Ollama, use tools_embed_pending_with_ollama.py or run ingestion with a valid Jina model tag (e.g., jina-embeddings-v4).")