#!/usr/bin/env python3
import os
import re
import json
from pathlib import Path
from datetime import datetime
from typing import List, Optional

import requests
import numpy as np
from dotenv import load_dotenv

ROOT = Path(__file__).parent
load_dotenv(ROOT / ".env")

DATASET_FOLDER = Path(os.getenv("DATASET_STORAGE_FOLDER", "datasets"))
OUTPUT_EMBEDDINGS_FILE = DATASET_FOLDER / "embeddings.jsonl"
EMBEDDINGS_NPY = DATASET_FOLDER / "embeddings.npy"
IDS_JSON = DATASET_FOLDER / "ids.json"
PENDING_JSONL = DATASET_FOLDER / "pending_embeddings.jsonl"
DEBUG_JSON = DATASET_FOLDER / "jina_debug_failed_request.json"

JINA_API_KEY = os.getenv("JINA_API_KEY")
JINA_API_ENDPOINT = os.getenv("JINA_API_ENDPOINT", "https://api.jina.ai/v1/embeddings")
JINA_EMBEDDING_MODEL = os.getenv("JINA_EMBEDDING_MODEL", os.getenv("EMBEDDING_MODEL", "mxbai-embed-large"))
BATCH_SIZE = int(os.getenv("JINA_BATCH_SIZE", "8"))

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))


def ensure_folder(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def read_text_files(folder: Path) -> List[Path]:
    if not folder.exists():
        return []
    return sorted([p for p in folder.glob("*.txt") if p.is_file()])


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []
    if chunk_size <= 0:
        return [text]
    chunks = []
    stride = chunk_size - overlap if chunk_size > overlap else chunk_size
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += stride
    return chunks


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


def send_embeddings_to_jina(texts: List[str]) -> Optional[List[List[float]]]:
    if not JINA_API_KEY:
        print("JINA_API_KEY not set; aborting embedding request.")
        return None

    headers = {
        "Authorization": f"Bearer {JINA_API_KEY}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

    variants = [
        {"model": JINA_EMBEDDING_MODEL, "input": texts},
        {"model": JINA_EMBEDDING_MODEL, "inputs": texts},
        {"model": JINA_EMBEDDING_MODEL, "data": texts},
        {"model": JINA_EMBEDDING_MODEL, "input": [{"text": t} for t in texts]},
        {"model": JINA_EMBEDDING_MODEL, "inputs": [{"text": t} for t in texts]},
        {"model": JINA_EMBEDDING_MODEL, "data": [{"text": t} for t in texts]},
        {"model": JINA_EMBEDDING_MODEL, "texts": texts},
    ]

    last_resp_text = None
    last_status = None
    attempted_model_override = None

    for payload in variants:
        try:
            resp = requests.post(JINA_API_ENDPOINT, headers=headers, json=payload, timeout=60)
            last_status = resp.status_code
            last_resp_text = resp.text
            if resp.status_code == 422:
                try:
                    resp_json = resp.json()
                    allowed = _extract_allowed_tags_from_error(resp_json)
                    if allowed:
                        suggested = allowed[0]
                        if suggested != payload.get("model"):
                            print(f"422 validation: server expects model tag '{suggested}'. Retrying with that tag...")
                            new_payload = dict(payload)
                            new_payload["model"] = suggested
                            attempted_model_override = suggested
                            r2 = requests.post(JINA_API_ENDPOINT, headers=headers, json=new_payload, timeout=60)
                            last_status = r2.status_code
                            last_resp_text = r2.text
                            if r2.status_code >= 400:
                                continue
                            data2 = r2.json()
                            embeddings2 = _parse_jina_response(data2, len(texts))
                            if embeddings2 is not None:
                                print(f"Success with suggested Jina tag '{suggested}'. Consider updating .env JINA_EMBEDDING_MODEL to this value.")
                                return embeddings2
                            else:
                                print("Retry returned no embeddings. Response:", data2)
                                continue
                except Exception as e:
                    print("422 received but could not extract allowed tags:", e)
                    continue
                print(f"422 Unprocessable Entity for payload shape {list(payload.keys())}; response: {resp.text}")
                continue

            if resp.status_code >= 500:
                print(f"Jina server error {resp.status_code} for payload shape {list(payload.keys())}")
                continue

            resp.raise_for_status()
            data = resp.json()
            embeddings = _parse_jina_response(data, len(texts))
            if embeddings is not None:
                if attempted_model_override:
                    print(f"Note: used runtime override model tag '{attempted_model_override}'. Update .env to persist.")
                return embeddings

            print("Successful response but could not parse embeddings. Response:", data)
        except requests.HTTPError as he:
            print("HTTP error calling Jina endpoint:", he)
        except Exception as e:
            print("Error calling Jina endpoint with payload keys", list(payload.keys()), ":", e)

    debug = {
        "endpoint": JINA_API_ENDPOINT,
        "model": JINA_EMBEDDING_MODEL,
        "batch_size": len(texts),
        "last_status": last_status,
        "last_response": last_resp_text,
        "attempted_model_override": attempted_model_override,
    }
    ensure_folder(DATASET_FOLDER)
    with DEBUG_JSON.open("w", encoding="utf-8") as df:
        json.dump(debug, df, ensure_ascii=False, indent=2)
    print(f"All payload variants failed. Debug info written to: {DEBUG_JSON}")
    return None


def main():
    ensure_folder(DATASET_FOLDER)
    files = read_text_files(DATASET_FOLDER)
    if not files:
        print(f"No .txt files found in {DATASET_FOLDER}. Run scraping step first.")
        return

    docs = []
    for f in files:
        txt = f.read_text(encoding="utf-8")
        chunks = chunk_text(txt)
        for i, c in enumerate(chunks):
            doc_id = f"{f.stem}_{i}"
            docs.append({"id": doc_id, "text": c, "metadata": {"source_file": str(f), "chunk_index": i}})

    if not docs:
        print("No document chunks to embed.")
        return

    all_embeddings = []
    all_ids = []
    with OUTPUT_EMBEDDINGS_FILE.open("w", encoding="utf-8") as out_f:
        for i in range(0, len(docs), BATCH_SIZE):
            batch = docs[i : i + BATCH_SIZE]
            texts = [d["text"] for d in batch]
            embeddings = send_embeddings_to_jina(texts)
            if embeddings is None:
                with PENDING_JSONL.open("w", encoding="utf-8") as pf:
                    for d in docs:
                        pf.write(json.dumps(d, ensure_ascii=False) + "\n")
                print(f"Failed to get embeddings from Jina. Wrote pending file: {PENDING_JSONL}")
                return
            for d, emb in zip(batch, embeddings):
                record = {
                    "id": d["id"],
                    "text": d["text"],
                    "metadata": d["metadata"],
                    "embedding": emb,
                    "timestamp": datetime.utcnow().isoformat(),
                }
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                all_embeddings.append(np.array(emb, dtype=np.float32))
                all_ids.append(d["id"])

    if all_embeddings:
        arr = np.stack(all_embeddings)
        np.save(EMBEDDINGS_NPY, arr)
        with IDS_JSON.open("w", encoding="utf-8") as idf:
            json.dump(all_ids, idf, ensure_ascii=False)
        print(f"Wrote {len(all_ids)} embeddings to {OUTPUT_EMBEDDINGS_FILE}")
        print(f"Numpy embeddings saved to {EMBEDDINGS_NPY}; ids saved to {IDS_JSON}")
    else:
        print("No embeddings produced.")


if __name__ == "__main__":
    main()