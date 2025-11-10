# ...existing code...
#!/usr/bin/env python3
import os
import re
import json
from pathlib import Path
from datetime import datetime
from typing import List, Optional
from urllib.parse import quote

import requests
import pandas as pd
from dotenv import load_dotenv

ROOT = Path(__file__).parent
load_dotenv(ROOT / ".env")

DATASET_FOLDER = Path(os.getenv("DATASET_STORAGE_FOLDER", "datasets"))
SNAPSHOT_FILE = Path(os.getenv("SNAPSHOT_STORAGE_FILE", "snapshot.txt"))

JINA_API_KEY = os.getenv("JINA_API_KEY")
JINA_API_ENDPOINT = os.getenv("JINA_API_ENDPOINT", "https://api.jina.ai/v1/embeddings")
JINA_EMBEDDING_MODEL = os.getenv("JINA_EMBEDDING_MODEL", os.getenv("EMBEDDING_MODEL", "mxbai-embed-large"))
BATCH_SIZE = int(os.getenv("JINA_BATCH_SIZE", "16"))

# Use a clear user-agent to avoid being blocked by Wikipedia
USER_AGENT = os.getenv("WIKI_USER_AGENT", "local-rag-wiki-scraper/1.0 (contact: none)")

def slugify(name: str) -> str:
    name = name.strip().lower()
    name = re.sub(r"[^\w\s-]", "", name)
    name = re.sub(r"[-\s]+", "_", name)
    return name[:200]

def ensure_folder(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def read_keywords_xlsx(xlsx_path: Path) -> List[str]:
    df = pd.read_excel(xlsx_path, engine="openpyxl")
    cols = [c.lower() for c in df.columns]
    if "keyword" in cols:
        col = df.columns[cols.index("keyword")]
        vals = df[col].dropna().astype(str).tolist()
    else:
        vals = df.iloc[:, 0].dropna().astype(str).tolist()
    cleaned = []
    for v in vals:
        s = v.strip().strip('"').strip("'")
        if s:
            cleaned.append(s)
    return cleaned

def _make_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({
        "User-Agent": USER_AGENT,
        "Accept": "application/json, text/plain, */*",
    })
    return s

def _query_extract_for_title(session: requests.Session, title: str) -> Optional[str]:
    params = {
        "action": "query",
        "format": "json",
        "prop": "extracts",
        "explaintext": 1,
        "titles": title,
        "redirects": 1,
        "formatversion": 2,
    }
    url = "https://en.wikipedia.org/w/api.php"
    try:
        r = session.get(url, params=params, timeout=20)
        r.raise_for_status()
        data = r.json()
        pages = data.get("query", {}).get("pages", [])
        if not pages:
            return None
        page = pages[0]
        if "missing" in page:
            return None
        text = page.get("extract", "")
        return text if text else None
    except Exception:
        return None

def _rest_summary(session: requests.Session, title: str) -> Optional[str]:
    try:
        enc = quote(title, safe="")
        url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{enc}"
        r = session.get(url, timeout=20)
        # provide a tiny debug line if non-200 to help trace problems
        if r.status_code != 200:
            return None
        data = r.json()
        if isinstance(data, dict) and data.get("extract"):
            # if type is disambiguation, prefer search fallback instead of disambig text
            if data.get("type") == "disambiguation":
                return None
            return data.get("extract")
        return None
    except Exception:
        return None

def _opensearch_title(session: requests.Session, query: str) -> Optional[str]:
    try:
        url = "https://en.wikipedia.org/w/api.php"
        params = {"action": "opensearch", "search": query, "limit": 3, "namespace": 0, "format": "json"}
        r = session.get(url, params=params, timeout=20)
        r.raise_for_status()
        data = r.json()
        if isinstance(data, list) and len(data) >= 2 and data[1]:
            # prefer exact-ish match: try to find one that contains the query token(s)
            qlow = query.lower()
            for t in data[1]:
                if qlow in t.lower():
                    return t
            return data[1][0]
        return None
    except Exception:
        return None

def _search_title_mediawiki(session: requests.Session, query: str) -> Optional[str]:
    try:
        url = "https://en.wikipedia.org/w/api.php"
        params = {"action": "query", "format": "json", "list": "search", "srsearch": query, "srlimit": 3, "formatversion": 2}
        r = session.get(url, params=params, timeout=20)
        r.raise_for_status()
        data = r.json()
        hits = data.get("query", {}).get("search", [])
        if hits:
            # pick the first non-disambiguation-like title if possible
            return hits[0].get("title")
        return None
    except Exception:
        return None

def fetch_wikipedia_extract(title: str) -> Optional[str]:
    session = _make_session()
    t = title.strip()
    if not t:
        return None

    candidates = [t]
    if "_" not in t:
        candidates.append(t.replace(" ", "_"))
    if t and t[0].islower():
        candidates.append(t[:1].upper() + t[1:])
    candidates.append(t.title())
    # dedupe preserving order
    seen = set()
    candidates = [x for x in candidates if not (x in seen or seen.add(x))]

    # 1) Try REST summary for candidates (skip if REST returns disambiguation)
    for cand in candidates:
        extract = _rest_summary(session, cand)
        if extract:
            if cand != title:
                print(f"  -> REST used variant: {cand}")
            return extract

    # 2) MediaWiki query extracts
    for cand in candidates:
        extract = _query_extract_for_title(session, cand)
        if extract:
            if cand != title:
                print(f"  -> Query used variant: {cand}")
            return extract

    # 3) Opensearch fallback -> try returned titles
    op_title = _opensearch_title(session, title)
    if op_title:
        extract = _rest_summary(session, op_title) or _query_extract_for_title(session, op_title)
        if extract:
            print(f"  -> Used opensearch fallback title: {op_title}")
            return extract

    # 4) MediaWiki search fallback (more flexible)
    mw_title = _search_title_mediawiki(session, title)
    if mw_title:
        extract = _rest_summary(session, mw_title) or _query_extract_for_title(session, mw_title)
        if extract:
            print(f"  -> Used mediawiki search fallback title: {mw_title}")
            return extract

    return None

def append_snapshot(text: str):
    ensure_folder(SNAPSHOT_FILE.parent)
    with SNAPSHOT_FILE.open("a", encoding="utf-8") as f:
        f.write(text + "\n")

def _parse_jina_response(resp_json, batch_len: int):
    if isinstance(resp_json, dict):
        if "embeddings" in resp_json and isinstance(resp_json["embeddings"], list):
            return resp_json["embeddings"]
        if "data" in resp_json and isinstance(resp_json["data"], list):
            embeddings = []
            for item in resp_json["data"]:
                if isinstance(item, dict) and ("embedding" in item and isinstance(item["embedding"], list)):
                    embeddings.append(item["embedding"])
            if len(embeddings) == batch_len:
                return embeddings
        if "results" in resp_json and isinstance(resp_json["results"], list):
            embeddings = []
            for item in resp_json["results"]:
                if isinstance(item, dict) and ("embedding" in item and isinstance(item["embedding"], list)):
                    embeddings.append(item["embedding"])
            if len(embeddings) == batch_len:
                return embeddings
    return None

def send_embeddings_to_jina(docs: List[dict]) -> Optional[List[dict]]:
    if not JINA_API_KEY:
        return None
    headers = {
        "Authorization": f"Bearer {JINA_API_KEY}",
        "Content-Type": "application/json",
    }
    results = []
    for i in range(0, len(docs), BATCH_SIZE):
        batch = docs[i : i + BATCH_SIZE]
        payload = {"model": JINA_EMBEDDING_MODEL, "input": [d["text"] for d in batch]}
        try:
            resp = requests.post(JINA_API_ENDPOINT, headers=headers, json=payload, timeout=60)
            resp.raise_for_status()
            data = resp.json()
            embeddings = _parse_jina_response(data, len(batch))
            if embeddings is None or len(embeddings) != len(batch):
                print("Unexpected embeddings response from Jina endpoint:", data)
                return None
            for doc, emb in zip(batch, embeddings):
                results.append({"id": doc["id"], "text": doc["text"], "metadata": doc.get("metadata", {}), "embedding": emb})
        except Exception as e:
            print("Failed to send batch to Jina endpoint:", e)
            return None
    return results

def main():
    xlsx = ROOT / "keywords.xlsx"
    if not xlsx.exists():
        print(f"keywords.xlsx not found at {xlsx}")
        return
    ensure_folder(DATASET_FOLDER)
    ensure_folder(SNAPSHOT_FILE.parent)
    keywords = read_keywords_xlsx(xlsx)
    if not keywords:
        print("No keywords found in keywords.xlsx")
        return

    docs_for_embedding = []
    saved = 0
    for kw in keywords:
        kw = kw.strip()
        if not kw:
            continue
        print(f"Processing: {kw}")
        content = fetch_wikipedia_extract(kw)
        if not content:
            print(f"  -> No page found for '{kw}'")
            append_snapshot(f"{datetime.utcnow().isoformat()} - MISSING - {kw}")
            continue
        filename = slugify(kw) + ".txt"
        outpath = DATASET_FOLDER / filename
        with outpath.open("w", encoding="utf-8") as f:
            f.write(content)
        meta = {"keyword": kw, "source": "wikipedia", "file": str(outpath), "timestamp": datetime.utcnow().isoformat()}
        append_snapshot(json.dumps({"timestamp": datetime.utcnow().isoformat(), "keyword": kw, "file": str(outpath)}))
        docs_for_embedding.append({"id": slugify(kw), "text": content, "metadata": meta})
        saved += 1
        print(f"  -> Saved to {outpath}")

    print(f"Done. Saved {saved} pages. Snapshot: {SNAPSHOT_FILE}")

    if not docs_for_embedding:
        return

    if JINA_API_KEY:
        print("JINA_API_KEY found, attempting to send texts to Jina endpoint...")
        sent = send_embeddings_to_jina(docs_for_embedding)
        if sent is None:
            pending_path = DATASET_FOLDER / "pending_for_jina.jsonl"
            with pending_path.open("w", encoding="utf-8") as f:
                for d in docs_for_embedding:
                    f.write(json.dumps({"id": d["id"], "text": d["text"], "metadata": d["metadata"]}) + "\n")
            print(f"Failed to get embeddings from Jina endpoint. Wrote pending file: {pending_path}")
        else:
            emb_path = DATASET_FOLDER / "embeddings.jsonl"
            with emb_path.open("w", encoding="utf-8") as f:
                for item in sent:
                    f.write(json.dumps(item) + "\n")
            print(f"Wrote embeddings to {emb_path}")
    else:
        pending_path = DATASET_FOLDER / "pending_for_jina.jsonl"
        with pending_path.open("w", encoding="utf-8") as f:
            for d in docs_for_embedding:
                f.write(json.dumps({"id": d["id"], "text": d["text"], "metadata": d["metadata"]}) + "\n")
        print(f"No JINA_API_KEY found. Wrote pending file for later upload: {pending_path}")

if __name__ == "__main__":
    main()
# ...existing code...