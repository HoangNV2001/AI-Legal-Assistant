#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ingest a whole folder of Vietnamese regulatory documents (Nghị định, Thông tư, ...)
into Milvus using:
  - MarkItDown to convert files -> plain text
  - OpenAI embeddings (text-embedding-3-small/large)
  - Milvus as the vector store

Collection schema (create beforehand, *dim must match model*):
  id:            VARCHAR (primary)
  chunk_id:      INT64
  embedding:     FLOAT_VECTOR(dim=1536|3072)
  text:          VARCHAR
  doc_type:      VARCHAR   # nghi_dinh | thong_tu | unknown
  so_hieu:       VARCHAR   # e.g., 24/2023/NĐ-CP, 08/2023/TT-BTC
  issued_date:   VARCHAR   # YYYY-MM-DD (optional)

Example:
  python ingest_openai_markitdown_to_milvus.py \
    --folder ./data/vbpq \
    --collection vn_regulations \
    --milvus-uri http://localhost:19530 \
    --openai-embed-model text-embedding-3-small \
    --chunk-tokens 600 --chunk-overlap 80 \
    --so-hieu-from content --auto-issued-date
"""
import os
import re
import sys
import glob
import math
import uuid
import json
import argparse
import hashlib
import traceback
from datetime import datetime
from typing import List, Iterable, Tuple, Optional

# OpenAI (v1)
from openai import OpenAI

# Milvus
from pymilvus import connections, Collection, utility

# MarkItDown
from markitdown import MarkItDown

# Tokenizer (optional)
try:
    import tiktoken
    ENCODING = tiktoken.get_encoding("cl100k_base")
except Exception:
    ENCODING = None


# ------------------------------
# Constants / Regex
# ------------------------------

EMBED_DIM_BY_MODEL = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
}

# Ex: "Nghị định số 24/2023/NĐ-CP", "Thông tư số 08/2023/TT-BTC", "Số: 10/2024/TT-BYT"
SO_HIEU_RE = re.compile(
    r"(?:Nghị\s*định|Nghi\s*dinh|Thông\s*tư|Thong\s*tu)?\s*(?:số|so)\s*[:\-]?\s*"
    r"([0-9]{1,4}/[0-9]{4}/[A-ZĐ\-]+)",
    flags=re.IGNORECASE
)

# "ngày dd tháng mm năm yyyy"
VN_LONG_DATE_RE = re.compile(
    r"ngày\s+(\d{1,2})\s+tháng\s+(\d{1,2})\s+năm\s+(\d{4})", flags=re.IGNORECASE
)
# numeric: dd/mm/yyyy | dd-mm-yyyy | dd.mm.yyyy | yyyy-mm-dd | yyyy/mm/dd
NUMERIC_DATE_RE = re.compile(
    r"(?:(\d{4})[\/\-.](\d{1,2})[\/\-.](\d{1,2}))|(?:(\d{1,2})[\/\-.](\d{1,2})[\/\-.](\d{4}))"
)


# ------------------------------
# File iteration / hashing
# ------------------------------

def iter_files(folder: str, recursive: bool = True,
               include_ext: Optional[List[str]] = None,
               exclude_ext: Optional[List[str]] = None):
    include_ext = [e.lower().lstrip(".") for e in (include_ext or [])]
    exclude_ext = [e.lower().lstrip(".") for e in (exclude_ext or [])]
    for root, _, files in os.walk(folder):
        for name in files:
            ext = os.path.splitext(name)[1].lower().lstrip(".")
            if include_ext and ext not in include_ext:
                continue
            if exclude_ext and ext in exclude_ext:
                continue
            yield os.path.join(root, name)
        if not recursive:
            break


def sha1_of_file(path: str, bufsize: int = 1 << 20) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        while True:
            b = f.read(bufsize)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


# ------------------------------
# VN helpers (doc type / số hiệu / ngày ban hành)
# ------------------------------

def detect_doc_type(text: str, filename: str) -> str:
    src = f"{filename}\n{text[:3000]}".lower()
    if "nghị định" in src or "nghi dinh" in src:
        return "nghi_dinh"
    if "thông tư" in src or "thong tu" in src:
        return "thong_tu"
    return "unknown"


def extract_so_hieu(text: str, filename: str, mode: str) -> str:
    """
    mode: content|filename|none
    """
    if mode == "none":
        return ""
    if mode == "filename":
        m = SO_HIEU_RE.search(filename)
        return m.group(1) if m else ""
    # content (default)
    m = SO_HIEU_RE.search(text)
    return m.group(1) if m else ""


def _safe_date(y: int, m: int, d: int) -> Optional[str]:
    try:
        return datetime(y, m, d).strftime("%Y-%m-%d")
    except Exception:
        return None


def extract_issued_date(text: str) -> Optional[str]:
    t = text.lower()
    m = VN_LONG_DATE_RE.search(t)
    if m:
        d, mo, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
        s = _safe_date(y, mo, d)
        if s:
            return s

    m2 = NUMERIC_DATE_RE.search(t)
    if m2:
        if m2.group(1) and m2.group(2) and m2.group(3):
            # yyyy-mm-dd
            y, mo, d = int(m2.group(1)), int(m2.group(2)), int(m2.group(3))
            s = _safe_date(y, mo, d)
            if s:
                return s
        elif m2.group(4) and m2.group(5) and m2.group(6):
            # dd-mm-yyyy
            d, mo, y = int(m2.group(4)), int(m2.group(5)), int(m2.group(6))
            s = _safe_date(y, mo, d)
            if s:
                return s
    return None


# ------------------------------
# Chunking / Embeddings
# ------------------------------

def to_chunks(text: str, chunk_tokens: int, chunk_overlap: int) -> List[str]:
    """
    Token-aware chunking with tiktoken if available; otherwise char-based heuristic.
    """
    if not text:
        return []
    text = text.strip()
    if not text:
        return []

    if ENCODING:
        toks = ENCODING.encode(text)
        if chunk_tokens <= 0:
            return [text]
        chunks = []
        start = 0
        step = max(1, chunk_tokens - max(0, chunk_overlap))
        while start < len(toks):
            end = min(len(toks), start + chunk_tokens)
            piece = ENCODING.decode(toks[start:end])
            chunks.append(piece)
            start += step
        return chunks
    else:
        if chunk_tokens <= 0:
            return [text]
        approx_chars = max(256, chunk_tokens * 4)          # ~4 chars/token
        overlap_chars = max(0, chunk_overlap * 4)
        chunks = []
        start = 0
        step = max(1, approx_chars - overlap_chars)
        while start < len(text):
            end = min(len(text), start + approx_chars)
            chunks.append(text[start:end])
            start += step
        return chunks


def embed_texts(client: OpenAI, model: str, texts: List[str], batch_size: int = 64) -> List[List[float]]:
    vectors: List[List[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        resp = client.embeddings.create(model=model, input=batch)
        vectors.extend([d.embedding for d in resp.data])  # same order
    return vectors


# ------------------------------
# Milvus helpers
# ------------------------------

def ensure_collection_dim_matches(col: Collection, expected_dim: int):
    """
    Verify the 'embedding' field dim matches the chosen embedding model.
    """
    for f in col.schema.fields:
        if f.name == "embedding":
            dim = None
            if hasattr(f, "params"):
                dim = (f.params or {}).get("dim", None)
            if dim is None:
                print("[warn] Could not read 'dim' param for embedding field; continuing.")
                return
            if int(dim) != int(expected_dim):
                raise RuntimeError(
                    f"Collection '{col.name}' embedding dim={dim}, but model requires {expected_dim}. "
                    "Please recreate the collection with the correct dim."
                )
            return
    print("[warn] No 'embedding' field found; insert will likely fail.")


def upsert_chunks(
    col: Collection,
    base_id: str,
    chunks: List[str],
    vectors: List[List[float]],
    doc_type: str,
    so_hieu: str,
    issued_date: str
) -> int:
    assert len(chunks) == len(vectors), "chunks and vectors length mismatch"
    ids = [f"{base_id}::{i}" for i in range(len(chunks))]
    chunk_ids = list(range(len(chunks)))
    insert_data = [
        ids,                 # id
        chunk_ids,           # chunk_id
        vectors,             # embedding
        chunks,              # text
        [doc_type] * len(chunks),
        [so_hieu] * len(chunks),
        [issued_date] * len(chunks),
    ]
    col.insert(insert_data)
    col.flush()
    return len(chunks)


# ------------------------------
# MarkItDown conversion
# ------------------------------

def convert_with_markitdown(md: MarkItDown, path: str) -> str:
    """
    Convert file to text using MarkItDown.
    """
    res = md.convert(path)
    txt = getattr(res, "text_content", None)
    if not txt:
        raise RuntimeError("No text extracted")
    return txt


# ------------------------------
# Main
# ------------------------------

def main():
    ap = argparse.ArgumentParser(description="Ingest Vietnamese regulations using MarkItDown + OpenAI embeddings -> Milvus")
    ap.add_argument("--folder", required=True, help="Folder path to ingest (recursive)")
    ap.add_argument("--collection", required=True, help="Milvus collection name (e.g., vn_regulations)")
    ap.add_argument("--milvus-uri", default="http://localhost:19530", help="Milvus URI (default: http://localhost:19530)")
    ap.add_argument("--openai-embed-model", default="text-embedding-3-small",
                    help="OpenAI embedding model: text-embedding-3-small|text-embedding-3-large")
    ap.add_argument("--chunk-tokens", type=int, default=600, help="Chunk size in tokens (default: 600)")
    ap.add_argument("--chunk-overlap", type=int, default=80, help="Token overlap (default: 80)")
    ap.add_argument("--batch-size", type=int, default=64, help="OpenAI embedding batch size (default: 64)")
    ap.add_argument("--include-ext", nargs="*", default=None, help="Whitelist extensions (e.g. pdf docx pptx html txt). No dots.")
    ap.add_argument("--exclude-ext", nargs="*", default=None, help="Blacklist extensions (e.g. png jpg mp3). No dots.")
    ap.add_argument("--dry-run", action="store_true", help="Do not insert into Milvus; just print stats")

    # VN-specific metadata
    ap.add_argument("--doc-type", default="auto", choices=["auto", "nghi_dinh", "thong_tu", "unknown"],
                    help="Document type (default: auto detect)")
    ap.add_argument("--so-hieu-from", default="content", choices=["content", "filename", "none"],
                    help="Extract 'số hiệu' from content|filename|none (default: content)")
    ap.add_argument("--issued-date", default="", help="Issued date YYYY-MM-DD (override, optional)")
    ap.add_argument("--auto-issued-date", action="store_true",
                    help="Try to auto-extract issued date from text if --issued-date is empty")

    # MarkItDown options
    ap.add_argument("--use-llm-for-images", action="store_true",
                    help="Pass OpenAI client to MarkItDown to caption images (uses your OpenAI API key)")
    args = ap.parse_args()

    # Validate embedding model / expected dim
    if args.openai_embed_model not in EMBED_DIM_BY_MODEL:
        print(f"[error] Unsupported embedding model: {args.openai_embed_model}. "
              f"Choose one of: {list(EMBED_DIM_BY_MODEL.keys())}")
        sys.exit(2)
    expected_dim = EMBED_DIM_BY_MODEL[args.openai_embed_model]

    # Check env
    if not os.getenv("OPENAI_API_KEY"):
        print("[error] OPENAI_API_KEY is not set")
        sys.exit(2)

    # Connect Milvus
    connections.connect(uri=args.milvus_uri)
    if not utility.has_collection(args.collection):
        print(f"[error] Milvus collection '{args.collection}' not found. "
              f"Create it first with embedding dim = {expected_dim}.")
        sys.exit(2)
    col = Collection(args.collection)
    try:
        ensure_collection_dim_matches(col, expected_dim)
    except Exception as e:
        print(f"[error] {e}")
        sys.exit(2)

    # OpenAI client
    client = OpenAI()

    # MarkItDown instance
    md_kwargs = {}
    if args.use_llm_for_images:
        md_kwargs.update({"llm_client": client, "llm_model": "gpt-4o"})
    md = MarkItDown(**md_kwargs)

    # Iterate & ingest
    total_files = 0
    total_chunks = 0
    failed_files = 0

    for path in iter_files(args.folder, recursive=True,
                           include_ext=args.include_ext,
                           exclude_ext=args.exclude_ext):
        total_files += 1
        rel = os.path.relpath(path, args.folder)
        try:
            base_id = sha1_of_file(path)  # stable ID per file content
            raw_text = convert_with_markitdown(md, path)

            # doc_type
            if args.doc_type == "auto":
                doc_type = detect_doc_type(raw_text, os.path.basename(path))
            else:
                doc_type = args.doc_type

            # so_hieu
            so_hieu = extract_so_hieu(raw_text, os.path.basename(path), args.so_hieu_from)

            # issued_date
            issued_date = args.issued_date.strip()
            if not issued_date and args.auto_issued_date:
                auto_date = extract_issued_date(raw_text)
                if auto_date:
                    issued_date = auto_date

            # chunk
            chunks = to_chunks(raw_text, args.chunk_tokens, args.chunk_overlap)
            if not chunks:
                print(f"[skip] No text after conversion: {rel}")
                continue

            # embed
            vectors = embed_texts(client, args.openai_embed_model, chunks, batch_size=args.batch_size)

            if args.dry_run:
                print(f"[dry-run] {rel} -> {len(chunks)} chunk(s) | type={doc_type} so_hieu={so_hieu} issued={issued_date}")
                continue

            # upsert
            n = upsert_chunks(col, base_id, chunks, vectors, doc_type, so_hieu, issued_date)
            total_chunks += n
            print(f"[ok] {rel} -> {n} chunk(s) | type={doc_type} so_hieu={so_hieu} issued={issued_date}")

        except KeyboardInterrupt:
            print("\n[interrupt] Stopping.")
            break
        except Exception as e:
            failed_files += 1
            print(f"[fail] {rel}: {e}")
            traceback.print_exc(limit=1)

    print("----")
    print(f"Files processed : {total_files}")
    print(f"Files failed    : {failed_files}")
    print(f"Chunks inserted : {total_chunks} (dry-run={args.dry_run})")


if __name__ == "__main__":
    main()
