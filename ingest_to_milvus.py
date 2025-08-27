#!/usr/bin/env python3
# ingest_openai_markitdown_to_milvus.py  (bản rút gọn, field = vector)
import os, argparse, hashlib, traceback
from typing import List, Optional
from openai import OpenAI
from pymilvus import connections, Collection, utility
from markitdown import MarkItDown

try:
    import tiktoken
    ENCODING = tiktoken.get_encoding("cl100k_base")
except Exception:
    ENCODING = None

EMBED_DIM_BY_MODEL = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
}

def iter_files(folder: str):
    for root, _, files in os.walk(folder):
        for n in files:
            yield os.path.join(root, n)

def sha1_of_file(path: str) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1<<20), b""):
            h.update(chunk)
    return h.hexdigest()

def to_chunks(text: str, chunk_tokens: int, chunk_overlap: int) -> List[str]:
    if not text: return []
    text = text.strip()
    if not text: return []
    if ENCODING:
        toks = ENCODING.encode(text)
        if chunk_tokens <= 0: return [text]
        chunks, start = [], 0
        step = max(1, chunk_tokens - max(0, chunk_overlap))
        while start < len(toks):
            end = min(len(toks), start + chunk_tokens)
            chunks.append(ENCODING.decode(toks[start:end]))
            start += step
        return chunks
    else:
        if chunk_tokens <= 0: return [text]
        approx_chars = max(256, chunk_tokens * 4)
        overlap_chars = max(0, chunk_overlap * 4)
        chunks, start = [], 0
        step = max(1, approx_chars - overlap_chars)
        while start < len(text):
            end = min(len(text), start + approx_chars)
            chunks.append(text[start:end])
            start += step
        return chunks

def embed_texts(client: OpenAI, model: str, texts: List[str], batch_size: int = 64) -> List[List[float]]:
    out = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        resp = client.embeddings.create(model=model, input=batch)
        out.extend([d.embedding for d in resp.data])
    return out

def ensure_vector_dim(col: Collection, expected_dim: int):
    # Kiểm tra field 'vector' có đúng dim không
    for f in col.schema.fields:
        if f.name == "vector":
            dim = (getattr(f, "params", {}) or {}).get("dim")
            if dim is not None and int(dim) != int(expected_dim):
                raise RuntimeError(f"Collection '{col.name}' vector dim={dim} != {expected_dim}")
            return
    raise RuntimeError("No 'vector' field found in schema.")

def upsert(col: Collection, base_id: str, chunks: List[str], vectors: List[List[float]], issued_date: str) -> int:
    assert len(chunks) == len(vectors)
    ids = [f"{base_id}::{i}" for i in range(len(chunks))]
    chunk_ids = list(range(len(chunks)))
    # Thứ tự insert phải trùng thứ tự schema đã tạo: id, chunk_id, vector, text, issued_date
    col.insert([ids, chunk_ids, vectors, chunks, [issued_date]*len(chunks)])
    col.flush()
    return len(chunks)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", required=True)
    ap.add_argument("--collection", required=True)
    ap.add_argument("--milvus-uri", default="http://localhost:19530")
    ap.add_argument("--openai-embed-model", default="text-embedding-3-large")
    ap.add_argument("--chunk-tokens", type=int, default=600)
    ap.add_argument("--chunk-overlap", type=int, default=80)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--issued-date", default="")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--use-llm-for-images", action="store_true")
    args = ap.parse_args()

    if args.openai_embed_model not in EMBED_DIM_BY_MODEL:
        raise SystemExit(f"Unsupported model: {args.openai_embed_model}")
    expected_dim = EMBED_DIM_BY_MODEL[args.openai_embed_model]

    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY not set")

    connections.connect(uri=args.milvus_uri)
    if not utility.has_collection(args.collection):
        raise SystemExit(f"Collection '{args.collection}' not found")
    col = Collection(args.collection)
    ensure_vector_dim(col, expected_dim)

    client = OpenAI()
    md = MarkItDown(**({"llm_client": client, "llm_model":"gpt-4o"} if args.use_llm_for_images else {}))

    total_files = total_chunks = failed = 0
    for path in iter_files(args.folder):
        total_files += 1
        rel = os.path.relpath(path, args.folder)
        try:
            base_id = sha1_of_file(path)
            text = md.convert(path).text_content
            if not text: 
                print(f"[skip] No text: {rel}")
                continue
            chunks = to_chunks(text, args.chunk_tokens, args.chunk_overlap)
            vecs = embed_texts(client, args.openai_embed_model, chunks, batch_size=args.batch_size)
            if args.dry_run:
                print(f"[dry-run] {rel}: {len(chunks)} chunks")
                continue
            n = upsert(col, base_id, chunks, vecs, args.issued_date.strip())
            total_chunks += n
            print(f"[ok] {rel} -> {n} chunks")
        except Exception as e:
            failed += 1
            print(f"[fail] {rel}: {e}")
            traceback.print_exc(limit=1)

    print("----")
    print(f"Files processed : {total_files}")
    print(f"Files failed    : {failed}")
    print(f"Chunks inserted : {total_chunks} (dry-run={args.dry_run})")

if __name__ == "__main__":
    main()
    from pymilvus import connections, Collection, utility

    connections.connect(uri="http://localhost:19530")

    col = Collection("vn_regulations")
    # Nếu bạn vừa create_index, nên release rồi load lại:
    try:
        col.release()
    except Exception:
        pass

    # Load collection vào RAM
    col.load()  # có thể kèm replica_num=1 nếu cần

    # Chờ load hoàn tất (tuỳ bản PyMilvus/Milvus, 1 trong 2 lệnh dưới sẽ có)
    try:
        utility.wait_for_loading_complete("vn_regulations", timeout=300)
    except Exception:
        pass

    # Kiểm tra nhanh
    print("Loaded segments:", utility.get_query_segment_info("vn_regulations"))
    print("Row count:", col.num_entities)