#!/usr/bin/env python3
# ingest_to_milvus_nemotron.py
import os, argparse, hashlib, traceback, math, random
from typing import List, Optional, Dict, Any
from openai import OpenAI
from pymilvus import connections, Collection, utility, DataType
from markitdown import MarkItDown

# --- optional: token-based chunking (fallback nếu không có tiktoken) ---
try:
    import tiktoken
    ENCODING = tiktoken.get_encoding("cl100k_base")
except Exception:
    ENCODING = None

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

def l2_normalize(vecs: List[List[float]]) -> List[List[float]]:
    out = []
    for v in vecs:
        s = math.sqrt(sum(x*x for x in v)) or 1.0
        out.append([x/s for x in v])
    return out

def get_nim_client(base_url: str, api_key: Optional[str]) -> OpenAI:
    # NIM dùng OpenAI-compatible API; api_key có thể không bắt buộc khi tự host
    return OpenAI(base_url=base_url, api_key=api_key or "nokey")

def embed_texts_nim(client: OpenAI, model: str, texts: List[str], batch_size: int = 64) -> List[List[float]]:
    out = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        resp = client.embeddings.create(
                model="nvidia/llama-3.2-nv-embedqa-1b-v2",
                input=batch,
                extra_body={
                    "input_type": "passage",      # hoặc "query" khi truy vấn
                    "dimensions": 2048            # nếu muốn chỉ định rõ
                })
        out.extend([d.embedding for d in resp.data])
    return out

def ensure_vector_dim_matches(col: Collection, expected_dim: int):
    for f in col.schema.fields:
        if f.name == "vector":
            dim = (getattr(f, "params", {}) or {}).get("dim")
            if dim is not None and int(dim) != int(expected_dim):
                raise RuntimeError(
                    f"Collection '{col.name}' vector dim={dim} != {expected_dim}. "
                    f"Bạn cần tạo collection/index đúng dim của embedder NIM."
                )
            return
    raise RuntimeError("Không tìm thấy field 'vector' trong schema.")

def dtype_defaults(dtype: DataType):
    if dtype in (DataType.INT64, DataType.INT8, DataType.INT16, DataType.INT32):
        return 0
    if dtype == DataType.FLOAT:
        return 0.0
    if dtype == DataType.BOOL:
        return False
    # VARCHAR hoặc không xác định → chuỗi rỗng
    return ""

def make_int_ids_from_sha1(base_hex: str, n: int) -> List[int]:
    # chuyển sha1 (40 hex) thành int64, cộng i để phân biệt từng chunk
    base_int = int(base_hex[:16], 16) & ((1<<63)-1)
    return [(base_int + i) & ((1<<63)-1) for i in range(n)]

def prepare_insert_payload(
    col: Collection,
    base_id: str,
    rel_path: str,
    chunks: List[str],
    vectors: List[List[float]],
    issued_date: str
) -> List[List[Any]]:
    """
    Trả về list dữ liệu theo đúng THỨ TỰ FIELD trong schema (bỏ qua PK auto_id).
    Hỗ trợ các field phổ biến:
      - id (INT64 hoặc VARCHAR)
      - chunk_id (INT64)
      - vector (FLOAT_VECTOR)
      - text (VARCHAR)
      - issued_date (VARCHAR)
      - source (VARCHAR)
    Field khác sẽ được điền giá trị mặc định theo kiểu dữ liệu.
    """
    n = len(chunks)
    data_by_field: Dict[str, List[Any]] = {}

    # Chuẩn bị sẵn map cho các field quen thuộc nếu có
    field_types = {f.name: f.dtype for f in col.schema.fields}
    is_auto_id_pk = None
    for f in col.schema.fields:
        if getattr(f, "is_primary", False):
            is_auto_id_pk = (f.name, getattr(f, "auto_id", False))

    # id
    if "id" in field_types:
        if field_types["id"] == DataType.INT64:
            data_by_field["id"] = make_int_ids_from_sha1(base_id, n)
        else:
            data_by_field["id"] = [f"{base_id}::{i}" for i in range(n)]

    # chunk_id
    if "chunk_id" in field_types:
        data_by_field["chunk_id"] = list(range(n))

    # vector
    if "vector" in field_types:
        data_by_field["vector"] = vectors

    # text
    if "text" in field_types:
        data_by_field["text"] = chunks

    # issued_date
    if "issued_date" in field_types:
        data_by_field["issued_date"] = [issued_date]*n

    # source (tuỳ schema)
    if "source" in field_types:
        data_by_field["source"] = [rel_path]*n

    # Lắp theo THỨ TỰ schema (bỏ qua primary auto_id nếu có)
    ordered = []
    for f in col.schema.fields:
        if getattr(f, "is_primary", False) and getattr(f, "auto_id", False):
            # Không truyền field auto_id
            continue
        if f.name in data_by_field:
            ordered.append(data_by_field[f.name])
        else:
            # field không phải auto_id nhưng không có trong map → điền mặc định
            ordered.append([dtype_defaults(f.dtype)]*n)
    return ordered

def embed_texts_nim(client: OpenAI, model: str, texts: List[str],
                    batch_size: int = 64, dimensions: Optional[int] = None,
                    input_type: str = "passage") -> List[List[float]]:
    out = []
    extra = {"input_type": input_type}
    if dimensions:
        extra["dimensions"] = dimensions

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        resp = client.embeddings.create(
            model=model,            # dùng model từ args
            input=batch,
            extra_body=extra        # truyền input_type/dimensions
        )
        out.extend([d.embedding for d in resp.data])
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", required=True)
    ap.add_argument("--collection", required=True)
    ap.add_argument("--milvus-uri", default="http://localhost:19530")

    # NIM embedder (OpenAI-compatible)
    ap.add_argument("--nim-embed-base-url", default=os.getenv("NIM_EMBED_BASE_URL", "http://localhost:8016/v1"))
    ap.add_argument("--nim-embed-model", default=os.getenv("NIM_EMBED_MODEL", "nvidia/llama-3.2-nv-embedqa-1b-v2"))
    ap.add_argument("--nim-api-key", default=os.getenv("NIM_API_KEY", ""))
    ap.add_argument("--nim-dimensions", type=int, default=2048,
                    help="Embedding dimensions (e.g., 384/512/768/1024/2048) – phải khớp schema Milvus.")
    ap.add_argument("--nim-input-type", default="passage", choices=["passage", "query"],
                    help="Asymmetric mode: 'passage' cho ingest, 'query' cho truy vấn.")
    # Chunking & batching
    ap.add_argument("--chunk-tokens", type=int, default=600)
    ap.add_argument("--chunk-overlap", type=int, default=80)
    ap.add_argument("--batch-size", type=int, default=64)

    # Metadata
    ap.add_argument("--issued-date", default="")
    ap.add_argument("--add-source", action="store_true", help="Nếu schema có field 'source', điền relative path.")
    ap.add_argument("--normalize", action="store_true", help="L2 normalize embedding (chủ yếu dùng nếu metric=IP).")

    # Behavior
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--use-llm-for-images", action="store_true")
    args = ap.parse_args()

    # Kết nối Milvus
    connections.connect(uri=args.milvus_uri)
    if not utility.has_collection(args.collection):
        raise SystemExit(f"Collection '{args.collection}' not found")
    col = Collection(args.collection)

    # NIM client (OpenAI-compatible)
    client = get_nim_client(args.nim_embed_base_url, args.nim_api_key)

    # MarkItDown (không cần LLM cho hầu hết PDF/Word)
    md_kwargs = {}
    if args.use_llm_for_images:
        # Nếu tự host LLM cũng qua OpenAI-compatible, bạn có thể set:
        # md_kwargs = {"llm_client": client, "llm_model": "<your-nim-vision-or-ocr-model>"}
        # Mặc định để trống cho an toàn.
        pass
    md = MarkItDown(**md_kwargs)

    total_files = total_chunks = failed = 0
    first_vec_dim: Optional[int] = None

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
            if not chunks:
                print(f"[skip] No chunks: {rel}")
                continue

            # Embed trước 1 lô nhỏ để lấy dim
            warmup = chunks[: min(len(chunks), max(1, 8))]
            warmup_vecs = embed_texts_nim(
                client, args.nim_embed_model, warmup,
                batch_size=args.batch_size,
                dimensions=args.nim_dimensions,
                input_type=args.nim_input_type
            )
            if not warmup_vecs:
                print(f"[fail] {rel}: empty embedding result")
                failed += 1
                continue

            # Xác định dim từ kết quả và khớp với schema
            if first_vec_dim is None:
                first_vec_dim = len(warmup_vecs[0])
                ensure_vector_dim_matches(col, first_vec_dim)

            # Embed phần còn lại (nếu có)
            if len(chunks) > len(warmup):
                rest = chunks[len(warmup):]
                rest_vecs = embed_texts_nim(
                    client, args.nim_embed_model, rest,
                    batch_size=args.batch_size,
                    dimensions=args.nim_dimensions,
                    input_type=args.nim_input_type
                )
                vecs = warmup_vecs + rest_vecs
            else:
                vecs = warmup_vecs

            if args.normalize:
                vecs = l2_normalize(vecs)

            if args.dry_run:
                print(f"[dry-run] {rel}: {len(chunks)} chunks, dim={first_vec_dim}")
                continue

            payload = prepare_insert_payload(
                col=col,
                base_id=base_id,
                rel_path=rel if args.add_source else "",
                chunks=chunks,
                vectors=vecs,
                issued_date=args.issued_date.strip()
            )

            col.insert(payload)
            col.flush()
            total_chunks += len(chunks)
            print(f"[ok] {rel} -> {len(chunks)} chunks (dim={first_vec_dim})")

        except Exception as e:
            failed += 1
            print(f"[fail] {rel}: {e}")
            traceback.print_exc(limit=1)
    try:
        col.release()
    except Exception:
        pass
    col.load()
    try:
        utility.wait_for_loading_complete(args.collection, timeout=300)
    except Exception:
        pass
    print("[OK] Collection reloaded.")
    print("----")
    print(f"Files processed : {total_files}")
    print(f"Files failed    : {failed}")
    print(f"Chunks inserted : {total_chunks} (dry-run={args.dry_run})")

if __name__ == "__main__":
    main()

    # Tải collection vào RAM để sẵn sàng truy vấn
    from pymilvus import connections, Collection, utility
    try:
        # đã connect ở trên; gọi lại cho chắc khi chạy độc lập
        connections.connect(uri=os.getenv("MILVUS_URI", "http://localhost:19530"))
        colname = os.getenv("MILVUS_COLLECTION", "")
        if not colname:
            # fallback: không có env thì thôi
            pass
        else:
            col = Collection(colname)
            try:
                col.release()
            except Exception:
                pass
            col.load()
            try:
                utility.wait_for_loading_complete(colname, timeout=300)
            except Exception:
                pass
            try:
                print("Loaded segments:", utility.get_query_segment_info(colname))
            except Exception:
                pass
            print("Row count:", col.num_entities)
    except Exception:
        pass
