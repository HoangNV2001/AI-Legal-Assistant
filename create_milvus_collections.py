#!/usr/bin/env python3
# create_milvus_collections.py
from pymilvus import (
    connections, utility,
    FieldSchema, CollectionSchema, DataType, Collection
)
import argparse

# ---------- DEFAULT CONFIG (override bằng CLI) ----------
DEFAULT_COLL_NAME = "vn_regulations"
DEFAULT_MILVUS_URI = "http://localhost:19530"

DEFAULT_EMBED_DIM = 2048               # phải khớp với dimensions của embedder NIM/OpenAI
PK_FIELD   = "id"
VEC_FIELD  = "vector"
TEXT_FIELD = "text"
DATE_FIELD = "issued_date"
SRC_FIELD  = "source"                   # tiện filter/xoá theo file khi re-ingest

DEFAULT_INDEX_TYPE = "IVF_PQ"
DEFAULT_METRIC = "COSINE"
DEFAULT_NLIST = 2048
DEFAULT_PQ_M = 128                      # yêu cầu EMBED_DIM % PQ_M == 0 (vd: 2048/128=16)
DEFAULT_SHARDS = 2

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--milvus-uri", default=DEFAULT_MILVUS_URI)
    ap.add_argument("--collection", default=DEFAULT_COLL_NAME)
    ap.add_argument("--embed-dim", type=int, default=DEFAULT_EMBED_DIM)

    ap.add_argument("--index-type", default=DEFAULT_INDEX_TYPE,
                    choices=["IVF_PQ", "IVF_FLAT", "IVF_SQ8", "HNSW", "DISKANN"])
    ap.add_argument("--metric", default=DEFAULT_METRIC,
                    choices=["COSINE", "IP", "L2"])
    ap.add_argument("--nlist", type=int, default=DEFAULT_NLIST)
    ap.add_argument("--pq-m", type=int, default=DEFAULT_PQ_M)
    ap.add_argument("--shards", type=int, default=DEFAULT_SHARDS)

    ap.add_argument("--text-maxlen", type=int, default=8192)
    ap.add_argument("--date-maxlen", type=int, default=32)
    ap.add_argument("--id-maxlen", type=int, default=128)
    ap.add_argument("--source-maxlen", type=int, default=512)

    return ap.parse_args()

def connect(uri: str):
    connections.connect(uri=uri)

def recreate_collection(args):
    coll_name = args.collection

    # Nếu tồn tại → drop để áp dụng cấu hình mới
    if utility.has_collection(coll_name):
        print(f"[DROP] Collection '{coll_name}' exists → dropping ...")
        try:
            utility.drop_collection(coll_name)
        except Exception as e:
            print(f"[WARN] Drop failed (try release then drop): {e}")
            try:
                Collection(coll_name).release()
            except Exception:
                pass
            utility.drop_collection(coll_name)
        print(f"[OK] Dropped '{coll_name}'")

    # Guardrail cho IVF_PQ
    if args.index_type == "IVF_PQ" and (args.embed_dim % args.pq_m != 0):
        raise SystemExit(
            f"[ERR] EMBED_DIM ({args.embed_dim}) must be divisible by pq_m ({args.pq_m}). "
            f"Ví dụ: 2048/128 = 16 (hợp lệ)."
        )

    # Tạo schema mới (PK = VARCHAR, auto_id=False)
    print(f"[CREATE] Creating collection '{coll_name}' (dim={args.embed_dim}) ...")

    pk = FieldSchema(
        name=PK_FIELD,
        dtype=DataType.VARCHAR,
        is_primary=True,
        auto_id=False,
        max_length=args.id_maxlen
    )
    vec = FieldSchema(
        name=VEC_FIELD,
        dtype=DataType.FLOAT_VECTOR,
        dim=args.embed_dim
    )
    text = FieldSchema(
        name=TEXT_FIELD,
        dtype=DataType.VARCHAR,
        max_length=args.text_maxlen
    )
    issued_date = FieldSchema(
        name=DATE_FIELD,
        dtype=DataType.VARCHAR,
        max_length=args.date_maxlen
    )
    source = FieldSchema(
        name=SRC_FIELD,
        dtype=DataType.VARCHAR,
        max_length=args.source_maxlen
    )

    schema = CollectionSchema(
        fields=[pk, vec, text, issued_date, source],
        description="Vietnamese regulations chunks with embeddings (RAG).",
        enable_dynamic_field=True  # vẫn cho phép thêm metadata linh hoạt
    )

    col = Collection(
        name=coll_name,
        schema=schema,
        shards_num=args.shards
    )
    print(f"[OK] Created '{coll_name}'")
    return col

def build_index_and_load(col: Collection, args):
    # Lựa chọn tham số index theo loại
    if args.index_type == "IVF_PQ":
        index_params = {
            "index_type": "IVF_PQ",
            "metric_type": args.metric,
            "params": {"nlist": args.nlist, "m": args.pq_m}  # subvector dim = embed_dim / m (vd: 2048/128=16)
        }
    elif args.index_type == "IVF_FLAT":
        index_params = {
            "index_type": "IVF_FLAT",
            "metric_type": args.metric,
            "params": {"nlist": args.nlist}
        }
    elif args.index_type == "IVF_SQ8":
        index_params = {
            "index_type": "IVF_SQ8",
            "metric_type": args.metric,
            "params": {"nlist": args.nlist}
        }
    elif args.index_type == "HNSW":
        index_params = {
            "index_type": "HNSW",
            "metric_type": args.metric,
            "params": {"M": 16, "efConstruction": 200}
        }
    elif args.index_type == "DISKANN":
        index_params = {
            "index_type": "DISKANN",
            "metric_type": args.metric,
            "params": {"search_list_size": 100}
        }
    else:
        raise SystemExit(f"Unsupported index type: {args.index_type}")

    print(f"[INDEX] Creating index on '{col.name}.{VEC_FIELD}': {index_params}")
    col.create_index(field_name=VEC_FIELD, index_params=index_params)
    print("[OK] Index created.")

    print("[LOAD] Loading collection into memory ...")
    col.load()
    try:
        utility.wait_for_loading_complete(col.name, timeout=300)
    except Exception:
        pass
    print("[OK] Loaded.")

def print_summary(col: Collection):
    print("\n=== COLLECTION SUMMARY ===")
    print(f"Name: {col.name}")
    print("Fields:")
    for f in col.schema.fields:
        extra = []
        if getattr(f, "is_primary", False):
            extra.append("PK")
        if getattr(f, "auto_id", False):
            extra.append("auto_id=True")
        if f.dtype == DataType.FLOAT_VECTOR:
            extra.append(f"dim={f.params.get('dim')}")
        if f.dtype == DataType.VARCHAR:
            ml = f.params.get('max_length')
            if ml:
                extra.append(f"max_length={ml}")
        extra_str = f" ({', '.join(extra)})" if extra else ""
        print(f"  - {f.name}: {f.dtype}{extra_str}")

    if col.indexes:
        print("Indexes:")
        for idx in col.indexes:
            print(f"  - field={idx.field_name}, params={idx.params}")
    else:
        print("Indexes: [none]")
    print("==========================\n")

def main():
    args = parse_args()
    connect(args.milvus_uri)
    col = recreate_collection(args)
    print_summary(col)
    build_index_and_load(col, args)
    print_summary(col)
    print("Create/Update collection complete.")

if __name__ == "__main__":
    main()
