# create_milvus_collections.py
from pymilvus import (
    connections, utility,
    FieldSchema, CollectionSchema, DataType, Collection
)

# ---------- CONFIG ----------
COLL_NAME = "vn_regulations"
EMBED_DIM = 3072  # OpenAI text-embedding-3-large
VEC_FIELD = "vector"
TEXT_FIELD = "text"
DATE_FIELD = "issued_date"
PK_FIELD = "id"

INDEX_PARAMS = {
    "index_type": "IVF_PQ",
    "metric_type": "COSINE",
    "params": {"nlist": 2048, "m": 128}  # m=128 => subvector dim = 3072/128 = 24
}

# ---------- CONNECT ----------
connections.connect(uri="http://localhost:19530")

def ensure_collection():
    if utility.has_collection(COLL_NAME):
        print(f"[OK] Collection '{COLL_NAME}' already exists.")
        return Collection(COLL_NAME)

    print(f"[CREATE] Creating collection '{COLL_NAME}' ...")

    pk = FieldSchema(
        name=PK_FIELD,
        dtype=DataType.INT64,
        is_primary=True,
        auto_id=True
    )
    vec = FieldSchema(
        name=VEC_FIELD,
        dtype=DataType.FLOAT_VECTOR,
        dim=EMBED_DIM
    )
    text = FieldSchema(
        name=TEXT_FIELD,
        dtype=DataType.VARCHAR,
        max_length=8192  # adjust as you need
    )
    issued_date = FieldSchema(
        name=DATE_FIELD,
        dtype=DataType.VARCHAR,
        max_length=32  # e.g. "2024-08-29" or "Circular 12/2024/TT-BCA"
    )

    schema = CollectionSchema(
        fields=[pk, vec, text, issued_date],
        description="Vietnamese regulations chunks with embeddings (RAG).",
        enable_dynamic_field=True  # allow extra metadata columns later
    )

    col = Collection(
        name=COLL_NAME,
        schema=schema,
        shards_num=2
    )
    print(f"[OK] Created collection '{COLL_NAME}'.")
    return col

def rebuild_index(col: Collection):
    # Always release before touching indexes
    try:
        col.release()
    except Exception:
        pass

    # Drop existing vector indexes if any
    if col.indexes:
        print("[INFO] Dropping existing indexes ...")
        for idx in col.indexes:
            try:
                col.drop_index(index_name=idx.index_name)
            except Exception:
                col.drop_index()
        print("[OK] Dropped indexes.")

    print("[BUILD] Creating IVF_PQ (COSINE) index ...")
    col.create_index(field_name=VEC_FIELD, index_params=INDEX_PARAMS)
    print("[OK] Index created.")

    print("[LOAD] Loading collection into memory ...")
    col.load()
    print("[OK] Loaded.")

def print_summary(col: Collection):
    print("\n=== COLLECTION SUMMARY ===")
    print(f"Name: {col.name}")
    print("Fields:")
    for f in col.schema.fields:
        extra = ""
        if f.dtype == DataType.FLOAT_VECTOR:
            extra = f" (dim={f.params.get('dim')})"
        if f.dtype == DataType.VARCHAR:
            extra += f" (max_length={f.params.get('max_length')})"
        print(f"  - {f.name}: {f.dtype}{extra}")
    if col.indexes:
        print("Indexes:")
        for idx in col.indexes:
            print(f"  - field={idx.field_name}, params={idx.params}")
    else:
        print("Indexes: [none]")
    print("==========================\n")

if __name__ == "__main__":
    col = ensure_collection()
    print_summary(col)
    rebuild_index(col)
    print_summary(col)
    print("Create/Update collection complete.")
