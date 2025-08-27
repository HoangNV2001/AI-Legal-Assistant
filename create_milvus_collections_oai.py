# create_vn_regulations_collection.py
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility

MILVUS_URI = "http://localhost:19530"
DIM = 3072  # dùng text-embedding-3-large; 

connections.connect(uri=MILVUS_URI)
NAME = "vn_regulations"
if utility.has_collection(NAME):
    utility.drop_collection(NAME)

fields = [
    FieldSchema("id", DataType.VARCHAR, max_length=256, is_primary=True),
    FieldSchema("chunk_id", DataType.INT64),
    FieldSchema("embedding", DataType.FLOAT_VECTOR, dim=DIM),
    FieldSchema("text", DataType.VARCHAR, max_length=8192),
    FieldSchema("doc_type", DataType.VARCHAR, max_length=32),       # "nghi_dinh" | "thong_tu" | ...
    FieldSchema("so_hieu", DataType.VARCHAR, max_length=64),         # ví dụ: 24/2023/NĐ-CP, 08/2023/TT-BTC
    FieldSchema("issued_date", DataType.VARCHAR, max_length=32),     # YYYY-MM-DD (tùy bạn)
]
col = Collection(NAME, CollectionSchema(fields, "VN regulations"))
col.create_index("embedding", {
    "metric_type":"COSINE", "index_type":"IVF_PQ",
    "params":{"nlist":2048, "m":96 if DIM==1536 else 128}
})
print("OK")
