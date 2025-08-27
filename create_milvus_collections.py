from pymilvus import connections, Collection

connections.connect(uri="http://localhost:19530")

col = Collection("vn_regulations")

# Tháo khỏi bộ nhớ trước khi đụng index
try:
    col.release()
except Exception:
    pass

# Xoá tất cả index hiện có của field 'vector'
for idx in col.indexes:
    # PyMilvus 2.3 thường có 1 index duy nhất; drop từng cái cho chắc
    try:
        col.drop_index(index_name=idx.index_name)
    except Exception:
        # fallback nếu version không hỗ trợ index_name
        col.drop_index()

# Tạo lại index với COSINE (khớp truy vấn của NAT)
col.create_index(
    "vector",
    {
        "index_type": "IVF_PQ",
        "metric_type": "COSINE",
        "params": {"nlist": 2048, "m": 128}  # m=128 phù hợp dim=3072
    }
)

# Load lại vào RAM
col.load()

print("Create collection complete.")
