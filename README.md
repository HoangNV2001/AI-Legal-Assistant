# AI-Legal-Assistant
Built with NeMo Agent Toolkit

# 1) Chuẩn bị

**Yêu cầu:**

* Python 3.11 hoặc 3.12
* Docker + docker-compose
* API keys:

  ```bash
  export OPENAI_API_KEY=...
  export NVIDIA_API_KEY=...      
  export TAVILY_API_KEY=...      
  ```

# 2) Khởi chạy hạ tầng vector DB (Milvus, MinIO, etcd, Attu, ES/Kibana – tuỳ chọn)
  ```bash
  docker compose -f deploy/docker-compose.yaml -f deploy/docker-compose-utils.yaml up -d
  ```

# 3) Tạo collections Milvus
  ```bash
  python3.12 -m pip install openai pymilvus tiktoken
  python3.12 create_milvus_collections_oai.py   
  ```

# 4) Ingest data with OpenAI Embeddings
  ```bash
  python3.12 -m pip install "markitdown[all]" 
  python3.12 ingest_to_milvus.py \
  --folder ./data/vbqp \
  --collection vn_regulations \
  --milvus-uri http://localhost:19530 \
  --openai-embed-model text-embedding-3-large \
  --chunk-tokens 600 --chunk-overlap 80 \
  --so-hieu-from content --auto-issued-date

  ```

# 5) Run NAT
  ```bash
    # chạy hỏi thử
    uv pip install -e ./
    nat run --config_file configs/legal_multi_agent.yml \
            --input "Tóm tắt điều kiện hiệu lực của hợp đồng lao động vô thời hạn theo pháp luật VN"
    nat eval --config_file configs/legal_multi_agent.yml
  ```