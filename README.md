# AI-Legal-Assistant
Built with NeMo Agent Toolkit

# 1) Preparation

**Requirements:**

* Python 3.11 or 3.12
* Docker + docker-compose
* [uv](https://docs.astral.sh/uv/getting-started/installation/)
* [NAT](https://github.com/NVIDIA/NeMo-Agent-Toolkit/blob/develop/docs/source/quick-start/installing.md)
* API keys:

  ```bash
  export OPENAI_API_KEY=...
  export NVIDIA_API_KEY=...      
  export TAVILY_API_KEY=...      
  ```

# 2) Start middleware services
  ```bash
  docker compose -f deploy/docker-compose.yaml -f deploy/docker-compose-utils.yaml up -d
  ```

# 3) Create Milvus collection
  ```bash
  python3.12 -m pip install openai pymilvus tiktoken
  python3.12 create_milvus_collections.py   
  ```

# 4) Ingest data with OpenAI Embeddings
  ```bash
  sudo apt-get update
  sudo apt-get install -y ffmpeg 
  python3.12 -m pip install "markitdown[all]" 
  python3.12 ingest_to_milvus.py \
  --folder ./data/vbqp \
  --collection vn_regulations \
  --openai-embed-model text-embedding-3-large \
  --chunk-tokens 600 --chunk-overlap 80
  ```

# 5) Run example using NAT's CLI
  ```bash
  uv pip install -e ./
  nat run --config_file configs/legal_multi_agent.yml \
          --input "Vượt đèn đỏ bị phạt bao tiền"
  ```

# 6) Launching NAT API Server & UI
## Start the NeMo Agent Toolkit Server
  ```bash
  nat serve --config_file configs/legal_multi_agent.yml --port 8006
  # Or nohup with log file
  nohup nat serve --config_file configs/legal_multi_agent.yml --port 8006 > "nat_serve_log_$(date +'%Y%m%d_%H%M%S').log" 2>&1 &

  ```
## UI
  [Reference](https://github.com/NVIDIA/NeMo-Agent-Toolkit/blob/develop/docs/source/quick-start/launching-ui.md)
