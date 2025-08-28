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

# 2) Start containers
  ```bash
  docker compose -f deploy/docker-compose.yaml -f deploy/docker-compose-utils.yaml up -d
  ```
## If self-host models
[embedding model](https://build.nvidia.com/nvidia/llama-3_2-nv-embedqa-1b-v2/deploy?environment=linux.md)
**NOTE:** set exposed port to 8016 *(-p 8016:8000 \)*
[LLM](https://build.nvidia.com/nvidia/llama-3_3-nemotron-super-49b-v1_5/deploy)
**NOTE:** set exposed port to 8015 *(-p 8015:8000 \)*

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

# 7) Evaluation & Sizing

  ```bash
  nat sizing calc --config_file configs/legal_multi_agent_eval.yml --calc_output_dir sizing_output --concurrencies 1,2,4,8,16 --num_passes 1
  ```