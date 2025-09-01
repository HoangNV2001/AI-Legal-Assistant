# AI-Legal-Assistant
Built with NeMo Agent Toolkit

# 1) Preparation

## 1.1) Requirements

* Python 3.11 or 3.12
* Docker + docker-compose + [nvidia container toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
* [uv](https://docs.astral.sh/uv/getting-started/installation/)
* [NAT](https://github.com/NVIDIA/NeMo-Agent-Toolkit/blob/develop/docs/source/quick-start/installing.md)

## 1.2) Clone 2 repos
  ```bash
  git clone https://github.com/NVIDIA/NeMo-Agent-Toolkit 
  git clone https://github.com/HoangNV2001/AI-Legal-Assistant
  cd AI-Legal-Assistant
  ```

## 1.3) API keys

  ```bash
  export NVIDIA_API_KEY=...      
  export TAVILY_API_KEY=...      
  ```
  
## 1.4) Create venv
  ```bash
  uv venv --python 3.11
  source .venv/bin/activate
  uv pip install -e ./
  ```

# 2) Start services & self-host models
## 2.1) Start docker containers
  ```bash
  docker compose -f deploy/docker-compose.yaml -f deploy/docker-compose-utils.yaml up -d
  ```

## 2.2) Self-host models
* [embedding model: llama-3_2-nv-embedqa-1b-v2](https://build.nvidia.com/nvidia/llama-3_2-nv-embedqa-1b-v2/deploy?environment=linux.md)
**NOTE:** set exposed port to 8016 *(-p 8016:8000 \)*
* [LLM: nvidia-nemotron-nano-9b-v2; section *Use it with vLLM*](https://build.nvidia.com/nvidia/nvidia-nemotron-nano-9b-v2/modelcard)
**NOTE:** set exposed port to 8015

# 3) Create Milvus collection
  ```bash
  python create_milvus_collections.py   
  ```

# 4) Ingest data with OpenAI Embeddings
  ```bash
  sudo apt-get update
  sudo apt-get install -y ffmpeg 
  python ingest_to_milvus.py \
    --folder ./data/vbqp \
    --collection vn_regulations \
    --milvus-uri http://localhost:19530 \
    --nim-embed-base-url http://localhost:8016/v1 \
    --nim-embed-model nvidia/llama-3.2-nv-embedqa-1b-v2 \
    --batch-size 64 \
    --chunk-tokens 600 --chunk-overlap 80 \
    --issued-date "" \
    --add-source
  ```

# 5) Run example using NAT's CLI
  ```bash
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