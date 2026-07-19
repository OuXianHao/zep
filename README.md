<p align="center">
  <a href="https://www.getzep.com/">
    <img src="https://github.com/user-attachments/assets/119c5682-9654-4257-8922-56b7cb8ffd73" width="150" alt="Zep Logo">
  </a>
</p>

<h1 align="center">Zep Cloud: Examples & Integrations</h1>

<p align="center">
  <a href="https://discord.gg/W8Kw6bsgXQ"><img
    src="https://img.shields.io/badge/Discord-%235865F2.svg?&logo=discord&logoColor=white"
    alt="Chat on Discord"
  /></a>
  <a href="https://twitter.com/intent/follow?screen_name=zep_ai" target="_new"><img alt="Twitter Follow" src="https://img.shields.io/twitter/follow/zep_ai"></a>
</p>

## About This Repository

This repository is **not** Zep's product or service. It contains **example code, framework
integrations, and tools** for building agent memory with [Zep Cloud](https://www.getzep.com/),
Zep's managed agent memory platform.

To use Zep Cloud, sign up at [www.getzep.com](https://www.getzep.com/) and read the
documentation at [help.getzep.com](https://help.getzep.com). Zep's official SDKs are:

- **Python**: `pip install zep-cloud`
- **TypeScript/JavaScript**: `npm install @getzep/zep-cloud`
- **Go**: `go get github.com/getzep/zep-go/v3`

> Looking for the open-source temporal knowledge graph framework that powers Zep? See
> [Graphiti](https://github.com/getzep/graphiti).

## Contents

| Directory | Description |
|-----------|-------------|
| [`examples/`](examples/) | Example apps and snippets in Python, TypeScript, and Go |
| [`integrations/`](integrations/) | Agent-framework integration packages |
| [`ontology/`](ontology/) | Default ontology definitions |
| [`plugins/`](plugins/) | Plugins for building with Zep |
| [`benchmarks/`](benchmarks/) | Memory benchmarks (LoCoMo, LongMemEval) |
| [`zep-eval-harness/`](zep-eval-harness/) | Evaluation harness for ingestion and retrieval |
| [`legacy/`](legacy/) | Deprecated Zep Community Edition (unsupported) |

## Integrations

Framework integration packages live under [`integrations/`](integrations/), organized
framework-first then language: `integrations/<framework>/<language>/`. Each package is built,
tested, and released independently.

- **Python**: Google ADK, Microsoft Agent Framework, Microsoft AutoGen, AG2, CrewAI, LangGraph, LiveKit, Pydantic AI
- **TypeScript**: Google ADK, Mastra, Vercel AI SDK
- **Go**: Google ADK

See [`integrations/README.md`](integrations/README.md) for packages, release status, and
links, and [`integrations/CLAUDE.md`](integrations/CLAUDE.md) for conventions.

## Contributing

We welcome contributions. See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines covering code,
documentation, bug reports, and community examples.

## Community Edition (Deprecated)

Zep Community Edition is no longer supported. Its code has been moved to the
[`legacy/`](legacy/) folder. Read more in
[Announcing a New Direction for Zep's Open Source Strategy](https://blog.getzep.com/announcing-a-new-direction-for-zeps-open-source-strategy/).

## Local LoCoMo research experiment

The LoCoMo adapter is an isolated reproducibility path for academic comparison. It uses the
open-source Zep Community Edition in `legacy/` as local message storage and a local SQLite
sentence-transformers index for retrieval. It never uses the `zep_cloud` SDK or hosted Zep.
The only non-local request made by the adapter is Qwen answer generation.

### Deployment checklist

Before running, verify all of the following:

- [ ] Docker Engine and the Docker Compose plugin are available (`docker --version` and
      `docker compose version`).
- [ ] Local Zep is healthy (`curl --fail http://127.0.0.1:8000/healthz`).
- [ ] `models/bge-base-zh-v1.5/config.json` exists locally.
- [ ] `data/locomo10.json` exists.
- [ ] `OPENAI_API_KEY`, `OPENAI_BASE_URL`, and `LLM_MODEL` are exported for the Qwen endpoint.
- [ ] Commands are run from the repository root so repository-relative config paths resolve.

### 1. Environment setup on clean Ubuntu

Python 3.12+, Docker, Docker Compose, and a downloaded embedding model are required:

```bash
sudo apt-get update
sudo apt-get install -y curl python3 python3-venv
# Install Docker Engine + Compose using Docker's official Ubuntu instructions, then verify:
docker --version
docker compose version

# Install uv and project dependencies.
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
git clone <repository-url> zep
cd zep
uv sync
```

The adapter targets `http://127.0.0.1:8000/api/v2` by default. Its storage requests include
`X-Zep-Skip-Processing: true`, so Zep stores the messages without invoking its historical
Graphiti extraction pipeline; benchmark retrieval is exclusively the configured local BGE
index. Other Community Edition clients retain the original processing behavior.

### 2. Required environment variables

Only the OpenAI-compatible Qwen endpoint needs credentials. No `ZEP_API_KEY` or cloud account
is required:

```bash
export OPENAI_API_KEY="your_qwen_api_key"
export OPENAI_BASE_URL="https://your-qwen-endpoint.example/v1"
export LLM_PROVIDER="qwen"
export LLM_MODEL="Qwen3-30B-A3B-Instruct-2507"
```

The variable names are required by the OpenAI-compatible Python client and the legacy local
Graphiti container, but their values point to Qwen; an OpenAI account or OpenAI-hosted API is
not required. Export these variables **before** starting Docker Compose:

```bash
docker compose -f legacy/docker-compose.ce.yaml up -d --build
curl --retry 30 --retry-delay 2 --retry-connrefused --fail \
  http://127.0.0.1:8000/healthz
```

### 3. Prepare the local embedding model and dataset

Download `BAAI/bge-base-zh-v1.5` and LoCoMo once during preparation:

```bash
mkdir -p models data
uv run python -c 'from huggingface_hub import snapshot_download; snapshot_download("BAAI/bge-base-zh-v1.5", local_dir="models/bge-base-zh-v1.5")'
curl --fail --location \
  https://raw.githubusercontent.com/snap-research/locomo/refs/heads/main/data/locomo10.json \
  --output data/locomo10.json
test -f models/bge-base-zh-v1.5/config.json
test -f data/locomo10.json
```

The evaluation itself passes `local_files_only=True` and therefore cannot download an embedding
model or call an embedding API. All paths in `configs/qwen_locomo.yaml` are repository-relative
and configurable; no developer-specific absolute path is present.

### 4. Run one sample

```bash
uv run python scripts/eval_locomo.py \
  --config configs/qwen_locomo.yaml \
  --max-samples 1 \
  --debug-output outputs/qwen_locomo_debug_one.json
```

### 5. Run the full evaluation

```bash
uv run python scripts/eval_locomo.py --config configs/qwen_locomo.yaml
```

To stop the local services after a run:

```bash
docker compose -f legacy/docker-compose.ce.yaml down
```

Predictions are saved at `outputs/qwen_locomo_predictions.json`. The optional debug JSON records
each question, category, retrieved memories, answer and prediction, inserted/retrieved counts,
and retrieval/generation latency. The local semantic memory persists at
`outputs/qwen_locomo_memory.sqlite3`; each sample is atomically replaced on rerun.

### Network dependency audit

At experiment runtime, Python makes one remote request type: OpenAI-compatible Chat Completions
to `OPENAI_BASE_URL` for Qwen generation. Zep requests are restricted in code to loopback hosts.
There is no `zep_cloud` import, `ZEP_API_KEY`, hosted memory call, OpenAI-hosted model, or external
embedding call in this adapter. Initial provisioning still needs network access to clone the
repository, install packages/pull containers, download LoCoMo, and download BGE; these artifacts
can instead be staged on an offline server before the run.
