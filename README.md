# Корпоративное развёртывание LightRAG

Репозиторий для переноса текущих RAG-экспериментов во внутреннюю корпоративную
среду.

Развёртывание разделено по целевым машинам:

| Директория | Целевая машина | Назначение |
| --- | --- | --- |
| `h100-llm/` | `msk2-prod-ais-llm03.unix.nspk.ru` | vLLM с Qwen на Nvidia H100 80GB VRAM |
| `rag-stack/` | `msk1-dev-gza-rag01.unix.nspk.ru` | LightRAG-инфраструктура, сначала LightRAG stack + embedding |

## Инфраструктура

### H100 LLM ВМ

Хост: `msk2-prod-ais-llm03.unix.nspk.ru`

Запускает vLLM в Docker. Стартовый compose-файл основан на текущем локальном
эксперименте из `C:\Users\gosto\Downloads\docker-compose.h100.yaml`.

Ожидаемая роль:

- отдавать Qwen через OpenAI-compatible API;
- держать основной LLM-инференс отдельно от LightRAG-инфраструктуры;
- предоставить model endpoint для LightRAG ВМ внутри корпоративной сети.

### LightRAG ВМ

Хост: `msk1-dev-gza-rag01.unix.nspk.ru`

Запрошенный sizing: 16 vCPU, 64 GB RAM, отдельный data disk 500 GB.

Планируемый полный стек:

- PostgreSQL;
- Qdrant;
- Neo4j;
- MinerU;
- embedding model;
- rerank model;
- VL model.

Начальная цель этого репозитория:

- LightRAG server;
- nginx proxy для внешнего доступа к LightRAG Web UI/API;
- H100 embedding service через vLLM;
- PostgreSQL вместо Redis для KV/doc-status storage;
- Neo4j для graph storage;
- Qdrant для vector storage;
- без rerank;
- без VL model;
- без MinerU;
- без переноса экспериментальных данных.

Embedding model надо выбрать до начала реальной индексации документов. Если
позже поменять embedding model или vector dimension, индексированные данные
придётся перестраивать.

## Быстрый старт

### H100 ВМ

```bash
cd h100-llm
./scripts/start.sh
./scripts/check.sh
```

### LightRAG ВМ

```bash
cd rag-stack
# отредактировать .env: внутренние hostnames, image tags, API keys и model paths
./scripts/start.sh
./scripts/check.sh
```

После запуска LightRAG Web UI/API доступен через nginx:

```text
https://<msk1-dev-gza-rag01.unix.nspk.ru>/
```

## Внешние материалы для первичного bootstrap

- LightRAG installation и Docker Compose notes:
  https://github.com/HKUDS/LightRAG
- LightRAG Docker deployment, local vLLM embedding и offline caveats:
  https://github.com/HKUDS/LightRAG/blob/main/docs/DockerDeployment.md
