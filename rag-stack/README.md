# LightRAG Stack ВМ

Целевой хост: `msk1-dev-gza-rag01.unix.nspk.ru`

Начальная цель: запустить рабочий стек:

- LightRAG server;
- nginx proxy для доступа к Web UI/API снаружи;
- embedding service на H100 через vLLM;
- PostgreSQL для KV/doc-status storage;
- Neo4j для graph storage;
- Qdrant для vector storage;
- без rerank;
- без MinerU;
- без VL processing.

В будущий полный стек можно добавить MinerU, rerank и VL models после
стабилизации запуска стека.

## Настройка

Перед первым запуском отредактировать `.env`:

- `LLM_BINDING_HOST`: H100 vLLM endpoint;
- `LLM_MODEL`: served model name с H100;
- `EMBEDDING_BINDING_HOST`: H100 vLLM embedding endpoint;
- `EMBEDDING_MODEL`: served embedding model name с H100;
- `EMBEDDING_DIM`: vector dimension для embedding model;
- `NGINX_HTTPS_PORT`: внешний HTTPS-порт Web UI/API;
- `POSTGRES_PASSWORD` и `NEO4J_PASSWORD`: заменить `change-me` перед запуском;
- Docker image variables, если internal registry tags отличаются.

LLM и embedding endpoints используют прямой HTTPS-маршрут с LightRAG ВМ до
H100 ВМ:

```env
LLM_BINDING_HOST=https://msk2-prod-ais-llm03.unix.nspk.ru/vllm/qwen/v1
EMBEDDING_BINDING_HOST=https://msk2-prod-ais-llm03.unix.nspk.ru/vllm/embed/v1
```

Перед запуском доступ надо проверить с `msk1-dev-gza-rag01`:

```bash
nc -vz -w 5 msk2-prod-ais-llm03.unix.nspk.ru 443
curl --fail --show-error --max-time 20 \
  https://msk2-prod-ais-llm03.unix.nspk.ru/vllm/qwen/v1/models
```

## Web UI auth

Если Web UI показывает ошибку вида:

```text
Failed to load documents 403 Forbidden {"detail":"Invalid API Key"} /documents/paginated
```

значит браузер отправляет заголовок `X-API-Key`, который не совпадает с
`LIGHTRAG_API_KEY` из `.env`.

Для текущего стенда ожидаемый ключ:

```env
LIGHTRAG_API_KEY=change-me
```

Нужно открыть настройки Web UI и указать этот API key. Если в браузере сохранён
старый ключ, можно очистить его через DevTools Console:

```javascript
localStorage.removeItem('settings-storage')
localStorage.removeItem('LIGHTRAG-API-TOKEN')
location.reload()
```

Проверка с RAG-машины:

```bash
curl -kfsS https://127.0.0.1/documents/paginated \
  -H 'X-API-Key: change-me' \
  -H 'Content-Type: application/json' \
  -d '{"page":1,"page_size":10,"sort_field":"updated_at","sort_direction":"desc"}'
```

## Миграция данных из corp-laptop

На corp-laptop старый эксперимент содержит file-based LightRAG storage в:

```text
~/Documents/LightRAG/storage
```

Скрипт `migrate_file_storage.py` из эксперимента переносил эти файлы в
`Redis + Neo4j + Qdrant`. В новом стеке Redis заменён на PostgreSQL, поэтому
для переноса используется отдельный мигратор:

```text
scripts/migrate-file-storage-to-stack.py
```

Он переносит:

- `kv_store_*.json` и `kv_store_doc_status.json` в PostgreSQL;
- `graph_chunk_entity_relation.graphml` в Neo4j;
- `vdb_entities.json`, `vdb_relationships.json`, `vdb_chunks.json` в Qdrant.

На corp-laptop:

```bash
cd ~/Documents/LightRAG
tar -czf /tmp/lightrag-file-storage.tar.gz storage
```

Передать архив на VDI, затем на RAG-машину и разложить:

```bash
mkdir -p ~/lightrag-rag-infra/rag-stack/imports/corp-laptop
tar -xzf /tmp/lightrag-file-storage.tar.gz \
  -C ~/lightrag-rag-infra/rag-stack/imports/corp-laptop
```

Перед импортом остановить LightRAG server, оставив базы запущенными:

```bash
cd ~/lightrag-rag-infra/rag-stack
docker-compose -f docker-compose.yaml stop lightrag
```

Dry-run:

```bash
python3 scripts/migrate-file-storage-to-stack.py \
  --env-file .env \
  --storage-dir imports/corp-laptop/storage \
  --dry-run
```

Импорт с очисткой текущего workspace:

```bash
python3 scripts/migrate-file-storage-to-stack.py \
  --env-file .env \
  --storage-dir imports/corp-laptop/storage \
  --drop-target
```

После импорта:

```bash
docker-compose -f docker-compose.yaml up -d lightrag
./scripts/check.sh
```

LightRAG требует offline tiktoken cache. Для `o200k_base.tiktoken` нужен файл:

```text
data/tiktoken/fb374d419588a4632f3f557e76b4b70aebbca790
```

Этот файл можно взять из старого эксперимента на corp-laptop:

```text
/home/gostokhovza/Documents/LightRAG/tiktoken_cache/fb374d419588a4632f3f557e76b4b70aebbca790
```

В контейнере cache монтируется в `/app/data/tiktoken`, а `TIKTOKEN_CACHE_DIR`
указывает туда же.

Локальный `ollama-embed` контейнер оставлен в compose как ручной fallback, но
LightRAG больше не зависит от него при запуске. Основной embedding endpoint -
H100 vLLM:

```text
https://msk2-prod-ais-llm03.unix.nspk.ru/vllm/embed/v1
```

Текущая embedding model:

```text
bge-m3
```

Данные не пересобираются при переключении с Ollama на vLLM, потому что model
name и dimension сохраняются: `bge-m3`, `1024`.

Также нужны TLS-файлы:

```text
certs/server.crt
certs/server.key
certs/llm-ca.crt
```

`server.crt` и `server.key` используются локальным RAG nginx. `llm-ca.crt` -
сертификат CA, которым подписан HTTPS-сертификат H100 endpoint. Он нужен
Python/httpx внутри LightRAG контейнера, чтобы доверять TLS при прямом
обращении к H100 LLM и embedding endpoints.

## Запуск

```bash
./scripts/start.sh
```

После запуска Web UI/API доступен через nginx:

```text
https://<host>/
```

Служебные веб-интерфейсы баз также доступны через этот nginx:

```text
https://<host>/dashboard
https://<host>/neo4j/browser/
```

Прямой порт LightRAG пробрасывается только на loopback для диагностики:

```text
http://127.0.0.1:9621/
```

## Проверка

```bash
./scripts/check.sh
```

Если после миграции Web UI показывает пустой список документов, проверить,
куда попали строки `doc_status`:

```bash
docker-compose -f docker-compose.yaml exec -T postgres \
  psql -U lightrag -d lightrag \
  -c "select workspace, status, count(*) from LIGHTRAG_DOC_STATUS group by workspace, status order by workspace, status;"
```

И сравнить с workspace, который видит LightRAG:

```bash
curl -kfsS -H "X-API-Key: change-me" \
  https://127.0.0.1/health
```

## Важно

Не индексировать реальные документы, пока не подтверждены embedding model и
`EMBEDDING_DIM`. Их последующее изменение означает пересборку vector index.

Данные из эксперимента на corp-laptop пока не переносятся. Запуск стека
поднимает пустой storage stack.
