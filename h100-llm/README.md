# H100 LLM ВМ

Целевой хост: `msk2-prod-ais-llm03.unix.nspk.ru`

Эта директория запускает Qwen LLM и bge-m3 embedding через vLLM и отдаёт
OpenAI-compatible API через nginx.
Compose-файл ожидает, что model files уже лежат на ВМ здесь:

```text
/apps/mgpt/models/llm
```

Текущие served model names:

```text
qwen3-30b-fp8
bge-m3
```

Qwen ограничен `--gpu-memory-utilization 0.85`, embedding service -
`--gpu-memory-utilization 0.10`. Целевой суммарный бюджет VRAM - около 75 ГБ
из 80 ГБ.

nginx endpoints:

```text
https://<host>/vllm/qwen/v1
https://<host>/vllm/embed/v1
```

## Запуск

Для быстрого внутреннего теста можно создать временный self-signed certificate,
если настоящий сертификат ещё не готов:

```bash
./scripts/create-self-signed-cert.sh
```

```bash
./scripts/start.sh
```

## Проверка

```bash
./scripts/check.sh
```

Скрипт проверки проверяет состояние локальных контейнеров и HTTPS endpoint,
который отдаётся через nginx.
