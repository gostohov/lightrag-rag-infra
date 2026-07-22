# pmi-generator

VDI-приложение для управляемой analyst-in-the-loop подготовки технического ПМИ
по спецификации.

Единственный поддерживаемый режим продукта — `PMI Workbench`. Архитектура
описана в
[`docs/architecture/pmi-test-design`](../docs/architecture/pmi-test-design/README.md),
а эксплуатация и восстановление — в
[`docs/operations/pmi-workbench.md`](../docs/operations/pmi-workbench.md).

## Установка

```bash
cd pmi-generator
python -m venv .venv
. .venv/bin/activate
python -m pip install -e .
cp .env.example .env
```

## Запуск

Новая работа из PDF:

```bash
pmi-generator \
  --pdf /path/to/specification.pdf \
  --run /tmp/pmi-workbench-YYYYMMDD
```

Продолжение существующей работы:

```bash
pmi-generator --run /tmp/pmi-workbench-YYYYMMDD
```

Первая команда локально извлекает text layer и outline через `pypdf`, копирует
исходный файл в `source/original.pdf` и атомарно создаёт
`source/document.sqlite3`. PDF без text layer не поддерживается. Содержание
страниц не анализируется для распознавания разделов: используется только
авторский outline либо навигация по страницам.

Persistent state хранится в `review/workbench.sqlite3`, результаты — в
`review/exports`, диагностика — в `review/diagnostics`. Повторный запуск с тем
же `--run` восстанавливает сохранённые диапазоны, карточки, сообщения, решения
и LangGraph checkpoints. Старые run с `pages.json` и
`structural_chunks.json` не импортируются.

URL, модели, ключи, TLS и timeout задаются через `.env` или переменные окружения
`PMI_*`; отдельные CLI-флаги для них отсутствуют.

## Локальный ручной UI-тест

Production TUI можно пройти без vLLM, LightRAG, URL и ключей:

```bash
pmi-generator \
  --pdf /path/to/specification.pdf \
  --run /tmp/pmi-workbench-mock-YYYYMMDD \
  --mock
```

Используйте отдельный run. Mock mode сохраняется в source metadata:
production run нельзя открыть с `--mock`, а mock run нельзя открыть без него.
Для повторного открытия передайте тот же PDF, run и `--mock`.

Ручной checklist:

1. Проверить текст `Тестовый режим: mock` на первом экране.
2. Выбрать произвольный непустой диапазон и дождаться Prompt 1 со spinner.
3. Убедиться, что каркасы содержательно связаны с первой и последней
   различающимися строками selection.
4. При наличии двух каркасов один взять в работу, второй исключить.
5. В session дождаться Prompt 2, mock-вопроса LightRAG и закрытия одного
   пробела.
6. Задать обычным текстом вопрос о состоянии и проверить read-only
   conversation-ответ без изменения карточки.
7. Отправить поисковую инструкцию и предметное уточнение обычным текстом;
   проверить объявление typed action и доработку карточки.
8. Выполнить `/include`, затем запустить проверку диапазона.
9. Проверить результат `approved` и сформировать полный ПМИ клавишей `E` на
   экране структуры.
10. Перезапустить ту же команду и проверить восстановление диапазона, session,
   решений и checkpoints.

Mock-ответы детерминированы и явно помечены. Этот режим проверяет UI и
оркестрацию, но не качество реальных Prompt 1–4 или retrieval.

## Проверка

Локальный suite без корпоративной сети:

```bash
.venv/bin/python -m unittest discover -s tests -p 'test_*.py'
.venv/bin/python -m unittest -v tests.test_workbench_mock_mode
.venv/bin/python -m compileall -q src/pmi_generator
```

Opt-in live smoke выполняется только на VDI с доступом к vLLM и LightRAG:

```bash
PMI_WORKBENCH_LIVE_SMOKE=1 \
  .venv/bin/python -m unittest -v \
  tests.test_workbench_end_to_end.LiveServicesSmokeTest
```
