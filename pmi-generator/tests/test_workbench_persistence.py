from __future__ import annotations

import json
import tempfile
import unittest
from datetime import UTC, datetime
from pathlib import Path
from typing import Callable

from pmi_generator.workbench.domain import (
    CardMutation,
    ContentField,
    Evidence,
    GapResolutionMode,
    RelatedGap,
    SourceAddress,
    TestCard,
)
from pmi_generator.workbench.infrastructure.storage.card_codec import (
    decode_card,
    encode_card,
)
from pmi_generator.workbench.infrastructure.storage import (
    AttemptRecord,
    AttemptStatus,
    InMemoryDatabase,
    InMemoryUnitOfWork,
    SessionRecord,
    SqliteUnitOfWork,
    StorageConflictError,
    StorageError,
    StorageSchemaError,
    StoredRecord,
    workbench_database_path,
)


NOW = datetime(2026, 7, 16, 12, 0, tzinfo=UTC)


def make_card() -> TestCard:
    return TestCard.create(
        card_id="CARD_0001",
        selection_id="SELECTION_0001",
        title="Проверка PUT DATA",
        section_number="4.16.5",
        changed_factors=("первый байт данных",),
        consequences=("карта возвращает 6987",),
    )


def add_source_field(test_card: TestCard, suffix: str = "1") -> None:
    evidence = Evidence.source_fragment(
        evidence_id=f"EVIDENCE_000{suffix}",
        card_id=test_card.card_id,
        selection_id=test_card.selection_id,
        quote="Первый байт данных должен быть равен 81.",
        address=SourceAddress(
            document_id="spec_2.3.pdf",
            document_version="2.3",
            page=283,
            line_start=19,
            line_end=24,
            chunk_id="section-0270",
        ),
        collected_at=NOW,
    )
    test_card.apply(
        CardMutation(
            evidence=(evidence,),
            fields={
                "requirement.condition": ContentField.confirmed(
                    f"первый байт равен 8{suffix}",
                    evidence_ids=(evidence.evidence_id,),
                )
            },
        )
    )


class RepositoryContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.sqlite_path = Path(self.temp_dir.name) / "workbench.sqlite3"
        self.memory = InMemoryDatabase()

    def factories(self) -> list[tuple[str, Callable[[], object]]]:
        return [
            ("memory", lambda: InMemoryUnitOfWork(self.memory)),
            ("sqlite", lambda: SqliteUnitOfWork(self.sqlite_path)),
        ]

    def test_card_round_trip_preserves_types_revision_and_decision(self) -> None:
        for name, factory in self.factories():
            with self.subTest(backend=name):
                test_card = make_card()
                add_source_field(test_card)
                test_card.include_incomplete(author="Аналитик", reason="Технический экспорт", at=NOW)

                with factory() as uow:
                    uow.cards.save(test_card)

                with factory() as uow:
                    loaded = uow.cards.get(test_card.card_id)

                self.assertIsNotNone(loaded)
                self.assertEqual(loaded.revision, test_card.revision)
                self.assertEqual(loaded.decision, test_card.decision)
                self.assertEqual(loaded.field("requirement.condition"), test_card.field("requirement.condition"))
                self.assertEqual(loaded.evidence, test_card.evidence)

    def test_legacy_gap_without_resolution_mode_restores_as_source_fact(self) -> None:
        test_card = make_card()
        test_card.apply(
            CardMutation(
                gaps=(
                    RelatedGap(
                        gap_id="GAP_0001",
                        card_id=test_card.card_id,
                        question="Какое значение указано в источнике?",
                        blocking_reason="Значение неизвестно",
                        allowed_paths=("test.control_values",),
                        dependencies=(),
                        closure_criterion="Значение подтверждено",
                    ),
                )
            )
        )
        payload = json.loads(encode_card(test_card))
        del payload["gaps"][0]["resolution_mode"]

        restored = decode_card(json.dumps(payload, ensure_ascii=False))

        self.assertIs(
            restored.gaps["GAP_0001"].resolution_mode,
            GapResolutionMode.SOURCE_FACT,
        )
        self.assertEqual(
            json.loads(encode_card(restored))["gaps"][0]["resolution_mode"],
            "source_fact",
        )

    def test_legacy_human_only_source_status_migrates_to_analyst_confirmed(
        self,
    ) -> None:
        test_card = make_card()
        add_source_field(test_card)
        payload = json.loads(encode_card(test_card))
        payload["evidence"][0].update(
            {
                "kind": "экспертное знание",
                "address": None,
                "author": "Аналитик",
                "message_id": "MSG_LEGACY",
            }
        )

        restored = decode_card(json.dumps(payload, ensure_ascii=False))

        field = restored.field("requirement.condition")
        self.assertEqual(field.status.value, "подтверждено аналитиком")
        self.assertEqual(field.value, "первый байт равен 81")
        resolution = next(iter(restored.resolutions.values()))
        self.assertEqual(
            resolution.target_paths,
            ("requirement.condition",),
        )
        self.assertEqual(resolution.evidence_ids, field.evidence_ids)

    def test_legacy_mixed_source_and_human_evidence_is_not_auto_migrated(
        self,
    ) -> None:
        test_card = make_card()
        add_source_field(test_card)
        payload = json.loads(encode_card(test_card))
        human = {
            **payload["evidence"][0],
            "evidence_id": "EVIDENCE_HUMAN",
            "kind": "экспертное знание",
            "address": None,
            "author": "Аналитик",
            "message_id": "MSG_LEGACY",
        }
        payload["evidence"].append(human)
        payload["fields"]["requirement.condition"]["evidence_ids"].append(
            "EVIDENCE_HUMAN"
        )

        with self.assertRaisesRegex(StorageError, "смешан"):
            decode_card(json.dumps(payload, ensure_ascii=False))

    def test_transaction_rolls_back_all_changes(self) -> None:
        for name, factory in self.factories():
            with self.subTest(backend=name):
                with self.assertRaisesRegex(RuntimeError, "abort"):
                    with factory() as uow:
                        uow.cards.save(make_card())
                        uow.sessions.save(
                            SessionRecord(
                                session_id="SESSION_0001",
                                selection_id="SELECTION_0001",
                                card_id="CARD_0001",
                                current_stage="подготовка",
                                payload={"step": 1},
                                updated_at=NOW,
                            )
                        )
                        raise RuntimeError("abort")

                with factory() as uow:
                    self.assertIsNone(uow.cards.get("CARD_0001"))
                    self.assertIsNone(uow.sessions.get("SESSION_0001"))

    def test_older_revision_cannot_overwrite_newer_card(self) -> None:
        for name, factory in self.factories():
            with self.subTest(backend=name):
                old = make_card()
                current = make_card()
                add_source_field(current)
                with factory() as uow:
                    uow.cards.save(current)

                with self.assertRaises(StorageConflictError), factory() as uow:
                    uow.cards.save(old)

    def test_content_change_does_not_restore_old_decision(self) -> None:
        for name, factory in self.factories():
            with self.subTest(backend=name):
                test_card = make_card()
                add_source_field(test_card)
                test_card.include_incomplete(author="Аналитик", reason="Первая версия", at=NOW)
                with factory() as uow:
                    uow.cards.save(test_card)

                add_source_field(test_card, "2")
                with factory() as uow:
                    uow.cards.save(test_card)
                with factory() as uow:
                    loaded = uow.cards.get(test_card.card_id)

                self.assertIsNone(loaded.decision)
                self.assertEqual(loaded.revision, 2)

    def test_session_and_active_stage_survive_restart(self) -> None:
        for name, factory in self.factories():
            with self.subTest(backend=name):
                session = SessionRecord(
                    session_id="SESSION_0001",
                    selection_id="SELECTION_0001",
                    card_id="CARD_0001",
                    current_stage="исследуется пробел",
                    payload={"gap_id": "GAP_0001"},
                    updated_at=NOW,
                )
                with factory() as uow:
                    uow.sessions.save(session)
                with factory() as uow:
                    loaded = uow.sessions.get(session.session_id)

                self.assertEqual(loaded, session)

    def test_cancelled_attempt_is_diagnostic_but_not_active(self) -> None:
        for name, factory in self.factories():
            with self.subTest(backend=name):
                attempt = AttemptRecord(
                    attempt_id="ATTEMPT_0001",
                    session_id="SESSION_0001",
                    stage="Промпт 2",
                    status=AttemptStatus.ACTIVE,
                    payload={},
                    updated_at=NOW,
                )
                with factory() as uow:
                    uow.attempts.save(attempt)
                    uow.attempts.save(attempt.with_status(AttemptStatus.CANCELLED, NOW))
                with factory() as uow:
                    loaded = uow.attempts.get(attempt.attempt_id)
                    active = uow.attempts.active_for_session(attempt.session_id)

                self.assertEqual(loaded.status, AttemptStatus.CANCELLED)
                self.assertIsNone(active)

    def test_generic_records_and_events_use_same_transaction(self) -> None:
        for name, factory in self.factories():
            with self.subTest(backend=name):
                record = StoredRecord(
                    kind="selection",
                    record_id="SELECTION_0001",
                    payload={"page": 283},
                )
                with factory() as uow:
                    uow.records.save(record)
                    sequence = uow.events.append(
                        "SELECTION_0001",
                        "диапазон подтвержден",
                        {"page": 283},
                    )
                with factory() as uow:
                    loaded = uow.records.get(record.kind, record.record_id)
                    events = uow.events.list_for(record.record_id)

                self.assertEqual(loaded, record)
                self.assertEqual(events[0].sequence, sequence)
                self.assertEqual(events[0].event_type, "диапазон подтвержден")


class SqliteLayoutTests(unittest.TestCase):
    def test_database_is_created_inside_run_and_legacy_json_is_ignored(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            legacy = run_dir / "review" / "review_decisions.json"
            legacy.parent.mkdir(parents=True)
            legacy.write_text('[{"legacy": true}]', encoding="utf-8")
            database_path = workbench_database_path(run_dir)

            with SqliteUnitOfWork(database_path) as uow:
                self.assertEqual(uow.cards.list_all(), [])

            self.assertTrue(database_path.exists())
            self.assertEqual(database_path, run_dir / "review" / "workbench.sqlite3")

    def test_incompatible_schema_version_has_clear_error(self) -> None:
        import sqlite3

        with tempfile.TemporaryDirectory() as tmp:
            database_path = Path(tmp) / "workbench.sqlite3"
            connection = sqlite3.connect(database_path)
            connection.execute("CREATE TABLE schema_meta (version INTEGER NOT NULL)")
            connection.execute("INSERT INTO schema_meta(version) VALUES (99)")
            connection.commit()
            connection.close()

            with self.assertRaisesRegex(StorageSchemaError, "несовместима"):
                with SqliteUnitOfWork(database_path):
                    pass


if __name__ == "__main__":
    unittest.main()
