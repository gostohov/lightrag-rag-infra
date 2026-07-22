from __future__ import annotations

import copy
import unittest

from pmi_generator.workbench.application.evaluation import (
    CorpusValidationError,
    QualityCorpusCodec,
)


def selection(start: int, end: int) -> dict[str, object]:
    return {
        "start": {"page": 1, "line": start},
        "end": {"page": 1, "line": end},
        "lines": [
            {"page": 1, "line": line, "text": f"synthetic line {line}"}
            for line in range(start, end + 1)
        ],
    }


def corpus_fixture() -> dict[str, object]:
    return {
        "schema_version": 1,
        "cases": [
            {
                "case_id": "SYNTHETIC_CASE_001",
                "source_hash": "a" * 64,
                "reason": "Synthetic contract fixture",
                "comparison_selection": selection(1, 3),
                "baseline_selection": selection(1, 2),
                "free_selection": selection(1, 3),
                "expected_outcome": "skeletons_created",
                "expected_obligations": [
                    {
                        "obligation_id": "OBL_001",
                        "description": "Synthetic obligation",
                        "evidence": [{"page": 1, "line_start": 2, "line_end": 3}],
                        "context_loss": True,
                    }
                ],
                "forbidden_claims": [
                    {
                        "claim_id": "CLAIM_001",
                        "description": "Synthetic forbidden claim",
                    }
                ],
                "allowed_groupings": [["OBL_001"]],
                "line_expectations": [
                    {"page": 1, "line": 1, "role": "context"},
                    {"page": 1, "line": 2, "role": "evidence"},
                    {"page": 1, "line": 3, "role": "evidence"},
                ],
                "analyst_decision": {
                    "version": 1,
                    "status": "approved",
                    "comment": "Synthetic fixture only",
                },
            }
        ],
    }


class QualityCorpusCodecTests(unittest.TestCase):
    def test_loads_strict_typed_case_and_computes_content_version(self) -> None:
        corpus = QualityCorpusCodec.load(corpus_fixture())

        self.assertEqual(corpus.schema_version, 1)
        self.assertEqual(corpus.cases[0].case_id, "SYNTHETIC_CASE_001")
        self.assertTrue(corpus.version.startswith("sha256:"))
        self.assertFalse(corpus.ready_for_full_run)
        self.assertEqual(corpus.missing_case_count, 11)

    def test_expectation_change_changes_corpus_version(self) -> None:
        original = QualityCorpusCodec.load(corpus_fixture())
        changed_payload = copy.deepcopy(corpus_fixture())
        changed_payload["cases"][0]["expected_obligations"][0][  # type: ignore[index]
            "description"
        ] = "Changed synthetic obligation"
        changed = QualityCorpusCodec.load(changed_payload)

        self.assertNotEqual(original.version, changed.version)

    def test_rejects_case_without_source_hash(self) -> None:
        payload = corpus_fixture()
        del payload["cases"][0]["source_hash"]  # type: ignore[index]

        with self.assertRaisesRegex(CorpusValidationError, "source_hash"):
            QualityCorpusCodec.load(payload)

    def test_requires_obligation_except_for_no_testable_behavior(self) -> None:
        payload = corpus_fixture()
        payload["cases"][0]["expected_obligations"] = []  # type: ignore[index]

        with self.assertRaisesRegex(CorpusValidationError, "expected_obligations"):
            QualityCorpusCodec.load(payload)

        payload["cases"][0]["expected_outcome"] = (  # type: ignore[index]
            "no_testable_behavior"
        )
        payload["cases"][0]["allowed_groupings"] = []  # type: ignore[index]
        corpus = QualityCorpusCodec.load(payload)
        self.assertEqual(corpus.cases[0].expected_obligations, ())

    def test_rejects_unknown_keys_and_incomplete_line_expectations(self) -> None:
        payload = corpus_fixture()
        payload["cases"][0]["unexpected"] = True  # type: ignore[index]
        with self.assertRaisesRegex(CorpusValidationError, "unexpected"):
            QualityCorpusCodec.load(payload)

        payload = corpus_fixture()
        payload["cases"][0]["line_expectations"].pop()  # type: ignore[index]
        with self.assertRaisesRegex(CorpusValidationError, "line_expectations"):
            QualityCorpusCodec.load(payload)


if __name__ == "__main__":
    unittest.main()
