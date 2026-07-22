from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from pmi_generator.core.env import load_env_files


class EnvTests(unittest.TestCase):
    def test_load_env_files_reads_values_without_overriding_environment(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env_path = Path(tmp) / ".env"
            env_path.write_text(
                "\n".join(
                    [
                        "PMI_TEST_FROM_FILE=file-value",
                        "PMI_TEST_QUOTED='quoted value'",
                        "PMI_TEST_EXISTING=file-value",
                    ]
                ),
                encoding="utf-8",
            )
            old_values = {
                key: os.environ.get(key)
                for key in ["PMI_TEST_FROM_FILE", "PMI_TEST_QUOTED", "PMI_TEST_EXISTING"]
            }
            try:
                os.environ.pop("PMI_TEST_FROM_FILE", None)
                os.environ.pop("PMI_TEST_QUOTED", None)
                os.environ["PMI_TEST_EXISTING"] = "env-value"

                loaded = load_env_files([env_path])

                self.assertEqual(loaded, [env_path.resolve()])
                self.assertEqual(os.environ["PMI_TEST_FROM_FILE"], "file-value")
                self.assertEqual(os.environ["PMI_TEST_QUOTED"], "quoted value")
                self.assertEqual(os.environ["PMI_TEST_EXISTING"], "env-value")
            finally:
                for key, value in old_values.items():
                    if value is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = value


if __name__ == "__main__":
    unittest.main()
