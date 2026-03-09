import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import validate_schema, verify_cases


class VerifyCasesTests(unittest.TestCase):
    def test_find_duplicate_case_ids_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            first = root / "a.jsonl"
            second = root / "b.jsonl"
            first.write_text(json.dumps({"case_id": "dup"}) + "\n", encoding="utf-8")
            second.write_text(json.dumps({"case_id": "dup"}) + "\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "duplicate case_id"):
                verify_cases.find_duplicate_case_ids([first, second])

    def test_load_jsonl_records_reads_multiple_lines(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "cases.jsonl"
            path.write_text(
                "\n".join(
                    [
                        json.dumps({"case_id": "a"}),
                        json.dumps({"case_id": "b"}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            records = verify_cases.load_jsonl_records(path)

            self.assertEqual([record["case_id"] for record in records], ["a", "b"])


class ValidateSchemaTests(unittest.TestCase):
    def test_require_jsonschema_dependency_raises_clear_error(self) -> None:
        real_import = __import__

        def fake_import(name, *args, **kwargs):
            if name == "jsonschema":
                raise ModuleNotFoundError("No module named 'jsonschema'")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=fake_import):
            with self.assertRaisesRegex(RuntimeError, "jsonschema is required"):
                validate_schema.require_jsonschema()


if __name__ == "__main__":
    unittest.main()
