import tempfile
import unittest
from pathlib import Path

from scripts import report_complex_support


REPO_ROOT = Path(__file__).resolve().parents[1]
CHECKED_IN_REPORT = REPO_ROOT / "docs" / "generated" / "complex-support.md"


class ComplexSupportReportTests(unittest.TestCase):
    def test_build_report_includes_expected_sections(self) -> None:
        report = report_complex_support.build_report_text()

        self.assertIn("# Complex Support Report", report)
        self.assertIn("## Summary", report)
        self.assertIn("## Full Ledger", report)
        self.assertIn("## Ready For Downstream", report)
        self.assertIn("## Unsupported", report)
        self.assertIn("## Still Pending", report)

    def test_build_report_highlights_representative_entries(self) -> None:
        report = report_complex_support.build_report_text()

        self.assertIn(
            "| svd | u_abs | reviewed | covered | complex128, complex64 | - | yes |",
            report,
        )
        self.assertIn(
            "| atan2 | identity | not_required | unsupported | - | "
            "float-only in pinned PyTorch upstream AD coverage | no |",
            report,
        )
        self.assertIn("## Still Pending\n\nNone.", report)

    def test_main_writes_report_and_matches_checked_in_copy(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "complex-support.md"

            self.assertEqual(
                report_complex_support.main(["--output", str(output_path)]),
                0,
            )
            self.assertEqual(
                output_path.read_text(encoding="utf-8"),
                CHECKED_IN_REPORT.read_text(encoding="utf-8"),
            )


if __name__ == "__main__":
    unittest.main()
