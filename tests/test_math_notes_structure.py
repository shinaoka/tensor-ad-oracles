import re
import unittest
from pathlib import Path

from validators import math_registry


REPO_ROOT = Path(__file__).resolve().parents[1]
REQUIRED_HEADINGS = (
    "## Forward",
    "## Linearization",
    "## JVP",
    "## Transpose",
    "## VJP (JAX convention)",
    "## VJP (PyTorch convention)",
)


class MathNoteStructureTests(unittest.TestCase):
    def test_published_note_paths_use_unified_heading_order(self) -> None:
        entries = math_registry.load_registry(REPO_ROOT)["entries"]
        note_paths = sorted({REPO_ROOT / row["note_path"] for row in entries})

        for note_path in note_paths:
            text = note_path.read_text(encoding="utf-8")
            last_index = -1
            for heading in REQUIRED_HEADINGS:
                current_index = text.find(heading)
                self.assertNotEqual(
                    current_index,
                    -1,
                    msg=f"{note_path.name} is missing required heading {heading!r}",
                )
                self.assertGreater(
                    current_index,
                    last_index,
                    msg=f"{note_path.name} does not preserve unified heading order",
                )
                last_index = current_index

    def test_published_note_paths_state_raw_output_space_and_complex_convention(self) -> None:
        entries = math_registry.load_registry(REPO_ROOT)["entries"]
        note_paths = sorted({REPO_ROOT / row["note_path"] for row in entries})

        for note_path in note_paths:
            text = note_path.read_text(encoding="utf-8")
            self.assertIn(
                "raw-output-space",
                text,
                msg=f"{note_path.name} must describe raw-output-space rules",
            )
            self.assertRegex(
                text,
                re.compile(r"real Frobenius inner\s+product"),
                msg=f"{note_path.name} must state the transpose convention",
            )


if __name__ == "__main__":
    unittest.main()
