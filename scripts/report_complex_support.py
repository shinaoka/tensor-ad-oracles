"""Generate a Markdown report for complex-support coverage."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPO_ROOT / "docs" / "generated" / "complex-support.md"

if str(REPO_ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(REPO_ROOT))

from validators.complex_support import (  # noqa: E402
    load_complex_support,
    published_complex_dtype_index,
)


@dataclass(frozen=True)
class ComplexSupportRow:
    op: str
    family: str
    note_status: str
    db_status: str
    complex_published_dtypes: tuple[str, ...]
    unsupported_reason: str | None

    @property
    def ready(self) -> bool:
        return self.db_status == "covered" and self.note_status != "pending"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Markdown output path.",
    )
    return parser.parse_args(argv)


def _format_dtypes(names: tuple[str, ...]) -> str:
    if not names:
        return "-"
    return ", ".join(names)


def _format_reason(reason: str | None) -> str:
    if not reason:
        return "-"
    return reason


def collect_rows(root: Path = REPO_ROOT) -> list[ComplexSupportRow]:
    ledger = load_complex_support(root)
    dtype_index = published_complex_dtype_index(root / "cases")
    rows: list[ComplexSupportRow] = []
    for entry in ledger["entries"]:
        key = (entry["op"], entry["family"])
        rows.append(
            ComplexSupportRow(
                op=entry["op"],
                family=entry["family"],
                note_status=entry["note"]["status"],
                db_status=entry["db"]["status"],
                complex_published_dtypes=dtype_index.get(key, ()),
                unsupported_reason=entry["unsupported_reason"],
            )
        )
    return sorted(rows, key=lambda row: (row.op, row.family))


def _append_table(lines: list[str], rows: list[ComplexSupportRow]) -> None:
    lines.extend(
        [
            "| op | family | note status | db status | complex published dtypes | unsupported reason | ready |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in rows:
        lines.append(
            f"| {row.op} | {row.family} | {row.note_status} | {row.db_status} | "
            f"{_format_dtypes(row.complex_published_dtypes)} | "
            f"{_format_reason(row.unsupported_reason)} | "
            f"{'yes' if row.ready else 'no'} |"
        )


def build_report_text(root: Path = REPO_ROOT) -> str:
    rows = collect_rows(root)
    ready_rows = [row for row in rows if row.ready]
    unsupported_rows = [row for row in rows if row.db_status == "unsupported"]
    pending_rows = [row for row in rows if not row.ready and row.db_status != "unsupported"]
    pending_note = sum(1 for row in rows if row.note_status == "pending")
    pending_db = sum(1 for row in rows if row.db_status == "pending")

    lines = [
        "# Complex Support Report",
        "",
        "Generated from the checked-in complex-support ledger and the published `cases/` tree.",
        "",
        "## Summary",
        "",
        f"- Total tracked families: {len(rows)}",
        f"- Ready for downstream: {len(ready_rows)}",
        f"- Unsupported: {len(unsupported_rows)}",
        f"- Pending note review: {pending_note}",
        f"- Pending DB coverage: {pending_db}",
        "",
        "## Full Ledger",
        "",
    ]
    _append_table(lines, rows)

    lines.extend(["", "## Ready For Downstream", ""])
    if ready_rows:
        _append_table(lines, ready_rows)
    else:
        lines.append("None.")

    lines.extend(["", "## Unsupported", ""])
    if unsupported_rows:
        _append_table(lines, unsupported_rows)
    else:
        lines.append("None.")

    lines.extend(["", "## Still Pending", ""])
    if pending_rows:
        _append_table(lines, pending_rows)
    else:
        lines.append("None.")

    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_report_text()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(f"complex_support_report={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
