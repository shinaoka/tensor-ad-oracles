# Complex Support Ledger Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a machine-readable complex-support ledger, validate it against the published DB and math-note registry, generate a human-facing complex-support report, and complete the remaining note/unsupported classifications needed for issue `#14`.

**Architecture:** Keep `docs/math/registry.json` responsible only for note linkage and add a separate `docs/math/complex-support.json` ledger at exact `(op, family)` granularity. Validate the ledger against the current published family surface from `generators.pytorch_v1.build_case_spec_index()`, the checked-in `cases/` tree, and note anchors on disk. Generate `docs/generated/complex-support.md` from that ledger and finish the note/unsupported audit using the local `../pytorch` tree as the upstream review reference.

**Tech Stack:** Python 3.12, `uv`, pinned `torch==2.10.0`, `unittest`, JSON, Markdown generation, existing validators/scripts/test patterns, local upstream references under `../pytorch`.

---

### Task 1: Add ledger-validator unit tests

**Files:**
- Create: `tests/test_complex_support.py`
- Modify: `tests/test_scripts.py`

**Step 1: Write the failing test**

Add tests that require a complex-support validator to reject:

- duplicate `(op, family)` entries
- missing reviewed note paths or anchors
- `not_required` entries with non-null note metadata
- `unsupported` DB entries without a reason
- `covered` entries whose case tree does not include the expected complex dtypes

Use a temporary repository fixture similar to `tests/test_math_registry.py`,
with a tiny injected spec surface such as:

```python
spec_index = {
    ("solve", "identity"): StubSpec(
        op="solve",
        family="identity",
        supported_dtype_names=("float64", "complex128"),
    )
}
```

Also add one script test asserting that `scripts/check_complex_support.py`
exits successfully on the checked-in repo once implemented.

**Step 2: Run test to verify it fails**

Run:

```bash
uv run python -m unittest tests.test_complex_support tests.test_scripts -v
```

Expected: FAIL because no complex-support validator or script exists yet.

**Step 3: Write minimal implementation**

Create:

- `validators/complex_support.py`
- `scripts/check_complex_support.py`

Implement helpers to:

- load the ledger
- compute published complex dtypes from `cases/*/*.jsonl`
- validate note/db invariants against an injected or default spec surface

Keep the API close to `validators/math_registry.py` so the tests can use
temporary repos without mocking the full application.

**Step 4: Run test to verify it passes**

Run:

```bash
uv run python -m unittest tests.test_complex_support tests.test_scripts -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add tests/test_complex_support.py tests/test_scripts.py validators/complex_support.py scripts/check_complex_support.py
git commit -m "feat: validate complex support ledger"
```

### Task 2: Add the checked-in ledger and repo-level classification tests

**Files:**
- Create: `docs/math/complex-support.json`
- Modify: `tests/test_complex_support.py`
- Modify: `README.md`

**Step 1: Write the failing test**

Add repo-level tests that require:

- the checked-in ledger to exist
- every current `(op, family)` from `build_case_spec_index()` to appear exactly once
- representative complex-covered families such as `svd/u_abs`, `solve/identity`,
  `eig/values_vectors_abs`, and `sum/identity` to be marked `reviewed + covered`
- representative DB-only note cases such as `lu_factor_ex/identity`,
  `solve_ex/identity`, and `inv_ex/identity` to be marked `note.status == "not_required"`
- representative float-only families such as `atan2/identity`,
  `copysign/identity`, and `special_ndtr/identity` to be marked
  `db.status == "unsupported"` with a reason

Document the new ledger in `README.md`.

**Step 2: Run test to verify it fails**

Run:

```bash
uv run python -m unittest tests.test_complex_support tests.test_repo_config -v
```

Expected: FAIL because the ledger file is not checked in yet and the README does
not mention it.

**Step 3: Write minimal implementation**

Add `docs/math/complex-support.json` with one flat entry per published
`(op, family)`:

- `reviewed + covered` for families whose math note and checked-in DB are ready
- `not_required + covered` for DB-only wrapper families that do not need a
  separate complex note review
- `reviewed + unsupported` or `not_required + unsupported` for float-only or
  intentionally deferred complex families

The file should contain no `pending` entries in the final checked-in state.

Update `README.md` to document:

- the new ledger path
- the validation script
- the generated complex-support report

**Step 4: Run test to verify it passes**

Run:

```bash
uv run python -m unittest tests.test_complex_support tests.test_repo_config -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add docs/math/complex-support.json tests/test_complex_support.py README.md
git commit -m "docs: add checked-in complex support ledger"
```

### Task 3: Add report-generation tests and script

**Files:**
- Create: `tests/test_complex_support_report.py`
- Create: `scripts/report_complex_support.py`
- Create: `docs/generated/complex-support.md`

**Step 1: Write the failing test**

Add tests that require the report generator to:

- emit a Markdown report with summary counts
- include a full table with `op`, `family`, `note status`, `db status`,
  `complex published dtypes`, `unsupported reason`, and `ready`
- include representative rows such as:

```text
| svd | u_abs | reviewed | covered | complex128, complex64 | - | yes |
```

- include representative unsupported rows such as `atan2/identity`
- reproduce the checked-in `docs/generated/complex-support.md`

**Step 2: Run test to verify it fails**

Run:

```bash
uv run python -m unittest tests.test_complex_support_report -v
```

Expected: FAIL because the report script and checked-in report do not exist yet.

**Step 3: Write minimal implementation**

Create `scripts/report_complex_support.py` that:

- loads `docs/math/complex-support.json`
- loads the current DB/spec surface
- computes published complex dtypes from the case tree
- emits `docs/generated/complex-support.md`

Check in the generated report.

**Step 4: Run test to verify it passes**

Run:

```bash
uv run python -m unittest tests.test_complex_support_report -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add tests/test_complex_support_report.py scripts/report_complex_support.py docs/generated/complex-support.md
git commit -m "docs: add complex support report"
```

### Task 4: Audit note coverage and lock representative reviewed classifications

**Files:**
- Modify: `docs/math/scalar_ops.md`
- Modify: `docs/math/svd.md`
- Modify: `docs/math/qr.md`
- Modify: `docs/math/lu.md`
- Modify: `docs/math/solve.md`
- Modify: `docs/math/cholesky.md`
- Modify: `docs/math/inv.md`
- Modify: `docs/math/det.md`
- Modify: `docs/math/eig.md`
- Modify: `docs/math/eigen.md`
- Modify: `docs/math/lstsq.md`
- Modify: `docs/math/matrix_exp.md`
- Modify: `docs/math/norm.md`
- Modify: `docs/math/pinv.md`
- Modify: `docs/math/complex-support.json`
- Modify: `tests/test_math_registry.py`
- Modify: `tests/test_complex_support.py`

**Step 1: Write the failing test**

Add targeted tests that pin representative complex-reviewed note coverage:

- `eig.md` still mentions normalization correction and gauge invariance
- `eigen.md` still documents the Hermitian complex rule
- `svd.md`, `solve.md`, `pinv.md`, and `norm.md` still retain the key complex
  derivation markers relied on by the audit
- representative ledger entries for those families are marked
  `note.status == "reviewed"`

For any note found incomplete during the audit, first add an assertion for the
missing complex detail before editing the note.

**Step 2: Run test to verify it fails**

Run:

```bash
uv run python -m unittest tests.test_math_registry tests.test_complex_support -v
```

Expected: FAIL on the first audited family whose note-review status or note
content is still incomplete for complex mode.

**Step 3: Write minimal implementation**

Audit the shared scalar note and the dedicated linalg notes against the local
upstream references under `../pytorch`, then:

- strengthen any note whose complex explanation is still missing
- update the ledger from provisional classifications to final `reviewed` or
  `not_required` values

Do not create new dedicated note files unless the audit actually needs them;
prefer strengthening the current shared-note layout first.

**Step 4: Run test to verify it passes**

Run:

```bash
uv run python -m unittest tests.test_math_registry tests.test_complex_support -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add docs/math/scalar_ops.md docs/math/svd.md docs/math/qr.md docs/math/lu.md docs/math/solve.md docs/math/cholesky.md docs/math/inv.md docs/math/det.md docs/math/eig.md docs/math/eigen.md docs/math/lstsq.md docs/math/matrix_exp.md docs/math/norm.md docs/math/pinv.md docs/math/complex-support.json tests/test_math_registry.py tests/test_complex_support.py
git commit -m "docs: audit complex note coverage"
```

### Task 5: Wire the new checks into repository verification

**Files:**
- Modify: `tests/test_scripts.py`
- Modify: `tests/test_repo_config.py`
- Modify: `README.md`
- Modify: `.github/workflows/oracle-integrity.yml`

**Step 1: Write the failing test**

Add tests that require repository verification to mention and run the new
ledger/report checks:

- `README.md` documents `uv run python scripts/check_complex_support.py`
- `README.md` documents `uv run python scripts/report_complex_support.py`
- `.github/workflows/oracle-integrity.yml` includes the complex-support check

**Step 2: Run test to verify it fails**

Run:

```bash
uv run python -m unittest tests.test_repo_config tests.test_scripts -v
```

Expected: FAIL because the README and workflow do not yet mention the new check.

**Step 3: Write minimal implementation**

Update:

- `README.md`
- `.github/workflows/oracle-integrity.yml`

so the repository contract includes the new validator and report artifact.

**Step 4: Run test to verify it passes**

Run:

```bash
uv run python -m unittest tests.test_repo_config tests.test_scripts -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add tests/test_scripts.py tests/test_repo_config.py README.md .github/workflows/oracle-integrity.yml
git commit -m "ci: enforce complex support checks"
```

### Task 6: Full verification and checked-artifact regeneration

**Files:**
- Modify: `docs/generated/complex-support.md`
- Modify: `docs/math/complex-support.json`
- Modify: any files touched in Tasks 1-5

**Step 1: Run the focused tests**

Run:

```bash
uv run python -m unittest tests.test_complex_support tests.test_complex_support_report tests.test_math_registry tests.test_scripts tests.test_repo_config -v
```

Expected: PASS

**Step 2: Run the ledger and docs checks**

Run:

```bash
uv run python scripts/check_complex_support.py
uv run python scripts/check_math_registry.py
uv run python scripts/report_complex_support.py
```

Expected:

- `complex_support_ok=1`
- `math_registry_ok=1`
- `complex_support_report=.../docs/generated/complex-support.md`

**Step 3: Run repository contract verification**

Run:

```bash
uv run python scripts/validate_schema.py
uv run python scripts/verify_cases.py
uv run python scripts/check_replay.py
uv run python scripts/check_regeneration.py
```

Expected: PASS

**Step 4: Commit**

```bash
git add docs/generated/complex-support.md docs/math/complex-support.json README.md .github/workflows/oracle-integrity.yml validators/complex_support.py scripts/check_complex_support.py scripts/report_complex_support.py tests/test_complex_support.py tests/test_complex_support_report.py tests/test_math_registry.py tests/test_repo_config.py tests/test_scripts.py
git commit -m "feat: track complex support across the published DB surface"
```
