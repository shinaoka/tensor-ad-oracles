# Complex Support Ledger Design

## Goal

Make this repository the machine-readable source of truth for complex AD
capability on the full published `(op, family)` surface before downstream
`tenferro-rs` implementation work proceeds.

The repository should be able to answer, from checked-in artifacts and CI
alone:

- which families have a complex-reviewed math note
- which families have full checked-in complex DB coverage
- which families are explicitly unsupported for complex and why

## Problem

The repository already has two partial sources of truth:

- `docs/math/registry.json` answers where published DB families map into math
  notes
- `cases/*/*.jsonl` answers which concrete records are currently materialized

That is not enough for issue `#14`.

The missing piece is an explicit ledger for complex readiness. Without it:

- the issue checklist is the only place that tries to summarize readiness
- CI cannot decide whether a family is reviewed, covered, unsupported, or
  simply forgotten
- downstream replay consumers cannot distinguish "not supported" from "not yet
  audited"
- the current publish-coverage report can show dtype presence, but not whether
  a family is mathematically reviewed for complex mode

## Design Principles

1. Keep the source of truth inside the repository.
2. Preserve single responsibility:
   - `registry.json` keeps note-location linkage only
   - the new ledger tracks complex capability state
3. Track status at exact `(op, family)` granularity.
4. Derive readiness from explicit note and DB states rather than storing a
   redundant `ready` bit.
5. Generate human-facing reports from the ledger; do not maintain large status
   tables by hand.
6. Reuse the current repo surface as the canonical family universe instead of
   manually duplicating the issue checklist in code.

## Upstream Review Inputs

The note audit for this work should use the local PyTorch checkout at
`../pytorch` as the upstream reference corpus alongside the pinned runtime
contract in this repository.

Relevant local upstream files include:

- `../pytorch/torch/csrc/autograd/FunctionsManual.cpp`
- `../pytorch/tools/autograd/derivatives.yaml`
- `../pytorch/torch/testing/_internal/opinfo/definitions/linalg.py`
- `../pytorch/torch/testing/_internal/common_methods_invocations.py`
- `../pytorch/test/test_linalg.py`

These paths provide the complex manual formulas, wrapper conventions, OpInfo
family inventory, and gauge-related failure behavior that the local math notes
and ledger statuses need to reflect.

## Scope

### Included

- Every currently published `(op, family)` in
  `generators.pytorch_v1.build_case_spec_index()`
- All existing math-note linkage in `docs/math/registry.json`
- A new complex-support ledger under `docs/math/`
- A validator script and tests for ledger integrity
- A generated human-facing complex-support report under `docs/generated/`
- Math-note updates where the complex audit shows that existing prose is still
  incomplete for a family we intend to mark reviewed
- Explicit unsupported classification for float-only or intentionally deferred
  complex families

### Excluded

- Any schema change to the JSONL case format
- Any attempt to merge the complex ledger into `docs/math/registry.json`
- GitHub issue text as a machine-readable source of truth
- New downstream replay behavior inside `tenferro-rs`

## Canonical Ledger

Add a new checked-in artifact:

- `docs/math/complex-support.json`

The canonical shape is:

```json
{
  "version": 1,
  "entries": [
    {
      "op": "svd",
      "family": "u_abs",
      "note": {
        "path": "docs/math/svd.md",
        "anchor": "family-u-abs",
        "status": "reviewed"
      },
      "db": {
        "status": "covered"
      },
      "unsupported_reason": null
    }
  ]
}
```

Each entry represents exactly one `(op, family)` pair.

### Note Axis

`note.status` is one of:

- `reviewed`
- `pending`
- `not_required`

Rules:

- `reviewed` requires non-null `note.path` and `note.anchor`
- `pending` also carries the intended `note.path` and `note.anchor` when a note
  target already exists
- `not_required` requires `note.path = null` and `note.anchor = null`

`not_required` is for families whose complex capability depends on an existing
shared or adjacent rule but does not require a distinct complex note-review
decision at this family boundary, for example wrapper-style DB-only families.

### DB Axis

`db.status` is one of:

- `covered`
- `pending`
- `unsupported`

Rules:

- `covered` means all publishable complex dtypes for this family are present in
  the checked-in DB
- `pending` means the family is intended to be supported but the checked-in DB
  is not yet complete
- `unsupported` means the family is intentionally out of current complex scope

### Unsupported Reason

`unsupported_reason` is required iff `db.status == "unsupported"`.

It is not a free-floating override; it is the required justification for an
unsupported DB decision.

## Derived Semantics

The ledger does not store a separate readiness field.

Downstream-ready status is derived as:

```text
note.status == reviewed && db.status == covered
```

Unsupported status is derived as:

```text
db.status == unsupported && unsupported_reason is non-empty
```

Everything else is incomplete.

## Relationship To Existing Artifacts

### `docs/math/registry.json`

The registry remains the source of truth for mapping published DB families to
note locations and anchors.

It continues to answer:

- where the note lives
- which stable anchor corresponds to a DB family

It does not answer:

- whether the family was complex-reviewed
- whether the DB has complex coverage
- whether the family is intentionally unsupported

### `cases/*/*.jsonl`

The checked-in JSONL case tree remains the source of truth for what numeric
coverage actually exists.

The ledger does not duplicate published dtype payloads. Instead, the validator
derives complex coverage from the case tree and checks that against the ledger's
declared `db.status`.

## Validation And CI

Add a dedicated validator module and script:

- `validators/complex_support.py`
- `scripts/check_complex_support.py`

Validation should enforce:

- the ledger exists and has versioned structure
- every repo-tracked `(op, family)` appears exactly once
- there are no duplicate entries
- reviewed note targets exist on disk and expose the declared anchor
- reviewed note targets match `docs/math/registry.json` when a registry entry is
  present for that family
- `covered` DB entries have all publishable complex dtypes materialized in the
  checked-in case tree
- `unsupported` DB entries have a non-empty reason
- checked-in repository state contains no `pending` entries once this issue is
  completed

The canonical family universe for these checks is
`generators.pytorch_v1.build_case_spec_index()`. This keeps the validator tied
to the same surface the repository already publishes and verifies elsewhere.

## Generated Report

Add a generated report:

- `docs/generated/complex-support.md`

Generated from:

- `docs/math/complex-support.json`
- `docs/math/registry.json`
- `generators.pytorch_v1.build_case_spec_index()`
- the checked-in `cases/` tree

The report should include:

- summary counts for ready, unsupported, and pending families
- a full flat table of family status
- focused sections for ready, unsupported, and pending subsets
- published complex dtype coverage per family

The report is the human-facing artifact. The JSON ledger is the machine-readable
artifact.

## Note Audit Strategy

The math-note audit should be structured around the existing shared-note layout:

- dedicated linalg notes such as `svd.md`, `qr.md`, `lu.md`, `solve.md`,
  `cholesky.md`, `inv.md`, `det.md`, `eig.md`, `eigen.md`, `lstsq.md`,
  `matrix_exp.md`, `norm.md`, and `pinv.md`
- shared scalar/wrapper note `scalar_ops.md`

Families should only be marked `reviewed` when the checked-in note is adequate
for complex mode relative to the local upstream references in `../pytorch`.
Otherwise:

- update the note and then mark `reviewed`, or
- keep `pending`, or
- mark DB support `unsupported` with a reason when complex support is not
  intended yet

## Expected Initial Audit Outcome

Current repo state already appears to have:

- full registry coverage for all published `(op, family)` pairs
- full checked-in complex DB coverage for the complex-publishable family set
- a large set of scalar and linalg notes that are already explicitly
  complex-aware

This suggests that the main missing artifact is the explicit complex-support
ledger and its generated report, plus any residual note edits discovered during
the manual review.

Float-only published families should generally end up as explicit complex
`unsupported` entries rather than appearing silent or pending forever.

## Risks

### Risk: conflating generic note linkage with complex review state

Mitigation:

- keep the ledger separate from `registry.json`
- validate only reviewed entries against the registry

### Risk: calling a family "covered" when only partial complex dtype support is present

Mitigation:

- derive coverage from the case tree and current generator spec
- require all publishable complex dtypes for `covered`

### Risk: audit drift between notes, ledger, and report

Mitigation:

- generate the report from the ledger
- validate note targets and DB coverage in CI

## Deliverables

1. `docs/math/complex-support.json`
2. `validators/complex_support.py`
3. `scripts/check_complex_support.py`
4. `scripts/report_complex_support.py`
5. `docs/generated/complex-support.md`
6. Tests covering validator and report behavior
7. Any required math-note updates discovered during the complex audit
