# Third-party notices

This repository does not currently grant a project-wide license. The notices
below apply only to the identified third-party material and do not license the
repository's original solver code.

## IntelLabs/FloorSet

Source: <https://github.com/IntelLabs/FloorSet>

License: Apache License 2.0; see [`LICENSES/Apache-2.0.txt`](LICENSES/Apache-2.0.txt).

The following files contain or reproduce FloorSet material:

- `contest_solution/iccad2026_evaluate.py` — adapted evaluator. Local changes
  replace two loader imports and expand constraint-count diagnostics.
- `contest_solution/lite_dataset.py` — adapted dataset loader. The local copy
  makes the large-download decision non-interactive.
- `contest_solution/lite_dataset_test.py` — unmodified dataset loader snapshot.
- `packaging/eval_stub.py` — derived minimal interface and metric helpers used
  by the self-contained submission executable.
- `packaging/op_wrapper.py` — organizer-provided wrapper retained byte-for-byte.

## ICCAD 2026 contest documents

The text files under `docs/extracted/` are searchable extracts of official
Problem C materials supplied for contest participation. Their source URLs and
downloaded-file hashes are recorded in `docs/official_sources.json`. No claim
of authorship or additional license is made for those materials.

## FloorSet dataset

Source: <https://huggingface.co/datasets/IntelLabs/FloorSet>

License: [Creative Commons Attribution 4.0 International
(CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/).

The embedded learned-order model was trained from FloorSet examples. The
submission does not redistribute those examples; it contains only the derived
model parameters and inference code. Attribution is to Intel Labs and the
FloorSet authors identified by the source repository and dataset card. The
derived model and its compact deployment format are modifications created for
this contest entry and are not an official Intel release.
