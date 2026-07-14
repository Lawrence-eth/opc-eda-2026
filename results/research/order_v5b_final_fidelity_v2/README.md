# v5b final-output fidelity evidence

This directory is the immutable evidence chain for the v5b paid-slot
replacement policy. It separates three questions that must not be conflated:

1. Can a legacy paid slot be removed without changing the completed incumbent
   layout on source-disjoint training panels?
2. Does that observation survive a rejection-only fold-3 confirmation?
3. Does the confirmed removal preserve the public incumbent without using
   public golden cost?

## Bound snapshots

- Audited solver: `9a58b727b9c61d654406e8d0ca10eae312ec3a86`.
- Public-audit harness snapshot:
  `c57e19a339ec1d76e0e9ec438a462645653f2060`.
- All six live component hashes are recorded in every raw panel. The public
  snapshot has the same six hashes; only non-live audit tooling changed.
- Frozen dataset: FloorSet commit
  `aadddcc2238695eb21e6542b8a6cd9e9fe6b80fa` and the schema-3 clean/raw
  manifests under `results/folds/`.

## Contents and verdict

- `audit_{clean,raw}_fold{0,1,2}.json`: six development panels, 30 observed
  removals per block count in total.
- `audit_{clean,raw}_fold3.json`: confirmation panels used only to reject a
  development mapping; they cannot retune it.
- `derived_slot_map.json`: the fail-closed reduction of the six-plus-two
  matrix.
- `public_slot_audit.json`: one public structural check for every retained
  size. It reconstructs fixed/preplaced hard targets from polygon geometry but
  neither reads stored golden metrics nor computes golden HPWL/area.

Across the eight training panels, 840 baselines and all 4,200 slot-removal
reruns were feasible. The retained 13-size map preserved packed final
positions and inference-visible metrics in every observation. Fold 3 rejected
sizes `105, 109, 113, 118, 119, 120`; development abstained at `101, 112`.
The public rejection audit preserved all 13 retained sizes and added no
abstentions.

The promoted composition is
`results/models/order_v5b_final_fidelity_v2.json`. It binds the exact
derivation and public-audit bytes plus the reducer, audit, and finalizer
harness hashes.

This is evidence of observed final-output redundancy, not a proof that the
same slot is redundant on every hidden layout and not a claim of global
floorplan optimality. Production remains fail-closed to the displaced paid
pass when learned inference is unavailable or malformed.
