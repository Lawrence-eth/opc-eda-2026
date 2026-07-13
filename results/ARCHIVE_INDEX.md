# Off-repository evidence archive index

Large scratch outputs and abandoned worktree patches are preserved outside Git
instead of being mixed with the curated R0-R2 result set. This index is the
durable receipt for the pre-cleanup snapshot made on 2026-07-13.

## Repository-reset archive

Private release asset name: `opc-eda-2026-repo-reset-20260713.tar.gz`

- SHA-256: `780365e927e472e6091a0e823904f557f5cbcbf1f93c5a7cca4f049bee5bf9b7`
- Size: 5,504,238 bytes
- Local staging copy: `/home/ubuntu/artifacts/opc-eda-2026-repo-reset-20260713.tar.gz`
- Contents: 85 ephemeral v32/v33 files, eight verified binary worktree
  patches, full per-file checksums, and a verified thin remote-history bundle.

The local path is informational and is not assumed to exist in a fresh clone.
Upload the archive as a private release asset before removing any source
worktree, scratch output, or obsolete remote ref. After upload, download it
again and verify the digest above.

Critical evidence preserved in the archive:

| File | SHA-256 |
|---|---|
| `v32_fuzz_400.log` | `a0dc28eedecc058e219bde4f2ebb1317fbd8e03f3ddd78b5e1e8e3d21af02b9b` |
| `v33_low_inprocess_timing.json` | `4318251288740c3bf30b3f64a556ff8bacb60f72a1c4708142546593572d7057` |
| `v33_primary_p1_public.json` | `c78e324629e380e9965b7b3ee8d41f17313ebb889d0b65093728d439743d48b4` |
| `v33_primary_p1_clean_dev_summary.json` | `ccf7ca40b8920d38e73ba743a31a983fb1d59d98073248828ccd4e5c128f4bad` |
| `v33_primary_p1_raw_dev_summary.json` | `cf458a1db56891a29773cdc40b517cffab35bee0e53a3e19e3161c350f4c34ac` |

## Dirty worktree patches

Each patch was generated with `git diff --binary HEAD` and passed
`git apply --cached --check` against a temporary index initialized from its
recorded HEAD. Untracked `external` checkouts and symlinks were recorded in
status metadata but intentionally excluded.

| Worktree | HEAD | Branch | Patch SHA-256 |
|---|---|---|---|
| `opc-v32` | `e96d9045d5c30ac15046d65cab41e2d6c5c6c5a0` | `exp/v32-fast-topology` | `ac312e7bcb44b633fe9de658dedd6fa160ff38f2f0c4dab46f8ad8d344019a46` |
| `opc-v32-constraints` | `a3db2ac674a8efa7e6172d14974ad1179a0b06a7` | detached | `e36e65fa154e7e0c96c43cd564f17c6af368d88c3d8659a9fa389aa2a72ebc70` |
| `opc-v32-topology` | `58894c47d6ed1017107977ce1c53c99999e5f83d` | `exp/v32-topology-clean2` | `52356dbf6bce70a9c830dc0b36b09f552bc9630c1dd8d7173f5ae634652cc7b2` |
| `opc-v33-p1` | `ae483a2cddc8dcb9ef6f7da7945d39ca714ec4ab` | `research/v33-zero-cost-p1` | `52aae2000cc76b7810630a4e0276c54802155879d65fb1189c602a40ea436af1` |
| `v32_p1` | `99f439d8d67208d9bb5a15affc6af5004600ac1c` | detached | `9312fc3103edd0faf662d7600f6b6bdcde4dfa91186f102a61c38408575627d3` |
| `v32_replace` | `99f439d8d67208d9bb5a15affc6af5004600ac1c` | detached | `26efbdd02be7a5eda939d9cc33eb73f31f82272ca7bafba6156c98adbd339115` |
| `v32_reuse` | `99f439d8d67208d9bb5a15affc6af5004600ac1c` | detached | `66e2bf26de9920c927f6be0a024516c05dee9d12da0c0bcd0a35b5e9179565bc` |
| `v32_swap` | `99f439d8d67208d9bb5a15affc6af5004600ac1c` | detached | `a132f375eb74472c1293b5f58097d58c972b4ccc06e2dde9e237d42f53f15e96` |

## Obsolete remote-history receipt

`archive-remote-2026-05-30-thin.bundle` is 183,827 bytes with SHA-256
`ba2ab11ed7e9b9795bdbc938c3144f3556c2b138077149ebf652c7338ceb8565`.
`git bundle verify` passed. It contains the four commits through
`d03f5c5ca6b538edcfcf259d7c24d327118f08ca` and requires prerequisite
`fc28d6d4a7e173c572329e1c9326160ad0ce90ac` from IntelLabs/FloorSet.
