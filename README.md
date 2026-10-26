# trixtom-dataset

Trixtom dataset build pipeline, run on GitHub-hosted runners (4 vCPU /
16 GB RAM) instead of the 2-CPU / 7.9 GB VM, where two parallel builds
kept running out of RAM.

Each batch gets its own workflow (`.github/workflows/dataset-vNN.yml`).
Inputs (hash cache, previous-batch hashes) come from Google Drive
("anyone with the link", read-only); outputs (packed shards, manifest,
hash list for the next batch) go to a GitHub release for the batch.
The main agent downloads the release assets, verifies SHA256 against
the manifest, and uploads them to Drive.

## Batches

- **v16** — HPLT-id + Taiga-ru + mC4 (session 2 chain). Workflow:
  `dataset-v16.yml`. Cache pre-extended on the VM with v13+v14 text.
