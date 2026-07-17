This directory stores historical, blocked frontier-incompatible configs.

- `honesty_auxiliary_manifest.yaml`: MASK auxiliary benchmark definition that
  used authored completions and belief-conditioned labels.
- `final_release_manifest.yaml`: legacy release config with authored responses
  and target-derived operating points.

The active frontier-facing `experiments/controls/honesty_auxiliary_manifest.yaml`
contains the on-policy auxiliary path.
