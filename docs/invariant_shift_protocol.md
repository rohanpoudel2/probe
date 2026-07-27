# Executable Invariant Shift Protocol

The maintained shift generators do not require subjective ratings. They test exact, machine-verifiable transformations of a frozen source prompt.

## Presentation shift

`verbatim_wrapper_v1` places the original request between fixed boundary markers and appends a fixed instruction to follow that request. The source payload remains byte-for-byte unchanged.

The validator requires:

- an exact parent scenario ID;
- parent and variant prompt hashes;
- the hash of the embedded source payload;
- an executable inverse that recovers the original payload;
- an unchanged answer-metadata hash;
- no second held-out transformed axis.

## Obfuscation shift

`reversible_rot13_v1` applies ROT13 to the original request and places it inside a fixed decoding instruction. Applying the registered inverse must recover the exact source payload.

The same parentage, hashing, answer-metadata, and single-axis checks apply.

## Generation

Create variants before model rollout generation:

```bash
uv run python -m cli.build_invariant_shift_scenarios \
  --base_scenarios data/scenarios/sycophancy.jsonl \
  --axis presentation \
  --registry experiments/protocol/falsification_registry.yaml \
  --output data/scenarios/sycophancy_with_presentation.jsonl

uv run python -m cli.build_invariant_shift_scenarios \
  --base_scenarios data/scenarios/sycophancy_with_presentation.jsonl \
  --axis obfuscation \
  --registry experiments/protocol/falsification_registry.yaml \
  --output data/scenarios/sycophancy_with_shifts.jsonl
```

The builder assigns the transformed scenario and every member of its parent group to `test`.

## Claim boundary

These transformations support robustness claims about verbatim presentation wrapping and reversible encoding. They do not establish semantic equivalence for free-form paraphrases and must not be described that way.
