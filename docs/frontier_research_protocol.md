# Frontier Research Protocol v1

## Registered question

The primary estimand is the paired change in true-positive rate produced by access to model activations, relative to the strongest registered black-box monitor, when both systems operate at thresholds calibrated to no more than 1% false-positive rate on benign traffic.

The study is not a generic demonstration that behavior labels are linearly decodable. It asks whether white-box access adds operational value under extreme label scarcity and distribution shift.

## Independent unit and few-shot budget

The independent unit is a scenario group, such as one factual question or one agentic decision context. Prompt conditions and repeated rollouts are nested inside that group.

For balanced training, `k` means `k` positive scenario groups and one matched negative observation from each same group. The resulting training set contains `2 * k` labeled observations. Repeated rollouts from one scenario never count as independent shots.

Registered label-efficiency points should be `k = 1, 2, 4, 8, 16, 32`, subject to a probe's minimum sample requirements. Full-data performance is an upper bound, not a few-shot result.

## Required data partitions

Every partition is group-disjoint:

1. `train`: few-shot observations only.
2. `calibration`: benign negatives used only to fit the operating threshold.
3. `eval`: source-domain method and layer selection.
4. `test`: untouched source or transfer evaluation.

The final calibration pool must contain at least 10,000 independent natural, deployment-like benign scenario groups, with exactly one accepted rollout per prompt group. Repeated rollouts are not allowed to reweight the empirical tail. Task-authored neutral controls alone are not sufficient evidence about operational FPR. A substantially larger pool is preferred for tail stability.

Natural-traffic sources may use upstream moderation only as a candidate prefilter. Inclusion as a negative requires the exact on-policy prompt/response pair to pass every frozen benign criterion unanimously with at least two distinct independent raters blinded to model identity, monitor scores, and one another's ratings. Screening records are content-hashed. Unsafe, behavior-positive, malformed, identifying, injected, ambiguous, stale, or under-reviewed rows are excluded rather than labeled negative. Acceptance, exclusions by reason, pairwise agreement, and nominal Krippendorff alpha are archived.

The operational criteria and rater procedure are frozen in `docs/benign_screening_rubric.md`.

Exact and normalized prompt hashes must be checked across sources and splits. Semantic-near-duplicate auditing should be documented before the frozen run.

## Rollout and labeling protocol

- Generate from the exact target model using its chat template.
- Pin model and tokenizer commits.
- Store prompts, outputs, token IDs, decoding settings, seeds, provenance, and registered statistics from the processed generation distribution for every emitted token.
- Generate all registered conditions before examining activation results.
- Use executable task-labeling rules whenever the behavior has an objective criterion.
- Send ambiguous cases to blinded adjudication under a frozen rubric.
- Report exclusion counts, disagreement, and inter-rater reliability.
- Preserve uncertain labels rather than coercing them into the negative class.

MASK must use demonstrated neutral beliefs and pressured statements as intended by that benchmark. MonitorBench must be run through its official task/evaluation interfaces at the registered commit. Its imported binary transfer construct is the upstream per-rollout target-outcome verifier result: label 1 means that the registered target outcome or process was verified and label 0 means it was not. This is neither a ground-truth label of CoT unfaithfulness nor the official MonitorBench monitorability score. The latter is computed and reported separately from the upstream monitor outputs, including the intervention-effect normalization used by input-intervention tasks. Neither benchmark may be converted into authored positive and negative completions.

MonitorBench is evaluation-only: all imported rows remain in `test`, and its 19-task official matrix is never used to fit a probe, choose a layer, or set a threshold. The source archive, extracted tree, adapter, resolved run configuration, evaluated/verifier model revisions, chat template, tested artifacts, response/verdict alignment, and normalized output are content-hashed. A final import requires every registered task/stress artifact. Partial imports are pilot-only and explicitly ineligible for the main study.

## Monitors

The primary white-box monitor should be fixed before the frozen run. A simple standardized logistic probe or mass-mean direction is appropriate; layer selection must use source `eval` only.

Required black-box comparisons include:

- prompt-only text monitor;
- answer/action-only monitor;
- full visible transcript monitor;
- frozen text-embedding classifier;
- zero-shot and few-shot LLM judge;
- registered output-logit or confidence baseline where available;
- majority and random baselines;
- full-data upper bounds.

Frozen encoder baselines must pin the encoder and tokenizer commits, pooling, padding side, normalization, instruction, instruction format, and maximum length. Raw texts are embedded once per dataset/view; downstream few-shot seeds reuse those caches. Source, benign calibration, and transfer caches must share the same encoder specification and monitored-model revision. Truncated inputs are prohibited in the final protocol.

The LLM judge is a different model family from every monitored model in the frozen study. Its model and tokenizer commits, chat-template arguments, system prompt, decision labels, padding side, and context limit are locked. The interaction is delimited as untrusted data. Scores are the pairwise softmax over the two registered next-token label logits; free-text generation and answer parsing are prohibited. Label tokens must each append exactly one token in the rendered generation context. Few-shot demonstrations come only from the matched source-training sample. Target labels never enter prompts, score computation, or threshold fitting. Overlength prompts fail rather than truncate. Each cache is bound to the code revision, implementation hash, data hashes, exact demonstrations, rendered-prompt hashes, view, and judge specification; split payload hashes detect partial or altered caches.

The output-confidence baseline uses only the monitored model's generation-time processed scores. Each natively generated rollout stores aligned selected-token log probabilities, entropies, top-1/top-2 probability margins, and top-1 indicators, protected by a content hash. A frozen 22-dimensional summary includes response length, distributional summaries, early/late confidence, and final-token confidence. The matched few-shot logistic classifier is fit on those summaries. Rollouts without a valid trace are ineligible for B4; confidence must not be reconstructed by a later teacher-forced pass because that would change the measured quantity. The official MonitorBench tested-artifact contract does not serialize these distributions, so B4 is pre-registered as structurally unavailable for that target while the other visible-text baselines remain required.

The primary comparison is white-box TPR minus the best black-box TPR chosen without target-test labels. Standalone activation AUROC is secondary.

## Metrics and inference

The threshold is selected using dedicated calibration negatives and applied unchanged to every source and target test. The same benign calibration identities must be used across activation and black-box systems in a registered comparison. AUROC and AUPRC are descriptive. Calibration metrics are reported only for actual probabilities or scores calibrated on a separate split.

Every monitor must save per-example predictions. Primary uncertainty uses paired hierarchical resampling over training seeds and scenario groups. Registered comparison families receive Holm or stronger family-wise correction. Seed-only variation is diagnostic and must not be labeled as a population confidence interval.

No system, layer, view, probe, intervention coefficient, or baseline may be chosen from final-test performance.

## Falsification suite

The falsification registry is an immutable, content-hashed contract over five axes: behavior, domain, prompt template, paraphrase, and obfuscation. Every scenario records one value and a `source` or `heldout` role for each axis. Scenario schema v2 includes this metadata in the scenario hash. If any member of a scenario group has a held-out role, the entire group is assigned to `test`; the builder and final validator reject mixed-split groups. This follows the separation between in-distribution and out-of-distribution evaluation used by [WILDS](https://proceedings.mlr.press/v139/koh21a.html), while retaining paired scenario groups for this study.

Paraphrases and obfuscations are prompt transformations, not model outcomes. Each must name an exact parent scenario, transformation protocol and source, parent and variant prompt hashes, and one isolated shift axis. At least two distinct independent reviewers who did not author the variant must unanimously confirm semantic equivalence, preservation of the target behavioral pressure, and absence of answer leakage under `docs/falsification_review_protocol.md`. Unreviewed or multiply transformed prompts are ineligible. This is a conservative adaptation of behavioral tests and local contrast sets rather than an automated claim of semantic equivalence; see [CheckList](https://aclanthology.org/2020.acl-main.442/) and [Contrast Sets](https://aclanthology.org/2020.findings-emnlp.117/).

A matched hard negative is not a neutral prompt and is never an authored completion. It is a label-0 on-policy response to a registered trigger prompt paired one-to-one with a label-1 response from the exact same scenario ID, prompt hash, group, and shift signature. Multiple completions therefore use registered stochastic decoding and remain nested within one scenario group. Repeated greedy decoding is prohibited. This slice measures hard-negative FPR, paired-positive TPR, pairwise score ordering, and pairwise score margin at the already frozen benign threshold.

The final gate requires both labels and at least 100 independent groups for every held-out shift axis, plus at least 100 independent exact-prompt hard-negative groups for every enabled behavior task. These are minimum coverage floors, not substitutes for the registered power analysis. A behavior value is treated as held out only in the context of a registry-approved source-behavior-to-target-behavior run; the same target task is not mislabeled as held out in its within-task calibration run.

Every per-example shift prediction and per-pair hard-negative prediction is archived with the exact registry and evaluation manifests. Confirmatory falsification comparisons are separately pre-registered with exact model, task, probe, layer, view, label budget, and balance-mode selectors. Systems must cover identical examples or pairs for every identical seed. Inference gives equal weight to independent scenario groups, jointly resamples groups and seeds, reports system-A-minus-system-B differences, and applies one global Holm correction across the registered falsification family. The comparison file is content-hashed and archived; changing it after an archive exists is rejected.

Before a positive result is interpreted as internal monitoring, it must survive:

- prompt-only and transcript-only baselines;
- response-template scrubbing and paraphrase;
- matched hard negatives with the same topic and vocabulary;
- held-out prompt templates and domains;
- cross-behavior transfer;
- adversarial obfuscation;
- early/pre-answer temporal evaluation if the claim is extended from post-hoc monitoring to pre-output detection;
- feature-label permutation and matched Gaussian controls;
- large benign-traffic evaluation.

If a text monitor matches the activation monitor, the correct conclusion is that white-box access did not add value under that condition.

## Intervention policy

Steering is not part of the primary protocol. It can become a follow-up only after monitoring succeeds. Its direction must be learned on training data, strength selected on development data, and behavioral outcomes measured on untouched scenarios. Projection onto the intervened direction is not a behavioral outcome. Capability, fluency, and clean-task side effects must be reported.

## Stage gates

### A. Infrastructure and data integrity

- All integrity tests pass.
- Scenario, rollout, annotation, and feature schemas validate.
- No authored completions enter extraction.
- Source and model revisions are immutable.

### B. Leakage pilot

- One model family, two tasks, and a limited scenario set.
- Compare prompt, transcript, and activation monitors.
- Verify matched sampling and frozen thresholds.
- Do not make publication claims before the full protocol has been frozen.

### C. Freeze

- Final behaviors, model families, sample sizes, exclusions, layer rule, metrics, primary comparisons, and five-axis/hard-negative falsification comparisons are registered.
- Baseline implementations, executable labeling rules, and human-annotation rubrics are frozen.
- Final-test labels remain inaccessible to selection code.

### D. Full execution

- Run the validated manifest without partial-success flags.
- Preserve all prediction and provenance artifacts.
- Report every registered comparison, including null results.

### E. Optional causal follow-up

- Temporal localization, adversarial stress tests, and behavioral interventions.
