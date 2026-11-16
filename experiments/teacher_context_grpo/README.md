# Teacher-assisted actor and memory-manager training

This source-domain pilot distills verified Qwen3.8-flash-next answers into the actor, freezes it, and collects fresh memory-manager GRPO trajectories. It evaluates actor-only distillation, subsequent student GRPO, and subsequent GRPO with validated teacher proposals.

The first run uses the retained step-20600 checkpoint, eight successful teacher answers on student-generated contexts, the same 64 HotpotQA training questions as the preceding manager experiment, and 48 fixed development questions. These development questions have already been used in earlier diagnostics.

## Supervision and invariants

data.py verifies diagnostic result hashes, public-context identity, strict EM, supporting-sentence coverage, termination, and split membership. Official support annotations are used for offline auditing. Actor prompts contain the original public question and excerpts. Citations are programmatically audited against official annotations.

The accepted answers receive three fixed SFT epochs with batch size two and learning rate 1e-5. Only actor completion tokens enter the loss. The manager remains frozen. Saving after update one, restoring optimizer and parameters, and continuing to update twelve are tested.

Each manager arm loads the same manager and Adam state, plus the same distilled actor checkpoint. The actor is frozen throughout collection and optimization. Both arms traverse the same 64 questions using the parent alternating full-trajectory and local GRPO protocol.

A teacher is queried when four student candidates and the no-op control all fail strictly. Teacher proposals are checked against public evidence and the legal menu. A successful forced branch must pass evidence removal before becoming auxiliary supervision. Only sampled student actions enter GRPO. The auxiliary optimizer is independent, and a proposed update is rolled back if the target action probability fails to increase. Missing menu actions are recorded. Free-form graph repair and menu extension remain future work.

## Execution

Use the existing SCOPE Python environment. PYTHONPATH must include the pinned full-run source directory, its optimization directory, and optimization/training_speed. Allocate CUDA_VISIBLE_DEVICES to the selected GPU.

Run run.py with --base pointing to the full training run, --parent pointing to the completed corrected teacher validation, --output pointing to a new directory, and --config pointing to the private USTC configuration.

The parent supplies verified diagnostic results, the retained initial checkpoint, baseline development predictions, and the reusable paired-collection implementation. Their identities are checked. Receipt, data validation, SFT restoration checks, checkpoints, per-question logs, development predictions, and summary are retained in the output directory.

The combined recipe has eight additional actor supervision examples. Comparisons report that budget explicitly. Development evaluation uses the student alone. Fitting the selected training examples and improving independent development performance are measured separately.

Expanded source evidence training (2026-11-12)
--------------------------------------------
The new source-only pipeline and its controls are documented in EVIDENCE_TRAINING.md.
- evidence_train.py: fixed 128-question teacher collection, 96-question development set, answer-only vs evidence-and-answer actor SFT.
- evidence_teacher.py: cached batches of up to four independently checked USTC requests.
- analyze_evidence.py: fact-access, answer-format and token-cost diagnostics; these diagnostics do not replace the reward.
- expanded_grpo.py: exploratory 256-question-per-arm manager GRPO after actor distillation, without online teacher calls.
- audit_evidence.py and audit_expanded_grpo.py: coverage, provenance, resume and frozen-parameter checks.
The original run.py remains the previous eight-example/64-question pilot for reproducibility.