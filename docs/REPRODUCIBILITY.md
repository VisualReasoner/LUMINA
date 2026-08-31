# Reproducibility

This repository implements the LUMINA method described in the paper. It does
not reimplement the comparison baselines.

## Paper-to-Code Map

| Paper component | Implementation |
| --- | --- |
| inspection and current-visit grounding | `src/lumina/agent/controller.py`, `src/lumina/prompts/builders.py` |
| past same-modality comparison and visit memory | `EvidenceController._compare`, `LocalComparison` |
| trajectory memory and SMC-guided selection | `src/lumina/memory/trajectory.py` |
| optional cross-subject memory | `src/lumina/memory/cross_subject.py` |
| multimodal evidence reconciliation | `PromptBuilder.reconcile`, `ReconciledEvidence` |
| final belief, verification, and audit | `EvidenceController.run_visit`, `BeliefState`, `AuditRecord` |
| dataset/task configuration | `configs/adapters/` |

The controller order is inspect, observe, optional reference calibration,
same-modality comparison, trajectory review, optional cross-subject retrieval,
reconciliation, final belief, bounded repair, audit, and trajectory update.

## Fixed Configuration

`configs/benchmark/default.yaml` contains the controller, trajectory-memory, and
evaluation defaults. In particular, it fixes the active-event budget, SMC
utility weights, recency half-life, schema retry limit, repair limit, bootstrap
sample count, and bootstrap seed.

Component ablations are overlays in `configs/ablations/`. They are applied to
the same runner and controller:

```bash
python scripts/run_ablation.py \
  --visit-index-csv /path/to/visit_index.csv \
  --routed-targets-csv /path/to/routed_targets.csv \
  --adapter-yaml configs/adapters/adni_ad_continuum.yaml \
  --backend transformers \
  --model /path/to/model \
  --output-root /path/to/ablation-results
```

The `no_references` configuration is a reference-protocol control and is not
part of the component-ablation suite used by default.

## Input Boundaries

Model-facing prompts receive processed images, visit order and dates, available
modalities, adapter-approved pre-target context, and structured evidence states.
They do not receive subject identifiers, routed labels, future visits, future
outcomes, diagnosis/CDR fields used for labels, or cognitive and functional
scores.

Reference candidates are non-label-bearing visual comparators. Cross-subject
records are built from reserved subjects, exclude target labels, and are checked
for overlap with scored subjects before evaluation.

## Validation

The test suite does not require medical data or model weights:

```bash
pytest -q
```

It covers all included adapters, routed-target eligibility, reference matching,
identifier exclusion, staged controller order, cached trajectory execution,
cross-subject leakage checks, SMC selection, preprocessing, output schemas, and
the data-free replay path.

Paper datasets, derived restricted tables, model weights, and benchmark outputs
are not distributed in this repository.
