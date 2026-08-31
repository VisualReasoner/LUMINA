# Adapters

An adapter defines the dataset-specific interface around the shared LUMINA
controller. Porting does not require edits to the controller, memory modules,
evidence schemas, or model clients.

## What an Adapter Defines

- task name, label column, and allowed labels
- target eligibility and minimum visible history
- modality names and image-path columns
- modality roles and observation/comparison questions
- reconciliation rules and the final decision rubric
- optional reference matching and cross-subject memory policy
- allowed pre-target context fields

Label construction and dataset-specific cohort selection remain offline data
preparation steps.

## Minimal Example

```yaml
adapter_name: example_longitudinal_task
context_columns: [age]

task:
  name: example_task
  label_column: example_label
  labels: [class_a, class_b]
  target_policy: latest_eligible_visit
  minimum_prefix_visits: 2
  require_any_target_modality: [Modality_A]
  inspection_rules:
    - Identify current evidence and valid same-modality anchors.
  reconciliation_rules:
    - Preserve agreement, conflict, and missing evidence explicitly.
  belief_rubric:
    - Use class_b only when the visible prefix provides sufficient support.
  belief_dimensions:
    current_state: Current visible evidence.
    longitudinal_change: Supported same-modality change.
    overall_evidence_quality: Sufficiency and uncertainty.

modalities:
  - name: Modality_A
    path_column: modality_a_paths
    role: primary task evidence
    observation_questions:
      - What is visible at the current visit?
    comparison_questions:
      - What changed relative to the latest prior Modality_A study?

references:
  enabled: false

memory:
  cross_subject_enabled: false
```

The strings in `inspection_rules`, `reconciliation_rules`, `belief_rubric`, and
`belief_dimensions` form the task-specific reasoning primer. Keep them short,
ordered, and limited to evidence available at inference time.

## Eligibility

Adapters may require:

- all modalities listed in `required_target_modalities` at the target visit;
- at least one modality in `require_any_target_modality`; and
- prior same-modality studies through `required_prior_modalities`.

The router evaluates these conditions after truncating the subject history at
each candidate target.

## Reference Comparators

For a reference-enabled adapter, declare a `reference_column` on each eligible
modality and configure two matching fields:

```yaml
references:
  enabled: true
  match_primary: age
  match_secondary: sex
```

Both fields must appear in `context_columns`. Candidate records contain image
paths and matching fields only; labels, diagnoses, outcomes, and future
information are rejected from the reference interface.

## Included Adapters

- `configs/adapters/adni_ad_continuum.yaml`
- `configs/adapters/adni_progression_forecasting.yaml`
- `configs/adapters/ppmi_parkinsonian_state.yaml`
- `configs/adapters/acrin6698_pcr.yaml`

Run `pytest tests/test_all_adapters.py -q` after adding an adapter. The test
exercises routing, the shared controller, structured states, and identifier
exclusion with saved responses rather than a live model.
