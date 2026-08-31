# Usage

LUMINA operates on processed image views and two CSV tables: a longitudinal
visit index and a routed target table. Dataset access and label construction
remain the user's responsibility.

## Input Tables

### Visit index

Each row represents one visible visit and requires:

- `subject_id`
- `visit_id`
- `visit_date`
- one or more modality path columns declared by the active adapter

For example, the ADNI adapters use `mri_paths`, `amyloid_pet_paths`, and
`tau_pet_paths`. A path field may contain one path, pipe-separated paths, or a
JSON list of paths. The standard three-view representation uses axial, coronal,
and sagittal PNG files.

### Routed targets

Each row represents one scored subject target and requires:

- `subject_id`
- `target_visit_id`
- `routed_label`

Optional columns provide adapter-declared context, reference candidates, or a
cached `prior_trajectory_state`. Labels are used only by evaluation and are not
included in model-facing prompts.

## Image Preparation

Create native-space center views from DICOM or NIfTI:

```bash
python scripts/preprocess_study.py \
  --dicom-dir /path/to/dicom \
  --mode native \
  --stem subject_visit_modality \
  --output-dir /path/to/processed-study
```

For T1-weighted MRI with MNI normalization:

```bash
python scripts/preprocess_study.py \
  --nifti /path/to/t1.nii.gz \
  --mode t1_mri_mni \
  --mni-template /path/to/MNI152_T1_1mm.nii.gz \
  --stem subject_visit_mri \
  --output-dir /path/to/processed-study
```

The MRI mode requires ANTs and FreeSurfer SynthStrip. DICOM conversion requires
`dcm2niix`. Four-dimensional inputs require an explicit `--volume-index`.

## Visit and Target Construction

Build the visit index from a study catalog containing `subject_id`,
`study_date`, `modality`, and `image_path`:

```bash
python scripts/build_visit_index.py \
  --study-catalog-csv /path/to/study_catalog.csv \
  --output-csv /path/to/visit_index.csv
```

ADNI label construction accepts diagnosis, CDR, and a Centiloid-derived binary
amyloid-status table. Route one eligible target per subject with the selected
adapter:

```bash
python scripts/build_labels.py \
  --visit-index-csv /path/to/visit_index.csv \
  --diagnosis-csv /path/to/diagnosis.csv \
  --cdr-csv /path/to/cdr.csv \
  --amyloid-csv /path/to/amyloid_status.csv \
  --output-csv /path/to/visit_labels.csv

python scripts/build_routed_targets.py \
  --visit-index-csv /path/to/visit_index.csv \
  --visit-labels-csv /path/to/visit_labels.csv \
  --adapter-yaml configs/adapters/adni_ad_continuum.yaml \
  --output-csv /path/to/routed_targets.csv
```

For reference-enabled adapters, the candidate table contains `modality`,
`candidate_id`, `image_path`, and the adapter's matching fields. Reference
records must not contain labels or outcomes.

```bash
python scripts/build_reference_index.py \
  --routed-targets-csv /path/to/routed_targets.csv \
  --reference-candidates-csv /path/to/reference_candidates.csv \
  --demographics-csv /path/to/subject_context.csv \
  --adapter-yaml configs/adapters/adni_ad_continuum.yaml \
  --output-csv /path/to/routed_targets_with_references.csv
```

## Model Backends

### Local Hugging Face model

```bash
python scripts/run_benchmark.py \
  --visit-index-csv /path/to/visit_index.csv \
  --routed-targets-csv /path/to/routed_targets.csv \
  --adapter-yaml configs/adapters/adni_ad_continuum.yaml \
  --backend transformers \
  --model /path/to/model \
  --device-map auto \
  --output-dir /path/to/results
```

`--model` may be a Hugging Face repository name, a downloaded model directory,
or a Hugging Face cache directory containing `refs/main` and `snapshots/`.

### OpenAI-compatible API

```bash
export OPENAI_API_KEY=...

python scripts/run_benchmark.py \
  --visit-index-csv /path/to/visit_index.csv \
  --routed-targets-csv /path/to/routed_targets.csv \
  --adapter-yaml configs/adapters/adni_ad_continuum.yaml \
  --backend openai \
  --model gpt-4o-mini \
  --output-dir /path/to/results
```

Use `--base-url` for another OpenAI-compatible endpoint.

## Trajectory Execution

If a routed target contains a serialized `prior_trajectory_state`, the runner
updates only the target visit. Otherwise, it processes the visible prefix in
chronological order to construct trajectory memory before the target decision.
`metrics.json` reports target, history, and total model-call counts separately.

## Cross-Subject Memory

Cross-subject memory is optional and must be built from reserved subjects:

```bash
python scripts/build_cross_subject_bank.py \
  --traces-jsonl /path/to/development_traces.jsonl \
  --development-subjects /path/to/development_subjects.txt \
  --adapter-yaml configs/adapters/adni_ad_continuum.yaml \
  --output-jsonl /path/to/cross_subject_bank.jsonl
```

Pass the frozen bank with `--cross-subject-bank`. The benchmark runner rejects
overlap between memory-bank subjects and scored subjects.

## Outputs

- `predictions.csv`: normalized prediction, target label, call counts, and runtime
- `traces.jsonl`: structured evidence states and action traces
- `errors.jsonl`: per-subject failures without stopping the full run
- `metrics.json`: aggregate metrics and subject-level bootstrap intervals

Use `--fail-fast` during setup to stop on the first invalid input or model output.
