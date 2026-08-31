# LUMINA

**LUMINA: LongitUdinal MultImodal reasoNing Agent for Medical Imaging**

EMNLP 2026 Main Conference

Zhaoqi An, Yunchang Xie, Xiaobing Yu, Yuyi Yang, Ahmad Chowdhury, and Peijie Qiu

[Project page](https://visualreasoner.github.io/LUMINA/) | [Citation](#citation)

LUMINA is a longitudinal multimodal reasoning agent that builds an explicit
evidence state from an irregular subject-history prefix before making a final
prediction. This repository contains the official reference implementation.

<p align="center">
  <img src="docs/images/LUMINA-main.jpg" width="95%" alt="LUMINA architecture">
</p>

## Method

LUMINA keeps the main operations of longitudinal reasoning explicit:

1. ground the currently available studies by modality;
2. compare each current study with the latest valid prior study from the same modality;
3. maintain visit-level and compact trajectory memory;
4. optionally retrieve label-free summaries from a subject-disjoint memory bank;
5. reconcile convergent, complementary, and conflicting evidence before prediction; and
6. run one bounded verification and repair pass when the evidence state is inconsistent.

The controller is shared across datasets. Dataset-specific modalities, labels,
eligibility rules, reference policy, and decision rubric are defined in a small
adapter YAML file.

## Installation

```bash
git clone https://github.com/VisualReasoner/LUMINA.git
cd LUMINA

conda env create -f environment.yml
conda activate lumina-public
pip install -e .
```

The environment includes the local Transformers and OpenAI-compatible model
backends. For a smaller installation, install only the backend you need:

```bash
pip install -e .          # controller, data utilities, and replay backend
pip install -e ".[local]"  # local Hugging Face MLLMs
pip install -e ".[api]"    # OpenAI-compatible APIs
```

Raw-image preparation additionally requires `dcm2niix`; T1 MRI normalization
requires ANTs and FreeSurfer SynthStrip. These system tools are not installed by
`pip` or the Conda environment.

## How to Run

### Data-free check

The replay example exercises the complete controller without loading a model or
using medical data:

```bash
python scripts/create_toy_example.py --output-dir /tmp/lumina_toy

python scripts/run_benchmark.py \
  --visit-index-csv /tmp/lumina_toy/visit_index.csv \
  --routed-targets-csv /tmp/lumina_toy/routed_targets.csv \
  --adapter-yaml configs/adapters/adni_ad_continuum.yaml \
  --backend replay \
  --replay-json /tmp/lumina_toy/replay_responses.json \
  --output-dir /tmp/lumina_toy/results
```

The synthetic fixture verifies execution and output contracts; it is not a
performance example.

### Longitudinal inference

Given a processed visit index and one routed target per subject:

```bash
python scripts/run_benchmark.py \
  --visit-index-csv /path/to/visit_index.csv \
  --routed-targets-csv /path/to/routed_targets.csv \
  --adapter-yaml configs/adapters/adni_ad_continuum.yaml \
  --backend transformers \
  --model /path/to/Qwen3-VL-8B-Thinking \
  --output-dir /path/to/results
```

The runner writes `predictions.csv`, `traces.jsonl`, `errors.jsonl`, and
`metrics.json`. Input preparation, API-backed inference, cached trajectory
states, and cross-subject memory are documented in [docs/USAGE.md](docs/USAGE.md).

## Adapters

The repository includes four examples:

- `adni_ad_continuum.yaml`
- `adni_progression_forecasting.yaml`
- `ppmi_parkinsonian_state.yaml`
- `acrin6698_pcr.yaml`

All four use the same controller and evidence schemas. See
[docs/ADAPTERS.md](docs/ADAPTERS.md) to configure another disease, modality
set, or prediction target.

## Repository Layout

```text
configs/adapters/      task and dataset adapters
configs/ablations/     component-level ablation overlays
scripts/               preprocessing, indexing, inference, and evaluation entry points
src/lumina/agent/      evidence controller and longitudinal runtime
src/lumina/memory/     trajectory and cross-subject memory
src/lumina/prompts/    staged prompt construction
src/lumina/schemas/    structured evidence states
tests/                 unit and data-free integration tests
```

The mapping between the paper and the implementation is summarized in
[docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md).

## Data

ADNI, PPMI, and ACRIN-6698 data are not redistributed. Users must obtain the
datasets under their respective access terms and prepare local image paths and
task labels. Model weights and paper benchmark outputs are also not included.

## Citation

```bibtex
@inproceedings{an2026lumina,
  title     = {{LUMINA}: {L}ongit{U}dinal {M}ult{I}modal reaso{N}ing {A}gent for Medical Imaging},
  author    = {An, Zhaoqi and Xie, Yunchang and Yu, Xiaobing and Yang, Yuyi and Chowdhury, Ahmad and Qiu, Peijie},
  booktitle = {Proceedings of the 2026 Conference on Empirical Methods in Natural Language Processing},
  year      = {2026}
}
```

## License

The code is released under the MIT License. LUMINA is research software and is
not intended for clinical diagnosis or treatment decisions.
