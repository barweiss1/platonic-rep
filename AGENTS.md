# AGENTS.md

Guidance for coding agents working in this repository.

## Project Overview

This repository studies representation alignment for the Platonic Representation Hypothesis. The main workflow is:

1. Extract per-layer features from language and vision models.
2. Measure alignment between feature sets with metrics in `metrics.py`.
3. Run locality/globality sweeps over metric parameters.
4. Analyze and plot the resulting alignment tensors in notebooks under `post_processing/`.

The code is research-oriented and experiment-driven. Prefer small, readable changes that preserve existing experiment outputs and naming conventions.

## Important Files

- `extract_features.py`: extracts language and vision features into `results/features/...`.
- `measure_alignment.py`: computes a single alignment result for one metric/top-k setting.
- `measure_alignment_locality_sweep.py`: computes alignment across a sweep of metric parameters such as `topk`, `temperature`, or `rbf_sigma`.
- `metrics.py`: contains `AlignmentMetrics`, supported metric names, sweep ranges, and helper kernels/neighborhood functions.
- `tasks.py`: defines model sets (`val`, `test`, `mini`, and `custom`) and modality-specific model lists.
- `utils.py`: path construction for feature/alignment files plus language-model loss helpers.
- `models.py`: language model/tokenizer loading helpers.
- `platonic/`: small package interface for external use.
- `examples/`: minimal examples for using language, vision, and convnet feature extraction.
- `post_processing/similarity_globality.ipynb`: main notebook for sweep/globality analysis and plotting.
- `results/`: generated features and alignment arrays. Treat these as large experiment artifacts, not ordinary source files.

## Data And Output Conventions

Feature files are written by `utils.to_feature_filename`:

```text
results/features/<dataset>/<subset>/<model_name>_pool-<pool>[_prompt-...][_cid-...].pt
```

Alignment sweep files are written by `utils.to_alignment_filename`:

```text
results/alignment/<dataset>/<modelset>/<run_name>/<modality_x>_pool-..._<modality_y>_pool-..._layer-<mode>/<metric>_sweep<sweep_len>.npy
```

Sweep `.npy` files are dictionaries with at least:

- `scores`: alignment scores, usually shaped `[n_x, n_y, n_param]`.
- `indices`: best layer indices, usually shaped `[n_x, n_y, n_param, 2]`.
- `param_vec`: swept parameter values.
- `param_name`: swept parameter name.

Do not change file naming conventions casually. Many notebooks and scripts depend on them.

Sweep layer modes:

- `max`: choose the best layer pair independently at each parameter value.
- `final`: use the final layer pair for every parameter value.
- `max_auc`: choose the one layer pair with maximal arithmetic mean score across `param_vec`, then use that pair for the whole sweep. In this repo, AUC usually means this mean-over-sweep convention unless stated otherwise.

## Running Common Tasks

Install dependencies:

```bash
pip install -r requirements.txt
```

Extract features:

```bash
python extract_features.py --dataset minhuh/prh --subset wit_1024 --modelset val --modality language --pool avg
python extract_features.py --dataset minhuh/prh --subset wit_1024 --modelset val --modality vision --pool cls
```

Measure a single alignment:

```bash
python measure_alignment.py --dataset minhuh/prh --subset wit_1024 --modelset val \
  --modality_x language --pool_x avg --modality_y vision --pool_y cls
```

Run a parameter sweep:

```bash
python measure_alignment_locality_sweep.py --dataset minhuh/prh --subset wit_1024 --modelset val \
  --modality_x language --pool_x avg --modality_y vision --pool_y cls \
  --metric mutual_knn --sweep_len 400 --layer_mode max_auc --run_name lin_rbf
```

Most core scripts assume CUDA is available. If a command cannot be run locally because dependencies, GPU, or data are missing, say so explicitly and still do static validation where possible.

## Coding Guidelines

- Follow the current plain Python style. Avoid introducing new frameworks or large abstractions unless they clearly reduce experiment complexity.
- Keep changes scoped. Research notebooks and generated outputs can be noisy, so avoid unrelated notebook reformatting or output churn.
- Use `tasks.py` when adding/removing model sets or model names.
- Use `metrics.AlignmentMetrics.SUPPORTED_METRICS` and `SWEEP_PARAMS` when adding a metric or sweepable parameter.
- Preserve modality assumptions unless deliberately changing them:
  - language features use `avg` pooling;
  - vision features use `cls` pooling;
  - current vision feature extraction assumes ViT/timm models.
- Be careful with argument order in alignment computations. `measure_alignment.py` calls `compute_score(y_feats, x_feats, ...)` and then stores results in `[i, j]`.
- Avoid broad refactors in `metrics.py`; metric changes should include a focused sanity check or a small synthetic example when feasible.
- Do not delete or overwrite experiment artifacts in `results/` unless the user explicitly asks.

## Notebook Guidelines

- Prefer adding separate cells for new analysis rather than rewriting existing cells.
- Guard plots that only make sense for language-vs-vision with:

```python
if set([sim_params["modality_x"], sim_params["modality_y"]]) != {"language", "vision"}:
    ...
```

- Save generated figures under the existing `figure_path`.
- When editing notebooks programmatically, preserve valid JSON and check that all code cells parse.
- If outputs become huge or unrelated to the requested change, avoid re-executing the whole notebook.

## Writing Style For Future Requests

When responding to the user:

- Be concise but clear. Lead with what changed or what was found.
- Use concrete file paths and command names.
- Mention verification performed, and separately mention anything that could not be run.
- Keep explanations practical and research-workflow aware.
- Do not over-polish experimental code into production architecture.
- If changing plots or analysis, name the saved figure files and explain how to adjust the main parameter.
- If a request is ambiguous, make a reasonable assumption when the blast radius is small; otherwise ask one direct question.

When writing code/comments/docs in this repo:

- Use straightforward research-code prose.
- Prefer descriptive variable names like `param_vec`, `alignment_scores`, `model_names_x`, and `figure_path`.
- Keep comments short and useful, especially around tensor shapes, layer indexing, sweep parameters, or modality-specific behavior.
- Match existing terminology: alignment, metric, globality/locality sweep, language, vision, model family, layer mode.

## Maintenance Rule

Update this file when significant repository behavior changes, especially when:

- a new top-level workflow or script is added;
- output path conventions change;
- metrics or sweep parameter semantics change;
- model set conventions change;
- notebook analysis conventions change;
- dependency, CUDA, or data assumptions change.
