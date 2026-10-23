<h1 align="center">What Makes a Good Unmasking Policy? A Controlled Study of Learned Unmasking in Diffusion Language Models</h1>

<div align="center">

**[Jinya Sakurai](https://jinya1013.github.io/)**<sup>1,2</sup>, **[Xun Xu](https://alex-xun-xu.github.io/)**<sup>2,3</sup>, **[Jaehong Yoon](https://jaehong31.github.io/)**<sup>1</sup>

<sup>1</sup>NTU Singapore &nbsp;
<sup>2</sup>IAIC, A\*STAR &nbsp;
<sup>3</sup>CFAR, A\*STAR

</div>

<div align="center">

[![arXiv](https://img.shields.io/badge/arXiv-2610.31576-b31b1b?style=flat&logo=arxiv)](https://arxiv.org/abs/2610.31576)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Weights-jinyas%2Fontrack-yellow)](https://huggingface.co/jinyas/ontrack)
</div>

Code for training and evaluating learned unmasking policies for
blockwise decoding of masked diffusion language models (LLaDA-8B-Instruct and
Dream-v0-Instruct-7B).

At every denoising step a small policy network decides which masked positions of
the active block to commit. Our policy, **OnTrack**, reads the frozen denoiser's
top-1 confidence and its final hidden state at every response position. It is
trained by offline teacher forcing on teacher trajectories derived from
deterministic Fast-dLLM decoding: from each visited state, the teacher commits
the tokens that rejoin the trajectory as far ahead as possible (shortest
exact-rejoin path), without changing the final output.

## Setup

```bash
uv sync            # Python 3.12, CUDA 12.6 wheels (torch 2.7.1, flash-attn)
source .venv/bin/activate
```

Models and datasets are downloaded from the Hugging Face Hub.  HumanEval and
MBPP are read from the pinned copies in `data/datasets/`.

## Layout

| Path | Contents |
|---|---|
| `common/generation/generation.py` | Blockwise decoding: learned policy, Fast-dLLM threshold, EB-Sampler, one-by-one; optional Fast-dLLM prefix KV cache |
| `common/models/` | Policy networks (`policy.py`, `hidden_residual.py` = OnTrack, `learn2pd.py`), `factory.py`, Fast-dLLM KV-cache LLaDA port |
| `teacher/` | Trace collection and teacher-dataset construction |
| `train/` | Offline teacher-forcing trainer |
| `eval/` | Evaluation CLI, protocols, scoring (math-verify, HumanEval/MBPP pass@1) |
| `scripts/` | Teacher pipeline, threshold sweeps, Pareto tables, Learn2PD weight import |
| `configs/` | Training configs (`llada/`, `dream/`) and the Learn2PD baseline config |

## Pretrained policies

Trained OnTrack policies are on the Hugging Face Hub at
[jinyas/ontrack](https://huggingface.co/jinyas/ontrack).

| Model | Policy | Weights |
|---|---|---|
| LLaDA-8B-Instruct | OnTrack | [`llada/`](https://huggingface.co/jinyas/ontrack/tree/main/llada) |
| Dream-v0-Instruct-7B | OnTrack | [`dream/`](https://huggingface.co/jinyas/ontrack/tree/main/dream) |

```bash
hf download jinyas/ontrack --local-dir policies   # -> policies/llada, policies/dream
```

The evaluation commands in [step 3](#3-evaluation) use these weights by default,
so steps 1 and 2 are only needed to retrain the policies.

## 1. Teacher data

```bash
MODEL=llada scripts/build_teachers.sh   # -> data/teachers/llada/<cell>_correct
MODEL=dream scripts/build_teachers.sh   # -> data/teachers/dream/student_rejoin_correct
```

The pipeline records deterministic Fast-dLLM trajectories (threshold 0.9,
generation length 256, block length 32) on the GSM8K and MATH training
prompts, keeps trajectories with a correct final answer, and builds four
teacher cells: supervision on the student (`student_*`) or teacher
(`teacher_*`) trajectory states, labeled by the shortest exact-rejoin oracle
(`*_rejoin`) or by agreement with the final tokens (`*_terminal`).  OnTrack
uses `student_rejoin_correct`.

## 2. Training

One epoch of teacher forcing on one GPU; the policy is saved to
`<output_dir>/final`.

```bash
python -m train.train --config configs/llada/ontrack.yaml                 # OnTrack (ours)
python -m train.train --config configs/llada/ontrack_cache_aware.yaml     # features as seen with the prefix KV cache
python -m train.train --config configs/llada/conf.yaml                    # confidence only
python -m train.train --config configs/llada/conf_margin.yaml             # top-1 confidence + margin
python -m train.train --config configs/llada/hidden_only.yaml             # hidden state only
python -m train.train --config configs/dream/ontrack.yaml                 # OnTrack, Dream-7B
```

Teacher-cell ablations and seeds override the config:

```bash
for cell in student_terminal teacher_rejoin teacher_terminal; do
  python -m train.train --config configs/llada/ontrack.yaml \
    --g_trace_teacher_data data/teachers/llada/${cell}_correct \
    --output_dir outputs/llada_ontrack_${cell} --run_name llada_ontrack_${cell}
done
python -m train.train --config configs/llada/ontrack.yaml --seed 456 \
  --output_dir outputs/llada_ontrack_seed456 --run_name llada_ontrack_seed456
```

## 3. Evaluation

Protocol `zero-shot-tps-math500-v1`: GSM8K (1,319), MATH500 (500), HumanEval
(164) and MBPP-sanitized (257), full (untruncated) prompts, generation length
256, block length 32, batch size 1, greedy tokens, seed 42.  `scripts/sweep.py` sweeps a method over the
threshold grid 0.50-0.99 (EB-Sampler: gamma 10-0), stops a dataset once one
token per step is reached, and scores generations on CPU while the GPU
continues:

```bash
# OnTrack, released weights (see Pretrained policies)
python scripts/sweep.py --output results/ontrack \
  --policy-config configs/llada/ontrack.yaml --policy-path policies/llada
python scripts/sweep.py --output results/ontrack_dream \
  --policy-config configs/dream/ontrack.yaml --policy-path policies/dream

# Baselines
python scripts/sweep.py --output results/fastdllm --fastdllm
python scripts/sweep.py --output results/eb_sampler --eb-sampler
python scripts/sweep.py --output results/one_by_one --one-by-one

# Learn2PD: import the authors' released filter, then sweep it like a policy
curl -L -o layer_2_flan.pth \
  https://raw.githubusercontent.com/ims-kdks/Learning-to-Parallel-Decoding/main/llada/layer_2_flan.pth
python scripts/import_learn2pd_checkpoint.py --source layer_2_flan.pth --output outputs/learn2pd
python scripts/sweep.py --output results/learn2pd \
  --policy-config configs/learn2pd.yaml --policy-path outputs/learn2pd
```

To evaluate a policy you trained in step 2, pass its output instead, e.g.
`--policy-path outputs/llada_ontrack/final`.

Further settings:

* Fast-dLLM prefix KV cache: add `--kv-cache prefix_cache` (LLaDA; Fast-dLLM,
  EB-Sampler and policies).
* Generation lengths 128 / 512: `--protocol zero-shot-tps-length-transfer-v1 --gen-length 128`.
* Four-shot prompts: `--protocol four-shot-tps-gsm8k-v1 --datasets gsm8k` or
  `--protocol four-shot-tps-math500-v1 --datasets math`.
* Dream baselines: add
  `--model-path Dream-org/Dream-v0-Instruct-7B --model-revision 05334cb9faaf763692dcf9d8737c642be2b2a6ae`.
* Evaluation feeds full prompts. (Training prompts are truncated to the last
  256 tokens when the teacher data is built.) `--truncate-prompt 256`
  reproduces truncated evaluation.

Single runs use `python -m eval.eval` directly (see `python -m eval.eval --help`).

## 4. Pareto frontiers

```bash
python scripts/pareto.py results/ontrack results/fastdllm results/eb_sampler results/learn2pd
```

prints, per dataset, the best measured accuracy within NFE budgets
(length/8, /4, /2, length) and each method's accuracy-NFE Pareto frontier,
using measured sweep points only.

## Acknowledgements

This codebase builds on [apple/ml-rl-dllm](https://github.com/apple/ml-rl-dllm),
the official implementation of
[Learning Unmasking Policies for Diffusion Language Models](https://arxiv.org/abs/2512.09106)
(Jazbec et al., 2025); its license notice is reproduced in
[THIRD_PARTY_NOTICES](THIRD_PARTY_NOTICES).  We also use code from d1, Fast-dLLM
and LLaDA (see THIRD_PARTY_NOTICES).

## Citation

```bibtex
@article{sakurai2026unmasking,
  title   = {What Makes a Good Unmasking Policy? A Controlled Study of Learned Unmasking in Diffusion Language Models},
  author  = {Sakurai, Jinya and Xu, Xun and Yoon, Jaehong},
  journal = {arXiv preprint arXiv:2610.31576},
  year    = {2026},
}
```

## License

This code is released under the [Apache License 2.0](LICENSE).
Third-party code included in this repository remains under its original
license; see [THIRD_PARTY_NOTICES](THIRD_PARTY_NOTICES).
