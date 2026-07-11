# jina-v5-omni-nano test-time image tagging

Open-vocabulary, **multi-label** image tagging with a **single frozen model** —
[`jinaai/jina-embeddings-v5-omni-nano-mlx`](https://huggingface.co/jinaai/jina-embeddings-v5-omni-nano-mlx)
(~1B, MLX). **Zero training. No second model. No WordNet / regex dictionary /
POS tagger.** Every signal — the label space, the visual features, the
word filter — comes from that one model.

The label space is not a hand-curated tag list: it is the model's **own
tokenizer vocabulary**. You give it an image, it returns tags.

```
python src/tag_image.py assets/examples/cat.jpg --topk 6
# cat.jpg (86ms): kitty, cosy, plush, crib, paw, sleeps

python src/tag_image.py assets/examples/zebra.jpg --topk 5 --hq
# zebra.jpg (1016ms): fur, bear, puppy, muzzle, leather   # <- CWR recovers the small bear

python src/tag_image.py assets/examples/cat.jpg --topk 4 --adj
# cat.jpg (113ms): couch kitty, grey cosy, cat plush, sleeping crib
```

![pipeline](docs/figures/pipeline.png)

## Results (real benchmark: 150 COCO-val images, 80-category closed set)

| method | P@1 | P@3 | R@5 | mAP |
|---|---|---|---|---|
| global pooled ([CLS]-style) | 0.433 | 0.289 | 0.449 | 0.264 |
| patch-level max (a=0.7) | 0.753 | 0.427 | 0.631 | 0.635 |
| softpool T=0.05 | 0.773 | 0.449 | 0.649 | 0.608 |
| **+ CWR multi-crop (`--hq`)** | **0.813** | **0.476** | **0.680** | **0.710** |

Latency (M3 Ultra, MLX): fast mode ~60–90 ms/image; `--hq` ~1 s/image (14 crops).

## Why this design (the short version)

The whole project is an exercise in **finding what the model already knows** and
not adding machinery it doesn't need. Key grounded findings (full log in
[`docs/findings-v5omni-nano.md`](docs/findings-v5omni-nano.md)):

1. **Label = tokenizer vocab, via `encode_text` (NOT `embed_tokens`).**
   `tie_word_embeddings=false`, so the raw input-embedding table lives in a
   different space than the pooled output — dotting the image against it returns
   garbage (`Tutor, avatar, PyTuple`). Encoding each token *as text* puts labels
   in the same aligned space as the image. ✅

2. **Patch-level >> global.** The global (last-token) vector is a weak
   multi-label classifier (mAP 0.264). Scoring each patch against labels and
   class-wise max-pooling, fused with global at a=0.7, jumps to mAP 0.635. This
   is the PIAA "[CLS] is not enough" thesis, confirmed on v5-omni.

3. **Background centering** removes base-rate bias (some words are just closer
   to the image modality). Subtract a per-label prior estimated from neutral
   images.

4. **`Ġ` word-start gate.** The tokenizer's own byte-BPE space marker (`Ġcat` =
   a whole word, `GetComponent` = a fragment) filters 128k tokens → ~25k clean
   words. No external dictionary — the tokenizer already knows what a word is.

5. **embedding-NMS** dedups synonyms / multilingual variants (猫/Cat/кот/cat)
   using cosine in the model's own embedding space.

6. **CWR multi-crop (`--hq`)** is the one lever that genuinely lifts accuracy:
   re-encode the image as a 3×3+2×2+center grid of crops and take the per-label
   **max** across crops. A small object becomes large in whichever crop contains
   it, so its signal goes from weak to strong. This is **test-time augmentation
   done right** — full-coverage grid + per-label max (take the strongest
   evidence), not blind center-crop averaging (which *hurt*).

7. **Patch-local adjectives (`--adj`)** attach modifiers without a POS tagger:
   find the patches where a noun fires, pool that local region, and score
   attribute words *only there*. "grey cat", "fur bear" — the modifier is tied
   to the object's pixels, so bag-of-words noise can't win.

## What did NOT work (and why it's the interesting part)

We systematically tried the obvious "test-time compute" upgrades from recent
literature. **None improved over CWR.** This is a feature, not a failure — it
maps the boundary of what helps a strong single-model embedding pipeline.
Full analysis in [`docs/ttc-paper-eval.md`](docs/ttc-paper-eval.md) and
[`docs/accuracy-design-memo.md`](docs/accuracy-design-memo.md).

| lever | result | why it didn't transfer |
|---|---|---|
| use penultimate layer patches (TagCLIP) | mAP 0.16 (collapse) | v5-omni is a bidirectional embedding model w/ last-token pooling — its final layer *is* the trained output space, unlike CLIP-ViT |
| softmax-over-classes (TagCLIP) | flat (0.636) | same reason; cross-class normalization not needed |
| whitening / GDA (PIAA) | mAP 0.06 (collapse) | image & text already share one space; whitening destroys it |
| ZLaP label propagation | mAP 0.14 | graph built on the weak global vector; multi-label breaks the manifold assumption |
| OTTER optimal transport | flat (0.699), P@3↓ | scores already calibrated; OT's mass conservation fights multi-label recall |
| BCA adaptive prior | no-op | a per-class prior shift is absorbed by per-class centering |
| EM-Dirichlet | P@1 0.17 (collapse) | simplex / cross-class normalization destroys multi-label |
| soft-trim crop aggregation | worse than max | for small objects only one crop is right — the "outlier" *is* the signal |

**Meta-conclusion:** re-processing the features (re-layer, whiten, normalize,
propagate) does nothing or hurts, because v5-omni's space is already good. The
only thing that helps is **feeding the model new information** (CWR crops). The
ceiling is the 1B model itself.

## Install & run

```bash
pip install -r requirements.txt

# option A: let it auto-download the model from HuggingFace on first run
python src/tag_image.py your_image.jpg --topk 8

# option B: point at a local copy
export JINA_OMNI_NANO_DIR=/path/to/jina-embeddings-v5-omni-nano-mlx
python src/tag_image.py your_image.jpg --hq
```

First run encodes the whole vocabulary once (`src/label_cache.npz`, ~40 s) and
reuses it after. A small background prior ships in `src/bg_prior.npz`; regenerate
a stronger one by pointing `--bg-dir` at a folder of neutral images.

Flags: `--topk N` · `--hq` (CWR, higher accuracy) · `--adj` (adjectives) ·
`--soft` (softpool, better top-k precision).

## Repo layout

```
src/
  tag_image.py        # the tagger (fast / --hq / --adj / --soft), self-contained
  common.py           # portable model resolution + frozen forward passes
  bg_prior.npz        # small shipped background prior
eval/
  build_eval_set.py   # sample 150 COCO-val imgs + multi-label GT
  eval_coco.py        # the benchmark (pooled / patch / fusion, mAP/P@k)
  eval_hacks_coco.py  # anchor-prefix / softpool measured on the benchmark
  bench_latency.py    # per-image latency breakdown
experiments/
  probes/             # embed_tokens (rejected) vs encode_text (accepted)
  accuracy_levers/    # layer sweep, whitening, CWR (exp_cwr*.py = the win)
  test_time_compute/  # ZLaP, OTTER, soft-trim, BCA, EM-Dirichlet (all negative)
  tagger_*.py         # intermediate taggers (word-start, patch, adj-local)
docs/
  findings-v5omni-nano.md   # full grounded engineering log (conclusions 1-12)
  accuracy-design-memo.md   # birdview of accuracy levers
  ttc-paper-eval.md         # 9-paper test-time-compute evaluation + results
  figures/                  # pipeline diagrams
  paper-specs/              # method specs + diagrams for the reference papers
```

## References

The design is grounded in these papers (specs in `docs/paper-specs/`):

- Tag2Text (2303.05657), RAM (2306.03514), RAM++ (2310.15200) — image tagging foundation models
- **TagCLIP** (2312.12828, AAAI'24) — patch-level, drop last self-attention
- **PIAA** (2605.25821) — "[CLS] is not enough", patch inference + adaptive aggregation
- Improving Visual Discriminability of CLIP (2510.23894)
- ZLaP Label Propagation (CVPR'24), OTTER Optimal Transport (2404.08461),
  Bayesian TTA (CVPR'25), Transductive EM-Dirichlet (2405.18437) — test-time methods (evaluated, not adopted)

## Model & license

Code: Apache-2.0. Model `jinaai/jina-embeddings-v5-omni-nano`: CC-BY-NC-4.0
(non-commercial; commercial use needs a license from Elastic/Jina).

Built as a single-model, self-bootstrapping, test-time-compute study — no heavy
training, everything runs on a Mac with MLX.
