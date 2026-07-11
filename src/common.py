"""Shared helpers: portable model resolution + the frozen forward passes.

Model resolution order (so the repo runs out-of-box):
  1. env JINA_OMNI_NANO_DIR if set and exists
  2. common local caches (HF hub, /Volumes/vault, ~/models)
  3. huggingface_hub.snapshot_download("jinaai/jina-embeddings-v5-omni-nano-mlx")

Everything here uses ONLY the single model (jina-embeddings-v5-omni-nano-mlx).
No third-party model, no training.
"""
import os, sys, json
import numpy as np
import mlx.core as mx

MODEL_ID = "jinaai/jina-embeddings-v5-omni-nano-mlx"
IMAGE_TOKEN_ID = 128259


def resolve_model_dir() -> str:
    env = os.environ.get("JINA_OMNI_NANO_DIR")
    if env and os.path.isdir(env):
        return os.path.abspath(env)
    candidates = [
        os.path.expanduser("~/models/jina-embeddings-v5-omni-nano-mlx"),
        "/Volumes/vault/ai-models/jinaai/jina-embeddings-v5-omni-nano-mlx",
    ]
    for c in candidates:
        if os.path.isdir(c) and os.path.exists(os.path.join(c, "model.safetensors")):
            return c
    from huggingface_hub import snapshot_download
    return snapshot_download(MODEL_ID)


def load(model_dir=None):
    """Load model + tokenizer via the repo's own utils.load_model."""
    model_dir = model_dir or resolve_model_dir()
    if model_dir not in sys.path:
        sys.path.insert(0, model_dir)
    from utils import load_model  # provided inside the model repo
    m = load_model(model_dir)
    m.switch_task("retrieval")
    return m, model_dir


def _bidi_mask(am, ref):
    return mx.where(am == 0, -1e9, 0.0)[:, None, None, :].astype(ref.dtype)


def image_feats(model, proc, pil):
    """Return (patch_feats [n,768] L2-normed, global_vec [768] L2-normed).

    Replicates encode_image internals but keeps the FULL hidden sequence so we
    can read per-patch (contextualized, aligned-space) features. Each image-token
    row is one merged vision patch in the same 768-d space as text labels.
    """
    out = proc(images=pil, return_tensors="np")
    grid = np.array(out["image_grid_thw"]).astype(np.int32)
    t, h, w = [int(x) for x in grid[0]]
    n = (t * h * w) // 4
    pv = mx.array(out["pixel_values"].astype(np.float32))
    gthw = mx.array(grid)
    ids = mx.array([[IMAGE_TOKEN_ID] * n])
    am = mx.array([[1] * n])
    vf = model.merger(model.vision_tower(pv, gthw))
    emb = model.language_model.embed_tokens(ids)
    sm = ids == model.config.image_token_id
    pos = mx.array(np.where(mx.flatten(mx.broadcast_to(sm[..., None], emb.shape)))[0], mx.uint32)
    fe = mx.flatten(emb); fe[pos] = mx.flatten(vf); emb = mx.reshape(fe, emb.shape)
    lm = model.language_model
    hidden = lm(inputs_embeds=emb, mask=_bidi_mask(am, lm.layers[0].input_layernorm.weight))
    mx.eval(hidden)
    H = np.array(hidden[0].astype(mx.float32).tolist())
    P = H / (np.linalg.norm(H, axis=1, keepdims=True) + 1e-9)
    g = H[-1] / (np.linalg.norm(H[-1]) + 1e-9)
    return P.astype(np.float32), g.astype(np.float32)


def word_start_gate(model_dir, n_vocab):
    """Boolean mask over vocab: True for tokenizer 'Ġ' word-start alphabetic words.
    Tokenizer-internal signal for 'this is a whole word' — no external dictionary."""
    tj = json.load(open(os.path.join(model_dir, "tokenizer.json")))
    id2p = {v: k for k, v in tj["model"]["vocab"].items()}
    gate = np.zeros(n_vocab, bool)
    for tid in range(n_vocab):
        p = id2p.get(tid, "")
        if p.startswith("Ġ"):
            s = p[1:]
            if len(s) >= 3 and s.isalpha() and s.isascii() and s.islower():
                gate[tid] = True
    return gate
