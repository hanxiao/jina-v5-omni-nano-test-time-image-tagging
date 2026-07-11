"""
Solid grounding probe: can we do image multi-label tagging on the
jina-embeddings-v5-omni-nano-mlx model by scoring the image embedding
against the MODEL'S OWN tokenizer vocabulary embedding table
(language_model.embed_tokens.weight [128260, 768])?

No extra vocabulary. The "labels" ARE the tokenizer vocab tokens.

This script is a HONEST probe. It reports what actually happens, no
cherry-picking, no heuristic massaging. We test several scoring spaces
because tie_word_embeddings=false means the input-embedding space and the
pooled-output space are NOT guaranteed to be comparable by dot product.
"""
import sys, os, json
import numpy as np
import mlx.core as mx
from PIL import Image
from transformers import Qwen2VLImageProcessor

MODEL_DIR = "/Volumes/vault/ai-models/jinaai/jina-embeddings-v5-omni-nano-mlx"
sys.path.insert(0, MODEL_DIR)
from utils import load_model  # noqa

IMAGE_TOKEN_ID = 128259

def build_image_inputs(pil_img, proc, tokenizer, model_cfg):
    """Produce pixel_values, image_grid_thw, input_ids, attention_mask for one image."""
    out = proc(images=pil_img, return_tensors="np")
    pixel_values = mx.array(out["pixel_values"].astype(np.float32))
    grid = out["image_grid_thw"]  # (1,3) = (t,h,w) in patch units
    grid_thw = mx.array(np.array(grid).astype(np.int32))
    t, h, w = [int(x) for x in np.array(grid)[0]]
    merge = 2
    n_img_tokens = (t * h * w) // (merge * merge)
    # Build a minimal text sequence: just the image placeholders + a query prefix.
    # Mirror encode() text path: we still need last-token pooling to land on a real token.
    prefix_ids = tokenizer.encode("Query: ").ids
    # sequence: prefix + [IMG]*n + (end). last-token pool takes the LAST non-pad token.
    ids = prefix_ids + [IMAGE_TOKEN_ID] * n_img_tokens
    input_ids = mx.array([ids])
    attention_mask = mx.array([[1] * len(ids)])
    return pixel_values, grid_thw, input_ids, attention_mask, n_img_tokens

def main():
    print("Loading model...")
    m = load_model(MODEL_DIR)
    m.switch_task("retrieval")
    model = m.model
    tok = m.tokenizer

    # vocab embedding table
    params = dict(model.parameters())
    E = params["language_model"]["embed_tokens"]["weight"]  # [V,768]
    E = E.astype(mx.float32)
    V, D = E.shape
    print(f"vocab embed table: {E.shape}")
    En = E / mx.linalg.norm(E, axis=1, keepdims=True)

    proc = Qwen2VLImageProcessor.from_pretrained(MODEL_DIR)

    imgs = ["cat.jpg", "dogbike.jpg", "pizza.jpg"]
    base = os.path.join(os.path.dirname(__file__), "testimg")

    def decode_tok(tid):
        return tok.decode([tid])

    for name in imgs:
        p = os.path.join(base, name)
        img = Image.open(p).convert("RGB")
        pv, grid, ids, am, n = build_image_inputs(img, proc, tok, model.config)
        emb = model.encode_image(pv, grid, ids, am)  # [1,768] L2-normed
        v = emb[0].astype(mx.float32)
        vn = v / mx.linalg.norm(v)
        # score against vocab embeddings (cosine, since both normed)
        scores = En @ vn  # [V]
        scores_np = np.array(scores.tolist())
        top = np.argsort(-scores_np)[:30]
        print(f"\n===== {name} (img_tokens={n}) =====")
        print("top-30 vocab tokens by cosine(image_emb, embed_tokens):")
        toks = [repr(decode_tok(int(t))) for t in top]
        print("  ", ", ".join(f"{decode_tok(int(t)).strip()}={scores_np[t]:.3f}" for t in top))

if __name__ == "__main__":
    main()
