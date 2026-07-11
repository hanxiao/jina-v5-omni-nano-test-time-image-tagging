"""
Probe v2: the CORRECT space.

Since tie_word_embeddings=false, image pooled output lives in the same space
as TEXT POOLED OUTPUT (the model is trained image<->text aligned on pooled,
L2-normed, last-token vectors). So labels must be encoded via the full
encode_text path, not raw embed_tokens.

Han's intuition ("multi-label over the tokenizer vocab") lands as:
  - take candidate words (here: a curated probe set + later the real vocab)
  - encode each as text -> pooled label vector
  - cosine(image_vec, label_vecs) -> multi-label scores

This probe uses a small hand set of words spanning the 3 test images plus
distractors, to verify the aligned space actually ranks correct labels high.
Honest: distractors included, no cherry-picking.
"""
import sys, os
import numpy as np
import mlx.core as mx
from PIL import Image
from transformers import Qwen2VLImageProcessor

MODEL_DIR = "/Volumes/vault/ai-models/jinaai/jina-embeddings-v5-omni-nano-mlx"
sys.path.insert(0, MODEL_DIR)
from utils import load_model  # noqa

IMAGE_TOKEN_ID = 128259

LABELS = [
    "cat", "kitten", "dog", "puppy", "bicycle", "bike", "pizza", "food",
    "sofa", "couch", "remote control", "television", "person", "car",
    "airplane", "boat", "mountain", "ocean", "keyboard", "laptop",
    "flower", "tree", "guitar", "elephant", "horse", "table", "chair",
    "cheese", "tomato", "book", "clock", "bird", "fish", "snow",
]

def build_image_inputs(pil_img, proc, tokenizer):
    out = proc(images=pil_img, return_tensors="np")
    pixel_values = mx.array(out["pixel_values"].astype(np.float32))
    grid = np.array(out["image_grid_thw"]).astype(np.int32)
    grid_thw = mx.array(grid)
    t, h, w = [int(x) for x in grid[0]]
    n = (t * h * w) // 4
    prefix_ids = tokenizer.encode("Query: ").ids
    ids = prefix_ids + [IMAGE_TOKEN_ID] * n
    return pixel_values, grid_thw, mx.array([ids]), mx.array([[1]*len(ids)]), n

def main():
    print("Loading...")
    m = load_model(MODEL_DIR)
    proc = Qwen2VLImageProcessor.from_pretrained(MODEL_DIR)
    base = os.path.join(os.path.dirname(__file__), "testimg")

    # encode labels as text (retrieval.passage = "Document:" prefix, the doc side)
    m.switch_task("retrieval")
    label_emb = m.model.encode(LABELS, m.tokenizer, task_type="retrieval.passage")
    L = label_emb.astype(mx.float32)
    L = L / mx.linalg.norm(L, axis=1, keepdims=True)

    for name in ["cat.jpg", "dogbike.jpg", "pizza.jpg"]:
        img = Image.open(os.path.join(base, name)).convert("RGB")
        pv, grid, ids, am, n = build_image_inputs(img, proc, m.tokenizer)
        emb = m.model.encode_image(pv, grid, ids, am)[0].astype(mx.float32)
        emb = emb / mx.linalg.norm(emb)
        scores = np.array((L @ emb).tolist())
        order = np.argsort(-scores)
        print(f"\n===== {name} =====")
        print("  top-10:", ", ".join(f"{LABELS[i]}={scores[i]:.3f}" for i in order[:10]))
        print("  bottom-5:", ", ".join(f"{LABELS[i]}={scores[i]:.3f}" for i in order[-5:]))

if __name__ == "__main__":
    main()
