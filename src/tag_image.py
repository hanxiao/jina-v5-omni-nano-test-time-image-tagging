#!/usr/bin/env python3
"""
tag_image.py - open-vocabulary multi-label image tagging with a SINGLE model:
jina-embeddings-v5-omni-nano-mlx. Zero training, no third-party model, no
wordnet/regex-dict/POS. All signals are model-internal.

Pipeline:
- labels = the model's OWN tokenizer vocabulary (Ġ word-start words), encoded
  via encode_text (NOT embed_tokens; tie_word_embeddings=false so the raw table
  is unusable — see docs/findings-v5omni-nano.md).
- patch-level scoring: per-patch cosine to labels, class-wise max-pool, fused
  with the global (last-token) score at a=0.7.
- background centering: subtract a per-label prior estimated from neutral images
  (removes base-rate bias). A tiny built-in prior ships with the repo; you can
  regenerate a stronger one (see --rebuild-prior).
- embedding-NMS: drop synonym / multilingual near-duplicates.
- --hq: CWR multi-crop (3x3+2x2+center=14 crops, per-label MAX) -> mAP 0.635->0.710.
- --ngram N: patch-local n-grams (score modifier words only on the noun's peak
  patches; N-1 modifiers per noun, no word-class constraint).
- --beam: score ASSEMBLED phrases with the encoder itself (beam search over
  modifier slots; s(phrase) = cos(encode_text(phrase), region) - noun baseline).
  Slower (~1-2 s/image extra) but the model judges composition and order.

Usage:
    python tag_image.py IMG [IMG ...] [--topk 8] [--hq] [--ngram 2] [--beam] [--soft]

First run builds label_cache.npz (encodes the whole vocab once, ~40s) next to
this file, plus bg_prior.npz. Set JINA_OMNI_NANO_DIR to point at a local copy of
the model, else it is auto-downloaded from HuggingFace.
"""
import os, sys, glob, json, argparse, time
import numpy as np
from PIL import Image
from transformers import Qwen2VLImageProcessor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as C

CACHE_LABELS = os.path.join(HERE, "label_cache.npz")
CACHE_PRIOR = os.path.join(HERE, "bg_prior.npz")


def build_label_cache(m):
    """Encode the whole tokenizer vocab as text labels via encode_text (one-time)."""
    tok = m.tokenizer
    V = tok.get_vocab_size()
    strings = []
    for tid in range(V):
        s = tok.decode([tid]) or ""
        strings.append(s.strip() or "\u2205")
    print(f"[cache] encoding {V} vocab tokens as text labels (one-time)...")
    import mlx.core as mx
    rows = []
    B = 1024
    for i in range(0, V, B):
        e = m.model.encode(strings[i:i + B], tok, task_type="retrieval.passage").astype(mx.float32)
        e = e / mx.linalg.norm(e, axis=1, keepdims=True)
        rows.append(np.array(e.tolist()).astype(np.float16))
    E = np.vstack(rows)
    np.savez(CACHE_LABELS, E=E, strings=np.array(strings, dtype=object))
    return E.astype(np.float32), strings


def build_prior(m, proc, E, bg_dir=None):
    """Estimate per-label prior μ from neutral background images.
    Uses images in assets/examples if no bg_dir given (weak but functional prior)."""
    bg_dir = bg_dir or os.path.join(HERE, "..", "assets", "examples")
    imgs = sorted(glob.glob(os.path.join(bg_dir, "*.jpg")) + glob.glob(os.path.join(bg_dir, "*.png")))
    if not imgs:
        # zero prior fallback (still works, just no base-rate removal)
        return np.zeros(E.shape[0], np.float32), np.zeros(E.shape[0], np.float32)
    SP = []; SG = []
    for p in imgs:
        P, g = C.image_feats(m.model, proc, Image.open(p).convert("RGB"))
        SP.append((P @ E.T).max(0)); SG.append(g @ E.T)
    mup = np.mean(SP, 0).astype(np.float32); mug = np.mean(SG, 0).astype(np.float32)
    np.savez(CACHE_PRIOR, mup=mup, mug=mug)
    return mup, mug


def nms(order, En, keep, tau=0.6, avoid=None):
    kept = []; kv = [] if not avoid else [En[a] for a in avoid]
    for j in order:
        v = En[j]
        if any(float(v @ k) >= tau for k in kv):
            continue
        kept.append(j); kv.append(v)
        if len(kept) >= keep:
            break
    return kept


def grid_crops(img, ov=0.15):
    W, H = img.size; out = []
    for gx, gy in [(3, 3), (2, 2)]:
        cw = W / gx; ch = H / gy; ox = cw * ov; oy = ch * ov
        for j in range(gy):
            for i in range(gx):
                out.append(img.crop((max(0, int(i * cw - ox)), max(0, int(j * ch - oy)),
                                     min(W, int((i + 1) * cw + ox)), min(H, int((j + 1) * ch + oy)))))
    out.append(img.crop((int(W * .25), int(H * .25), int(W * .75), int(H * .75))))
    return out


def tag(m, proc, pil, E, En, strings, gate, mup, mug, topk=8, ngram=1, beam=False, soft=False, hq=False):
    P, g = C.image_feats(m.model, proc, pil)
    sim = P @ E.T
    if soft:
        w = np.exp((sim - sim.max(0, keepdims=True)) / 0.05); w /= w.sum(0, keepdims=True)
        sp = (w * sim).sum(0)
    else:
        sp = sim.max(0)
    cen = 0.7 * (sp - mup) + 0.3 * ((g @ E.T) - mug)
    if hq:
        crop_max = None
        for cr in grid_crops(pil):
            Pc, gc = C.image_feats(m.model, proc, cr)
            sc = np.maximum((Pc @ E.T).max(0), gc @ E.T)
            crop_max = sc if crop_max is None else np.maximum(crop_max, sc)
        cen = cen + 1.3 * (crop_max - mup)
    cen2 = cen.copy(); cen2[~gate] = -1e9
    nouns = nms(np.argsort(-cen2)[:400], En, keep=topk)
    if ngram <= 1:
        return [strings[j].strip().lower() for j in nouns]
    res = []
    for nid in nouns:
        nw = strings[nid].strip().lower()
        # pool the noun's support region; rank ALL gated words on it; take the
        # n-1 top survivors (noun cluster excluded, mutual NMS) as modifiers.
        ps = sim[:, nid]; topp = np.argsort(-ps)[:max(3, len(ps) // 10)]
        local = P[topp].mean(0); local /= np.linalg.norm(local) + 1e-9
        ls = local @ E.T; ls[~gate] = -1e9
        order = np.argsort(-ls)[:200]
        if not beam:
            mods = nms(order, En, keep=ngram - 1, tau=0.55, avoid=[nid])
            words = [strings[a].strip().lower() for a in mods] + [nw]
            res.append(" ".join(w for w in words if w))
            continue
        # --beam: the encoder itself scores assembled phrases against the region
        # (margin over the bare noun), searching modifier choice AND order.
        kv = En[nid]
        cands = [j for j in order if float(En[j] @ kv) < 0.55][:12]
        cw = [strings[j].strip().lower() for j in cands]
        base = float(_encode_texts(m, [nw])[0] @ local)
        beams = [([], 0.0)]
        for _ in range(ngram - 1):
            exp = []
            for mo, _sc in beams:
                used = [cands[cw.index(x)] for x in mo]
                for w, j in zip(cw, cands):
                    if w in mo: continue
                    if any(float(En[j] @ En[k]) >= 0.55 for k in used): continue
                    exp.append(mo + [w])
            if not exp: break
            phrases = [" ".join(list(reversed(mo)) + [nw]) for mo in exp]
            sc = _encode_texts(m, phrases) @ local - base
            beams = sorted(zip(exp, sc.tolist()), key=lambda t: -t[1])[:4]
        best = beams[0][0]
        res.append(" ".join(list(reversed(best)) + [nw]))
    return res


def _encode_texts(m, texts):
    import mlx.core as mx
    e = m.model.encode(texts, m.tokenizer, task_type="retrieval.passage").astype(mx.float32)
    e = e / mx.linalg.norm(e, axis=1, keepdims=True)
    return np.array(e.tolist(), dtype=np.float32)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("images", nargs="+")
    ap.add_argument("--topk", type=int, default=8)
    ap.add_argument("--ngram", type=int, default=1, help="emit patch-local n-grams: N-1 grounded modifiers per tag (1 = plain tags)")
    ap.add_argument("--beam", action="store_true", help="beam-search phrases scored by the encoder itself (with --ngram >= 2)")
    ap.add_argument("--soft", action="store_true", help="softpool (better top-k precision)")
    ap.add_argument("--hq", action="store_true", help="high-accuracy CWR multi-crop (mAP 0.635->0.710, ~14x slower)")
    ap.add_argument("--bg-dir", default=None, help="dir of neutral images for the background prior")
    args = ap.parse_args()

    m, model_dir = C.load()
    proc = Qwen2VLImageProcessor.from_pretrained(model_dir)

    if os.path.exists(CACHE_LABELS):
        d = np.load(CACHE_LABELS, allow_pickle=True)
        E = d["E"].astype(np.float32); strings = list(d["strings"])
    else:
        E, strings = build_label_cache(m)
    En = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)
    gate = C.word_start_gate(model_dir, len(strings))

    if os.path.exists(CACHE_PRIOR):
        z = np.load(CACHE_PRIOR); mup, mug = z["mup"], z["mug"]
    else:
        mup, mug = build_prior(m, proc, E, args.bg_dir)

    for path in args.images:
        pil = Image.open(path).convert("RGB")
        t0 = time.time()
        tags = tag(m, proc, pil, E, En, strings, gate, mup, mug,
                   topk=args.topk, ngram=args.ngram, beam=args.beam, soft=args.soft, hq=args.hq)
        dt = (time.time() - t0) * 1000
        print(f"{os.path.basename(path)} ({dt:.0f}ms): {', '.join(tags)}")


if __name__ == "__main__":
    main()
