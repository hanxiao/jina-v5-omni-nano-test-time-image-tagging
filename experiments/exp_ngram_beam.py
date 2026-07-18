#!/usr/bin/env python3
"""Beam-searched n-grams vs greedy --ngram.

Greedy (shipped): modifiers picked off ONE static region ranking (MMR-style).
Beam (this probe): expand modifier slots and let the frozen encoder score the
ASSEMBLED phrase against the noun's region, minus the noun-alone baseline:
    s(phrase) = cos(encode_text(phrase), local) - cos(encode_text(noun), local)
Model-internal only: same encoder, no external knowledge.
"""
import os, sys, argparse
import numpy as np
from PIL import Image
from transformers import Qwen2VLImageProcessor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
import common as C
import tag_image as T

def encode_texts(m, texts):
    import mlx.core as mx
    e = m.model.encode(texts, m.tokenizer, task_type="retrieval.passage").astype(mx.float32)
    e = e / mx.linalg.norm(e, axis=1, keepdims=True)
    return np.array(e.tolist(), dtype=np.float32)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="+")
    ap.add_argument("--topk", type=int, default=4)
    ap.add_argument("--n", type=int, default=3, help="max n-gram order")
    ap.add_argument("--cands", type=int, default=12)
    ap.add_argument("--beam", type=int, default=4)
    args = ap.parse_args()

    m, model_dir = C.load()
    proc = Qwen2VLImageProcessor.from_pretrained(model_dir)
    d = np.load(T.CACHE_LABELS, allow_pickle=True)
    E = d["E"].astype(np.float32); strings = list(d["strings"])
    En = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)
    gate = C.word_start_gate(model_dir, len(strings))
    z = np.load(T.CACHE_PRIOR); mup, mug = z["mup"].astype(np.float32), z["mug"].astype(np.float32)

    for path in args.images:
        pil = Image.open(path).convert("RGB")
        P, g = C.image_feats(m.model, proc, pil)
        sim = P @ E.T
        cen = 0.7 * (sim.max(0) - mup) + 0.3 * ((g @ E.T) - mug)
        cg = cen.copy(); cg[~gate] = -1e9
        nouns = T.nms(np.argsort(-cg)[:400], En, keep=args.topk)
        print(f"\n== {os.path.basename(path)} ==")
        for nid in nouns:
            nw = strings[nid].strip().lower()
            ps = sim[:, nid]; topp = np.argsort(-ps)[:max(3, len(ps) // 10)]
            local = P[topp].mean(0); local /= np.linalg.norm(local) + 1e-9
            ls = local @ E.T; ls[~gate] = -1e9
            # candidate modifiers: region-ranked, noun cluster excluded, NO mutual suppression
            order = np.argsort(-ls)[:200]
            kv = En[nid]
            cands = [j for j in order if float(En[j] @ kv) < 0.55][:args.cands]
            cwords = [strings[j].strip().lower() for j in cands]
            base = float(encode_texts(m, [nw])[0] @ local)
            # greedy (shipped) for reference
            mods_g = T.nms(order, En, keep=args.n - 1, tau=0.55, avoid=[nid])
            greedy = " ".join([strings[a].strip().lower() for a in mods_g] + [nw])
            # beam over modifier slots (prepend), phrase scored by the model itself
            beams = [([], 0.0)]
            best_per_n = {}
            for depth in range(1, args.n):
                expansions = []
                for mods, _ in beams:
                    for w, j in zip(cwords, cands):
                        if w in mods: continue
                        if any(float(En[j] @ En[k]) >= 0.55 for k in
                               [cands[cwords.index(x)] for x in mods]): continue
                        expansions.append(mods + [w])
                if not expansions: break
                phrases = [" ".join(list(reversed(mo)) + [nw]) for mo in expansions]
                sc = encode_texts(m, phrases) @ local - base
                ranked = sorted(zip(expansions, sc.tolist()), key=lambda t: -t[1])
                beams = ranked[:args.beam]
                best_per_n[depth + 1] = (" ".join(list(reversed(beams[0][0])) + [nw]), beams[0][1],
                                         [(" ".join(list(reversed(mo)) + [nw]), float(v)) for mo, v in ranked[1:4]])
            print(f"  noun={nw:12s} greedy(n={args.n}): {greedy}")
            for n_, (ph, sc_, alts) in best_per_n.items():
                alt = "  |  ".join(f"{p} ({v:+.3f})" for p, v in alts)
                print(f"    beam n={n_}: {ph} ({sc_:+.3f})   alts: {alt}")

if __name__ == "__main__":
    main()
