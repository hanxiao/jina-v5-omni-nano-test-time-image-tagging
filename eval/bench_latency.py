"""Measure per-image tagging latency (pooled vs patch), and component breakdown.
Label matrix + bg prior are precomputed/cached (one-time), so per-image cost =
vision encode + text-tower forward + score matmul + gate/nms.
"""
import sys, os, time, glob, json
import numpy as np
import mlx.core as mx
from PIL import Image
from transformers import Qwen2VLImageProcessor
MODEL_DIR=os.environ.get("JINA_OMNI_NANO_DIR","/Volumes/vault/ai-models/jinaai/jina-embeddings-v5-omni-nano-mlx")
sys.path.insert(0, MODEL_DIR)
from utils import load_model
IMAGE_TOKEN_ID=128259
HERE=os.path.dirname(__file__)

def _bidi_mask(am, ref): return mx.where(am==0,-1e9,0.0)[:,None,None,:].astype(ref.dtype)

def run(model, proc, pil):
    tv0=time.time()
    out=proc(images=pil, return_tensors="np")
    grid=np.array(out["image_grid_thw"]).astype(np.int32)
    t,h,w=[int(x) for x in grid[0]]; n=(t*h*w)//4
    pv=mx.array(out["pixel_values"].astype(np.float32)); gthw=mx.array(grid)
    ids=mx.array([[IMAGE_TOKEN_ID]*n]); am=mx.array([[1]*n])
    tprep=time.time()-tv0
    t0=time.time()
    vf=model.merger(model.vision_tower(pv,gthw)); mx.eval(vf)
    tvis=time.time()-t0
    t0=time.time()
    emb=model.language_model.embed_tokens(ids)
    sm=ids==model.config.image_token_id
    pos=mx.array(np.where(mx.flatten(mx.broadcast_to(sm[...,None],emb.shape)))[0],mx.uint32)
    fe=mx.flatten(emb); fe[pos]=mx.flatten(vf); emb=mx.reshape(fe,emb.shape)
    hidden=model.language_model(inputs_embeds=emb, mask=_bidi_mask(am,model.language_model.layers[0].input_layernorm.weight))
    mx.eval(hidden)
    ttext=time.time()-t0
    return n, tprep, tvis, ttext

def main():
    m=load_model(MODEL_DIR); m.switch_task("retrieval")
    proc=Qwen2VLImageProcessor.from_pretrained(MODEL_DIR)
    E=np.load(os.path.join(HERE,"full_vocab_emb.npz"),allow_pickle=True)["E"].astype(np.float32)
    base=os.path.join(HERE,"testimg")
    imgs=sorted(glob.glob(os.path.join(base,"*.jpg")))
    # warmup
    run(m.model, proc, Image.open(imgs[0]).convert("RGB"))
    tot={}
    for p in imgs:
        pil=Image.open(p).convert("RGB")
        n,tp,tv,tt=run(m.model,proc,pil)
        # score cost (patch matmul over full vocab)
        t0=time.time()
        P=np.random.randn(n,768).astype(np.float32)  # placeholder shape; real uses hidden
        _=(P@E.T).max(0); tscore=time.time()-t0
        print(f"{os.path.basename(p):20s} patches={n:4d} prep={tp*1000:.0f}ms vis={tv*1000:.0f}ms text={tt*1000:.0f}ms score={tscore*1000:.0f}ms total={(tp+tv+tt+tscore)*1000:.0f}ms")

if __name__=="__main__":
    main()
