"""
Patch-LOCAL adjective+noun tagging. No training/wordnet/regex-dict/POS.
Core idea to enforce the modifier relation without POS:
1. patch-level -> top nouns (a=0.7, bg-centered, gate, NMS).
2. for each noun, find the patches where that noun fires strongest (its
   support region), pool those patches into a LOCAL region vector.
3. score adjective candidates against the LOCAL region vector (not the whole
   image). A true attribute of the object scores high on ITS patches; a
   spurious co-occurring word (elsewhere in image) does not. This physically
   ties the adjective to the noun's pixels -> replaces POS.
Adjective candidates = word-start vocab, but ranked by local-region sim then
NMS-deduped against the noun (so we don't return a synonym of the noun).
Prints "adj noun" per subject on the 7 verified images (qualitative check).
"""
import sys, os, json
import numpy as np
import mlx.core as mx
from PIL import Image
from transformers import Qwen2VLImageProcessor
MODEL_DIR=os.environ.get("JINA_OMNI_NANO_DIR","/Volumes/vault/ai-models/jinaai/jina-embeddings-v5-omni-nano-mlx")
sys.path.insert(0, MODEL_DIR)
from utils import load_model
IMAGE_TOKEN_ID=128259
HERE=os.path.dirname(__file__)
TESTS=["cat.jpg","livingroom_tv.jpg","cat_shoe.jpg","tennis.jpg","skate.jpg","zebra.jpg","bus.jpg"]
def _bidi_mask(am,ref): return mx.where(am==0,-1e9,0.0)[:,None,None,:].astype(ref.dtype)

def forward(model, proc, pil):
    out=proc(images=pil, return_tensors="np")
    grid=np.array(out["image_grid_thw"]).astype(np.int32)
    t,h,w=[int(x) for x in grid[0]]; n=(t*h*w)//4
    pv=mx.array(out["pixel_values"].astype(np.float32)); gthw=mx.array(grid)
    ids=mx.array([[IMAGE_TOKEN_ID]*n]); am=mx.array([[1]*n])
    vf=model.merger(model.vision_tower(pv,gthw))
    emb=model.language_model.embed_tokens(ids)
    sm=ids==model.config.image_token_id
    pos=mx.array(np.where(mx.flatten(mx.broadcast_to(sm[...,None],emb.shape)))[0],mx.uint32)
    fe=mx.flatten(emb); fe[pos]=mx.flatten(vf); emb=mx.reshape(fe,emb.shape)
    hidden=model.language_model(inputs_embeds=emb, mask=_bidi_mask(am,model.language_model.layers[0].input_layernorm.weight))
    mx.eval(hidden)
    Hn=np.array(hidden[0].astype(mx.float32).tolist())
    P=Hn/(np.linalg.norm(Hn,axis=1,keepdims=True)+1e-9)
    g=Hn[-1]/(np.linalg.norm(Hn[-1])+1e-9)
    return P.astype(np.float32), g.astype(np.float32)

def main():
    m=load_model(MODEL_DIR); m.switch_task("retrieval")
    proc=Qwen2VLImageProcessor.from_pretrained(MODEL_DIR)
    d=np.load(os.path.join(HERE,"full_vocab_emb.npz"),allow_pickle=True)
    E=d["E"].astype(np.float32); strings=list(d["strings"])
    En=E/np.linalg.norm(E,axis=1,keepdims=True)
    tj=json.load(open(os.path.join(MODEL_DIR,"tokenizer.json")))
    id2p={v:k for k,v in tj["model"]["vocab"].items()}
    gate=np.zeros(len(strings),bool)
    for tid in range(len(strings)):
        p=id2p.get(tid,"")
        if p.startswith("Ġ"):
            s=p[1:]
            if len(s)>=3 and s.isalpha() and s.isascii() and s.islower(): gate[tid]=True
    z=np.load(os.path.join(HERE,"bg_patch.npz")); MUp=z["mup"]; MUg=z["mug"]
    gidx=np.where(gate)[0]

    def nms(order,keep,tau=0.6,avoid=None):
        kept=[];kv=[] if avoid is None else [En[a] for a in avoid]
        base=len(kv)
        for j in order:
            v=En[j]
            if any(float(v@k)>=tau for k in kv):continue
            kept.append(j);kv.append(v)
            if len(kept)>=keep:break
        return kept

    base=os.path.join(HERE,"testimg")
    for name in TESTS:
        P,g=forward(m.model,proc,Image.open(os.path.join(base,name)).convert("RGB"))
        sim=P@E.T                      # [n_patch, V]
        cen=0.7*(sim.max(0)-MUp)+0.3*((g@E.T)-MUg)
        cen2=cen.copy(); cen2[~gate]=-1e9
        nouns=nms(np.argsort(-cen2)[:300], keep=3)
        print(f"\n== {name}")
        for nid in nouns:
            nw=strings[nid].strip().lower()
            # noun support region: top patches for this noun
            ps=sim[:,nid]; topp=np.argsort(-ps)[:max(3,len(ps)//10)]
            local=P[topp].mean(0); local/=np.linalg.norm(local)+1e-9
            # score adjective candidates on LOCAL region (gate only)
            lsim=local@E.T
            lsim_g=lsim.copy(); lsim_g[~gate]=-1e9
            # avoid returning the noun or its near-duplicates
            adj=nms(np.argsort(-lsim_g)[:200], keep=3, tau=0.55, avoid=[nid])
            adjw=[strings[a].strip().lower() for a in adj]
            print(f"   {nw:14s} <- local-adj: {', '.join(adjw)}")

if __name__=="__main__":
    main()
