"""
Patch-level tagger (PIAA/TagCLIP idea) on v5-omni-nano-mlx.
No training / no wordnet / no regex-dict. Uses tokenizer word-start gate +
background centering, same as tagger_wordstart.py, but scores at PATCH level.

How we get patch features (aligned 768 space, no model surgery):
- encode_image injects merged vision features at image-token positions and runs
  the bidirectional text model. The text model returns hidden [1, L, 768] for
  ALL positions. We re-run the same forward but keep the full hidden sequence,
  then take the rows at the image-token positions = per-patch contextualized
  features in the SAME space as text labels.
- score each patch vs every label; MAX-POOL over patches per label (PIAA:
  small/non-salient objects fire on a few patches even if global pooled misses).
- calibrate by background prior (patch-maxpool over 50 bg imgs), word-start gate,
  embedding NMS. Fuse with global pooled score (PIAA PAA: a*patch + (1-a)*cls).
"""
import sys, os, glob, json
import numpy as np
import mlx.core as mx
from PIL import Image
from transformers import Qwen2VLImageProcessor

MODEL_DIR=os.environ.get("JINA_OMNI_NANO_DIR","/Volumes/vault/ai-models/jinaai/jina-embeddings-v5-omni-nano-mlx")
sys.path.insert(0, MODEL_DIR)
from utils import load_model
IMAGE_TOKEN_ID=128259
HERE=os.path.dirname(__file__)

GT={"cat.jpg":{"cat","couch","remote","television","sofa","kitten","plush"},
    "livingroom_tv.jpg":{"television","chair","table","couch","furniture","room","fireplace","lamp","dresser"},
    "cat_shoe.jpg":{"cat","shoe","sneaker","kitten","sock"},
    "zebra.jpg":{"bear","grass","zebra"},
    "bus.jpg":{"sign","pole","road","tree"},
    "tennis.jpg":{"snow","ski","mountain","skier","snowboard","hill","sled"},
    "skate.jpg":{"kitchen","refrigerator","oven","stove","cabinet","cupboard","dresser"}}

def _bidi_mask(am, ref):
    if am is None: return None
    pad=mx.where(am==0,-1e9,0.0)[:,None,None,:]
    return pad.astype(ref.dtype)

def patch_and_global(m, proc, path):
    """Return (patch_feats [P,768] normed, global_vec [768] normed)."""
    model=m.model
    out=proc(images=Image.open(path).convert("RGB"), return_tensors="np")
    grid=np.array(out["image_grid_thw"]).astype(np.int32)
    t,h,w=[int(x) for x in grid[0]]; n=(t*h*w)//4
    pv=mx.array(out["pixel_values"].astype(np.float32))
    gthw=mx.array(grid)
    input_ids=mx.array([[IMAGE_TOKEN_ID]*n]); am=mx.array([[1]*n])
    # replicate encode_image internals but keep full hidden
    vh=model.vision_tower(pv, gthw)
    vf=model.merger(vh)
    emb=model.language_model.embed_tokens(input_ids)
    smask=input_ids==model.config.image_token_id
    se=mx.broadcast_to(smask[...,None], emb.shape)
    pos=mx.array(np.where(mx.flatten(se))[0], mx.uint32)
    fe=mx.flatten(emb); fe[pos]=mx.flatten(vf); emb=mx.reshape(fe, emb.shape)
    mask=_bidi_mask(am, model.language_model.layers[0].input_layernorm.weight)
    hidden=model.language_model(inputs_embeds=emb, mask=mask)  # [1,L,768]
    mx.eval(hidden)
    H=np.array(hidden[0].astype(mx.float32).tolist())  # [L,768]
    # all positions are image tokens here (we built seq = only image tokens)
    P=H / (np.linalg.norm(H,axis=1,keepdims=True)+1e-9)
    # global = last-token pooled (matches encode_image)
    g=H[-1]; g=g/(np.linalg.norm(g)+1e-9)
    return P.astype(np.float32), g.astype(np.float32)

def main():
    m=load_model(MODEL_DIR); m.switch_task("retrieval")
    proc=Qwen2VLImageProcessor.from_pretrained(MODEL_DIR)
    d=np.load(os.path.join(HERE,"full_vocab_emb.npz"), allow_pickle=True)
    E=d["E"].astype(np.float32); strings=list(d["strings"])
    En=E/np.linalg.norm(E,axis=1,keepdims=True)

    # word-start gate (tokenizer-internal)
    tj=json.load(open(os.path.join(MODEL_DIR,"tokenizer.json")))
    id2piece={v:k for k,v in tj["model"]["vocab"].items()}
    gate=np.zeros(len(strings),dtype=bool)
    for tid in range(len(strings)):
        p=id2piece.get(tid,"")
        if p.startswith("Ġ"):
            s=p[1:]
            if len(s)>=3 and s.isalpha() and s.isascii() and s.islower():
                gate[tid]=True

    def score_img(P, g):
        # patch: max over patches of cos(patch,label)
        sp=(P@E.T).max(axis=0)      # [V]
        sg=g@E.T                    # [V]
        return sp, sg

    # background prior (patch-maxpool + global)
    bgp=os.path.join(HERE,"bg_patch.npz")
    bgfiles=sorted(glob.glob(os.path.join(HERE,"bg","*.jpg")))
    if os.path.exists(bgp):
        z=np.load(bgp); MUp=z["mup"]; MUg=z["mug"]
    else:
        SP=[]; SG=[]
        for p in bgfiles:
            P,g=patch_and_global(m,proc,p); sp,sg=score_img(P,g); SP.append(sp); SG.append(sg)
        MUp=np.mean(SP,axis=0); MUg=np.mean(SG,axis=0)
        np.savez(bgp, mup=MUp, mug=MUg)
    print("bg patch prior done")

    def nms(order, keep=8, tau=0.6):
        kept=[]; kv=[]
        for j in order:
            v=En[j]
            if any(float(v@k)>=tau for k in kv): continue
            kept.append(j); kv.append(v)
            if len(kept)>=keep: break
        return kept

    base=os.path.join(HERE,"testimg"); names=list(GT)
    for ALPHA in [1.0, 0.7]:
        hit=0
        print(f"\n########## ALPHA(patch weight)={ALPHA} ##########")
        for name in names:
            P,g=patch_and_global(m,proc,os.path.join(base,name))
            sp,sg=score_img(P,g)
            cen = ALPHA*(sp-MUp) + (1-ALPHA)*(sg-MUg)
            cen=cen.copy(); cen[~gate]=-1e9
            order=np.argsort(-cen)[:300]; kept=nms(order)
            tags=[strings[j].strip().lower() for j in kept]
            h=len(GT[name]&set(tags[:5]))>0; hit+=h
            print(f"  {name:20s} {'HIT ' if h else 'miss'} {', '.join(tags)}")
        print(f"  [hit@5]={hit}/{len(names)}")

if __name__=="__main__":
    main()
