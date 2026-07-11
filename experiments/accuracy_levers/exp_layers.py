"""
Accuracy experiment harness. Caches per-LAYER patch/global features for all 150
eval images (one forward each), then sweeps:
  - which text-tower layer to read patch features from (L0..L12, +final norm)
  - class-wise softmax vs raw cos
  - fusion alpha
against COCO-150 / 80-cat benchmark (mAP, P@1, P@3, R@5).
No training. Reuses full pipeline label matrix restricted to 80 COCO cats for eval.
"""
import sys, os, json, time
import numpy as np
import mlx.core as mx
from PIL import Image
from transformers import Qwen2VLImageProcessor
MODEL_DIR=os.environ.get("JINA_OMNI_NANO_DIR","/Volumes/vault/ai-models/jinaai/jina-embeddings-v5-omni-nano-mlx")
sys.path.insert(0, MODEL_DIR)
from utils import load_model
IMAGE_TOKEN_ID=128259
HERE=os.path.dirname(__file__)
TEMPLATES=["a photo of a {}.","a photo of the {}.","a picture of a {}."]

def _bidi_mask(am,ref): return mx.where(am==0,-1e9,0.0)[:,None,None,:].astype(ref.dtype)

def forward_all_layers(model, proc, pil):
    """Return list of per-layer patch feats [n,768] (after each block, +final norm)
       and per-layer global (last-token) vec."""
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
    lm=model.language_model
    mask=_bidi_mask(am, lm.layers[0].input_layernorm.weight)
    hs=[]
    x=emb
    for layer in lm.layers:
        x=layer(x, mask); hs.append(x)
    hs.append(lm.norm(x))  # final norm (index 12 = current baseline)
    feats=[]
    for hlv in hs:
        H=np.array(hlv[0].astype(mx.float32).tolist())
        feats.append(H.astype(np.float32))
    return feats, n

def enc_labels(m,cats):
    mats=[]
    for tpl in TEMPLATES:
        e=m.model.encode([tpl.format(c) for c in cats], m.tokenizer, task_type="retrieval.passage").astype(mx.float32)
        mats.append(np.array((e/mx.linalg.norm(e,axis=1,keepdims=True)).tolist()))
    E=np.mean(mats,0); return (E/np.linalg.norm(E,axis=1,keepdims=True)).astype(np.float32)

def metrics(S,Y,cats):
    Sc=S-S.mean(0,keepdims=True)
    def patk(k):
        idx=np.argsort(-Sc,1)[:,:k]
        p=np.mean([Y[i,idx[i]].sum()/k for i in range(len(Y))])
        r=np.mean([Y[i,idx[i]].sum()/max(Y[i].sum(),1) for i in range(len(Y))])
        return p,r
    aps=[]
    for c in range(len(cats)):
        if Y[:,c].sum()==0: continue
        o=np.argsort(-Sc[:,c]); yt=Y[o,c]; tp=np.cumsum(yt)
        aps.append((tp/(np.arange(len(yt))+1)*yt).sum()/yt.sum())
    p1,_=patk(1); p3,_=patk(3); _,r5=patk(5)
    return p1,p3,r5,float(np.mean(aps))

def main():
    m=load_model(MODEL_DIR); m.switch_task("retrieval")
    proc=Qwen2VLImageProcessor.from_pretrained(MODEL_DIR)
    ann=json.load(open(os.path.join(HERE,"annotations/instances_val2017.json")))
    cats=[c["name"] for c in ann["categories"]]; C={c:i for i,c in enumerate(cats)}
    E=enc_labels(m,cats)  # [80,768] normed
    gt=json.load(open(os.path.join(HERE,"eval_gt.json"))); ids=list(gt)
    Y=np.zeros((len(ids),len(cats)),bool)
    for i,iid in enumerate(ids):
        for c in gt[iid]: Y[i,C[c]]=True

    cache=os.path.join(HERE,"layer_feats.npz")
    nlayer=13
    if os.path.exists(cache):
        z=np.load(cache, allow_pickle=True)
        PL=z["pl"]; GL=z["gl"]  # PL: object array [img][layer]->[n,768]; GL [img,layer,768]
    else:
        PL=[]; GL=[]
        for k,iid in enumerate(ids):
            feats,n=forward_all_layers(m.model,proc,Image.open(os.path.join(HERE,"eval_imgs",f"{iid}.jpg")).convert("RGB"))
            # normalize patch + global per layer
            pl=[]; gl=[]
            for H in feats:
                Hn=H/(np.linalg.norm(H,axis=1,keepdims=True)+1e-9)
                pl.append(Hn.astype(np.float16)); gl.append((H[-1]/(np.linalg.norm(H[-1])+1e-9)).astype(np.float16))
            PL.append(pl); GL.append(np.array(gl))
            if k%25==0: print(f"  fwd {k}/{len(ids)}")
        GL=np.array(GL)  # [img,13,768]
        np.savez(cache, pl=np.array(PL,dtype=object), gl=GL)
    print("features cached. sweeping...\n")

    def score_layer(layer, use_softmax):
        SP=np.zeros((len(ids),len(cats))); SG=GL[:,layer,:]@E.T
        for i in range(len(ids)):
            P=PL[i][layer].astype(np.float32)
            sim=P@E.T  # [n,80]
            if use_softmax:
                sim=sim - sim.max(1,keepdims=True)
                w=np.exp(sim/0.01); w/=w.sum(1,keepdims=True)  # softmax over classes per patch
                sim=w
            SP[i]=sim.max(0)
        return SP, SG

    print(f"{'layer':6s}{'sm':4s}{'a':5s}  P@1   P@3   R@5   mAP")
    best=None
    for layer in [10,11,12]:  # L-2, L-1, final-norm(baseline)
        for sm in [False, True]:
            SP,SG=score_layer(layer, sm)
            for a in [0.5,0.7,1.0]:
                S=a*SP+(1-a)*SG
                p1,p3,r5,mp=metrics(S,Y,cats)
                tag=f"{layer:<6d}{'Y' if sm else 'N':<4s}{a:<5.1f}  {p1:.3f} {p3:.3f} {r5:.3f} {mp:.3f}"
                print(tag)
                if best is None or mp>best[0]: best=(mp,layer,sm,a)
    print(f"\nBEST mAP={best[0]:.3f} at layer={best[1]} softmax={best[2]} a={best[3]}")

if __name__=="__main__":
    main()
