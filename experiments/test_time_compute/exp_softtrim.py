"""
Soft-trim CWR aggregation (statistic stolen from Concept-Guided Bayesian).
Our CWR takes per-label MAX across crops -> sensitive to a single outlier crop.
Replace with robust aggregation over the crop score distribution:
- median + MAD based soft-trim weights w_j = sigmoid(-log((1-rho)/rho)*(S_j-m)/MAD)
  then weighted mean; rho = est contamination.
- also test: top-q quantile mean, trimmed mean, logsumexp (soft-max).
Needs per-crop per-label scores. We only cached per-label MAX (spc/sgc), not per
-crop. So recompute per-crop scores for the 150 eval imgs (14 crops each) once.
Compare aggregations at the fusion S = base + b*agg.
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
TEMPLATES=["a photo of a {}.","a photo of the {}.","a picture of a {}."]
def _bidi_mask(am,ref): return mx.where(am==0,-1e9,0.0)[:,None,None,:].astype(ref.dtype)
def feats(model, proc, pil):
    out=proc(images=pil, return_tensors="np"); grid=np.array(out["image_grid_thw"]).astype(np.int32)
    t,h,w=[int(x) for x in grid[0]]; n=(t*h*w)//4
    pv=mx.array(out["pixel_values"].astype(np.float32)); gthw=mx.array(grid)
    ids=mx.array([[IMAGE_TOKEN_ID]*n]); am=mx.array([[1]*n])
    vf=model.merger(model.vision_tower(pv,gthw)); emb=model.language_model.embed_tokens(ids)
    sm=ids==model.config.image_token_id
    pos=mx.array(np.where(mx.flatten(mx.broadcast_to(sm[...,None],emb.shape)))[0],mx.uint32)
    fe=mx.flatten(emb); fe[pos]=mx.flatten(vf); emb=mx.reshape(fe,emb.shape)
    lm=model.language_model; hidden=lm(inputs_embeds=emb, mask=_bidi_mask(am,lm.layers[0].input_layernorm.weight)); mx.eval(hidden)
    H=np.array(hidden[0].astype(mx.float32).tolist())
    P=H/(np.linalg.norm(H,axis=1,keepdims=True)+1e-9); g=H[-1]/(np.linalg.norm(H[-1])+1e-9)
    return P.astype(np.float32), g.astype(np.float32)
def enc_labels(m,cats):
    mats=[]
    for tpl in TEMPLATES:
        e=m.model.encode([tpl.format(c) for c in cats], m.tokenizer, task_type="retrieval.passage").astype(mx.float32)
        mats.append(np.array((e/mx.linalg.norm(e,axis=1,keepdims=True)).tolist()))
    E=np.mean(mats,0); return (E/np.linalg.norm(E,axis=1,keepdims=True)).astype(np.float32)
def metrics(S,Y,K=80):
    Sc=S-S.mean(0,keepdims=True)
    def patk(k):
        idx=np.argsort(-Sc,1)[:,:k]; return np.mean([Y[i,idx[i]].sum()/k for i in range(len(Y))]), np.mean([Y[i,idx[i]].sum()/max(Y[i].sum(),1) for i in range(len(Y))])
    aps=[]
    for c in range(K):
        if Y[:,c].sum()==0: continue
        o=np.argsort(-Sc[:,c]); yt=Y[o,c]; tp=np.cumsum(yt); aps.append((tp/(np.arange(len(yt))+1)*yt).sum()/yt.sum())
    p1,_=patk(1);p3,_=patk(3);_,r5=patk(5); return p1,p3,r5,float(np.mean(aps))
def grid_crops(img,gx,gy,ov=0.15):
    W,H=img.size; out=[]; cw=W/gx; ch=H/gy; ox=cw*ov; oy=ch*ov
    for j in range(gy):
        for i in range(gx):
            out.append(img.crop((max(0,int(i*cw-ox)),max(0,int(j*ch-oy)),min(W,int((i+1)*cw+ox)),min(H,int((j+1)*ch+oy)))))
    return out
def main():
    m=load_model(MODEL_DIR); m.switch_task("retrieval")
    proc=Qwen2VLImageProcessor.from_pretrained(MODEL_DIR)
    ann=json.load(open(os.path.join(HERE,"annotations/instances_val2017.json")))
    cats=[c["name"] for c in ann["categories"]]; C={c:i for i,c in enumerate(cats)}
    E=enc_labels(m,cats); gt=json.load(open(os.path.join(HERE,"eval_gt.json"))); ids=list(gt)
    Y=np.zeros((len(ids),80),bool)
    for i,iid in enumerate(ids):
        for c in gt[iid]: Y[i,C[c]]=True
    z=np.load(os.path.join(HERE,"cwr_scores.npz")); base=0.7*z["spb"]+0.3*z["sgb"]
    # per-crop scores cache: [n, ncrop, 80] using global-pooled per crop (fast)
    pc=os.path.join(HERE,"percrop_scores.npy")
    if os.path.exists(pc):
        CR=np.load(pc)
    else:
        allc=[]
        for k,iid in enumerate(ids):
            img=Image.open(os.path.join(HERE,"eval_imgs",f"{iid}.jpg")).convert("RGB")
            crs=grid_crops(img,3,3)+grid_crops(img,2,2)+[img.crop((int(img.size[0]*.25),int(img.size[1]*.25),int(img.size[0]*.75),int(img.size[1]*.75)))]
            sc=[]
            for cr in crs:
                Pc,gc=feats(m.model,proc,cr); sc.append(np.maximum((Pc@E.T).max(0), gc@E.T))
            allc.append(np.array(sc))
            if k%25==0: print(f"  {k}/{len(ids)}")
        CR=np.array(allc); np.save(pc,CR)  # [n,14,80]
    print("CR",CR.shape)
    def agg_max(): return CR.max(1)
    def agg_mean(): return CR.mean(1)
    def agg_trim(q): 
        s=np.sort(CR,1); k=int(CR.shape[1]*q); return s[:,k:,:].mean(1)
    def agg_lse(t): return t*np.log(np.exp(CR/t).mean(1))
    def agg_softtrim():
        m_=np.median(CR,1,keepdims=True); mad=np.median(np.abs(CR-m_),1,keepdims=True)+1e-6
        rho=0.3; w=1/(1+np.exp(np.log((1-rho)/rho)*(-(CR-m_)/mad)))  # up-weight high
        return (w*CR).sum(1)/w.sum(1)
    aggs={"max(base)":agg_max,"mean":agg_mean,"trim0.5":lambda:agg_trim(0.5),
          "trim0.7":lambda:agg_trim(0.7),"lse0.1":lambda:agg_lse(0.1),"softtrim":agg_softtrim}
    print(f"{'agg':14s}{'b':5s} P@1   P@3   R@5   mAP")
    for name,fn in aggs.items():
        A=fn()
        for b in [0.8,1.3]:
            r=metrics(base+b*A,Y); print(f"{name:14s}{b:<5.1f} {r[0]:.3f} {r[1]:.3f} {r[2]:.3f} {r[3]:.3f}")
if __name__=="__main__":
    main()
