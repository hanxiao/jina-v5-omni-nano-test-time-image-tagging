"""
Lever 3: whitening the aligned space to reduce residual modality-gap /
anisotropy before scoring. Model-agnostic (operates on embeddings).

Estimate a whitening transform W from BACKGROUND patch features (50 imgs, layer12):
  - mean m, covariance C over bg patches; W = C^{-1/2} (ZCA) or PCA-whiten.
Apply to BOTH image patch feats and label embeddings, then cosine.
Compare: baseline (no whiten) vs mean-only-center vs full whiten, at layer 12.
Also try shrinkage on C (Ledoit-Wolf style lambda) since 768-dim from ~15k bg patches.
Benchmark COCO-150.
"""
import sys, os, json, glob
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

def patchfeat(model, proc, pil):
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
    hidden=lm(inputs_embeds=emb, mask=_bidi_mask(am,lm.layers[0].input_layernorm.weight))
    mx.eval(hidden)
    H=np.array(hidden[0].astype(mx.float32).tolist())
    return H.astype(np.float32)  # raw (un-normed) patch feats [n,768]

def enc_labels(m,cats):
    mats=[]
    for tpl in TEMPLATES:
        e=m.model.encode([tpl.format(c) for c in cats], m.tokenizer, task_type="retrieval.passage").astype(mx.float32)
        mats.append(np.array(e.tolist()))
    return np.mean(mats,0).astype(np.float32)  # raw (un-normed) label vecs

def metrics(S,Y,cats):
    Sc=S-S.mean(0,keepdims=True)
    def patk(k):
        idx=np.argsort(-Sc,1)[:,:k]
        p=np.mean([Y[i,idx[i]].sum()/k for i in range(len(Y))]); r=np.mean([Y[i,idx[i]].sum()/max(Y[i].sum(),1) for i in range(len(Y))]); return p,r
    aps=[]
    for c in range(len(cats)):
        if Y[:,c].sum()==0: continue
        o=np.argsort(-Sc[:,c]); yt=Y[o,c]; tp=np.cumsum(yt); aps.append((tp/(np.arange(len(yt))+1)*yt).sum()/yt.sum())
    p1,_=patk(1);p3,_=patk(3);_,r5=patk(5); return p1,p3,r5,float(np.mean(aps))

def main():
    m=load_model(MODEL_DIR); m.switch_task("retrieval")
    proc=Qwen2VLImageProcessor.from_pretrained(MODEL_DIR)
    ann=json.load(open(os.path.join(HERE,"annotations/instances_val2017.json")))
    cats=[c["name"] for c in ann["categories"]]; C={c:i for i,c in enumerate(cats)}
    Eraw=enc_labels(m,cats)
    gt=json.load(open(os.path.join(HERE,"eval_gt.json"))); ids=list(gt)
    Y=np.zeros((len(ids),len(cats)),bool)
    for i,iid in enumerate(ids):
        for c in gt[iid]: Y[i,C[c]]=True

    # bg patch stats
    bgcache=os.path.join(HERE,"bg_rawpatch.npy")
    if os.path.exists(bgcache):
        BG=np.load(bgcache)
    else:
        allp=[]
        for p in sorted(glob.glob(os.path.join(HERE,"bg","*.jpg"))):
            allp.append(patchfeat(m.model,proc,Image.open(p).convert("RGB")))
        BG=np.vstack(allp); np.save(bgcache, BG)
    print("bg patches:", BG.shape)
    mean=BG.mean(0)
    Cov=np.cov((BG-mean).T)  # [768,768]

    # eval image raw patch feats cache
    ic=os.path.join(HERE,"eval_rawpatch.npz")
    if os.path.exists(ic):
        z=np.load(ic,allow_pickle=True); IP=z["ip"]
    else:
        IP=[patchfeat(m.model,proc,Image.open(os.path.join(HERE,"eval_imgs",f"{iid}.jpg")).convert("RGB")) for iid in ids]
        np.savez(ic, ip=np.array(IP,dtype=object))
    print("eval feats cached\n")

    def whiten_mat(lam):
        Cs=(1-lam)*Cov+lam*np.eye(768)*np.trace(Cov)/768
        u,s,_=np.linalg.svd(Cs)
        W=u@np.diag(1.0/np.sqrt(s+1e-6))@u.T
        return W.astype(np.float32)

    def run(mode, lam=0.1):
        if mode=="whiten":
            W=whiten_mat(lam)
            El=(Eraw-mean)@W; El/=np.linalg.norm(El,axis=1,keepdims=True)+1e-9
        else:
            El=Eraw/ (np.linalg.norm(Eraw,axis=1,keepdims=True)+1e-9)
        SP=np.zeros((len(ids),len(cats))); SG=np.zeros((len(ids),len(cats)))
        for i in range(len(ids)):
            P=IP[i].astype(np.float32)
            if mode=="whiten":
                Pp=(P-mean)@W
            elif mode=="center":
                Pp=P-mean
            else:
                Pp=P
            Pn=Pp/(np.linalg.norm(Pp,axis=1,keepdims=True)+1e-9)
            g=Pn[-1]
            sim=Pn@El.T
            SP[i]=sim.max(0); SG[i]=g@El.T
        best=None
        for a in [0.5,0.7,1.0]:
            S=a*SP+(1-a)*SG; r=metrics(S,Y,cats)
            if best is None or r[3]>best[0][3]: best=(r,a)
        return best

    print(f"{'mode':12s} a   P@1   P@3   R@5   mAP")
    for mode in ["none","center"]:
        (r,a)=run(mode); print(f"{mode:12s} {a} {r[0]:.3f} {r[1]:.3f} {r[2]:.3f} {r[3]:.3f}")
    for lam in [0.05,0.1,0.3,0.6]:
        (r,a)=run("whiten",lam); print(f"whiten l={lam:<4} {a} {r[0]:.3f} {r[1]:.3f} {r[2]:.3f} {r[3]:.3f}")

if __name__=="__main__":
    main()
