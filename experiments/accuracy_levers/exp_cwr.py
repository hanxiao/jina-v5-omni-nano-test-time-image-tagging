"""
Lever 1: CWR = response-guided crop + re-encode (TagCLIP CWR, done right).
Targets the ONLY failure mode (small/non-salient objects). Model-agnostic:
feeds a better-framed image to the same good model.

Per image:
1. baseline patch scoring (layer12, max-pool) -> candidate labels (top-Kc).
2. reshape patch scores to the (h,w) patch grid; for the union of top candidates'
   peak responses, compute a bounding box of high-response patches.
3. crop that bbox from the ORIGINAL image (with margin), re-encode via the model
   (global pooled = last-token), score vs labels -> S_crop.
4. fuse: S = base + beta * S_crop  (crop rescues small objects).
We test a simpler, robust variant: multi-crop over a coarse grid (2x2 + center)
and take per-label MAX of global scores across crops (object appears large in
whichever crop contains it). This is object-agnostic and needs no bbox math.
Benchmark COCO-150.
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
    P=H/(np.linalg.norm(H,axis=1,keepdims=True)+1e-9)
    g=H[-1]/(np.linalg.norm(H[-1])+1e-9)
    return P.astype(np.float32), g.astype(np.float32), (t,h//2,w//2)  # merged grid

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
        return np.mean([Y[i,idx[i]].sum()/k for i in range(len(Y))]), np.mean([Y[i,idx[i]].sum()/max(Y[i].sum(),1) for i in range(len(Y))])
    aps=[]
    for c in range(len(cats)):
        if Y[:,c].sum()==0: continue
        o=np.argsort(-Sc[:,c]); yt=Y[o,c]; tp=np.cumsum(yt); aps.append((tp/(np.arange(len(yt))+1)*yt).sum()/yt.sum())
    p1,_=patk(1);p3,_=patk(3);_,r5=patk(5); return p1,p3,r5,float(np.mean(aps))

def crops(img):
    W,H=img.size; out=[]
    # 2x2 quadrants + center 0.6
    boxes=[(0,0,W//2,H//2),(W//2,0,W,H//2),(0,H//2,W//2,H),(W//2,H//2,W,H),
           (int(W*0.2),int(H*0.2),int(W*0.8),int(H*0.8))]
    for b in boxes: out.append(img.crop(b))
    return out

def main():
    m=load_model(MODEL_DIR); m.switch_task("retrieval")
    proc=Qwen2VLImageProcessor.from_pretrained(MODEL_DIR)
    ann=json.load(open(os.path.join(HERE,"annotations/instances_val2017.json")))
    cats=[c["name"] for c in ann["categories"]]; C={c:i for i,c in enumerate(cats)}
    E=enc_labels(m,cats)
    gt=json.load(open(os.path.join(HERE,"eval_gt.json"))); ids=list(gt)
    Y=np.zeros((len(ids),len(cats)),bool)
    for i,iid in enumerate(ids):
        for c in gt[iid]: Y[i,C[c]]=True

    cache=os.path.join(HERE,"cwr_scores.npz")
    if os.path.exists(cache):
        z=np.load(cache); SPb=z["spb"];SGb=z["sgb"];SPc=z["spc"];SGc=z["sgc"]
    else:
        SPb=np.zeros((len(ids),80));SGb=np.zeros((len(ids),80))
        SPc=np.zeros((len(ids),80));SGc=np.zeros((len(ids),80))
        for i,iid in enumerate(ids):
            img=Image.open(os.path.join(HERE,"eval_imgs",f"{iid}.jpg")).convert("RGB")
            P,g,_=feats(m.model,proc,img)
            SPb[i]=(P@E.T).max(0); SGb[i]=g@E.T
            cP=[]; cG=[]
            for cr in crops(img):
                Pc,gc,_=feats(m.model,proc,cr)
                cP.append((Pc@E.T).max(0)); cG.append(gc@E.T)
            SPc[i]=np.max(cP,0); SGc[i]=np.max(cG,0)  # per-label max across crops
            if i%25==0: print(f"  {i}/{len(ids)}")
        np.savez(cache, spb=SPb,sgb=SGb,spc=SPc,sgc=SGc)
    print("scores cached\n")

    base=0.7*SPb+0.3*SGb
    print(f"{'variant':22s} P@1   P@3   R@5   mAP")
    print(f"{'baseline patch':22s}"+" %.3f %.3f %.3f %.3f"%metrics(base,Y,cats))
    for b in [0.3,0.5,0.8,1.0]:
        S=base + b*np.maximum(SPc,SGc)
        print(f"{'base+%.1f*cropmax'%b:22s}"+" %.3f %.3f %.3f %.3f"%metrics(S,Y,cats))
    # crop patch-max fused too
    for b in [0.3,0.5]:
        S=base + b*(0.7*SPc+0.3*SGc)
        print(f"{'base+%.1f*cropfuse'%b:22s}"+" %.3f %.3f %.3f %.3f"%metrics(S,Y,cats))
    # elementwise max of base and crop
    S=np.maximum(base, 0.7*SPc+0.3*SGc)
    print(f"{'max(base,crop)':22s}"+" %.3f %.3f %.3f %.3f"%metrics(S,Y,cats))

if __name__=="__main__":
    main()
