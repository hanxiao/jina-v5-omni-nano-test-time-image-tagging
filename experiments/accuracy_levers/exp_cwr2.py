"""CWR push: finer crop grids (latency free). 3x3 + 2x2 + center + full, per-label
max across all crops fused with base. Find the crop scheme that maxes mAP.
Reuses feats()/enc_labels from exp_cwr via import."""
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
def metrics(S,Y,cats):
    Sc=S-S.mean(0,keepdims=True)
    def patk(k):
        idx=np.argsort(-Sc,1)[:,:k]; return np.mean([Y[i,idx[i]].sum()/k for i in range(len(Y))]), np.mean([Y[i,idx[i]].sum()/max(Y[i].sum(),1) for i in range(len(Y))])
    aps=[]
    for c in range(len(cats)):
        if Y[:,c].sum()==0: continue
        o=np.argsort(-Sc[:,c]); yt=Y[o,c]; tp=np.cumsum(yt); aps.append((tp/(np.arange(len(yt))+1)*yt).sum()/yt.sum())
    p1,_=patk(1);p3,_=patk(3);_,r5=patk(5); return p1,p3,r5,float(np.mean(aps))
def grid_crops(img, gx, gy, overlap=0.15):
    W,H=img.size; out=[]; cw=W/gx; ch=H/gy; ox=cw*overlap; oy=ch*overlap
    for j in range(gy):
        for i in range(gx):
            l=max(0,int(i*cw-ox)); t=max(0,int(j*ch-oy)); r=min(W,int((i+1)*cw+ox)); b=min(H,int((j+1)*ch+oy))
            out.append(img.crop((l,t,r,b)))
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
    # base from cwr cache
    z=np.load(os.path.join(HERE,"cwr_scores.npz")); base=0.7*z["spb"]+0.3*z["sgb"]
    cache=os.path.join(HERE,"cwr2_scores.npz")
    if os.path.exists(cache):
        d=np.load(cache); SP33=d["sp33"];SG33=d["sg33"]
    else:
        SP33=np.zeros((len(ids),80));SG33=np.zeros((len(ids),80))
        for i,iid in enumerate(ids):
            img=Image.open(os.path.join(HERE,"eval_imgs",f"{iid}.jpg")).convert("RGB")
            cP=[];cG=[]
            for cr in grid_crops(img,3,3)+grid_crops(img,2,2)+[img.crop((int(img.size[0]*0.25),int(img.size[1]*0.25),int(img.size[0]*0.75),int(img.size[1]*0.75)))]:
                Pc,gc=feats(m.model,proc,cr); cP.append((Pc@E.T).max(0)); cG.append(gc@E.T)
            SP33[i]=np.max(cP,0); SG33[i]=np.max(cG,0)
            if i%25==0: print(f"  {i}/{len(ids)}")
        np.savez(cache, sp33=SP33, sg33=SG33)
    print("cached\n")
    print(f"{'variant':24s} P@1   P@3   R@5   mAP")
    print(f"{'baseline':24s}"+" %.3f %.3f %.3f %.3f"%metrics(base,Y,cats))
    for b in [0.5,0.8,1.0,1.3]:
        S=base+b*np.maximum(SP33,SG33)
        print(f"{'base+%.1f*grid33max'%b:24s}"+" %.3f %.3f %.3f %.3f"%metrics(S,Y,cats))
if __name__=="__main__":
    main()
