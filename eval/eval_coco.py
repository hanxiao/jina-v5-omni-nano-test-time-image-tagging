"""Proper multi-label eval on 150 COCO images, 80-category closed set.
Labels = 80 COCO category names, encoded via encode_text (prompt ensemble).
Compare pooled vs patch(a=0.7), with bg-centering. Metrics: P@1,P@3,R@5, mAP.
This is the real benchmark; every tweak measured here, not on 7 images.
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
TEMPLATES=["a photo of a {}.","a photo of the {}.","a picture of a {}."]

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

def enc_labels(m, cats):
    mats=[]
    for tpl in TEMPLATES:
        e=m.model.encode([tpl.format(c) for c in cats], m.tokenizer, task_type="retrieval.passage").astype(mx.float32)
        mats.append(np.array((e/mx.linalg.norm(e,axis=1,keepdims=True)).tolist()))
    E=np.mean(mats,0); return (E/np.linalg.norm(E,axis=1,keepdims=True)).astype(np.float32)

def main():
    m=load_model(MODEL_DIR); m.switch_task("retrieval")
    proc=Qwen2VLImageProcessor.from_pretrained(MODEL_DIR)
    ann=json.load(open(os.path.join(HERE,"annotations/instances_val2017.json")))
    cats=[c["name"] for c in ann["categories"]]; C={c:i for i,c in enumerate(cats)}
    E=enc_labels(m,cats)  # [80,768]
    gt=json.load(open(os.path.join(HERE,"eval_gt.json")))
    ids=list(gt)

    # forward all images once, cache patch-max sim and global sim to the 80 labels
    cache=os.path.join(HERE,"eval_sims.npz")
    if os.path.exists(cache):
        z=np.load(cache); SP=z["sp"]; SG=z["sg"]; ids=list(z["ids"])
    else:
        SP=[];SG=[]
        for k,iid in enumerate(ids):
            P,g=forward(m.model,proc,Image.open(os.path.join(HERE,"eval_imgs",f"{iid}.jpg")).convert("RGB"))
            SP.append((P@E.T).max(0)); SG.append(g@E.T)
            if k%25==0: print(f"  {k}/{len(ids)}")
        SP=np.array(SP); SG=np.array(SG)
        np.savez(cache, sp=SP, sg=SG, ids=np.array(ids))
    Y=np.zeros((len(ids),len(cats)),bool)
    for i,iid in enumerate(ids):
        for c in gt[iid]: Y[i,C[c]]=True

    def metrics(S, tag):
        # bg-centering = per-label mean over all eval images (proxy prior)
        Sc=S-S.mean(0,keepdims=True)
        # P@k, R@k
        def patk(k):
            idx=np.argsort(-Sc,1)[:,:k]
            p=np.mean([Y[i,idx[i]].sum()/k for i in range(len(ids))])
            r=np.mean([Y[i,idx[i]].sum()/max(Y[i].sum(),1) for i in range(len(ids))])
            return p,r
        # mAP per class
        aps=[]
        for c in range(len(cats)):
            if Y[:,c].sum()==0: continue
            order=np.argsort(-Sc[:,c]); yt=Y[order,c]
            tp=np.cumsum(yt); prec=tp/(np.arange(len(yt))+1); ap=(prec*yt).sum()/yt.sum()
            aps.append(ap)
        p1,r1=patk(1); p3,r3=patk(3); p5,r5=patk(5)
        print(f"  {tag:16s} P@1={p1:.3f} P@3={p3:.3f} R@5={r5:.3f} mAP={np.mean(aps):.3f}")

    print(f"eval {len(ids)} imgs, 80 cats:")
    metrics(SG,"global(pooled)")
    metrics(SP,"patch(max)")
    for a in [0.5,0.7,0.9]:
        metrics(a*SP+(1-a)*SG, f"fuse a={a}")

if __name__=="__main__":
    main()
