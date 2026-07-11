"""Measure training-free hacks on the REAL 150-img/80-cat benchmark.
Hacks: baseline patch(a=0.7); anchor-prefix; softpool temperatures; register-append.
All zero-training. Reports mAP/P@1/P@3/R@5 so differences are real, not 7-img noise.
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

def forward(model, proc, pil, prefix=None):
    out=proc(images=pil, return_tensors="np")
    grid=np.array(out["image_grid_thw"]).astype(np.int32)
    t,h,w=[int(x) for x in grid[0]]; n=(t*h*w)//4
    pv=mx.array(out["pixel_values"].astype(np.float32)); gthw=mx.array(grid)
    pre=prefix or []
    seq=pre+[IMAGE_TOKEN_ID]*n
    ids=mx.array([seq]); am=mx.array([[1]*len(seq)])
    vf=model.merger(model.vision_tower(pv,gthw))
    emb=model.language_model.embed_tokens(ids)
    sm=ids==model.config.image_token_id
    pos=mx.array(np.where(mx.flatten(mx.broadcast_to(sm[...,None],emb.shape)))[0],mx.uint32)
    fe=mx.flatten(emb); fe[pos]=mx.flatten(vf); emb=mx.reshape(fe,emb.shape)
    hidden=model.language_model(inputs_embeds=emb, mask=_bidi_mask(am,model.language_model.layers[0].input_layernorm.weight))
    mx.eval(hidden)
    Hn=np.array(hidden[0].astype(mx.float32).tolist())
    img=Hn[len(pre):len(pre)+n]
    P=img/(np.linalg.norm(img,axis=1,keepdims=True)+1e-9)
    g=Hn[-1]/(np.linalg.norm(Hn[-1])+1e-9)
    return P.astype(np.float32), g.astype(np.float32)

def enc_labels(m,cats):
    mats=[]
    for tpl in TEMPLATES:
        e=m.model.encode([tpl.format(c) for c in cats], m.tokenizer, task_type="retrieval.passage").astype(mx.float32)
        mats.append(np.array((e/mx.linalg.norm(e,axis=1,keepdims=True)).tolist()))
    E=np.mean(mats,0); return (E/np.linalg.norm(E,axis=1,keepdims=True)).astype(np.float32)

def met(S,Y,cats,tag):
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
    print(f"  {tag:20s} P@1={p1:.3f} P@3={p3:.3f} R@5={r5:.3f} mAP={np.mean(aps):.3f}")

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
    anchor=m.tokenizer.encode("a photo of").ids

    SPb=[];SGb=[];SPa=[];SGa=[]
    SP_soft={0.05:[],0.1:[],0.2:[]}
    for k,iid in enumerate(ids):
        pil=Image.open(os.path.join(HERE,"eval_imgs",f"{iid}.jpg")).convert("RGB")
        P,g=forward(m.model,proc,pil)
        SPb.append((P@E.T).max(0)); SGb.append(g@E.T)
        sim=P@E.T
        for T in SP_soft:
            wexp=np.exp((sim-sim.max(0,keepdims=True))/T); wexp/=wexp.sum(0,keepdims=True)
            SP_soft[T].append((wexp*sim).sum(0))
        Pa,ga=forward(m.model,proc,pil,prefix=anchor)
        SPa.append((Pa@E.T).max(0)); SGa.append(ga@E.T)
        if k%25==0: print(f"  {k}/{len(ids)}")
    SPb=np.array(SPb);SGb=np.array(SGb);SPa=np.array(SPa);SGa=np.array(SGa)
    print(f"\nbenchmark {len(ids)} imgs:")
    met(0.7*SPb+0.3*SGb,Y,cats,"H0 patch a=0.7")
    met(0.7*SPa+0.3*SGa,Y,cats,"H1 anchor-prefix")
    for T in SP_soft:
        met(0.7*np.array(SP_soft[T])+0.3*SGb,Y,cats,f"H2 softpool T={T}")

if __name__=="__main__":
    main()
