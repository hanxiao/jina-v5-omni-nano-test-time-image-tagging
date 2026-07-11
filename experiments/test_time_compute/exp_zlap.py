"""
ZLaP transductive label propagation (CVPR24) grafted onto our pipeline.
Single model, no training, multi-label (per-class propagation, no simplex норм).

Graph nodes = {80 label text vectors} U {150 image vectors}.
- separate kNN: image<->image edges and image<->label edges (modality gap).
- normalized adjacency Ŝ = D^-1/2 (S+S^T) D^-1/2 ; closed form ŷ = (I-αŜ)^-1 y.
- label seeds y: one-hot on the 80 label nodes; propagate to image nodes.
- read propagated scores at image nodes, per class -> mAP (no argmax).
We fuse propagated score with our base score and sweep k, alpha, fusion.

Image vector choice: we test (a) global pooled g, (b) CWR-enhanced representation.
Start from cached base scores + need actual vectors: recompute image global g and
label matrix E (cheap).
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
    g=H[-1]/(np.linalg.norm(H[-1])+1e-9)
    P=H/(np.linalg.norm(H,axis=1,keepdims=True)+1e-9)
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

def knn_row(sim, k):
    """keep top-k per row, zero else."""
    A=np.zeros_like(sim)
    idx=np.argsort(-sim,1)[:,:k]
    for i in range(sim.shape[0]): A[i,idx[i]]=sim[i,idx[i]]
    return A

def main():
    m=load_model(MODEL_DIR); m.switch_task("retrieval")
    proc=Qwen2VLImageProcessor.from_pretrained(MODEL_DIR)
    ann=json.load(open(os.path.join(HERE,"annotations/instances_val2017.json")))
    cats=[c["name"] for c in ann["categories"]]; C={c:i for i,c in enumerate(cats)}
    E=enc_labels(m,cats)  # [80,768]
    gt=json.load(open(os.path.join(HERE,"eval_gt.json"))); ids=list(gt)
    Y=np.zeros((len(ids),80),bool)
    for i,iid in enumerate(ids):
        for c in gt[iid]: Y[i,C[c]]=True

    gc=os.path.join(HERE,"eval_global.npy")
    if os.path.exists(gc):
        G=np.load(gc)
    else:
        G=np.stack([feats(m.model,proc,Image.open(os.path.join(HERE,"eval_imgs",f"{iid}.jpg")).convert("RGB"))[1] for iid in ids])
        np.save(gc,G)
    M=len(ids)  # images
    K=80        # labels
    # base score (our current best proxy: patch+global fuse from cache)
    z=np.load(os.path.join(HERE,"cwr_scores.npz")); base=0.7*z["spb"]+0.3*z["sgb"]
    print("baseline: P@1 %.3f P@3 %.3f R@5 %.3f mAP %.3f"%metrics(base,Y,cats))

    # cos blocks
    GG=G@G.T          # img-img [M,M]
    GL=G@E.T          # img-label [M,K]
    def run(kii, kil, alpha, beta):
        # build (M+K) graph, nodes 0..M-1 images, M..M+K-1 labels
        N=M+K
        S=np.zeros((N,N))
        # img-img knn
        Aii=knn_row(GG - np.eye(M)*2, kii)  # exclude self
        S[:M,:M]=Aii
        # img-label knn (each image connects to top-kil labels), and symmetric
        Ail=knn_row(GL, kil)  # [M,K]
        S[:M,M:]=Ail
        S[M:,:M]=Ail.T
        S=np.maximum(S,S.T)
        d=S.sum(1)+1e-9; Dm=1/np.sqrt(d)
        Sh=S*Dm[:,None]*Dm[None,:]
        L=np.eye(N)-alpha*Sh
        Linv=np.linalg.inv(L)
        # seeds: one-hot on label nodes
        Yseed=np.zeros((N,K)); 
        for c in range(K): Yseed[M+c,c]=1.0
        F=Linv@Yseed   # [N,K]
        prop=F[:M,:]   # propagated scores at image nodes [M,K]
        # normalize prop per column scale then fuse
        prop=(prop-prop.mean(0,keepdims=True))/(prop.std(0,keepdims=True)+1e-9)
        S2=base + beta*prop
        return metrics(S2,Y,cats)

    print(f"\n{'kii':4s}{'kil':4s}{'a':5s}{'b':5s}  P@1   P@3   R@5   mAP")
    best=None
    for kii in [5,10,20]:
        for kil in [3,5]:
            for alpha in [0.5,0.8,0.95]:
                for beta in [0.3,0.6,1.0]:
                    r=run(kii,kil,alpha,beta)
                    if best is None or r[3]>best[0][3]: best=(r,(kii,kil,alpha,beta))
    print("BEST", best[1], "-> P@1 %.3f P@3 %.3f R@5 %.3f mAP %.3f"%best[0])
    # print a small grid around best
    kii,kil,alpha,_=best[1]
    for beta in [0.2,0.4,0.6,0.8,1.0,1.5]:
        r=run(kii,kil,alpha,beta); print(f"{kii:<4d}{kil:<4d}{alpha:<5.2f}{beta:<5.1f}  {r[0]:.3f} {r[1]:.3f} {r[2]:.3f} {r[3]:.3f}")

if __name__=="__main__":
    main()
