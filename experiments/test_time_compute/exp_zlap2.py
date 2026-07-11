"""ZLaP diagnosis + fix. Check propagation ALONE, and proper ZLaP scoring.
Key fixes vs v1:
- ZLaP original reads the propagated label scores directly as the classifier
  (not as an additive term). Evaluate prop-only mAP first.
- Use the ZLaP dual/normalized form correctly: seeds one-hot at label nodes,
  but the useful signal is F[image, class]. Try WITHOUT z-scoring (raw), and
  try fusing multiplicatively / rank-blend.
- Also try the ZLaP recommended: sparsify + row-normalize similarities to [0,1]
  via max(0,sim) (negative cos edges are noise).
"""
import sys, os, json
import numpy as np
HERE=os.path.dirname(__file__)

def metrics(S,Y,cats_n):
    Sc=S-S.mean(0,keepdims=True)
    def patk(k):
        idx=np.argsort(-Sc,1)[:,:k]; return np.mean([Y[i,idx[i]].sum()/k for i in range(len(Y))]), np.mean([Y[i,idx[i]].sum()/max(Y[i].sum(),1) for i in range(len(Y))])
    aps=[]
    for c in range(cats_n):
        if Y[:,c].sum()==0: continue
        o=np.argsort(-Sc[:,c]); yt=Y[o,c]; tp=np.cumsum(yt); aps.append((tp/(np.arange(len(yt))+1)*yt).sum()/yt.sum())
    p1,_=patk(1);p3,_=patk(3);_,r5=patk(5); return p1,p3,r5,float(np.mean(aps))

def knn_row(sim,k,excl_self=False):
    A=np.zeros_like(sim); S=sim.copy()
    if excl_self: np.fill_diagonal(S,-9)
    idx=np.argsort(-S,1)[:,:k]
    for i in range(S.shape[0]):
        v=S[i,idx[i]]; v=np.maximum(v,0)  # drop negative edges
        A[i,idx[i]]=v
    return A

def main():
    ann=json.load(open(os.path.join(HERE,"annotations/instances_val2017.json")))
    cats=[c["name"] for c in ann["categories"]]; C={c:i for i,c in enumerate(cats)}
    gt=json.load(open(os.path.join(HERE,"eval_gt.json"))); ids=list(gt)
    Y=np.zeros((len(ids),80),bool)
    for i,iid in enumerate(ids):
        for c in gt[iid]: Y[i,C[c]]=True
    G=np.load(os.path.join(HERE,"eval_global.npy"))
    # rebuild E from cache label_cache? we need 80-cat E. recompute cheap via cwr? no.
    # We stored GL implicitly: base uses E. Instead reconstruct GL = sgb (global sim to labels)!
    z=np.load(os.path.join(HERE,"cwr_scores.npz"))
    GL=z["sgb"]              # [M,80] image-global vs label cos  == G@E.T
    base=0.7*z["spb"]+0.3*z["sgb"]
    M=len(ids); K=80
    GG=G@G.T
    print("baseline base: P@1 %.3f mAP %.3f"%(metrics(base,Y,80)[0],metrics(base,Y,80)[3]))
    print("GL(global-only): P@1 %.3f mAP %.3f\n"%(metrics(GL,Y,80)[0],metrics(GL,Y,80)[3]))

    def propagate(kii,kil,alpha):
        N=M+K; S=np.zeros((N,N))
        S[:M,:M]=knn_row(GG,kii,excl_self=True)
        Ail=knn_row(GL,kil)
        S[:M,M:]=Ail; S[M:,:M]=Ail.T
        S=np.maximum(S,S.T)
        d=S.sum(1)+1e-9; Dm=1/np.sqrt(d); Sh=S*Dm[:,None]*Dm[None,:]
        Yseed=np.zeros((N,K))
        for c in range(K): Yseed[M+c,c]=1.0
        F=np.linalg.solve(np.eye(N)-alpha*Sh, Yseed)
        return F[:M,:]

    print(f"{'mode':22s}{'kii':4s}{'kil':4s}{'a':5s} P@1   P@3   R@5   mAP")
    for kii in [10,20,40]:
        for kil in [5,10]:
            for alpha in [0.8,0.9,0.99]:
                prop=propagate(kii,kil,alpha)
                # prop-only
                r=metrics(prop,Y,80)
                print(f"{'prop-only':22s}{kii:<4d}{kil:<4d}{alpha:<5.2f} {r[0]:.3f} {r[1]:.3f} {r[2]:.3f} {r[3]:.3f}")

if __name__=="__main__":
    main()
