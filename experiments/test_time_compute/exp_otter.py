"""
OTTER (2404.08461) grafted: optimal-transport base-rate correction on SCORES.
Operates on our final score matrix (CWR-enhanced), not on weak global vecs -> avoids
the ZLaP failure mode. Multi-label adaptation: use unbalanced/relaxed Sinkhorn so
each image is NOT forced to sum to one class.

We test:
- balanced OTTER (row=uniform, col=nu) as ablation
- relaxed Sinkhorn (KL on marginals) with tau
- nu estimation: uniform vs batch-estimated from current scores
Compare against CWR baseline (mAP 0.710). Also A/B vs our background-centering.
"""
import sys, os, json
import numpy as np
HERE=os.path.dirname(__file__)

def metrics(S,Y,K=80):
    Sc=S-S.mean(0,keepdims=True)
    def patk(k):
        idx=np.argsort(-Sc,1)[:,:k]; return np.mean([Y[i,idx[i]].sum()/k for i in range(len(Y))]), np.mean([Y[i,idx[i]].sum()/max(Y[i].sum(),1) for i in range(len(Y))])
    aps=[]
    for c in range(K):
        if Y[:,c].sum()==0: continue
        o=np.argsort(-Sc[:,c]); yt=Y[o,c]; tp=np.cumsum(yt); aps.append((tp/(np.arange(len(yt))+1)*yt).sum()/yt.sum())
    p1,_=patk(1);p3,_=patk(3);_,r5=patk(5); return p1,p3,r5,float(np.mean(aps))

def softmax(x,ax=1):
    x=x-x.max(ax,keepdims=True); e=np.exp(x); return e/e.sum(ax,keepdims=True)

def sinkhorn(C, a, b, eps=0.05, iters=100):
    """entropic OT, returns plan pi. C cost [n,K]."""
    Kk=np.exp(-C/eps)
    u=np.ones(len(a)); v=np.ones(len(b))
    for _ in range(iters):
        u=a/(Kk@v+1e-12); v=b/(Kk.T@u+1e-12)
    return u[:,None]*Kk*v[None,:]

def sinkhorn_unbalanced(C, a, b, eps=0.05, tau=1.0, iters=100):
    """KL-relaxed marginals; tau controls relaxation (large tau -> balanced)."""
    Kk=np.exp(-C/eps); u=np.ones(len(a)); v=np.ones(len(b)); f=tau/(tau+eps)
    for _ in range(iters):
        u=(a/(Kk@v+1e-12))**f
        v=(b/(Kk.T@u+1e-12))**f
    return u[:,None]*Kk*v[None,:]

def main():
    ann=json.load(open(os.path.join(HERE,"annotations/instances_val2017.json")))
    cats=[c["name"] for c in ann["categories"]]; C={c:i for i,c in enumerate(cats)}
    gt=json.load(open(os.path.join(HERE,"eval_gt.json"))); ids=list(gt)
    Y=np.zeros((len(ids),80),bool)
    for i,iid in enumerate(ids):
        for c in gt[iid]: Y[i,C[c]]=True
    z=np.load(os.path.join(HERE,"cwr_scores.npz"))
    base=0.7*z["spb"]+0.3*z["sgb"]
    # current best = base + 1.3*max(spc,sgc)   (grid5 crop cache; use cwr2 if present)
    cwr=np.maximum(z["spc"],z["sgc"])
    best=base+0.8*cwr
    M=80  # classes
    n=len(ids)
    print("CWR baseline: P@1 %.3f P@3 %.3f R@5 %.3f mAP %.3f"%metrics(best,Y))

    P=softmax(best*20,1)   # per-image class prob (temp)
    Cost=-np.log(P+1e-9)   # cost matrix [n,80]
    a=np.ones(n)/n
    # nu estimates
    nu_uniform=np.ones(80)/80
    nu_batch=P.mean(0); nu_batch/=nu_batch.sum()

    print(f"\n{'variant':30s} P@1   P@3   R@5   mAP")
    for name,nu in [("uniform",nu_uniform),("batch-est",nu_batch)]:
        pi=sinkhorn(Cost,a,nu,eps=0.05,iters=200)
        # OTTER predicts via plan pi as score
        r=metrics(pi,Y); print(f"{'bal-OT nu='+name:30s} {r[0]:.3f} {r[1]:.3f} {r[2]:.3f} {r[3]:.3f}")
    for tau in [0.5,1.0,5.0]:
        for name,nu in [("uniform",nu_uniform),("batch",nu_batch)]:
            pi=sinkhorn_unbalanced(Cost,a,nu*n/1.0,eps=0.05,tau=tau,iters=200)
            r=metrics(pi,Y); print(f"{'unbal tau=%.1f nu=%s'%(tau,name):30s} {r[0]:.3f} {r[1]:.3f} {r[2]:.3f} {r[3]:.3f}")
    # OTTER as correction: blend plan with base
    pi=sinkhorn_unbalanced(Cost,a,nu_batch*n,eps=0.05,tau=1.0,iters=200)
    for b in [0.5,1.0,2.0]:
        r=metrics(best+b*pi,Y); print(f"{'best+%.1f*pi'%b:30s} {r[0]:.3f} {r[1]:.3f} {r[2]:.3f} {r[3]:.3f}")

if __name__=="__main__":
    main()
