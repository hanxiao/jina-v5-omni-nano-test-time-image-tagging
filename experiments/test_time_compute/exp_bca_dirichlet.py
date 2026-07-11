"""
Verify BCA (Gaussian likelihood + adaptive prior) and EM-Dirichlet, grafted.
No third-party model, no training. On CWR-enhanced scores / features.

BCA-A (Gaussian likelihood, DOTA-style): estimate per-class covariance Σ from
  background patch feats (shared or per-class shrinkage), score label via
  Mahalanobis: s_k = -(x-μ_k)^T Σ^-1 (x-μ_k). Here μ_k = label vec E_k.
  Compare to cos baseline.
BCA-B (adaptive prior): iteratively update per-class prior on the eval batch:
  posterior ∝ likelihood * prior; prior_k <- running mean of posterior_k.
  Multi-label friendly if we DON'T normalize across classes (per-class logistic).
EM-Dirichlet: batch simplex clustering (single-label; expected to conflict).

Everything measured on COCO-150 vs CWR baseline mAP 0.710.
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

def main():
    ann=json.load(open(os.path.join(HERE,"annotations/instances_val2017.json")))
    cats=[c["name"] for c in ann["categories"]]; C={c:i for i,c in enumerate(cats)}
    gt=json.load(open(os.path.join(HERE,"eval_gt.json"))); ids=list(gt)
    Y=np.zeros((len(ids),80),bool)
    for i,iid in enumerate(ids):
        for c in gt[iid]: Y[i,C[c]]=True
    z=np.load(os.path.join(HERE,"cwr_scores.npz"))
    base=0.7*z["spb"]+0.3*z["sgb"]
    best=base+0.8*np.maximum(z["spc"],z["sgc"])
    print("CWR baseline: P@1 %.3f P@3 %.3f R@5 %.3f mAP %.3f\n"%metrics(best,Y))

    # ---- BCA-A: Gaussian/Mahalanobis likelihood ----
    # need raw (unnormalized) global img feats + raw label vecs
    G=np.load(os.path.join(HERE,"eval_global.npy"))  # normed global img vecs [150,768]
    BG=np.load(os.path.join(HERE,"bg_rawpatch.npy")) # bg patch feats [~14k,768]
    # label vecs: reconstruct from GL? we need E. recompute not available here; use G? no.
    # We DO have GL = z["sgb"] = G @ E.T. Solve E via least squares from G? underdetermined.
    # Instead: Mahalanobis in the GLOBAL image space using shared Σ from bg patches,
    # applied to the cos-score refinement is not directly possible without E.
    # So test BCA-A only on global path where we can whiten G and labels consistently:
    # approximate label vec by class-mean of top images? skip — instead do the
    # adaptive-prior (BCA-B) and Dirichlet which operate on SCORES (available).
    print("(BCA-A Mahalanobis needs raw E; covered by earlier whitening exp which failed. skip.)\n")

    # ---- BCA-B: adaptive per-class prior (multi-label, per-class, no cross-norm) ----
    P=1/(1+np.exp(-best*10))  # per-class prob via logistic (NO cross-class norm)
    prior=np.ones(80)*0.5
    for it in range(20):
        post=P*prior[None,:]/ (P*prior[None,:]+(1-P)*(1-prior)[None,:]+1e-9)
        prior=0.5*prior+0.5*post.mean(0)
    Sb=np.log(P+1e-9)-np.log(1-P+1e-9)+np.log(prior+1e-9)-np.log(1-prior+1e-9)
    print("BCA-B adaptive prior (per-class): P@1 %.3f P@3 %.3f R@5 %.3f mAP %.3f"%metrics(Sb,Y))
    # blended
    for b in [0.3,0.6,1.0]:
        r=metrics(best+b*(np.log(prior+1e-9)[None,:]),Y)
        print(f"  best + {b}*log-prior:               P@1 %.3f P@3 %.3f R@5 %.3f mAP %.3f"%r)

    # ---- EM-Dirichlet (single-label simplex; expected conflict) ----
    Z=softmax(best*10,1)  # simplex per image
    a=np.ones(80)  # dirichlet params
    from scipy.special import digamma, gammaln
    u=Z.copy()
    for it in range(30):
        pi=u.mean(0)+1e-6
        # dirichlet loglik per class param (MM-ish crude): update a via moment approx
        m=(u[:,:,None]*Z[:,None,:]).sum(0)/ (u.sum(0)[:,None]+1e-9)  # not exact; crude
        # crude: set a_k prop to mean Z of assigned
        a=1.0+5*(u.T@Z).diagonal()/(u.sum(0)+1e-9)
        logp=(Z@ (a-1)[:,None]).squeeze() # crude dirichlet loglik surrogate
        u=softmax(np.log(Z+1e-9)@np.diag(a-1) + np.log(pi)[None,:],1)
    r=metrics(u,Y)
    print("\nEM-Dirichlet (crude, simplex): P@1 %.3f P@3 %.3f R@5 %.3f mAP %.3f"%r)

if __name__=="__main__":
    main()
