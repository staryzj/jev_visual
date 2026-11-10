"""Quantify candidate-score agreement and test cue sensitivity to photo/question pairing."""
import importlib.util
import numpy as np
from prepare import ROOT,R5,R6,R7,read,write
s=importlib.util.spec_from_file_location("old_analysis",R5/"analysis.py")
a=importlib.util.module_from_spec(s);s.loader.exec_module(a)

def main():
    data=[a.pack(R6,"photo-fit"),a.pack(R6,"photo-test"),a.pack(R7,"photo-more")]
    rows=sum((r for r,_,_ in data),[])
    pred=sum((p for _,p,_ in data),[])
    X=np.concatenate([x for _,_,x in data])
    q=a.question_kernel(rows)
    y=np.array([r["count"]-2 for r in rows])
    split=np.array([r["split"] for r in rows])
    fit=split=="fit";cal=split=="calibration"
    ix=np.arange(len(rows)-256,len(rows))
    gold=y[ix]
    V=a.image_kernel(X,fit)
    K=V*(.5+.5*q)+.5*q
    cue=a.solve(K,y,fit,4)
    raw=a.reorder(rows,pred,"llava",a.WORDS)
    bias=np.zeros(4); hot=np.eye(4)[y[cal]]
    for _ in range(500):
        c=raw[cal]+bias
        e=np.exp(c-c.max(1,keepdims=True))
        bias-=.25*((e/e.sum(1,keepdims=True)-hot).mean(0)+.01*bias)
    rec=raw+bias
    truth=read(R7/"photo_summary.json")["new"]["routes"]
    assert a.stats(cue[ix],gold)==truth["cue"] and a.stats(rec[ix],gold)==truth["bias"]
    left=cue[ix];right=rec[ix]
    # Each rank vector is a permutation of four candidate ranks.
    ranks=np.argsort(np.argsort(left,axis=1),axis=1)
    rr=np.argsort(np.argsort(right,axis=1),axis=1)
    corr=1-6*((ranks-rr)**2).sum(1)/(4*(4**2-1))
    pick_c=left.argmax(1);pick_r=right.argmax(1)
    yt=np.eye(4)[y[fit]]
    weights=np.linalg.solve(K[np.ix_(fit,fit)]+np.eye(fit.sum()),yt-yt.mean(0))
    v=V[np.ix_(ix,fit)];t=q[np.ix_(ix,fit)]
    def stat(v,t):
        z=(v*(.5+.5*t)+.5*t)@weights
        return [float((z.argmax(1)==gold).mean()),float((z.argmax(1)!=pick_c).mean())]
    assert stat(v,t)[0]==truth["cue"]["acc"]
    rng=np.random.default_rng(202611108)
    img=[]; ques=[]
    for _ in range(500):
        p=rng.permutation(len(ix))
        img.append(stat(v[p],t))
        ques.append(stat(v,t[p]))
    def summary(x):
        z=np.array(x)
        return dict(average_accuracy=float(z[:,0].mean()),prediction_change=float(z[:,1].mean()),
                    randomization_range=np.percentile(z[:,0],[2.5,97.5]).tolist())
    write(ROOT/"photo_diagnostics.json",dict(N=len(ix),top_agreement=float((pick_c==pick_r).mean()),
        mean_spearman=float(corr.mean()),matched_accuracy=truth["cue"]["acc"],
        image_permutation=summary(img),question_permutation=summary(ques),
        scope="500 test re-pairings per modality with frozen learned kernel coefficients, train-only projection. Mismatched inputs keep original labels to measure sensitivity."))
if __name__=="__main__":main()
