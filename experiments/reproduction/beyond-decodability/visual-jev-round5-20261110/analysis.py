"""Paired area/donor tests and fit/calibrate/test photo readouts."""
import argparse, hashlib, json, re
from pathlib import Path
import numpy as np

SEED=202611105
WORDS=("two","three","four","five")

def read(p):return json.loads(p.read_text(encoding="utf8"))
def readl(p):return [json.loads(x) for x in p.read_text(encoding="utf8").splitlines()]
def stats(score,y):
    p=score.argmax(1)
    return dict(N=len(y),correct=int((p==y).sum()),acc=float((p==y).mean()),
                pred=np.bincount(p,minlength=score.shape[1]).tolist(),
                correct_by_class=[int(((p==y)&(y==k)).sum()) for k in range(score.shape[1])])
def paired(a,b,y,group):
    x=a.argmax(1)==y
    z=b.argmax(1)==y
    v=np.array([(x[group==g].astype(float)-z[group==g]).mean() for g in np.unique(group)])
    rng=np.random.RandomState(SEED)
    boot=v[rng.randint(len(v),size=(10000,len(v)))].mean(1)
    return dict(gains=int((x&~z).sum()),losses=int((~x&z).sum()),N=len(y),clusters=len(v),
                delta=float(v.mean()),CI=np.percentile(boot,[2.5,97.5]).tolist())
def pack(path,kind):
    p=path/"reports"/kind
    rows=read(path/(kind+".json"))
    exit=read(p/"exit.json")
    result=read(p/"result.json")
    assert exit["exit_code"]==0 and result["status"]=="complete" and result["N"]==len(rows)
    pred=readl(p/"predictions.jsonl")
    f=np.load(p/"features.npz")
    u=[r["uid"] for r in rows]
    assert len(u)==len(set(u)) and u==f["uid"].tolist()==[x["uid"] for x in pred]
    return rows,pred,f["x"]

def image_kernel(X,tr):
    A=X[tr].astype(float)
    mu=A.mean(0)
    sd=np.maximum(A.std(0),1e-6)
    Y=(A-mu)/sd
    val,U=np.linalg.eigh(Y@Y.T)
    keep=np.flatnonzero(val>1e-8)[-32:]
    V=(Y.T@U[:,keep])/np.sqrt(val[keep])
    w=((X-mu)/sd)@V
    norm=np.square(w).sum(1)
    dist=np.maximum(norm[:,None]+norm[None,:]-2*w@w.T,0)
    m=np.median(dist[np.ix_(tr,tr)])
    assert m>0
    return np.exp(-dist/m)

def question_kernel(rows):
    A=np.zeros((len(rows),512))
    for i,r in enumerate(rows):
        tokens=re.findall(r"[a-z]+",r["row"]["question"].lower())
        tokens=[s for s in tokens if s not in {"how","many","are","is","there","the","in","of"}]
        terms=tokens+[a+" "+b for a,b in zip(tokens,tokens[1:])]
        for s in terms:
            j=int(hashlib.sha256(s.encode()).hexdigest()[:8],16)%512
            A[i,j]+=1
    A/=np.maximum(np.linalg.norm(A,axis=1,keepdims=True),1e-8)
    return A@A.T

def solve(K,y,tr,C):
    Y=np.eye(C)[y[tr]]
    return K[:,tr]@np.linalg.solve(K[np.ix_(tr,tr)]+np.eye(sum(tr)),Y-Y.mean(0))

def reorder(rows,pred,k,words):
    assert len(rows)==len(pred)
    score=np.array([[r["scores"][k][v["row"]["candidates"].index(w)] for w in words]
                    for v,r in zip(rows,pred)])
    assert score.shape==(len(rows),len(words)) and np.isfinite(score).all()
    return score

def synthetic(root):
    r,p,X=pack(root,"synthetic")
    y=np.array([0 if v["count"]==2 else 1 for v in r])
    t=np.array([v["split"]=="test" for v in r])
    tr=np.array([v["split"]=="train" for v in r])
    scenes=np.array([v["scene"] for v in r])
    v=[a for a,keep in zip(r,t) if keep]
    ps=[a for a,keep in zip(p,t) if keep]
    for a,b in zip(v,ps):
        assert a["uid"]==b["uid"] and b["audit"] is not None
    words=("two","four")
    out={}
    arms=["llava","native","same_obj","opposite_obj","same_bg","opposite_bg","different_same_obj"]
    scores={k:reorder(v,ps,k,words) for k in arms}
    cue=solve(image_kernel(X,tr),y,tr,2)
    scores["source_cue"]=cue[t]
    area=np.log([q["area"] for q in r])
    cent=[area[tr&(y==k)].mean() for k in range(2)]
    scores["area_only"]=-np.square(area[t,None]-np.array(cent)[None,:])
    for k,s in scores.items():out[k]=stats(s,y[t])
    src={a["uid"]:a for a in v}
    aud={b["uid"]:b["audit"] for b in ps}
    for a in v:
        me=aud[a["uid"]]
        donor=src[me["opposite"]]
        same=src[me["different_same"]]
        assert a["count"]!=donor["count"] and a["area"]==donor["area"]
        assert a["count"]==same["count"] and a["scene"]!=same["scene"]
        assert aud[donor["uid"]]["obj"]==me["obj"] and aud[donor["uid"]]["bg"]==me["bg"]
        assert aud[same["uid"]]["different_same"]==a["uid"]
    g=scenes[t]
    g2=np.array([min(a["scene"],src[aud[a["uid"]]["different_same"]]["scene"]) for a in v])
    contrasts={}
    for a,b in (("same_obj","opposite_obj"),("same_bg","opposite_bg"),("same_obj","same_bg"),
                ("different_same_obj","same_obj"),("different_same_obj","llava"),("source_cue","llava")):
        contrasts[a+"_vs_"+b]=paired(scores[a],scores[b],y[t],g2 if a=="different_same_obj" else g)
    idx={(a["scene"],a["scale"],a["count"]):j for j,a in enumerate(v)}
    separation={}
    for k,s in scores.items():
        margin=s[:,1]-s[:,0]
        order=[margin[idx[scene,scale,4]]-margin[idx[scene,scale,2]]
               for scene in np.unique(g) for scale in (0,1)]
        area_shift=[margin[idx[scene,1,c]]-margin[idx[scene,0,c]]
                    for scene in np.unique(g) for c in (2,4)]
        separation[k]=dict(correct_order=int(np.sum(np.array(order)>0)),tied=int(np.sum(np.array(order)==0)),
                           N=len(order),count_margin_delta=float(np.mean(order)),
                           area_margin_delta=float(np.mean(area_shift)))
    return dict(routes=out,comparisons=contrasts,separation=separation,
        opposite_obj_follows=stats(scores["opposite_obj"],1-y[t]),
        opposite_bg_follows=stats(scores["opposite_bg"],1-y[t]))

def natural(root,r4):
    r,p,X=pack(root,"natural")
    q=read(r4/"natural_cohort.json")
    archive=r4/"reports"/"tallyqa"
    assert read(archive/"exit.json")["exit_code"]==0
    qp=readl(archive/"predictions.jsonl")
    qx=np.load(archive/"pooled_features.npz")
    assert qx["uid"].tolist()==[a["uid"] for a in q]==[a["uid"] for a in qp]
    assert len(set(a["rgb"] for a in r+q))==len(r+q)
    q=[dict(x,split="prior") for x in q]
    r=r+q
    p=p+qp
    X=np.concatenate([X,qx["source_tap"]])
    y=np.array([a["count"]-2 for a in r])
    split=np.array([a["split"] for a in r])
    simple=np.array([a["simple"] for a in r],dtype=bool)
    group=np.array([a["uid"] for a in r])
    orig=reorder(r,p,"llava",WORDS)
    Kq=question_kernel(r)
    ans={}
    for name in ("all","simple_only"):
        dom=(simple if name=="simple_only" else np.ones(len(y),bool))
        tr=(split=="train")&dom
        cal=(split=="calibration")&dom
        Ki=image_kernel(X,tr)
        visual=solve(Ki,y,tr,4)
        question=solve(Kq,y,tr,4)
        cue=solve(Ki*(.5+.5*Kq)+.5*Kq,y,tr,4)
        b=np.zeros(4)
        hot=np.eye(4)[y[cal]]
        for _ in range(500):
            z=orig[cal]+b
            e=np.exp(z-z.max(1,keepdims=True))
            b-=.25*((e/e.sum(1,keepdims=True)-hot).mean(0)+.01*b)
        bias=orig+b
        def unit(a):
            a=a-a.mean(1,keepdims=True)
            return a/max(1e-8,np.median(a[cal].std(1)))
        A=unit(bias)
        B=unit(cue)
        blends={str(w):(1-w)*A+w*B for w in (0,.25,.5,.75,1)}
        selected=max(blends,key=lambda k:(int((blends[k][cal].argmax(1)==y[cal]).sum()),
                                           -abs(float(k)-1)))
        variants={"llava":orig,"bias":bias,"image_only":visual,"question_only":question,
                  "cue":cue,"fusion":blends[selected]}
        tests={}
        for tag,keep in (("fresh",split=="test"),("prior",split=="prior"),
                         ("fresh_complex",(split=="test")&~simple),
                         ("fresh_simple",(split=="test")&simple)):
            yy=y[keep]
            g=group[keep]
            tests[tag]=dict(routes={k:stats(s[keep],yy) for k,s in variants.items()},
                vs_cue=paired(variants["fusion"][keep],cue[keep],yy,g),
                vs_llava=paired(variants["fusion"][keep],orig[keep],yy,g),
                by_weight={w:dict(metrics=stats(s[keep],yy),vs_cue=paired(s[keep],cue[keep],yy,g),
                                 vs_llava=paired(s[keep],orig[keep],yy,g)) for w,s in blends.items()})
        ans[name]=dict(alpha=float(selected),train=int(tr.sum()),calibration=int(cal.sum()),tests=tests)
    return ans

if __name__=="__main__":
    a=argparse.ArgumentParser()
    a.add_argument("root",type=Path)
    a.add_argument("r4",type=Path)
    x=a.parse_args()
    report=dict(synthetic=synthetic(x.root),natural=natural(x.root,x.r4))
    (x.root/"summary.json").write_text(json.dumps(report,indent=2,allow_nan=False),encoding="utf8")

