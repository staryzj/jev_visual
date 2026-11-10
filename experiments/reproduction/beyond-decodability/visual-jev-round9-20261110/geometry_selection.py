"""Compare development-cohort selection with the original validation protocol."""
from collections import Counter
import numpy as np
from prepare import ROOT,R8,R6,R5,KEYS,read,write,SEED

def boot(delta,scene):
    g=np.unique(scene)
    v=np.array([delta[:,scene==s].mean() for s in g])
    b=np.random.default_rng(SEED).integers(len(g),size=(10000,len(g)))
    return float(v.mean()),np.percentile(v[b].mean(1),[2.5,97.5]).tolist()

def run():
    p=read(R8/"plan.json")
    rows=read(R5/"synthetic.json")+read(R6/"ood.json")+read(R8/"fresh.json")
    y=np.array([int(a["count"]==4) for a in rows])
    scene=np.array([a["scene"] for a in rows])
    dev=np.array([v["uid"].startswith("ood:") for v in rows])
    test={"fresh_equal":np.array([a["uid"].startswith("fresh:") and a.get("regime")=="equal_area" for a in rows]),
          "fresh_same_size":np.array([a["uid"].startswith("fresh:") and a.get("regime")=="same_instance_sizes" for a in rows])}
    assert dev.sum()==96 and all(m.sum()==48 and not (m&dev).any() for m in test.values())
    assert set(scene[dev]).isdisjoint(set(scene[test["fresh_equal"]]))
    out={}; detail=[]
    for k in KEYS:
        r=R8/"reports"/k
        a=read(r/"probe_results.json")
        z=np.load(r/"probe_scores.npz")
        assert z["uid"].tolist()==[v["uid"] for v in rows]
        stage_info={}
        for stage in ("source","slots"):
            src=[b for b in a if b["stage"]==stage]
            c=Counter();changed=0
            stats={n:[] for n in test}
            for v in src:
                pre=f"{stage}_{v['dimension']}_{v['projection_seed']}_{v['partition']}_"
                scores={fam:z[pre+fam] for fam in v["families"]}
                accuracy={fam:float((s[dev].argmax(1)==y[dev]).mean()) for fam,s in scores.items()}
                pick=min(accuracy,key=lambda f:(-accuracy[f],v["families"][f]["validation_ce"],f))
                base=v["chosen"]
                c[pick]+=1;changed+=pick!=base
                detail.append(dict(encoder=k,stage=stage,dimension=v["dimension"],seed=v["projection_seed"],
                   partition=v["partition"],original=base,geometry=pick,geometry_validation_accuracy=accuracy))
                for n,m in test.items():
                    stats[n].append([(scores[f][m].argmax(1)==y[m]) for f in (base,pick)])
            result={}
            for n,a0 in stats.items():
                b,c0=np.asarray(a0).transpose(1,0,2)
                d,ci=boot(c0.astype(float)-b,scene[test[n]])
                result[n]=dict(original=float(b.mean()),geometry=float(c0.mean()),difference=d,CI=ci)
            stage_info[stage]=dict(N=len(src),changed=changed,frequency=dict(c),evaluation=result)
        out[k]=stage_info
    write(ROOT/"geometry_selection.json",dict(routes=out,details=detail,
        scope="Retrospective only. The old shift is repurposed for development and the already reported new shift is used for comparison. Fits and regularizers stay fixed."))
if __name__=="__main__":run()
