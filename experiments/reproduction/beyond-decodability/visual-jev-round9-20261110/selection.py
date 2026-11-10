"""Compare validation-only rules using the same stored family scores."""
from collections import Counter
import numpy as np
from prepare import ROOT,R8,R6,R5,KEYS,read,write,SEED

def run():
    p=read(R8/"plan.json")
    rows=read(R5/"synthetic.json")+read(R6/"ood.json")+read(R8/"fresh.json")
    u=[a["uid"] for a in rows]
    y=np.array([int(a["count"]==4) for a in rows])
    scene=np.array([a["scene"] for a in rows])
    masks={"original_test":np.array([a["split"]=="test" for a in rows]),
           "prior_equal":np.array([a["uid"].startswith("ood:") and a.get("regime")=="equal_area" for a in rows]),
           "fresh_equal":np.array([a["uid"].startswith("fresh:") and a.get("regime")=="equal_area" for a in rows])}
    parts={v["name"]:v["validation"] for v in p["partitions"]}
    out={}
    details=[]
    for k in KEYS:
        d=R8/"reports"/k
        v=read(d/"probe_results.json")
        z=np.load(d/"probe_scores.npz")
        assert len(v)==72 and z["uid"].tolist()==u
        d0={}
        for stage in ("source","slots"):
            pool=[q for q in v if q["stage"]==stage]
            assert len(pool)==36
            data={n:[] for n in masks}; f0=Counter();f1=Counter()
            changes=0
            for a in pool:
                cal=np.isin(scene,parts[a["partition"]]);assert cal.sum()==36
                stem=f"{stage}_{a['dimension']}_{a['projection_seed']}_{a['partition']}_"
                s={f:z[stem+f] for f in a["families"]}
                c={f:float((q[cal].argmax(1)==y[cal]).mean()) for f,q in s.items()}
                ce=lambda f:a["families"][f]["validation_ce"]
                orig=min(c,key=lambda f:(ce(f),f))
                pick=min(c,key=lambda f:(-c[f],ce(f),f))
                assert orig==a["chosen"] and all(not (m&cal).any() for m in masks.values())
                changes+=orig!=pick;f0[orig]+=1;f1[pick]+=1
                details.append(dict(encoder=k,stage=stage,width=a["dimension"],seed=a["projection_seed"],
                    partition=a["partition"],ce_choice=orig,accuracy_choice=pick,validation_accuracy=c))
                for n,m in masks.items():
                    hit0=s[orig][m].argmax(1)==y[m];hit1=s[pick][m].argmax(1)==y[m]
                    data[n].append((hit0,hit1))
            r={}
            for n,arr in data.items():
                a,b=np.asarray(arr).transpose(1,0,2)
                delta=b.astype(float)-a
                sc=scene[masks[n]];gs=np.unique(sc)
                group=np.array([delta[:,sc==g].mean() for g in gs])
                idx=np.random.default_rng(SEED).integers(len(gs),size=(10000,len(gs)))
                r[n]=dict(ce=float(a.mean()),accuracy=float(b.mean()),delta=float(group.mean()),
                    CI=np.percentile(group[idx].mean(1),[2.5,97.5]).tolist(),configurations=len(arr),scenes=len(gs))
            d0[stage]=dict(changes=changes,ce_frequency=dict(f0),accuracy_frequency=dict(f1),metrics=r)
        out[k]=d0
    write(ROOT/"selection.json",dict(routes=out,details=details,
       policy="Within-family regularization remains as previously fitted. Re-rank families by validation accuracy, then cross entropy and family name. No test labels enter the choice."))
if __name__=="__main__":run()
