"""Audit validation-only selection, width effects, and same-image patch scores."""
from collections import Counter
from pathlib import Path
import argparse,json
import numpy as np
from prepare import ROOT,R5,R6,R7,KEYS,read,write
DOMAINS=("original_test","old_equal_area","old_same_instance_sizes","fresh_equal_area","fresh_same_instance_sizes")

def jl(p):return [json.loads(s) for s in Path(p).read_text(encoding="utf8").splitlines()]
def choice_scores(a,p,n):
    return np.array([p["scores"][n][a["row"]["candidates"].index(w)] for w in ("two","four")])
def scores(rs,data,name):
    by={v["uid"]:v for v in data}
    assert len(by)==len(data)
    return np.array([choice_scores(a,by[a["uid"]],name) for a in rs])
def accuracy(z,y):return float((z.argmax(1)==y).mean())
def gold_margin(z,y):return (z[:,1]-z[:,0])*(2*y-1)
def avg(z):
    v=np.asarray(z)
    return dict(mean=float(v.mean()),sd=float(v.std(ddof=1)),min=float(v.min()),max=float(v.max()),N=len(v))
def bootstrap(d,scene):
    gs=np.unique(scene)
    b=np.array([d[scene==g].mean() for g in gs])
    r=np.random.default_rng(202611108).integers(len(gs),size=(10000,len(gs)))
    return dict(delta=float(b.mean()),CI=np.percentile(b[r].mean(1),[2.5,97.5]).tolist(),scenes=len(gs))
def compare(a,b,y,sc):
    x,z=a.argmax(1)==y,b.argmax(1)==y
    return dict(acc=bootstrap(x.astype(float)-z,sc),margin=bootstrap(gold_margin(a,y)-gold_margin(b,y),sc),
        gains=int((x&~z).sum()),losses=int((z&~x).sum()))

def run(root,r5,r6,r7):
    plan=read(root/"plan.json")
    old=read(r5/"synthetic.json")
    shift=read(r6/"ood.json")
    fresh=read(root/"fresh.json")
    cover=[a for a in shift if a["regime"]=="same_instance_sizes"]
    rows=old+shift+fresh
    ids=[a["uid"] for a in rows]
    assert len(set(ids))==len(ids)==396
    y=np.array([int(r["count"]==4) for r in rows])
    sc=np.array([a["scene"] for a in rows])
    idx={v:i for i,v in enumerate(ids)}
    domain={}
    domain["original_test"]=np.array([v["split"]=="test" for v in rows])
    for c in ("old","fresh"):
        for reg in ("equal_area","same_instance_sizes"):
            domain[c+"_"+reg]=np.array([a.get("regime")==reg and (a["uid"].startswith("fresh:")==(c=="fresh")) for a in rows])
    assert tuple(domain)==DOMAINS
    assert [int(m.sum()) for m in domain.values()]==[96,48,48,48,48]
    pair=[idx[a["uid"]] for a in cover]
    ypatch=y[pair];scene=sc[pair]
    prior=read(r7/"summary.json")["routes"]
    original_llava=jl(r5/"reports/synthetic/predictions.jsonl")
    original_cross=jl(r6/"reports/cross/predictions.jsonl")
    ood=jl(r6/"reports/ood/predictions.jsonl")
    patches=[v for v in ("clean","cover","restore_local","restore_background","insert_local","insert_background")]
    out={}
    common_audit=None
    for k in KEYS:
        p=root/"reports"/k
        assert read(root/"reports"/("fit-"+k)/"exit.json")["exit_code"]==0 and read(p/"result.json")["N"]==len(rows)
        assert read(root/"reports"/("fit-"+k)/"resume.json")["reused_features"]
        pr=jl(p/"predictions.jsonl")
        assert len(pr)==144
        z=np.load(p/"probe_scores.npz")
        assert z["uid"].tolist()==ids
        re=read(p/"probe_results.json")
        assert len(re)==72
        assert len({(r["stage"],r["dimension"],r["projection_seed"],r["partition"]) for r in re})==72
        reference={}
        for tag,m in domain.items():
            rs=[a for a,ok in zip(rows,m) if ok]
            data=(original_llava if k=="llava" else original_cross) if tag=="original_test" else ood if tag.startswith("old") else pr
            reference[tag]=scores(rs,data,"clean" if tag.startswith("fresh") else k)
        condition={}
        for st in ("source","slots"):
            w={}
            for d in plan["dimensions"]:
                cell=[r for r in re if r["stage"]==st and r["dimension"]==d]
                assert len(cell)==12
                item=dict(frequency=dict(Counter(r["chosen"] for r in cell)),metrics={})
                for tag,m in domain.items():
                    selections=[]
                    for r in cell:
                        decision=min(r["families"],key=lambda fam:(r["families"][fam]["validation_ce"],fam))
                        assert decision==r["chosen"]
                        prefix=f"{st}_{d}_{r['projection_seed']}_{r['partition']}"
                        for fam,v in r["families"].items():
                            a=z[prefix+"_"+fam][m]
                            assert accuracy(a,y[m])==v["accuracy"][tag]
                        selections.append(z[prefix+"_"+decision][m])
                    estimates=[accuracy(a,y[m]) for a in selections]
                    ar=np.stack(selections)
                    dlt=(ar.argmax(2)==y[m]).mean(0)-(reference[tag].argmax(1)==y[m])
                    item["metrics"][tag]=dict(accuracy=avg(estimates),receiver=accuracy(reference[tag],y[m]),
                        compare=bootstrap(dlt,sc[m]),
                        individual_seed_sds=avg([float(np.std(r["families"][r["chosen"]]["neural_seed_accuracy"][tag],ddof=1))
                            if len(r["families"][r["chosen"]]["neural_seed_accuracy"][tag])>1 else 0.0 for r in cell]))
                w[str(d)]=item
            condition[st]=w
        patch={tag:scores(cover,pr,tag) for tag in patches}
        by={p["uid"]:p for p in pr}
        audits=[by[r["uid"]]["audit"] for r in cover]
        assert all(a["local"] and len(a["local"])==len(a["background"]) and not (set(a["local"])&set(a["background"])) for a in audits)
        if common_audit is None:common_audit=audits
        else:assert common_audit==audits
        clean_prior=scores(cover,ood,k)
        assert (patch["clean"].argmax(1)==clean_prior.argmax(1)).all()
        old_pick=prior[k]["readouts"]["source"]["18"]
        vals={tag:accuracy(patch[tag],ypatch) for tag in patches}
        contrast={n:compare(patch[a],patch[b],ypatch,scene) for n,a,b in (
            ("cover","cover","clean"),("restored","restore_local","cover"),
            ("restore_vs_background","restore_local","restore_background"),
            ("insert_vs_background","insert_local","insert_background"))}
        # Associations compare signed changes for the same image. They do not pool score scales across receivers.
        x=gold_margin(patch["clean"],ypatch)-gold_margin(patch["cover"],ypatch)
        z1=gold_margin(patch["restore_local"],ypatch)-gold_margin(patch["restore_background"],ypatch)
        v=float(np.corrcoef(x,z1)[0,1])
        assert np.isfinite(v)
        out[k]=dict(conditions=condition,prior18=dict(family=old_pick["selected"],
            accuracy=old_pick["families"][old_pick["selected"]]["metrics"]["equal_area"]["acc"]),
            frozen={n:accuracy(a,y[domain[n]]) for n,a in reference.items()},
            patches=dict(accuracy=vals,contrast=contrast,correlation=v,
                slots=[len(a["local"]) for a in audits],max_replay_difference=float(np.abs(patch["clean"]-clean_prior).max())))
    write(root/"summary.json",dict(routes=out,repeated_partitions=4,projection_seeds=3,eval_scenes=24))
if __name__=="__main__":
    p=argparse.ArgumentParser()
    for name,value in (("root",ROOT),("r5",R5),("r6",R6),("r7",R7)):p.add_argument("--"+name,type=Path,default=value)
    a=p.parse_args();run(a.root,a.r5,a.r6,a.r7)
