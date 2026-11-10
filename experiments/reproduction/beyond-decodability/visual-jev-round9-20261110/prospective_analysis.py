"""Check replay audits and summarize accuracy on the untouched confirmation cohort."""
from pathlib import Path
import json,numpy as np
from prepare import ROOT,KEYS,SEED,read,write
def mean_ci(diff,scene):
    g=np.unique(scene)
    v=np.array([diff[:,scene==s].mean() for s in g])
    b=np.random.default_rng(SEED).integers(len(v),size=(10000,len(v)))
    return dict(delta=float(v.mean()),CI=np.percentile(v[b].mean(1),[2.5,97.5]).tolist())
def run():
    rows=read(ROOT/"prospective.json")
    plan=read(ROOT/"prospective_plan.json")
    assert plan["N"]==len(rows)==96 and len({a["uid"] for a in rows})==96
    Y=np.array([int(v["count"]==4) for v in rows])
    scene=np.array([v["scene"] for v in rows])
    out={}
    for k in KEYS:
        d=ROOT/"reports"/("confirm-"+k)
        assert read(d/"exit.json")["exit_code"]==0
        result=read(d/"result.json");assert result["status"]=="complete" and result["choices"]==72
        assert all(a["max_abs_replay"]<.002 and a["old_argmax_disagreement"]<=1 for a in result["replay"])
        z=np.load(d/"scores.npz")
        u=[a["uid"] for a in rows]
        assert z["uid"].tolist()==u
        p=[json.loads(s) for s in (d/"predictions.jsonl").read_text(encoding="utf8").splitlines()]
        assert u==[a["uid"] for a in p]
        ref=np.array([[q["scores"][a["row"]["candidates"].index(t)] for t in ("two","four")] for q,a in zip(p,rows)])
        r={}
        for stage in ("source","slots"):
            old=sorted(n for n in z.files if n.startswith(stage+"_") and n.endswith("_original"))
            assert len(old)==36
            metrics={}
            for name in ("equal_area","same_instance_sizes"):
                m=np.array([v["regime"]==name for v in rows])
                y=Y[m];sc=scene[m]
                t=np.stack([z[n][m].argmax(1)==y for n in old])
                g=np.stack([z[n.removesuffix("_original")+"_geometry"][m].argmax(1)==y for n in old])
                f=ref[m].argmax(1)==y
                metrics[name]=dict(N=len(y),scenes=len(set(sc)),original=float(t.mean()),geometry=float(g.mean()),
                    frozen=float(f.mean()),original_sd=float(t.mean(1).std(ddof=1)),geometry_sd=float(g.mean(1).std(ddof=1)),
                    geometry_minus_original=mean_ci(g.astype(float)-t,sc),
                    geometry_minus_frozen=mean_ci(g.astype(float)-f[None],sc))
            r[stage]=metrics
        out[k]=dict(stages=r,replay_max=max(v["max_abs_replay"] for v in result["replay"]),
                    replay_argmax_mismatches=sum(v["old_argmax_disagreement"] for v in result["replay"]))
    write(ROOT/"prospective_summary.json",dict(routes=out,design="36 locked choices per stage and encoder. Scene intervals condition on these choices."))
if __name__=="__main__":run()
