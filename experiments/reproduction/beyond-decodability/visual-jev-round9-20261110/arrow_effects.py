"""Measure how much of a full counterfactual shift reaches matched slots."""
import json,numpy as np
from prepare import ROOT,KEYS,SEED,read,write
def run():
    a=read(ROOT/"arrows.json")[::2];out={}
    for k in KEYS:
        r=[json.loads(x) for x in (ROOT/"reports"/k/"predictions.jsonl").read_text(encoding="utf8").splitlines()]
        assert [v["scene"] for v in a]==[v["scene"] for v in r]
        def margin(r,a,n):
            z=r["scores"][n];c=a["row"]["candidates"];return z[c.index("down")]-z[c.index("up")]
        x=np.array([margin(r0,a0,"down_clean")-margin(r0,a0,"up_clean") for r0,a0 in zip(r,a)])
        def effect(tag):
            return np.array([.5*(margin(r0,a0,"up_"+tag)-margin(r0,a0,"up_clean")
                -margin(r0,a0,"down_"+tag)+margin(r0,a0,"down_clean")) for r0,a0 in zip(r,a)])
        u,v=effect("local"),effect("background")
        idx=np.random.default_rng(SEED).integers(48,size=(10000,48))
        ratio=u[idx].mean(1)/x[idx].mean(1)
        out[k]=dict(full=float(x.mean()),local=float(u.mean()),background=float(v.mean()),
            local_share=float(u.mean()/x.mean()),local_share_CI=np.percentile(ratio,[2.5,97.5]).tolist())
    write(ROOT/"arrow_effects.json",out)
if __name__=="__main__":run()
