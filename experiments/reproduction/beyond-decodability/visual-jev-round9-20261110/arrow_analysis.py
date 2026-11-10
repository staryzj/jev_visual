"""Summarize bidirectional arrow changes and paired site effects."""
import json
import numpy as np
from prepare import ROOT,KEYS,SEED,read,write,WORDS
def jl(p):return [json.loads(x) for x in p.read_text(encoding="utf8").splitlines()]
def boot(d):
    assert len(d)>1
    i=np.random.default_rng(SEED).integers(len(d),size=(10000,len(d)))
    return dict(mean=float(d.mean()),CI=np.percentile(d[i].mean(1),[2.5,97.5]).tolist())
def run():
    rows=read(ROOT/"arrows.json")
    out={}; mask0=None
    for k in KEYS:
        p=ROOT/"reports"/k
        assert read(p/"exit.json")["exit_code"]==0 and read(p/"result.json")["N"]==48
        a=jl(p/"predictions.jsonl")
        assert len(a)==48 and len({q["scene"] for q in a})==48
        assert [r["uid"] for r in rows]==[u for q in a for u in q["uid"]]
        masks=[(q["grid"],q["local"],q["background"]) for q in a]
        assert all(len(local)==len(bg)>0 and set(local).isdisjoint(bg) for _,local,bg in masks)
        if mask0 is None:mask0=masks
        else:assert masks==mask0
        r=rows[::2]
        def collect(tag):
            z=np.array([[[q["scores"][o+"_"+tag][u["row"]["candidates"].index(t)] for t in WORDS] for o in WORDS] for q,u in zip(a,r)])
            assert z.shape==(48,2,2) and np.isfinite(z).all()
            return z
        z={n:collect(n) for n in ("clean","local","background")}
        true=np.array([0,1])
        m={}
        for subset,sel in (("all",np.ones(len(r),bool)),("plain",np.array([not v["clutter"] for v in r])),
                           ("clutter",np.array([v["clutter"] for v in r]))):
            g={n:a[sel] for n,a in z.items()}
            p0=g["clean"].argmax(2)
            clean=p0==true
            a0=g["local"].argmax(2)==1-true
            b0=g["background"].argmax(2)==1-true
            def toward_donor(z):return (z[:,:,1]-z[:,:,0])*np.array([1,-1])
            margin=(toward_donor(g["local"])-toward_donor(g["background"])).mean(1)
            flip=(a0&clean).mean(1)-(b0&clean).mean(1)
            m[subset]=dict(scenes=int(sel.sum()),clean_accuracy=float(clean.mean()),clean_both=int(clean.all(1).sum()),
                local_donor_both=int(a0.all(1).sum()),background_donor_both=int(b0.all(1).sum()),
                local_flips=int((a0&clean).sum()),background_flips=int((b0&clean).sum()),
                paired_margin=boot(margin),paired_flip=boot(flip),slot_count=[len(v[1]) for v,b in zip(masks,sel) if b])
        out[k]=m
    write(ROOT/"arrow_summary.json",dict(routes=out,primary="Local minus background patch. Positive margin points to the donor answer, averaged over two swaps per scene. Pointwise paired intervals resample scenes."))
if __name__=="__main__":run()
