"""Seal projection, split, and linked patch tests before reading their results."""
import hashlib, importlib.util, json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent
R7=ROOT.parent/"visual-jev-round7-20261110"
R6=ROOT.parent/"visual-jev-round6-20261110"
R5=ROOT.parent/"visual-jev-round5-20261110"
R2=ROOT.parent/"visual-jev-round2-controls-20261110"
KEYS=("llava","internvl","siglip2","qwen")
PROJ=(107,211,503)
DIMS=(64,128,256)
SPLIT=(1287,2399,4127)
FAMILIES=("linear","kernel","mlp","attention")

def read(p):return json.loads(Path(p).read_text(encoding="utf8"))
def write(p,v):
    with Path(p).open("x",encoding="utf8") as f:json.dump(v,f,indent=2,allow_nan=False)
def sha(p):
    with Path(p).open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()
def seal():
    s=importlib.util.spec_from_file_location("r6_render",R6/"prepare.py")
    m=importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    m.ROOT=ROOT
    m.SEED=202611108
    m.render()
    x=read(ROOT/"ood.json")
    a=read(R5/"synthetic.json")
    b=read(R6/"ood.json")
    p=read(R7/"interventions.json")
    assert len(x)==96 and {r["rgb"] for r in x}.isdisjoint({r["rgb"] for r in a+b+p})
    fresh=[]
    for r in x:
        uid=r["uid"]
        new=uid.replace("ood:","fresh:",1)
        fresh.append(dict(r,uid=new,scene=r["scene"].replace("ood:","fresh:",1),row=dict(r["row"],uid=new)))
    write(ROOT/"fresh.json",fresh)
    dev=sorted({v["scene"] for v in a if v["split"]!="test"})
    assert len(dev)==27
    split=[dict(name="original",fit=sorted({v["scene"] for v in a if v["split"]=="train"}),
                   validation=sorted({v["scene"] for v in a if v["split"]=="calibration"}))]
    for seed in SPLIT:
        order=np.random.default_rng(seed).permutation(dev).tolist()
        split.append(dict(name=str(seed),fit=order[:18],validation=order[18:]))
    assert len({frozenset(v["fit"]) for v in split})==len(split)
    assert all(len(v["fit"])==18 and len(v["validation"])==9 and set(v["fit"]).isdisjoint(v["validation"]) for v in split)
    routes=read(R6/"plan.json")["routes"]
    inp=[ROOT/n for n in ("prepare.py","worker.py","fresh.json","fit.py")]
    inp += [R5/"synthetic.json",R6/"ood.json",R7/"interventions.json",R6/"plan.json"]
    write(ROOT/"plan.json",dict(routes=routes,inputs={str(f):sha(f) for f in inp},dimensions=DIMS,
       projection_seeds=PROJ,partitions=split,probe_families=FAMILIES,neural_seeds=[13,37,73],
       ridge_grid=[.01,.1,1,10],decay_grid=[.001,.01],epochs=120,selection="Minimum validation cross entropy over families and regularizers, with family name breaking ties.",
       study="Both visual stages. Same 18/9 budget. Old test cohort and new independent render with locked model selection. All configurations reported.",
       patch="Original and target-covered version of each same-size image. Restore or replace the same number of slots at the covered region or at a nearby background control. Fixed weights and scoring context."))
if __name__=="__main__":seal()
