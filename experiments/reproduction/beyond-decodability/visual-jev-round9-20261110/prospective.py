"""Seal a new count cohort and the two already fixed selection rules."""
import importlib.util
from prepare import ROOT,R8,R6,R5,read,write,sha,KEYS
def run():
    o=ROOT/"prospective-data";o.mkdir()
    s=importlib.util.spec_from_file_location("r6_renderer",R6/"prepare.py")
    m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
    m.ROOT=o;m.SEED=20261110937;m.render()
    rs=read(o/"ood.json");out=[]
    for a in rs:
        u=a["uid"].replace("ood:","confirm:",1)
        out.append(dict(a,uid=u,scene=a["scene"].replace("ood:","confirm:",1),row=dict(a["row"],uid=u)))
    old=read(R5/"synthetic.json")+read(R6/"ood.json")+read(R8/"fresh.json")+read(ROOT/"arrows.json")
    assert len(out)==96 and {v["rgb"] for v in out}.isdisjoint({v["rgb"] for v in old})
    write(ROOT/"prospective.json",out)
    f=[ROOT/k for k in ("prospective.py","prospective_worker.py","prospective.json","geometry_selection.json")]
    f+=[R8/"fit.py",R8/"plan.json"]
    for k in KEYS:
        d=R8/"reports"/k
        f += [d/"probe_results.json",d/"probe_scores.npz",d/"source.npy",d/"slots.npy"]
    write(ROOT/"prospective_plan.json",dict(seed=20261110937,N=96,inputs={str(p):sha(p) for p in f},
        decision="Existing original validation choices vs. geometry-validation choices locked before evaluation. Fit only the recorded families and regularizers on the original training partitions.",
        primary="Accuracy over 24 new scene pairs per regime, averaged over the same 36 configurations per encoder/stage. Frozen scores share the candidate and image protocol."))
if __name__=="__main__":run()
