"""Audit fixed-weight answer changes on a label-changing image pair."""
import argparse,json,os,sys,time,importlib.util
from collections import defaultdict
import numpy as np
import torch
from prepare import ROOT,R2,R8,KEYS,read,write,sha
sys.path.insert(0,str(R2))
from independent_worker import ExternalEncoder,attach,external_only,from_checkpoint,score_many
from u_official import prepare_external

def load_helpers():
    p=R8/"worker.py"
    # Load only the invariant image, slot and graft helpers, with no imported project state.
    content=p.read_text(encoding="utf8")
    ns=dict(np=np)
    from PIL import Image
    ns.update(Image=Image,sha=sha)
    exec(content[content.index("def opened("):content.index("def run(")],ns)
    return ns["opened"],ns["matching_slots"],ns["graft"]

def run(k,gpu):
    assert os.environ["CUDA_VISIBLE_DEVICES"]==str(gpu) and torch.cuda.device_count()==1
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    p=read(ROOT/"plan.json")
    assert all(sha(f)==v for f,v in p["inputs"].items())
    route=p["routes"][k];assert sha(route["checkpoint"])==route["checkpoint_sha256"]
    opened,matching,graft=load_helpers()
    rows=read(ROOT/"arrows.json")
    by=defaultdict(list)
    for a in rows:by[a["scene"]].append(a)
    assert len(by)==48
    r=attach("27b");e=ExternalEncoder(k);bridge=from_checkpoint(route["checkpoint"],"cuda:0")
    out=ROOT/"reports"/k
    t=time.monotonic()
    with torch.inference_mode(),torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH),\
        (out/"predictions.jsonl").open("x",encoding="utf8") as f:
        for i,(scene,items) in enumerate(by.items(),1):
            a,b=items; u,v=(opened(q) for q in items)
            assert a["box"]==b["box"] and a["row"]["candidates"]==b["row"]["candidates"]
            batch,grid=prepare_external(r,a["row"],u.size)
            local,bg=matching(u,v,a["box"],grid)
            zu,zv=[bridge([e.encode(im,route["tap"])],[grid])[0] for im in (u,v)]
            values={"up_clean":zu,"down_clean":zv}
            for tag,z,z2 in (("up",zu,zv),("down",zv,zu)):
                values[tag+"_local"]=graft(z,z2,local)
                values[tag+"_background"]=graft(z,z2,bg)
            with external_only(r):s=score_many(r,batch,values)
            p0={n:z.tolist() for n,z in s.items()}
            assert all(len(z)==2 and np.isfinite(z).all() for z in p0.values())
            f.write(json.dumps(dict(scene=scene,uid=[a["uid"],b["uid"]],scores=p0,grid=grid,local=local,background=bg),allow_nan=False)+"\n");f.flush()
            r.unchanged()
            if i==1 or i%4==0:
                tmp=out/"live.tmp";write(tmp,dict(done=i,N=48,seconds=time.monotonic()-t));tmp.replace(out/"live.json")
    write(out/"result.json",dict(status="complete",N=48,checkpoint_sha256=route["checkpoint_sha256"],optimizer_steps=0,receiver=r.identity))
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("encoder",choices=KEYS);p.add_argument("--gpu",type=int,required=True)
    a=p.parse_args();run(a.encoder,a.gpu)
