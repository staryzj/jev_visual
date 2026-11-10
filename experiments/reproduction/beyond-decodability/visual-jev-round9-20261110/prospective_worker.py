"""Evaluate the sealed choices on a new set while verifying all original predictions."""
import argparse,importlib.util,json,os,sys,time
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from prepare import ROOT,R2,R5,R6,R8,KEYS,read,write,sha,pixel
sys.path.insert(0,str(R2))
from independent_worker import ExternalEncoder,attach,external_only,from_checkpoint,score_many
from u_official import prepare_external

def recover_fit():
    s=importlib.util.spec_from_file_location("locked_r8_fit",R8/"fit.py")
    m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
    return m

def score_neural(m,x,y,tr,fam,p,reg):
    one=dict(p,decay_grid=[reg])
    return m.neural(x,y,tr,fam,one,"cuda:0")[0][1]

def reconstruct(m,x,y,tr,old,needed,p):
    a=np.concatenate([x.mean(1),x.std(1),x.max(1),x[:,:32].mean(1)-x[:,32:].mean(1)],1)
    a=m.norm(a,tr,0).astype(np.float64)
    out={}
    for f in needed:
        reg=old[f]["regularizer"]
        if f in ("linear","kernel"):
            K=a@a[tr].T/a.shape[1]
            if f=="kernel":
                dis=np.maximum((a*a).sum(1)[:,None]/a.shape[1]+(a[tr]**2).sum(1)[None,:]/a.shape[1]-2*K,0)
                K=np.exp(-dis/np.median(dis[tr]))
            one=np.eye(2)[y[tr]]
            out[f]=K@np.linalg.solve(K[tr]+reg*np.eye(tr.sum()),one-one.mean(0))
        else:out[f]=score_neural(m,a if f=="mlp" else m.norm(x,tr,(0,1)),y,tr,f,p,reg)
    return out

def extract(k,rows,p,out):
    route=p["routes"][k];assert sha(route["checkpoint"])==route["checkpoint_sha256"]
    r=attach("27b"); enc=ExternalEncoder(k); bridge=from_checkpoint(route["checkpoint"],"cuda:0")
    proj={}; data={"source":[],"slots":[]}
    def packed(v,stage):
        w=v.shape[-1]
        if stage not in proj:
            ix=KEYS.index(k)*10+("source","slots").index(stage)
            a=[np.random.default_rng(s+ix*10000).standard_normal((w,256),dtype=np.float32)/np.float32(np.sqrt(w))
                for s in p["projection_seeds"]]
            proj[stage]=torch.from_numpy(np.concatenate(a,axis=1)).to("cuda:0")
        f=F.adaptive_avg_pool1d(v.float().T[None],64)[0].T@proj[stage]
        assert torch.isfinite(f).all()
        return f.T.reshape(3,256,64).transpose(1,2).cpu().numpy()
    ts=time.monotonic()
    with torch.inference_mode(),torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH),\
            (out/"predictions.jsonl").open("x",encoding="utf8") as f:
        for i,a in enumerate(rows,1):
            path=a["row"]["images"][0]
            with Image.open(path) as u:im=u.convert("RGB")
            assert pixel(im)==a["rgb"] and sha(path)==a["file"]
            b,g=prepare_external(r,a["row"],im.size)
            x=enc.encode(im,route["tap"])
            z=bridge([x],[g])[0]
            data["source"].append(packed(x["tensor"],"source"))
            data["slots"].append(packed(z,"slots"))
            with external_only(r):v=score_many(r,b,dict(clean=z))["clean"].tolist()
            assert len(v)==2 and np.isfinite(v).all()
            f.write(json.dumps(dict(uid=a["uid"],scores=v),allow_nan=False)+"\n");f.flush()
            r.unchanged()
            if i==1 or i%8==0:
                tmp=out/"live.tmp";write(tmp,dict(stage="score",done=i,N=96,seconds=time.monotonic()-ts));tmp.replace(out/"live.json")
    return {k:np.stack(v,axis=1) for k,v in data.items()}

def run(k,gpu):
    assert os.environ["CUDA_VISIBLE_DEVICES"]==str(gpu) and torch.cuda.device_count()==1
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    plan=read(ROOT/"prospective_plan.json")
    assert all(sha(f)==h for f,h in plan["inputs"].items())
    # Import before legacy model loading changes the module search order.
    m=recover_fit()
    p=read(R8/"plan.json")
    confirm=read(ROOT/"prospective.json")
    old=read(R5/"synthetic.json")+read(R6/"ood.json")+read(R8/"fresh.json")
    allr=old+confirm
    y=np.array([int(v["count"]==4) for v in allr])
    scene=np.array([v["scene"] for v in allr])
    out=ROOT/"reports"/("confirm-"+k)
    olddir=R8/"reports"/k
    base=read(olddir/"probe_results.json")
    archived=np.load(olddir/"probe_scores.npz")
    assert archived["uid"].tolist()==[a["uid"] for a in old]
    sel={(v["stage"],v["dimension"],v["seed"],v["partition"]):(v["original"],v["geometry"])
        for v in read(ROOT/"geometry_selection.json")["details"] if v["encoder"]==k}
    assert len(sel)==len(base)==72
    incoming=extract(k,confirm,p,out)
    maps={st:np.load(olddir/(st+".npy"),mmap_mode="r") for st in ("source","slots")}
    vals={"uid":np.array([a["uid"] for a in confirm])}
    audit=[]
    ts=time.monotonic()
    for i,v in enumerate(base,1):
        st,d,sd,part=(v["stage"],v["dimension"],v["projection_seed"],v["partition"])
        key=(st,d,sd,part); orig,geo=sel[key]
        assert orig==v["chosen"]
        ix=p["projection_seeds"].index(sd)
        x=np.ascontiguousarray(np.concatenate([maps[st][ix,:,:,:d],incoming[st][ix,:,:,:d]]))
        tr=np.isin(scene,next(q["fit"] for q in p["partitions"] if q["name"]==part))
        assert tr.sum()==72 and not tr[len(old):].any()
        got=reconstruct(m,x,y,tr,v["families"],sorted({orig,geo}),p)
        stem=f"{st}_{d}_{sd}_{part}"
        for fam,s in got.items():
            oldscore=archived[stem+"_"+fam]
            error=float(np.abs(oldscore-s[:len(old)]).max())
            disagree=int((oldscore.argmax(1)!=s[:len(old)].argmax(1)).sum())
            assert error<.002 and disagree<=1,(k,stem,fam,error,disagree)
            audit.append(dict(config=stem,family=fam,max_abs_replay=error,old_argmax_disagreement=disagree))
        for name,fam in (("original",orig),("geometry",geo)):
            vals[stem+"_"+name]=got[fam][len(old):].astype(np.float32)
        if i==1 or i%4==0:
            tmp=out/"live.tmp";write(tmp,dict(stage="refit",done=i,N=72,seconds=time.monotonic()-ts));tmp.replace(out/"live.json")
    np.savez_compressed(out/"scores.npz",**vals)
    write(out/"result.json",dict(status="complete",N=96,choices=72,replay=audit))
if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("encoder",choices=KEYS);p.add_argument("--gpu",type=int,required=True)
    a=p.parse_args();run(a.encoder,a.gpu)
