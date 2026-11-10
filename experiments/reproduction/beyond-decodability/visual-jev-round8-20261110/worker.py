"""Extract nested projections and score linked same-image token interventions."""
import argparse,json,os,sys,time
from collections import defaultdict
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from prepare import ROOT,R2,R5,R6,R7,KEYS,read,sha,write
sys.path.insert(0,str(R2))
from independent_worker import ExternalEncoder,attach,external_only,from_checkpoint,score_many
from u_official import prepare_external

def opened(a):
    p=a["row"]["images"][0]
    with Image.open(p) as im:v=im.convert("RGB")
    import hashlib
    h=hashlib.sha256(str((v.size,hashlib.sha256(v.tobytes()).hexdigest())).encode()).hexdigest()
    assert h==a["rgb"] and sha(p)==a["file"]
    return v

def matching_slots(image,changed,box,grid):
    x0,y0,x1,y1=box
    t,hh,ww=grid
    assert t==1 and hh%2==ww%2==0
    h,w=hh//2,ww//2
    W,H=image.size
    u,v=np.asarray(image),np.asarray(changed)
    bg=u[0,0]
    assert not np.array_equal(u,v)
    active=[]; free=set()
    for j in range(h):
        for i in range(w):
            l,r=round(i*W/w),round((i+1)*W/w)
            a,b=round(j*H/h),round((j+1)*H/h)
            k=j*w+i
            if max(l,x0)<min(r,x1) and max(a,y0)<min(b,y1):active.append(k)
            elif np.all(u[a:b,l:r]==bg) and np.all(v[a:b,l:r]==bg):free.add(k)
    assert active and len(free)>=len(active)
    control=[]
    for k in active:
        z=min(free,key=lambda i:((i//w-k//w)**2+(i%w-k%w)**2,i))
        control.append(z);free.remove(z)
    assert len(active)==len(control) and set(active).isdisjoint(control)
    return active,control

def graft(z,other,slots):
    y=z.clone()
    y[slots]=other[slots].to(y.dtype)
    return y

def run(k,gpu):
    assert os.environ["CUDA_VISIBLE_DEVICES"]==str(gpu) and torch.cuda.device_count()==1
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    p=read(ROOT/"plan.json")
    assert all(sha(f)==h for f,h in p["inputs"].items())
    route=p["routes"][k];assert sha(route["checkpoint"])==route["checkpoint_sha256"]
    old=read(R5/"synthetic.json")
    ood=read(R6/"ood.json")
    fresh=read(ROOT/"fresh.json")
    cover={a["parent"]:a for a in read(R7/"interventions.json") if a["variant"]=="target_cover"}
    rows=old+ood+fresh
    assert len(rows)==396 and len({v["uid"] for v in rows})==396
    r=attach("27b"); enc=ExternalEncoder(k); bridge=from_checkpoint(route["checkpoint"],"cuda:0")
    projection={}; seen_width={}
    out=ROOT/"reports"/k
    dst={}
    for stage in ("source","slots"):
        dst[stage]=np.lib.format.open_memmap(out/(stage+".npy"),mode="w+",dtype=np.float32,
            shape=(len(p["projection_seeds"]),len(rows),64,max(p["dimensions"])))
    t=time.monotonic()
    def packed(v,stage):
        w=v.shape[-1]
        if stage not in projection:
            ix=KEYS.index(k)*10+("source","slots").index(stage)
            proj=[np.random.default_rng(s+ix*10000).standard_normal((w,max(p["dimensions"])),dtype=np.float32)/np.float32(np.sqrt(w))
                  for s in p["projection_seeds"]]
            projection[stage]=torch.from_numpy(np.concatenate(proj,axis=1)).to("cuda:0")
            seen_width[stage]=w
        assert w==seen_width[stage]
        f=F.adaptive_avg_pool1d(v.float().T[None],64)[0].T@projection[stage]
        assert torch.isfinite(f).all()
        return f.T.reshape(len(p["projection_seeds"]),max(p["dimensions"]),64).transpose(1,2).cpu().numpy()
    with torch.inference_mode(),torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH),\
        (out/"predictions.jsonl").open("x",encoding="utf8") as f:
        for i,a in enumerate(rows):
            im=opened(a)
            batches,g=prepare_external(r,a["row"],im.size)
            x=enc.encode(im,route["tap"])
            z=bridge([x],[g])[0]
            dst["source"][:,i]=packed(x["tensor"],"source")
            dst["slots"][:,i]=packed(z,"slots")
            values=None; audit=None
            if a["uid"] in cover:
                item=cover[a["uid"]]
                covered=opened(item)
                obj,bg=matching_slots(im,covered,item["rectangle"],g)
                zz=bridge([enc.encode(covered,route["tap"])],[g])[0]
                assert z.shape==zz.shape==(g[0]*g[1]*g[2]//4,5120)
                values=dict(clean=z,cover=zz,restore_local=graft(zz,z,obj),restore_background=graft(zz,z,bg),
                            insert_local=graft(z,zz,obj),insert_background=graft(z,zz,bg))
                audit=dict(local=obj,background=bg,grid=g,cover_uid=item["uid"],rectangle=item["rectangle"])
            elif a["split"]=="ood" and a["uid"].startswith("fresh:"):
                values=dict(clean=z)
            if values is not None:
                with external_only(r):
                    res=score_many(r,batches,values)
                scores={tag:u.tolist() for tag,u in res.items()}
                assert all(len(v)==2 and np.isfinite(v).all() for v in scores.values())
                f.write(json.dumps(dict(uid=a["uid"],scores=scores,audit=audit),allow_nan=False)+"\n");f.flush()
                r.unchanged()
            if i==0 or (i+1)%8==0 or i+1==len(rows):
                tmp=out/"live.tmp";write(tmp,dict(done=i+1,N=len(rows),seconds=time.monotonic()-t));tmp.replace(out/"live.json")
    for ar in dst.values():ar.flush()
    write(out/"ids.json",[v["uid"] for v in rows])
    write(out/"result.json",dict(status="complete",N=len(rows),scored=144,dimensions=p["dimensions"],
        projection_seeds=p["projection_seeds"],source_widths=seen_width,checkpoint_sha256=route["checkpoint_sha256"],
        optimizer_steps=0,receiver=r.identity))
if __name__=="__main__":
    a=argparse.ArgumentParser()
    a.add_argument("encoder",choices=KEYS);a.add_argument("--gpu",type=int,required=True)
    v=a.parse_args();run(v.encoder,v.gpu)
    import fit
    fit.run(v.encoder)
