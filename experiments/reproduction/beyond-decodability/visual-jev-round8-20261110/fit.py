"""Fit matched probes and lock each family and regularizer by validation only."""
import time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from prepare import ROOT,R5,R6,read,write

def acc(z,y):return float((z.argmax(1)==y).mean())
def ce(z,y):
    a=z-z.max(1,keepdims=True)
    return float(np.mean(np.log(np.exp(a).sum(1))-a[np.arange(len(y)),y]))
def norm(z,fit,axes):
    return (z-z[fit].mean(axis=axes))/np.maximum(z[fit].std(axis=axes),1e-5)

class MLP(nn.Module):
    def __init__(self,c):
        super().__init__();self.f=nn.Linear(c,64);self.out=nn.Linear(64,2)
    def forward(self,x):return self.out(F.gelu(self.f(x)))

class Attn(nn.Module):
    def __init__(self,c):
        super().__init__();self.key=nn.Linear(c,32);self.value=nn.Linear(c,32)
        self.query=nn.Linear(32,1);self.out=nn.Linear(32,2)
    def forward(self,x):return self.out((self.query(self.key(x).tanh()).softmax(1)*self.value(x)).sum(1))

def neural(x,y,tr,net,p,device):
    x=torch.from_numpy(x.astype(np.float32)).to(device)
    yy=torch.from_numpy(y[tr]).to(device)
    xx=x[tr]
    model=MLP if net=="mlp" else Attn
    layers=[]
    for seed in p["neural_seeds"]:
        torch.manual_seed(seed);layers.append(model(x.shape[-1]).to(device))
    params,buffers=torch.func.stack_module_state(layers)
    def evaluate(state,b,data):return torch.func.functional_call(layers[0],(state,b),(data,))
    vmap=torch.vmap(evaluate,in_dims=(0,0,None))
    n=len(layers);fits=[]
    for wd in p["decay_grid"]:
        q={k:v.detach().clone().requires_grad_(True) for k,v in params.items()}
        opt=torch.optim.AdamW(q.values(),lr=.005,weight_decay=wd)
        for _ in range(p["epochs"]):
            opt.zero_grad(set_to_none=True)
            log=F.log_softmax(vmap(q,buffers,xx),dim=-1)
            loss=-log.gather(-1,yy[None,:,None].expand(n,-1,1)).mean(1).sum()
            loss.backward();opt.step()
        with torch.no_grad():z=vmap(q,buffers,x).cpu().numpy()
        assert z.shape==(n,len(x),2) and np.isfinite(z).all()
        fits.append((wd,z.mean(0),z))
    return fits

def all_families(x,y,tr,cal,p,device):
    a=np.concatenate([x.mean(1),x.std(1),x.max(1),x[:,:32].mean(1)-x[:,32:].mean(1)],1)
    a=norm(a,tr,0).astype(np.float64)
    kernel=a@a[tr].T/a.shape[1]
    dis=np.maximum((a*a).sum(1)[:,None]/a.shape[1]+(a[tr]**2).sum(1)[None,:]/a.shape[1]-2*kernel,0)
    med=np.median(dis[tr]);assert med>0
    ans={}
    one=np.eye(2)[y[tr]]
    for fam,K in (("linear",kernel),("kernel",np.exp(-dis/med))):
        records=[]
        for reg in p["ridge_grid"]:
            z=K@np.linalg.solve(K[tr]+reg*np.eye(tr.sum()),one-one.mean(0))
            records.append((ce(z[cal],y[cal]),reg,z,z[None]))
        ans[fam]=min(records,key=lambda v:v[:2])
    for fam,a0 in (("mlp",a),("attention",norm(x,tr,(0,1)))):
        ans[fam]=min(((ce(z[cal],y[cal]),wd,z,individual) for wd,z,individual in neural(a0,y,tr,fam,p,device)),key=lambda v:v[:2])
    return ans

def run(k,device="cuda:0"):
    p=read(ROOT/"plan.json"); out=ROOT/"reports"/k
    start=time.monotonic()
    rows=read(R5/"synthetic.json")+read(R6/"ood.json")+read(ROOT/"fresh.json")
    ids=[v["uid"] for v in rows]
    assert read(out/"ids.json")==ids
    y=np.array([int(v["count"]==4) for v in rows])
    scene=np.array([v["scene"] for v in rows])
    domains={"original_test":np.array([v["split"]=="test" for v in rows])}
    for c in ("old","fresh"):
        for reg in ("equal_area","same_instance_sizes"):
            domains[c+"_"+reg]=np.array([v.get("regime")==reg and (v["uid"].startswith("fresh:")== (c=="fresh")) for v in rows])
    assert [int(a.sum()) for a in domains.values()]==[96,48,48,48,48]
    table=[];values={"uid":np.array(ids)}
    for stage in ("source","slots"):
        a=np.load(out/(stage+".npy"),mmap_mode="r")
        for ii,seed in enumerate(p["projection_seeds"]):
            for d in p["dimensions"]:
                x=np.ascontiguousarray(a[ii,:,:,:d])
                for split in p["partitions"]:
                    tr=np.isin(scene,split["fit"]);cal=np.isin(scene,split["validation"])
                    assert (int(tr.sum()),int(cal.sum()))==(72,36) and not (tr&cal).any()
                    assert all(not (m&(tr|cal)).any() for m in domains.values())
                    fits=all_families(x,y,tr,cal,p,device)
                    choice=min(fits,key=lambda family:(fits[family][0],family))
                    r=dict(stage=stage,dimension=d,projection_seed=seed,partition=split["name"],chosen=choice,families={})
                    for fam,(c,reg,z,individual) in fits.items():
                        tag=f"{stage}_{d}_{seed}_{split['name']}_{fam}"
                        values[tag]=z.astype(np.float32)
                        r["families"][fam]=dict(validation_ce=c,regularizer=reg,
                            accuracy={key:acc(z[m],y[m]) for key,m in domains.items()},
                            neural_seed_accuracy={key:[acc(v[m],y[m]) for v in individual] for key,m in domains.items()})
                    table.append(r)
                    tmp=out/"fit_live.tmp";write(tmp,dict(done=len(table),N=72,seconds=time.monotonic()-start));tmp.replace(out/"fit_live.json")
    assert len(table)==72
    np.savez_compressed(out/"probe_scores.npz",**values)
    write(out/"probe_results.json",table)

if __name__=="__main__":
    import sys
    run(sys.argv[1],sys.argv[2] if len(sys.argv)>2 else "cuda:0")
