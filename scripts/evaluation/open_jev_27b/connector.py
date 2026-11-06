"""One bridge API; preserves selected legacy tensor layouts and state-dict names."""
import torch
from torch import nn
from torch.nn import functional as F
from .geometry import align

class Connector(nn.Module):
    def __init__(self,input_dim,output_dim,arm='linear',width=512,backbone=None,residual_width=None):
        super().__init__(); assert arm in ['linear','geometric']
        self.spec=dict(input_dim=input_dim,output_dim=output_dim,arm=arm,width=width)
        if backbone is not None: self.spec['backbone']=backbone
        if residual_width is not None: self.spec['residual_width']=residual_width
        self.norm=nn.LayerNorm(input_dim); self.projection=nn.Linear(input_dim,output_dim)
        if residual_width is not None:
            self.residual_in=nn.Linear(input_dim,residual_width)
            self.residual_out=nn.Linear(residual_width,output_dim)
            nn.init.zeros_(self.residual_out.weight); nn.init.zeros_(self.residual_out.bias)
        self.register_buffer('teacher_mean',torch.zeros(output_dim))
        self.register_buffer('teacher_std',torch.ones(output_dim))

    def forward(self,records,grids):
        assert len(records)==len(grids); result=[]
        for record,grid in zip(records,grids):
            value=record['tensor'] if isinstance(record,dict) else record
            assert value.ndim==2 and value.shape[-1]<=self.spec['input_dim']
            # Historical original-width linear baseline must not change memory layout.
            if 'backbone' in self.spec:
                value=F.pad(value,(0,self.spec['input_dim']-value.shape[-1]))
            value=self.norm(value); t,h,w=map(int,grid)
            assert t==1 and h%2==0 and w%2==0
            if self.spec['arm']=='geometric':
                assert isinstance(record,dict)
                normalized=dict(record,tensor=value); normalized['_maps']=record.setdefault('_maps',{})
                sampled=align(normalized,grid)
            else:
                sampled=F.interpolate(value.T[None],size=h*w//4,mode='linear',align_corners=False)[0].T
            raw=self.projection(sampled)
            if 'residual_width' in self.spec:
                raw=raw+self.residual_out(F.gelu(self.residual_in(sampled)))
            result.append(raw*self.teacher_std+self.teacher_mean)
        return result

def feature_loss(pred,target,std):
    return torch.stack([((a-b.float())/std).square().mean() for a,b in zip(pred,target)]).mean()

def relation_loss(pred,target,std):
    values=[]
    for a,b in zip(pred,target):
        a=F.normalize(a.float()/std,dim=-1,eps=1e-6)
        b=F.normalize(b.detach().float()/std,dim=-1,eps=1e-6)
        assert len(a)==len(b) and len(a)>1; total=a.sum()*0
        for start in range(0,len(a),128):
            diff=a[start:start+128]@a.T-b[start:start+128]@b.T
            mask=torch.ones_like(diff,dtype=torch.bool)
            ix=torch.arange(len(diff),device=a.device); mask[ix,ix+start]=False
            total=total+diff[mask].square().sum()
        values.append(total/(len(a)*(len(a)-1)))
    return torch.stack(values).mean()

def alignment_loss(pred,target,std,mode):
    if mode=='mse': return feature_loss(pred,target,std)
    if mode=='mse_relation': return feature_loss(pred,target,std)+relation_loss(pred,target,std)
    assert mode=='spatial'; values=[]
    for a,b in zip(pred,target):
        a=a.float()/std; b=b.float()/std
        token=(1-F.cosine_similarity(a,b,dim=-1,eps=1e-4)).clamp_min(0).mean()
        ac=a-a.mean(0,keepdim=True); bc=b-b.mean(0,keepdim=True); mask=bc.norm(dim=-1)>1e-4
        spatial=(1-F.cosine_similarity(ac[mask],bc[mask],dim=-1,eps=1e-4)).clamp_min(0).mean() if mask.any() else a.sum()*0
        values.append(.45*token+.45*spatial+.10*F.smooth_l1_loss(a,b,reduction='mean'))
    return torch.stack(values).mean()

def from_checkpoint(path,device='cpu',frozen=True):
    c=torch.load(path, map_location='cpu', weights_only=True)
    bridge=Connector(**c['spec']).to(device)
    bridge.load_state_dict(c['bridge'],strict=True)
    bridge.train(not frozen).requires_grad_(not frozen)
    return bridge
