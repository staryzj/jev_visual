"""Recover geometry from recorded processor metadata and explicit model configs.

No square-root inference from sequence length. LLaVA views are mapped through
the exact resize/pad/tile order; its overview and covering tile are averaged.
InternVL's CLS token is not assigned a fake pixel coordinate.
"""
import math
from functools import lru_cache
import torch
from torch.nn import functional as F
import json
from pathlib import Path

@lru_cache(None)
def source_spec(backbone):return json.loads(Path(__file__).with_name('source_geometry.json').read_text(encoding='utf-8'))[backbone]

def best_resolution(original,choices):
    oh,ow=original;best=None;maximum=0;waste=float('inf')
    for h,w in choices:
        scale=min(w/ow,h/oh);effective=min(int(ow*scale)*int(oh*scale),ow*oh);unused=w*h-effective
        if effective>maximum or (effective==maximum and unused<waste):best=(h,w);maximum=effective;waste=unused
    return best

def resize_and_padding(image_size,spec):
    ow,oh=image_size;th,tw=best_resolution((oh,ow),spec['pinpoints']);sw=tw/ow;sh=th/oh
    if sw<sh:nw=tw;nh=min(math.ceil(oh*sw),th)
    else:nh=th;nw=min(math.ceil(ow*sh),tw)
    return dict(target_h=th,target_w=tw,new_h=nh,new_w=nw,pad_x=(tw-nw)//2,pad_y=(th-nh)//2)

def bilinear(h,w,row,col,offset=0):
    row=row.clamp(0,h-1);col=col.clamp(0,w-1);r0=row.floor().long();c0=col.floor().long()
    r1=(r0+1).clamp_max(h-1);c1=(c0+1).clamp_max(w-1);dy=row-r0;dx=col-c0
    indices=torch.stack([r0*w+c0,r0*w+c1,r1*w+c0,r1*w+c1],-1)+offset
    weights=torch.stack([(1-dy)*(1-dx),(1-dy)*dx,dy*(1-dx),dy*dx],-1)
    return indices,weights

def make_mapping(layout,target_grid,spec):
    t,gh,gw=map(int,target_grid);assert t==1 and gh%2==0 and gw%2==0
    h,w=gh//2,gw//2;n=int(layout['shape'][0]);b=layout['encoder'];assert b==spec['backbone']
    yy,xx=torch.meshgrid((torch.arange(h).float()+.5)/h,(torch.arange(w).float()+.5)/w,indexing='ij');u=xx.flatten();v=yy.flatten()
    audit=dict(encoder=b,source_tokens=n,target_grid=list(target_grid),target_tokens=h*w,identity=False,source_CLS_excluded=0)
    if b=='qwen':
        grid=layout['processor_geometry']['image_grid_thw'].tolist()[0];st,sh,sw=map(int,grid)
        assert st==1 and sh%2==0 and sw%2==0 and n==sh*sw//4
        sh//=2;sw//=2
        if (sh,sw)==(h,w):
            return dict(identity=True,indices=torch.arange(n)[:,None],weights=torch.ones(n,1),audit=dict(audit,identity=True))
        idx,weight=bilinear(sh,sw,v*sh-.5,u*sw-.5)
    elif b in ['internvl','siglip2']:
        sh,sw=layout.get('canonical_grid',[spec['grid_h'],spec['grid_w']]);cls=0 if 'canonical_grid' in layout else spec['CLS_tokens'];assert n==sh*sw+cls
        if b=='internvl':assert layout['processor_geometry']['num_patches'].tolist()==[1]
        idx,weight=bilinear(sh,sw,v*sh-.5,u*sw-.5,cls);audit['source_CLS_excluded']=cls
    elif b=='llava':
        g=spec['grid_h'];assert g==spec['grid_w'];tile=spec['image_h'];patch=spec['patch_h'];assert tile==spec['image_w']
        geo=resize_and_padding(layout['image_size'],spec);nh,nw=geo['new_h'],geo['new_w'];th,tw=geo['target_h'],geo['target_w']
        assert th%tile==0 and tw%tile==0
        rows,cols=th//tile,tw//tile;assert n==(1+rows*cols)*g*g,(n,geo)
        # Global overview is stretched to image_h x image_w, not padded.
        base_i,base_w=bilinear(g,g,v*tile/patch-.5,u*tile/patch-.5)
        px=u*nw+geo['pad_x'];py=v*nh+geo['pad_y'];tc=(px/tile).floor().long().clamp_max(cols-1);tr=(py/tile).floor().long().clamp_max(rows-1)
        local_i,local_w=bilinear(g,g,(py-tr*tile)/patch-.5,(px-tc*tile)/patch-.5,(1+tr*cols+tc)[:,None]*g*g)
        # Exclude patch centers wholly in padding, retaining all supported views.
        inner=local_i%(g*g);patch_x=(inner%g+.5)*patch+tc[:,None]*tile;patch_y=(inner//g+.5)*patch+tr[:,None]*tile
        valid=(patch_x>=geo['pad_x'])&(patch_x<geo['pad_x']+nw)&(patch_y>=geo['pad_y'])&(patch_y<geo['pad_y']+nh)
        local_w=local_w*valid;mass=local_w.sum(-1,keepdim=True);supported=mass>0
        local_w=local_w/mass.clamp_min(1e-12)
        views=1+supported.float();base_w=base_w/views;local_w=local_w/views
        idx=torch.cat([base_i,local_i],-1);weight=torch.cat([base_w,local_w],-1)
        audit.update(geo,tiles=rows*cols,local_unsupported_queries=int((~supported).sum()),overview_plus_local_coverage_average=True)
    else:raise ValueError(b)
    assert idx.min()>=0 and idx.max()<n and torch.isfinite(weight).all() and (weight>=0).all()
    assert torch.allclose(weight.sum(-1),torch.ones(h*w),atol=2e-6)
    return dict(identity=False,indices=idx,weights=weight,audit=audit)

def align(record,grid):
    # Records are task-local containers; cache only the tiny sampling map.
    key=tuple(map(int,grid));maps=record.setdefault('_maps',{})
    if key not in maps:maps[key]=make_mapping(record['layout'],grid,source_spec(record['layout']['encoder']))
    mapping=maps[key];x=record['tensor']
    if mapping['identity']:
        # Preserve the reference memory layout as well as values: GEMM kernels
        # may round differently for row-major vs transposed identical inputs.
        return F.interpolate(x.T[None],size=len(x),mode='linear',align_corners=False)[0].T
    i=mapping['indices'].to(x.device);w=mapping['weights'].to(x.device)
    return (x[i]*w[...,None]).sum(1)
