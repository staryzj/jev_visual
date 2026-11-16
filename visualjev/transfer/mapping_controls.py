"""Recorded dense 2D controls at fixed connector and receiver weights."""
import torch
from torch.nn import functional as F
from .geometry import bilinear, make_mapping, resize_and_padding, source_spec, align

ARMS = ("grid_2d", "unpad_grid_2d", "processor_2d")
SPEC = source_spec("llava")

def target_coordinates(grid):
    t,h,w=grid
    assert t==1 and h%2==0 and w%2==0
    h,w=h//2,w//2
    y,x=torch.meshgrid((torch.arange(h).float()+.5)/h,(torch.arange(w).float()+.5)/w,indexing="ij")
    return x.flatten(),y.flatten()


def maps(layout,grid):
    x,y=target_coordinates(grid)
    g,tile,patch=SPEC["grid_h"],SPEC["image_h"],SPEC["patch_h"]
    geo=resize_and_padding(layout["image_size"],SPEC)
    nr,nc=geo["target_h"]//tile,geo["target_w"]//tile
    nw,nh,px,py=[geo[k] for k in ("new_w","new_h","pad_x","pad_y")]
    H,W=nr*g,nc*g
    assert layout["shape"][0]==(1+nr*nc)*g*g
    processor=make_mapping(layout,grid,SPEC)
    assert not processor["identity"]
    overview_i,overview_w=bilinear(g,g,y*tile/patch-.5,x*tile/patch-.5)
    assert torch.equal(processor["indices"][:,:4],overview_i)
    X=(torch.arange(W)//g)*tile+(torch.arange(W)%g+.5)*patch
    Y=(torch.arange(H)//g)*tile+(torch.arange(H)%g+.5)*patch
    valid_x=torch.where((X>=px)&(X<px+nw))[0]
    valid_y=torch.where((Y>=py)&(Y<py+nh))[0]
    assert len(valid_x) and len(valid_y)
    assert int(valid_x[-1]-valid_x[0]+1)==len(valid_x)
    assert int(valid_y[-1]-valid_y[0]+1)==len(valid_y)

    def dense(top,left,h,w):
        local_i,local_w=bilinear(h,w,y*h-.5,x*w-.5)
        rr,cc=local_i//w+top,local_i%w+left
        native_i=(1+(rr//g)*nc+cc//g)*g*g+(rr%g)*g+cc%g
        return dict(indices=torch.cat((overview_i,native_i),1),
                    weights=torch.cat((overview_w,local_w),1)*.5)
    padded=dense(0,0,H,W)
    unpadded=dense(int(valid_y[0]),int(valid_x[0]),len(valid_y),len(valid_x))
    rows=[]
    yy,xx=torch.meshgrid((torch.arange(g).float()+.5)*patch,(torch.arange(g).float()+.5)*patch,indexing="ij")
    rows.append(torch.stack((xx/tile,yy/tile),-1).reshape(-1,2))
    for r in range(nr):
        for c in range(nc):
            rows.append(torch.stack(((xx+c*tile-px)/nw,(yy+r*tile-py)/nh),-1).reshape(-1,2))
    source_xy=torch.cat(rows)
    target_xy=torch.stack((x,y),1)
    output=dict(zip(ARMS,(padded,unpadded,processor)))
    for m in output.values():
        i,w=m["indices"],m["weights"]
        assert i.min()>=0 and i.max()<len(source_xy) and (w>=0).all()
        assert torch.allclose(w.sum(1),torch.ones(len(target_xy)),atol=2e-6)
        m["mean_displacement"]=float(((source_xy[i]-target_xy[:,None,:]).norm(dim=-1)*w).sum(1).mean())
        outside=((source_xy<0)|(source_xy>=1)).any(1)
        m["padding_mass"]=float((outside[i]*w).sum(1).mean())
    assert output["processor_2d"]["padding_mass"]==0.
    assert output["unpad_grid_2d"]["padding_mass"]==0.
    return output



def map_with_fixed_weights(bridge, record, grid, mapping="selected"):
    if mapping == "selected":
        return bridge([record], [grid])[0]
    value = record["tensor"]
    if "backbone" in bridge.spec:
        value = F.pad(value, (0, bridge.spec["input_dim"] - value.shape[-1]))
    value = bridge.norm(value)
    if mapping == "flat":
        sampled = F.interpolate(value.T[None], size=int(grid[1])*int(grid[2])//4,
                                mode="linear", align_corners=False)[0].T
    elif mapping in ("crop", "coordinates"):
        if record["layout"]["encoder"] != "llava":
            raise ValueError("The dense crop control is defined for LLaVA")
        table = maps(record["layout"], grid)["unpad_grid_2d" if mapping == "crop" else "processor_2d"]
        sampled = (value[table["indices"].to(value.device)] *
                   table["weights"].to(value.device)[..., None]).sum(1)
    else:
        raise ValueError(mapping)
    raw = bridge.projection(sampled)
    if "residual_width" in bridge.spec:
        raw = raw + bridge.residual_out(F.gelu(bridge.residual_in(sampled)))
    return raw * bridge.teacher_std + bridge.teacher_mean
