"""Seal a paired orientation task with exactly matched global RGB histograms."""
import hashlib,json,random
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parent
BASE=ROOT.parent
R8=BASE/"visual-jev-round8-20261110"
R6=BASE/"visual-jev-round6-20261110"
R5=BASE/"visual-jev-round5-20261110"
R2=BASE/"visual-jev-round2-controls-20261110"
KEYS=("llava","internvl","siglip2","qwen")
SEED=202611109
WORDS=("up","down")

def read(p):return json.loads(Path(p).read_text(encoding="utf8"))
def write(p,v):
    with Path(p).open("x",encoding="utf8") as f:json.dump(v,f,indent=2,allow_nan=False)
def sha(p):
    with Path(p).open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()
def pixel(im):
    return hashlib.sha256(str((im.size,hashlib.sha256(im.tobytes()).hexdigest())).encode()).hexdigest()

def glyph(size,color,background,direction):
    p=Image.new("RGB",(size,size),background)
    d=ImageDraw.Draw(p)
    m=size//2
    d.polygon([(m,3),(size-4,m),(m+size//6,m),(m+size//6,size-4),
       (m-size//6,size-4),(m-size//6,m),(3,m)],fill=color)
    return p if direction=="up" else p.transpose(Image.Transpose.FLIP_TOP_BOTTOM)

def render():
    (ROOT/"images").mkdir()
    rows=[]
    for i in range(48):
        rng=random.Random(SEED+i)
        w,h=((448,448),(576,432),(432,576))[i%3]
        x=rng.randint(w//3,2*w//3); y=rng.randint(h//3,2*h//3)
        bg=tuple(rng.randint(220,249) for _ in range(3))
        base=Image.new("RGB",(w,h),bg)
        dr=ImageDraw.Draw(base)
        r=63
        dr.rectangle((x-r,y-r,x+r,y+r),outline=(16,22,24),width=4)
        noise=(i//3)%2==1
        palette=((5,107,136),(185,49,55),(21,133,79),(100,62,159))
        if noise:
            for cx,cy in ((58,58),(w-58,58),(58,h-58),(w-58,h-58)):
                s=rng.randint(29,54)
                p=(cx-s//2,cy-s//2)
                base.paste(glyph(s,rng.choice(palette),bg,rng.choice(WORDS)),p)
        z=rng.randint(45,68)
        left,top=x-z//2,y-z//2
        p=(left,top,left+z,top+z)
        color=rng.choice(palette)
        opts=list(WORDS);rng.shuffle(opts)
        pair=[]
        for direction in WORDS:
            im=base.copy();im.paste(glyph(z,color,bg,direction),(left,top))
            uid=f"arrow:{i:02}:{direction}"
            dst=ROOT/"images"/(uid.replace(":","-")+".png")
            im.save(dst)
            row=dict(uid=uid,dataset="arrow_orientation",images=[str(dst)],candidates=opts,
                label=opts.index(direction),question="Which way does the arrow inside the black square point?",
                metadata={},reference_answers=[],source_split="test")
            pair.append((np.array(im),dict(uid=uid,row=row,scene=f"arrow:{i:02}",box=p,
                direction=direction,clutter=noise,rgb=pixel(im),file=sha(dst))))
        u,v=(a for a,_ in pair)
        m=np.any(u!=v,axis=2)
        assert m.sum()>0 and m[top:top+z,left:left+z].sum()==m.sum()
        assert np.array_equal(np.sort(u.reshape(-1,3),axis=0),np.sort(v.reshape(-1,3),axis=0))
        assert {(tuple(c),int(n)) for c,n in zip(*np.unique(u[top:top+z,left:left+z].reshape(-1,3),axis=0,return_counts=True))}=={
                (tuple(c),int(n)) for c,n in zip(*np.unique(v[top:top+z,left:left+z].reshape(-1,3),axis=0,return_counts=True))}
        rows.extend(a for _,a in pair)
    assert len(rows)==len({a["rgb"] for a in rows})==96
    assert sum(a["clutter"] for a in rows)==48
    write(ROOT/"arrows.json",rows)

def seal():
    render()
    a=read(ROOT/"arrows.json")
    old=read(R8/"fresh.json")+read(R6/"ood.json")+read(R5/"synthetic.json")
    assert {v["rgb"] for v in a}.isdisjoint({v["rgb"] for v in old})
    routes=read(R6/"plan.json")["routes"]
    inputs=[ROOT/n for n in ("prepare.py","worker.py","arrows.json")]
    write(ROOT/"plan.json",dict(routes=routes,seed=SEED,N=96,inputs={str(p):sha(p) for p in inputs},
        primary="Paired clean correctness on both labels and label-directed margin for local vs. matched background swapping. Score both directions.",
        controlled="Only arrow pixels inside the target box change. Each pair preserves image dimensions, exact RGB histogram, candidates, prompt, and background."))
if __name__=="__main__":seal()
