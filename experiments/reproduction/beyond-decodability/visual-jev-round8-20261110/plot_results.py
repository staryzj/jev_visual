"""Draw primary validation-selected readouts and the paired patch contrast."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
D=Path(__file__).resolve().parent
R=json.loads((D/"summary.json").read_text(encoding="utf8"))["routes"]
keys=("llava","internvl","siglip2","qwen")
names=("LLaVA","InternVL","SigLIP2","Qwen")
rc="#145780";pc="#B04D2D";gc="#187365";ic="#73518F"
def combined(cells):
    n=sum(v["N"] for v in cells)
    mean=sum(v["mean"]*v["N"] for v in cells)/n
    var=sum((v["N"]-1)*v["sd"]**2+v["N"]*(v["mean"]-mean)**2 for v in cells)/(n-1)
    return 100*mean,100*np.sqrt(var)
def h(name,c,m):
    return Line2D([],[],color=c,marker=m,linestyle="none",markersize=4,label=name)
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":8,"axes.labelsize":8,
    "axes.titlesize":8.3,"pdf.fonttype":42,"axes.linewidth":.6})
fig,axs=plt.subplots(1,3,figsize=(6.95,2.03))
fig.subplots_adjust(left=.106,right=.988,top=.79,bottom=.30,wspace=.28)
y=np.arange(4)
for i,a in enumerate(axs):
    a.set_ylim(3.58,-.58);a.set_axisbelow(True)
    a.grid(axis="x",color="#e7ebf0",lw=.55)
    for sp in ("top","right","left"):a.spines[sp].set_visible(False)
    a.tick_params(axis="y",length=0,pad=3,labelsize=7.8)
    a.tick_params(axis="x",length=2,width=.6,pad=2,labelsize=7.2)
    a.set_yticks(y);a.set_yticklabels(names if i==0 else [""]*4)
for i,(a,dom,title) in enumerate(zip(axs[:2],("old_equal_area","fresh_equal_area"),
    ("(a) Prior shift, locked choice","(b) New shift, repeated choices"))):
    a.set_title(title,loc="left",fontweight="bold",pad=7)
    ref=100*np.array([R[k]["frozen"][dom] for k in keys])
    if i==0: mu=100*np.array([R[k]["prior18"]["accuracy"] for k in keys]);sd=np.zeros(len(keys))
    else: mu,sd=np.array([combined([v["metrics"][dom]["accuracy"] for v in R[k]["conditions"]["source"].values()]) for k in keys]).T
    for j in range(len(y)):
        a.plot([ref[j],mu[j]],[y[j]-.105,y[j]+.105],color="#aab4be",lw=.65,zorder=1)
    a.scatter(ref,y-.105,c=rc,marker="o",s=23,zorder=3)
    a.errorbar(mu,y+.105,xerr=sd,color=pc,fmt="s",ms=3.1,elinewidth=.8,capsize=2,linestyle="none")
    a.axvline(50,color="#777777",ls=(0,(2,2)),lw=.65)
    a.set_xlim(min(45,5*np.floor((mu-sd).min()/5)),106)
    a.set_xticks([50,75,100]);a.set_xlabel("Accuracy (%)",labelpad=2)
a=axs[2]
a.set_title("(c) Local slot effects",loc="left",fontweight="bold",pad=7)
bounds=[]
for tag,color,m,o in (("restore_vs_background",gc,"^",-.12),("insert_vs_background",ic,"D",.12)):
    v=[R[k]["patches"]["contrast"][tag]["margin"] for k in keys]
    d=np.array([q["delta"] for q in v]);lo,hi=np.array([q["CI"] for q in v]).T
    bounds+=list(lo)+list(hi)
    a.errorbar(d,y+o,xerr=[d-lo,hi-d],color=color,fmt=m,ms=3,elinewidth=.85,capsize=2,linestyle="none")
a.axvline(0,color="#777777",ls=(0,(2,2)),lw=.65)
a.set_xlim(min(-.25,min(bounds)-.10),max(.25,max(bounds)+.10))
a.set_xlabel("Gold-margin difference",labelpad=2)
fig.legend(handles=[h("Frozen receiver",rc,"o"),h("Validation-selected source probe",pc,"s")],
    loc="lower left",bbox_to_anchor=(.113,.021),ncol=2,frameon=False,
    handletextpad=.25,columnspacing=.95,fontsize=7.3)
fig.legend(handles=[h("Restore clean",gc,"^"),h("Insert cover",ic,"D")],
    loc="lower right",bbox_to_anchor=(.986,.021),ncol=2,frameon=False,
    handletextpad=.25,columnspacing=.85,fontsize=7.3)
fig.savefig(D/"round8-validation-mechanism.pdf")
