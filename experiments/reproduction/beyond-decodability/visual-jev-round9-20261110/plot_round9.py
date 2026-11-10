"""Plot confirmation accuracy and matched arrow counterfactuals."""
from pathlib import Path
import json,numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
D=Path(__file__).resolve().parent
def data(n):return json.loads((D/n).read_text(encoding="utf8"))
A=data("arrow_summary.json")["routes"];E=data("arrow_effects.json")
P=data("prospective_summary.json")["routes"]
K=("llava","internvl","siglip2","qwen")
names=("LLaVA","InternVL","SigLIP2","Qwen")
blue="#155982";red="#a94c30";green="#1b7768"
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":8,"axes.labelsize":8,
                      "axes.titlesize":8.2,"pdf.fonttype":42,"axes.linewidth":.6})
f,axs=plt.subplots(1,3,figsize=(6.95,2.08))
f.subplots_adjust(left=.106,right=.988,top=.80,bottom=.30,wspace=.26)
y=np.arange(4)
for i,ax in enumerate(axs):
    ax.set_ylim(3.55,-.55);ax.set_axisbelow(True);ax.grid(axis="x",lw=.55,color="#e7ebf0")
    for s in ("top","right","left"):ax.spines[s].set_visible(False)
    ax.tick_params(axis="y",length=0,pad=3,labelsize=7.8)
    ax.tick_params(axis="x",length=2,width=.6,pad=2,labelsize=7.2)
    ax.set_yticks(y);ax.set_yticklabels(names if i==0 else [""]*4)
a=axs[0]
a.set_title("(a) New count scenes",loc="left",fontweight="bold",pad=7)
vals=[]
for k,c,m,o in (("frozen",blue,"o",-.20),("original",red,"s",0),("geometry",green,"^",.20)):
    score=100*np.array([P[t]["stages"]["source"]["equal_area"][k] for t in K]);vals.extend(score)
    a.scatter(score,y+o,color=c,marker=m,s=19,zorder=3)
a.set_xlim(min(45,5*np.floor(min(vals)/5)),105);a.set_xticks([50,75,100])
a.axvline(50,color="#777777",ls=(0,(2,2)),lw=.65);a.set_xlabel("Accuracy (%)",labelpad=2)
b=axs[1]
b.set_title("(b) Local share of image effect",loc="left",fontweight="bold",pad=7)
z=np.array([[E[k]["local_share"],*E[k]["local_share_CI"]] for k in K])*100
b.errorbar(z[:,0],y,xerr=[z[:,0]-z[:,1],z[:,2]-z[:,0]],fmt="^",color=green,
           ms=3.7,elinewidth=.8,capsize=2,linestyle="none")
b.set_xlim(-3,105);b.set_xticks([0,25,50,75,100]);b.set_xlabel("Full margin shift recovered (%)",labelpad=2)
c=axs[2]
c.set_title("(c) Two-direction arrow test",loc="left",fontweight="bold",pad=7)
for key,co,mark,o in (("clean_both",blue,"o",-.20),("background_donor_both",red,"s",0),("local_donor_both",green,"^",.20)):
    c.scatter([A[k]["all"][key] for k in K],y+o,color=co,marker=mark,s=19,zorder=3)
c.set_xlim(-2,50);c.set_xticks([0,12,24,36,48]);c.set_xlabel("Pairs following both labels",labelpad=2)
def h(name,color,marker):return Line2D([],[],color=color,marker=marker,linestyle="none",markersize=4,label=name)
f.legend(handles=[h("Frozen receiver",blue,"o"),h("Original validation",red,"s"),h("Geometry validation",green,"^")],
         loc="lower left",bbox_to_anchor=(.103,.010),ncol=2,frameon=False,handletextpad=.30,columnspacing=.82,fontsize=7.15)
f.legend(handles=[h("Clean images",blue,"o"),h("Background swap",red,"s"),h("Local swap",green,"^")],
         loc="lower right",bbox_to_anchor=(.985,.010),ncol=2,frameon=False,handletextpad=.30,columnspacing=.78,fontsize=7.15)
f.savefig(D/"round9-confirmation.pdf")
