"""Resume only the sealed fit after completed extraction, with explicit module resolution."""
import argparse,importlib.util,os
import numpy as np
import torch
from prepare import ROOT,KEYS,read,sha,write

def main(k,gpu):
    assert os.environ["CUDA_VISIBLE_DEVICES"]==str(gpu) and torch.cuda.device_count()==1
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    p=read(ROOT/"plan.json")
    assert all(sha(f)==h for f,h in p["inputs"].items())
    src=ROOT/"reports"/k
    assert read(src/"exit.json")["exit_code"]==1
    assert "run() missing 4 required positional arguments" in (src/"worker.log").read_text()
    assert read(src/"result.json")["status"]=="complete" and len(read(src/"ids.json"))==396
    assert all(np.load(src/(st+".npy"),mmap_mode="r").shape==(3,396,64,256) for st in ("source","slots"))
    assert len((src/"predictions.jsonl").read_text().splitlines())==144
    assert not (src/"probe_results.json").exists()
    d=ROOT/"reports"/("fit-"+k)
    f=ROOT/"fit.py"
    s=importlib.util.spec_from_file_location("sealed_round8_fit",f)
    m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
    write(d/"resume.json",dict(original_report=k,fit_sha256=sha(f),complete_scored_rows=144,reused_features=True))
    m.run(k)
    assert len(read(src/"probe_results.json"))==72
    write(d/"result.json",dict(status="complete",N=72))
if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("encoder",choices=KEYS);p.add_argument("--gpu",type=int,required=True)
    a=p.parse_args();main(a.encoder,a.gpu)
