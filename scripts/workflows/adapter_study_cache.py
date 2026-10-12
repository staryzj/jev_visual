"""Frozen feature computation/cache; never cache model decisions or targets."""
import hashlib
import json
from pathlib import Path
from scripts.evaluation.jev_pair_protocol import sha256, write_json


def key(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False).encode()).hexdigest()


def resolve_rows(rows, runtime):
    """Map transferred absolute image paths at load, preserving original JSONL bytes."""
    transfer=runtime/'transfer_manifest.json'
    source=json.loads(transfer.read_text())['source_root'] if transfer.exists() else str(runtime)
    for row in rows:
        images=[]
        for value in row.get('images',[]):
            path=Path(value)
            if not path.is_file():
                if path.is_absolute():
                    try: path=runtime/path.relative_to(source)
                    except ValueError: raise ValueError('Image is outside declared transfer root: '+value) from None
                else: path=runtime/path
            if not path.is_file(): raise FileNotFoundError(path)
            images.append(str(path.resolve()))
        yield {**row,'images':images}


class FrozenEncoders:
    def __init__(self,runtime,config,backbone,device):
        import torch
        from transformers import AutoProcessor,AutoModelForImageTextToText
        from types import SimpleNamespace
        from visualjev.models.visual_jev_v2 import Qwen3VLFeatureExtractor
        self.runtime,self.config,self.backbone,self.device=runtime,config,backbone,device
        path=runtime/config['text_model']
        processor=AutoProcessor.from_pretrained(path,local_files_only=True,min_pixels=256*28*28,max_pixels=512*28*28)
        processor.tokenizer.padding_side='right'
        if processor.tokenizer.pad_token_id is None: processor.tokenizer.pad_token=processor.tokenizer.eos_token
        self.full=AutoModelForImageTextToText.from_pretrained(path,local_files_only=True,dtype=torch.bfloat16,
                    attn_implementation='sdpa',device_map={'':device}).eval().requires_grad_(False)
        self.text=Qwen3VLFeatureExtractor(SimpleNamespace(processor=processor,backbone=self.full.model))
        self.visual=None
        if backbone!='qwen':
            from jev.hf_vision_backends import HFVisionTokenExtractor
            self.visual=HFVisionTokenExtractor(backbone,runtime/config['visual_models'][backbone],device=device)
        self.checked_large=False

    def encode_text(self,row):
        import torch
        from scripts.training.train_visual_jev_v2 import format_candidate
        from scripts.evaluation.cached_candidate_features import cached_features
        texts=[format_candidate(row['question'],c) for c in row['candidates']]
        batch=self.config['candidate_batch']
        with torch.inference_mode():
            features=cached_features(self.text,texts,batch) if len(texts)>64 else torch.cat([self.text.encode_candidates(texts[j:j+batch]) for j in range(0,len(texts),batch)])
            if len(texts)>64 and not self.checked_large:
                original=torch.cat([self.text.encode_candidates(texts[j:j+8]) for j in range(0,len(texts),8)])
                cosine=torch.nn.functional.cosine_similarity(original.float(),features.float(),dim=-1)
                if float(cosine.min())<.999: raise ValueError('All-candidate cached text numerical check failed')
                print(f'All {len(texts)} candidate feature check: cosine min={float(cosine.min()):.6f}',flush=True)
                self.checked_large=True
        return features.detach().cpu().to(torch.bfloat16)

    def encode_image(self,image):
        import torch
        from jev.hf_vision_backends import resample_visual_tokens
        with torch.inference_mode():
            tokens=self.text.encode_image_pre_merger(image) if self.visual is None else self.visual.encode(image)
            return resample_visual_tokens(tokens,self.config['token_count']).detach().cpu().to(torch.bfloat16)

    def close(self):
        import gc,torch
        if self.visual is not None: self.visual.close()
        del self.text,self.full; gc.collect(); torch.cuda.empty_cache()


class FeaturePool:
    def __init__(self,root,identity):
        self.root,self.identity=Path(root),identity
        self.root.mkdir(parents=True,exist_ok=True)
        ip=self.root/'identity.json'
        if ip.exists() and json.loads(ip.read_text())!=identity:
            raise ValueError('Frozen encoder/data/cache identity changed; use a new cache')
        write_json(ip,identity); self.checked={}; self.image_hashes={}

    def stored(self,category,identifier,compute):
        import torch
        path=self.root/category/(identifier+'.pt'); info=path.with_suffix('.json')
        if path.exists() and info.exists():
            metadata=json.loads(info.read_text())
            stamp=(path.stat().st_size,path.stat().st_mtime_ns)
            if self.checked.get(str(path))!=stamp:
                if sha256(path)!=metadata['sha256']: raise ValueError('Feature cache checksum changed')
                self.checked[str(path)]=stamp
            payload=torch.load(path,map_location='cpu',weights_only=True)
            if payload['key']!=identifier: raise ValueError('Feature cache key differs')
            tensor=payload['tensor']
        else:
            tensor=compute()
            if tensor.ndim!=2 or not torch.isfinite(tensor).all(): raise ValueError('Invalid frozen feature tensor')
            path.parent.mkdir(parents=True,exist_ok=True); temp=path.with_suffix('.tmp')
            torch.save(dict(key=identifier,tensor=tensor),temp); temp.replace(path)
            write_json(info,dict(sha256=sha256(path),shape=list(tensor.shape),dtype=str(tensor.dtype)))
        return tensor,str(path.relative_to(self.root))

    def get(self,row,encoder,backbone,controls=True):
        from PIL import Image
        from scripts.evaluation.evaluate_submission_benchmarks import load_image
        tk=key([row['question'],row['candidates']])
        text,tp=self.stored('text',tk,lambda:encoder.encode_text(row))
        hashes=[]
        for path in row['images']:
            if path not in self.image_hashes: self.image_hashes[path]=sha256(path)
            hashes.append(self.image_hashes[path])
        vk=key(hashes or ['intrinsic_white_224'])
        image=None
        def visual():
            nonlocal image
            image=load_image(row); return encoder.encode_image(image)
        tokens,vp=self.stored(backbone+'/visual',vk,visual)
        blank=None; bp=None
        if controls:
            if image is None: image=load_image(row)
            blank,bp=self.stored(backbone+'/blank',key(['white',image.size]),
                          lambda:encoder.encode_image(Image.new('RGB',image.size,'white')))
        return dict(visual=tokens,text=text,blank=blank),dict(uid=row['uid'],visual=vp,text=tp,blank=bp)


def load_feature(pool,record):
    import torch
    result={}
    for field in ('visual','text','blank'):
        value=record[field]
        if value is None: result[field]=None; continue
        path=(pool.root/value).resolve()
        if not path.is_relative_to(pool.root.resolve()): raise ValueError('Unsafe feature reference')
        metadata=json.loads(path.with_suffix('.json').read_text())
        stamp=(path.stat().st_size,path.stat().st_mtime_ns)
        if pool.checked.get(str(path))!=stamp:
            if sha256(path)!=metadata['sha256']: raise ValueError('Cached feature checksum mismatch')
            pool.checked[str(path)]=stamp
        result[field]=torch.load(path,map_location='cpu',weights_only=True)['tensor']
    return result


def build_training_cache(runtime,config,rows,backbone,pool,index_path,device):
    from scripts.workflows.progress_eta import ProgressETA
    index_path=Path(index_path)
    if index_path.exists():
        payload=json.loads(index_path.read_text())
        if payload['identity']!=pool.identity or [r['uid'] for r in payload['rows']]!=[r['uid'] for r in rows]:
            raise ValueError('Whole training cache membership differs')
        return payload['rows']
    encoder=FrozenEncoders(runtime,config,backbone,device)
    progress=ProgressETA({'cache_'+backbone:len(rows)},{'cache_'+backbone:sum(len(r['candidates']) for r in rows)/len(rows)})
    references=[]
    try:
        for i,row in enumerate(rows,1):
            _,reference=pool.get(row,encoder,backbone); references.append(reference)
            if i%100==0 or i==len(rows): progress.update('cache_'+backbone,dict(N_complete=i),force=i==len(rows))
        write_json(index_path,dict(status='complete',identity=pool.identity,rows=references,N=len(rows),
                                  policy='Every training UID; frozen encoder computation only; no decision/label cache'))
        return references
    finally: encoder.close()
