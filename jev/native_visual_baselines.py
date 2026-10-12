"""Released-model candidate scoring, with NO adapter/JEV decision head.

InternVL/LLaVA: mean conditional answer-token log-likelihood, sharing only the
image/question prefix KV computation. SigLIP2: native image/text similarity;
explicit full-token window averaging is used beyond its finite text context.
These are specified candidate-constrained baselines, not leaderboard claims.
"""
import copy
import math
from pathlib import Path

import torch


PROMPT_SUFFIX = '\nAnswer the question with a short answer.'


def move_inputs(inputs, device, dtype):
    return {k: v.to(device=device, dtype=dtype if v.is_floating_point() else v.dtype)
            if torch.is_tensor(v) else v for k,v in inputs.items()}


def answer_tokens(tokenizer, candidates):
    result=[]
    for answer in candidates:
        ids=tokenizer.encode(answer,add_special_tokens=False)
        if not ids:
            if tokenizer.eos_token_id is None: raise ValueError('Empty answer requires an EOS token')
            ids=[tokenizer.eos_token_id]
        result.append(ids)
    return result


@torch.inference_mode()
def conditional_scores(model, encoded, tokens, pad_token_id, batch_size):
    """One actual image/question prefix, all complete answer token sequences."""
    prefix=model(**encoded,use_cache=True,logits_to_keep=1,return_dict=True)
    first=prefix.logits[0,-1].float().log_softmax(-1)
    cache=prefix.past_key_values
    if cache is None: raise ValueError('Native conditional model did not return a prefix cache')
    scores=[]; offset=0; batch=batch_size
    while offset<len(tokens):
        suffix=repeated=None
        chunk=tokens[offset:offset+batch]
        try:
            sums=torch.stack([first[t[0]] for t in chunk])
            maximum=max(map(len,chunk))
            if maximum>1:
                ids=torch.full((len(chunk),maximum-1),pad_token_id,device=first.device,dtype=torch.long)
                mask=torch.zeros_like(ids)
                for j,t in enumerate(chunk):
                    if len(t)>1:
                        ids[j,:len(t)-1]=torch.tensor(t[:-1],device=first.device)
                        mask[j,:len(t)-1]=1
                repeated=copy.deepcopy(cache); repeated.batch_repeat_interleave(len(chunk))
                attention=torch.cat([encoded['attention_mask'].expand(len(chunk),-1),mask],dim=1)
                positions=(attention.long().cumsum(-1)-1)[:,-ids.shape[1]:]
                suffix=model(input_ids=ids,attention_mask=attention,position_ids=positions,
                             past_key_values=repeated,use_cache=True,logits_to_keep=0,return_dict=True)
                for j,t in enumerate(chunk):
                    if len(t)>1:
                        lp=suffix.logits[j,:len(t)-1].float().log_softmax(-1)
                        sums[j]+=lp.gather(1,torch.tensor(t[1:],device=first.device)[:,None]).sum()
            scores.extend(float(s)/len(t) for s,t in zip(sums,chunk)); offset+=len(chunk)
        except torch.cuda.OutOfMemoryError:
            suffix=repeated=None; torch.cuda.empty_cache()
            if batch==1: raise
            batch=max(1,batch//2)  # reduce compute batch, never N, K or answer length
    if len(scores)!=len(tokens) or not all(math.isfinite(s) for s in scores):
        raise ValueError('Incomplete/nonfinite native candidate scores')
    return scores


@torch.inference_mode()
def cold_conditional_scores(model, encoded, tokens):
    """Uncached numerical fixture for a few candidates; NOT partial evaluation."""
    scores=[]
    n=encoded['input_ids'].shape[1]
    for t in tokens:
        inputs={**encoded,'input_ids':torch.cat([encoded['input_ids'],torch.tensor([t[:-1]],device=encoded['input_ids'].device,dtype=torch.long)],dim=1),
                'attention_mask':torch.cat([encoded['attention_mask'],torch.ones((1,len(t)-1),device=encoded['attention_mask'].device,dtype=encoded['attention_mask'].dtype)],dim=1)}
        out=model(**inputs,use_cache=False,logits_to_keep=len(t),return_dict=True)
        lp=out.logits[0,-len(t):].float().log_softmax(-1)
        scores.append(float(lp.gather(1,torch.tensor(t,device=lp.device)[:,None]).sum())/len(t))
    return scores


def text_windows(tokenizer, text, max_length):
    """Keep all tokens. Supported SigLIP2 Gemma template is body + EOS."""
    if tokenizer('',add_special_tokens=True)['input_ids']!=[tokenizer.eos_token_id]:
        raise ValueError('Unsupported SigLIP text special-token template; no silent token changes')
    body=tokenizer(text,add_special_tokens=False,truncation=False)['input_ids']
    capacity=max_length-1
    if capacity<1: raise ValueError('Invalid native text context capacity')
    chunks=[body[i:i+capacity] for i in range(0,len(body),capacity)] or [[]]
    return [c+[tokenizer.eos_token_id] for c in chunks],[max(1,len(c)) for c in chunks]


class NativeVisualBaseline:
    def __init__(self,backbone,path,device='cuda:0',batch_size=32,dtype=torch.bfloat16):
        from transformers import AutoProcessor,AutoModel,InternVLForConditionalGeneration,LlavaOnevisionForConditionalGeneration
        self.backbone,self.device,self.batch_size,self.dtype=backbone,device,batch_size,dtype
        self.processor=AutoProcessor.from_pretrained(Path(path),local_files_only=True)
        if self.processor.tokenizer.pad_token_id is None:
            self.processor.tokenizer.pad_token=self.processor.tokenizer.eos_token
        self.processor.tokenizer.padding_side='right'
        cls={'siglip2':AutoModel,'internvl':InternVLForConditionalGeneration,'llava':LlavaOnevisionForConditionalGeneration}[backbone]
        self.model=cls.from_pretrained(Path(path),local_files_only=True,dtype=dtype).to(device).eval().requires_grad_(False)
        self.checked=False; self.windowed_texts=0

    @property
    def protocol(self):
        if self.backbone=='siglip2':
            return dict(name='native_siglip_image_text_candidate_similarity',prompt='Question: {question}\nCandidate: {candidate}',
                        max_text_tokens=self.model.config.text_config.max_position_embeddings,
                        long_text_policy='All text tokens retained in consecutive body+EOS windows; token-count-weighted mean native text embedding then normalize',
                        note='Retrieval/candidate-ranking baseline, not a generative VQA model; multiwindow aggregation is explicit')
        return dict(name='native_conditional_mean_answer_log_likelihood',prompt_suffix=PROMPT_SUFFIX,
                    answer_tokens='Tokenizer encode(add_special_tokens=False); empty answer=EOS; no answer truncation',
                    cached_prefix_check='First row: up to 8 candidates compared with uncached full forward, absolute tolerance 0.02; numerical fixture only',
                    note='Released native LM head; no JEV features/head; candidate-constrained, not unconstrained generation')

    @torch.inference_mode()
    def scores(self,image,question,candidates):
        if self.backbone=='siglip2': return self.siglip_scores(image,question,candidates)
        content=[dict(type='image',image=image),dict(type='text',text=question+PROMPT_SUFFIX)]
        encoded=self.processor.apply_chat_template([dict(role='user',content=content)],tokenize=True,
                   add_generation_prompt=True,return_dict=True,return_tensors='pt',processor_kwargs={'truncation':False})
        encoded=move_inputs(encoded,self.device,self.dtype)
        tokens=answer_tokens(self.processor.tokenizer,candidates)
        limit=getattr(self.model.config.text_config,'max_position_embeddings',None)
        if limit and encoded['input_ids'].shape[1]+max(map(len,tokens))-1>limit:
            raise ValueError('Native model context capacity exceeded; do not truncate questions or candidates silently')
        result=conditional_scores(self.model,encoded,tokens,self.processor.tokenizer.pad_token_id,self.batch_size)
        self.last_check=None
        if not self.checked:
            k=min(8,len(tokens)); cold=cold_conditional_scores(self.model,encoded,tokens[:k])
            difference=max(abs(a-b) for a,b in zip(cold,result[:k]))
            if difference>.02: raise ValueError(f'Native cached/uncached score difference {difference} exceeds tolerance')
            self.last_check=dict(kind='numerical software fixture only',N_candidates_checked=k,
                                 N_candidates_evaluated=len(tokens),max_absolute_difference=difference,tolerance=.02)
            self.checked=True
        return result

    @torch.inference_mode()
    def siglip_scores(self,image,question,candidates):
        image_inputs=move_inputs(self.processor(images=image,return_tensors='pt'),self.device,self.dtype)
        visual=self.model.get_image_features(pixel_values=image_inputs['pixel_values']).pooler_output.float()
        visual=torch.nn.functional.normalize(visual,dim=-1)
        maximum=self.model.config.text_config.max_position_embeddings
        texts=[f'Question: {question}\nCandidate: {c}' for c in candidates]
        windows=[]; owners=[]; weights=[]
        for index,text in enumerate(texts):
            seqs,w=text_windows(self.processor.tokenizer,text,maximum)
            self.windowed_texts+=len(seqs)>1
            windows.extend(seqs); weights.extend(w); owners.extend([index]*len(seqs))
        sums=None; total_weights=torch.zeros(len(candidates),device=self.device)
        offset=0; batch=self.batch_size
        while offset<len(windows):
            try:
                seqs=windows[offset:offset+batch]
                ids=torch.full((len(seqs),maximum),self.processor.tokenizer.pad_token_id,device=self.device,dtype=torch.long)
                mask=torch.zeros_like(ids)
                for i,s in enumerate(seqs):
                    ids[i,:len(s)]=torch.tensor(s,device=self.device); mask[i,:len(s)]=1
                features=self.model.get_text_features(input_ids=ids,attention_mask=mask).pooler_output.float()
                if sums is None: sums=torch.zeros((len(candidates),features.shape[-1]),device=self.device)
                for i,f in enumerate(features):
                    owner=owners[offset+i]; weight=weights[offset+i]
                    sums[owner]+=f*weight; total_weights[owner]+=weight
                offset+=len(seqs)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if batch==1: raise
                batch=max(1,batch//2)
        features=torch.nn.functional.normalize(sums/total_weights[:,None],dim=-1)
        scores=((visual@features.T)*self.model.logit_scale.float().exp()+self.model.logit_bias.float()).flatten().cpu().tolist()
        if len(scores)!=len(candidates) or not all(math.isfinite(s) for s in scores): raise ValueError('Invalid SigLIP candidate scores')
        self.last_check=None
        return scores

    def close(self):
        del self.model,self.processor
        import gc
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
