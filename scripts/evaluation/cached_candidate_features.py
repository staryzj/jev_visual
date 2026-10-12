"""Exact text-feature boundary using a per-question live common-prefix KV cache.

Only computation is shared. Every candidate's complete, untruncated token
sequence is retained. This is not a cache of training features or predictions.
"""
import copy
import torch
from scripts.training.train_visual_jev_v2 import format_candidate


@torch.inference_mode()
def cached_features(extractor,candidates,batch_size=32):
    if not candidates:raise ValueError('No candidates')
    conversations=[[{'role':'user','content':[{'type':'text','text':text}]}] for text in candidates]
    encoded=extractor.processor.apply_chat_template(conversations,tokenize=True,add_generation_prompt=True,
                    processor_kwargs={'padding':True},return_dict=True,return_tensors='pt')
    sequences=[ids[mask.bool()].tolist() for ids,mask in zip(encoded['input_ids'],encoded['attention_mask'])]
    prefix_length=0
    for column in zip(*sequences):
        if any(t!=column[0] for t in column):break
        prefix_length+=1
    prefix_length=min(prefix_length,min(map(len,sequences))-1)
    if prefix_length<=0:
        return torch.cat([extractor.encode_candidates(candidates[j:j+batch_size]) for j in range(0,len(candidates),batch_size)])
    model=extractor.language_model
    device=next(model.get_input_embeddings().parameters()).device
    prefix_ids=torch.tensor([sequences[0][:prefix_length]],device=device)
    prefix_positions=torch.arange(prefix_length,device=device).view(1,1,-1).expand(3,1,-1)
    prefix=model(input_ids=prefix_ids,attention_mask=torch.ones_like(prefix_ids),position_ids=prefix_positions,
                 use_cache=True,return_dict=True)
    cache=prefix.past_key_values
    features=[];i=0;batch=batch_size
    while i<len(sequences):
        suffixes=[s[prefix_length:] for s in sequences[i:i+batch]]
        length=max(map(len,suffixes))
        repeated=None;output=None
        try:
            ids=torch.full((len(suffixes),length),extractor.processor.tokenizer.pad_token_id,dtype=torch.long,device=device)
            mask=torch.zeros_like(ids)
            lengths=[]
            for k,suffix in enumerate(suffixes):
                ids[k,:len(suffix)]=torch.tensor(suffix,device=device);mask[k,:len(suffix)]=1;lengths.append(len(suffix))
            repeated=copy.deepcopy(cache);repeated.batch_repeat_interleave(len(suffixes))
            attention=torch.cat([torch.ones((len(suffixes),prefix_length),device=device,dtype=mask.dtype),mask],dim=1)
            positions=torch.arange(prefix_length,prefix_length+length,device=device).view(1,1,-1).expand(3,len(suffixes),-1)
            output=model(input_ids=ids,attention_mask=attention,position_ids=positions,past_key_values=repeated,
                         use_cache=True,return_dict=True)
            rows=torch.arange(len(suffixes),device=device)
            features.append(output.last_hidden_state[rows,torch.tensor(lengths,device=device)-1].clone())
            i+=len(suffixes)
            del output,repeated
        except torch.cuda.OutOfMemoryError:
            output=None;repeated=None;torch.cuda.empty_cache()
            if batch==1:raise
            batch=max(1,batch//2)
    return torch.cat(features)


@torch.inference_mode()
def cached_jev_scores(extractor,jev,image,row,batch):
    visual=extractor.encode_image_pre_merger(image)
    aligned=jev.alignment(visual)
    texts=cached_features(extractor,[format_candidate(row['question'],c) for c in row['candidates']],batch_size=max(batch,32))
    # The head is shared but candidate-independent; avoid an N*K*d expansion.
    scores=[]
    for chunk in texts.split(batch):
        scores.extend(jev.score_aligned(aligned,chunk).scores.detach().float().cpu().tolist())
    return scores
