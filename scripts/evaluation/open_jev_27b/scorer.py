"""Prepare candidate inputs without reading image pixels in the receiver."""
from contextlib import contextmanager

import torch
from PIL import Image

SIZE = {"shortest_edge": 65536, "longest_edge": 262144}

@contextmanager
def external_only(model):
    original=model.mm.visual.forward
    def denied(*a,**kw): raise RuntimeError('Native visual forward called in external-only inference')
    model.mm.visual.forward=denied
    try: yield
    finally: model.mm.visual.forward=original

def prepare_external(model,row,image_size):
    """Only dimensions, not image content, construct placeholders and M-RoPE."""
    from release_loader.api import candidate_prompts
    blank=Image.new('RGB',image_size,'white')
    proc=model.processor.image_processor(images=[blank],size=SIZE,return_tensors='pt')
    grid=proc['image_grid_thw'].to('cuda:0')
    replacement=model.processor.replace_image_token(proc,image_idx=0); batches=[]
    for prompt in candidate_prompts(model.reference.record(row)):
        text=model.processor.tokenizer.apply_chat_template(
            [{'role':'user','content':'<|vision_start|><|image_pad|><|vision_end|>\n'+prompt}],
            tokenize=False,add_generation_prompt=True,enable_thinking=False)
        ids=model.processor.tokenizer(text.replace(model.processor.image_token,replacement),return_tensors='pt')['input_ids'].to('cuda:0')
        assert ids.shape[1]<=2048,'No silent truncation'
        assert int(ids.eq(model.mm.config.image_token_id).sum())==int(grid.prod()//4)
        batches.append(dict(input_ids=ids,attention_mask=torch.ones_like(ids),image_grid_thw=grid,
            mm_token_type_ids=torch.tensor(model.processor.create_mm_token_type_ids(ids.tolist()),device='cuda:0')))
    return batches,grid[0].tolist()
