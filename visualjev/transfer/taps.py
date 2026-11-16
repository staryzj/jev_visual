"""Experiment-identical native pre/post-projector feature taps."""
import copy
import torch
from .geometry import source_spec, resize_and_padding

def downsampled_grid(cfg):
    image=cfg.vision_config.image_size;patch=cfg.vision_config.patch_size
    image=[image,image] if isinstance(image,int) else list(image)
    patch=[patch,patch] if isinstance(patch,int) else list(patch)
    return [int((s//p)*cfg.downsample_ratio) for s,p in zip(image,patch)]

def checked_llava_layout(tensor,inp,image_size,spec):
    geo=resize_and_padding(image_size,spec);tile=spec['image_h']
    views=1+(geo['target_h']//tile)*(geo['target_w']//tile)
    expected=views*spec['grid_h']*spec['grid_w']
    native_max=(1+max((h//tile)*(w//tile) for h,w in spec['pinpoints']))*spec['grid_h']*spec['grid_w']
    assert tensor.ndim==2 and len(tensor)==expected and expected<=native_max
    assert torch.isfinite(tensor).all()
    return dict(shape=list(tensor.shape),image_size=list(image_size),encoder='llava',
        source_order='native selected layer, overview then native tile order',token_resampling=False,
        native_views=views,native_expected_tokens=expected,native_config_max_tokens=native_max,
        processor_geometry={k:v.detach().cpu() for k,v in inp.items() if torch.is_tensor(v) and k!='pixel_values' and v.numel()<8192})

def external_encode_for_taps(external,b,image):
    if b!='llava':return external.encode(image)
    # Preserve native processor/model calls. The legacy generic 12288-token
    # guard rejects legal17-view images (17*729=12393). Validate the exact
    # configured view/grid count instead; never truncate or resize differently.
    processor=getattr(external.processor,'image_processor',external.processor)
    inp=processor(images=[image],return_tensors='pt')
    inp={k:v.to(external.device,dtype=external.dtype if v.is_floating_point() else v.dtype) if torch.is_tensor(v) else v for k,v in inp.items()}
    from torch.nn.attention import sdpa_kernel,SDPBackend
    with torch.inference_mode(),sdpa_kernel(SDPBackend.MATH):
        out=external.model.get_image_features(pixel_values=inp['pixel_values'],image_sizes=inp.get('image_sizes',torch.tensor([[image.height,image.width]],device=external.device)))
    x=out.last_hidden_state.reshape(-1,out.last_hidden_state.shape[-1]).detach().cpu()
    return dict(tensor=x,layout=checked_llava_layout(x,inp,image.size,source_spec('llava')))

def capture_pair(external,b,image):
    model=external.model
    projector=model.merger if b=='qwen' else model.model.multi_modal_projector
    captured={}
    def before(module,args):
        assert 'input' not in captured,'Projector called more than once'
        captured['input']=args[0].detach().clone()
    def after(module,args,output):captured['output']=output.detach().clone()
    a=projector.register_forward_pre_hook(before);z=projector.register_forward_hook(after)
    try:
        old=external_encode_for_taps(external,b,image)
    finally:a.remove();z.remove()
    x=captured['input'];y=captured['output']
    from torch.nn.attention import sdpa_kernel,SDPBackend
    with torch.inference_mode(),sdpa_kernel(SDPBackend.MATH):again=projector(x)
    assert torch.equal(again,y),'Pre/post tap must exactly reproduce native projector'
    if b=='qwen':
        assert not projector.use_postshuffle_norm
        # Lossless 2x2 channel concatenation; same output sites on both sides.
        raw=x.reshape(-1,projector.hidden_size)
        assert torch.equal(y.reshape(-1,y.shape[-1]).cpu(),old['tensor'])
        grid=old['layout']['processor_geometry']['image_grid_thw'][0].tolist()
        shape=[int(grid[1])//2,int(grid[2])//2]
    else:
        raw=x.reshape(-1,x.shape[-1])
        shape=[16,16] if b=='internvl' else [27,27]
        if b=='internvl':
            cfg=model.config
            assert cfg.downsample_ratio==.5
            shape=downsampled_grid(cfg)
            assert old['layout']['processor_geometry']['num_patches'].tolist()==[1]
    post=y.reshape(-1,y.shape[-1]);assert len(raw)==len(post)
    if b=='internvl':assert len(raw)==shape[0]*shape[1]
    layout=copy.deepcopy(old['layout']);layout['canonical_grid']=shape
    layout['tap_definition']='raw: native projector input before learned projection; projected: exact native projector output'
    layout['shared_spatial_preprocessing']=dict(qwen='lossless native 2x2 group/channel concat before merger norm/MLP',internvl='native selected layer, CLS removal, lossless pixel shuffle',llava='native configured layer, identical overview and tiles before pack/newline')[b]
    return {'raw':raw.cpu().contiguous(),'projected':post.cpu().contiguous()},layout
