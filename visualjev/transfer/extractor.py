"""Fresh unpooled external sequence caches or frozen native teacher targets."""
from pathlib import Path
from .common import read
SIZE = {"shortest_edge": 65536, "longest_edge": 262144}
import torch
from PIL import Image

def flatten_native(o):
    for key in ['last_hidden_state','image_embeds','pooler_output']:
        if hasattr(o,key) and getattr(o,key) is not None:return flatten_native(getattr(o,key))
    if isinstance(o,(tuple,list)):
        vv=[flatten_native(v) for v in o];assert len({v.shape[-1] for v in vv})==1
        return torch.cat(vv)
    assert isinstance(o,torch.Tensor) and 2<=o.ndim<=4
    z=o.reshape(-1,o.shape[-1]);assert z.numel()>0 and torch.isfinite(z).all();return z

class External:
    def __init__(self,b,model_path,device="cuda:0"):
        self.b=b;self.path=Path(model_path);self.dtype=torch.bfloat16;self.device=device
        from transformers import AutoProcessor
        self.processor=AutoProcessor.from_pretrained(str(self.path),local_files_only=True,trust_remote_code=(b=='internvl'))
        if b=='qwen':
            from transformers import Qwen3VLConfig
            from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel
            from safetensors import safe_open
            cfg=Qwen3VLConfig.from_pretrained(str(self.path),local_files_only=True);cfg.vision_config._attn_implementation='sdpa'
            self.model=Qwen3VLVisionModel(cfg.vision_config);state={}
            index=read(self.path/'model.safetensors.index.json')['weight_map']
            for file in sorted({v for k,v in index.items() if k.startswith('model.visual.')}):
                with safe_open(str(self.path/file),framework='pt',device='cpu') as f:
                    for k in f.keys():
                        if k.startswith('model.visual.'):state[k[len('model.visual.'):]]=f.get_tensor(k)
            assert set(state)==set(self.model.state_dict());self.model.load_state_dict(state,strict=True,assign=True)
            self.model.to(self.device);assert all(p.dtype==self.dtype for p in self.model.parameters())
        else:
            # Existing audited HF API, extraction below deliberately bypasses its196-token resampling.
            from jev.hf_vision_backends import HFVisionTokenExtractor
            self.backend=HFVisionTokenExtractor(b,self.path,device=self.device);self.model=self.backend.model
        self.model.eval().requires_grad_(False)

    @torch.inference_mode()
    def encode(self,image):
        from torch.nn.attention import sdpa_kernel,SDPBackend
        processor=getattr(self.processor,'image_processor',self.processor)
        inp=processor(images=[image],**({'size':SIZE} if self.b=='qwen' else {}),return_tensors='pt')
        inp={k:v.to(self.device,dtype=self.dtype if v.is_floating_point() else v.dtype) if torch.is_tensor(v) else v for k,v in inp.items()}
        with sdpa_kernel(SDPBackend.MATH):
            if self.b=='qwen':o=self.model(inp['pixel_values'],grid_thw=inp['image_grid_thw'],return_dict=True).pooler_output
            elif self.b=='siglip2':o=self.model.vision_model(pixel_values=inp['pixel_values'],return_dict=True).last_hidden_state
            else:
                kw=dict(pixel_values=inp['pixel_values'])
                if self.b=='llava':kw['image_sizes']=inp.get('image_sizes',torch.tensor([[image.height,image.width]],device=self.device))
                o=self.model.get_image_features(**kw)
        tokens=flatten_native(o).detach().cpu();assert len(tokens)<=12288,'Refuse unplanned memory or silent token truncation'
        return dict(tensor=tokens,layout=dict(shape=list(tokens.shape),image_size=list(image.size),
            source_order='native API order, leading axes flattened, no invented2D source grid',
            processor_geometry={k:v.detach().cpu() for k,v in inp.items() if torch.is_tensor(v) and k!='pixel_values' and v.numel()<8192},
            token_resampling=False,encoder=self.b))
