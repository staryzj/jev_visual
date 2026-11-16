"""One image-only extractor API for four encoders and the preserved feature taps.

Official native extraction primitives are pinned by SHA256 in plan.json; no
language decoding from the external models is used as JEV's answer.
"""
import torch
from .extractor import External
from . import taps

class ExternalEncoder:
    def __init__(self,name,model_path,device="cuda:0"):
        self.name=name; self.device=device; self.taps=taps
        self.external=External(name,model_path,device=device)
        self.model=self.external.model
        assert all(not p.requires_grad for p in self.model.parameters())

    @torch.inference_mode()
    def encode(self,image,tap):
        if tap in ['raw','projected'] and self.name!='siglip2':
            pair,layout=self.taps.capture_pair(self.external,self.name,image)
            value=pair[tap]
            layout=dict(layout,shape=list(value.shape))
        else:
            assert tap=='native' or (self.name=='siglip2' and tap=='raw')
            # The guarded LLaVA path retains all legal high-resolution tiles.
            rec=self.taps.external_encode_for_taps(self.external,self.name,image)
            value=rec['tensor']; layout=rec['layout']
            if self.name=='siglip2': layout=dict(layout,canonical_grid=[14,14])
        assert value.ndim==2 and torch.isfinite(value).all()
        assert value.numel()*value.element_size()<=256*1024*1024
        return dict(tensor=value.float().to(self.device),layout=layout,_maps={})
