"""Three explicit adapter arms sharing the existing Visual-JEV decision scorer."""
from dataclasses import asdict
import torch
from torch import nn
from visualjev.models.visual_jev_v3 import VisualJEVV3Config
from visualjev.models.visual_jev_v3_pipeline import AlignmentAdapterConfig, ThreeStageVisualJEV

ARMS = ('linear_ce', 'mlp_ce', 'mlp_invalid')


class LinearAlignment(nn.Module):
    """Dimension-matching baseline: LayerNorm followed by one linear map."""
    def __init__(self, config):
        super().__init__(); self.config = config
        self.input_norm = nn.LayerNorm(config.pre_merger_dim)
        self.projection = nn.Linear(config.pre_merger_dim, config.teacher_dim)

    def forward(self, tokens):
        parameter = next(self.parameters())
        return self.projection(self.input_norm(tokens.to(parameter.device, parameter.dtype)))


class AdapterStudyJEV(ThreeStageVisualJEV):
    def __init__(self, alignment_config, decision_config, arm):
        if arm not in ARMS:
            raise ValueError('Unknown adapter arm')
        super().__init__(alignment_config, decision_config); self.arm = arm
        if arm == 'linear_ce':
            self.alignment = LinearAlignment(alignment_config)
        self.decision.eval().requires_grad_(False)

    def save(self, path, metadata):
        torch.save(dict(model_type='jev-adapter-study-v1', format_version=1, arm=self.arm,
                        alignment_config=asdict(self.alignment.config), decision_config=asdict(self.decision.config),
                        state_dict=self.state_dict(), metadata=metadata), path)


def load(path, device='cpu'):
    payload = torch.load(path, map_location='cpu', weights_only=False)
    if payload.get('model_type') == 'visual-jev-v3-three-stage':
        return ThreeStageVisualJEV.from_checkpoint(path, map_location=device)
    if payload.get('model_type') != 'jev-adapter-study-v1' or payload.get('format_version') != 1:
        raise ValueError('Unsupported JEV comparison checkpoint')
    model = AdapterStudyJEV(AlignmentAdapterConfig(**payload['alignment_config']),
                            VisualJEVV3Config(**payload['decision_config']), payload['arm'])
    model.load_state_dict(payload['state_dict']); return model.to(device), payload


def fresh_arms(source, input_dim, seed, hidden_dim=1024):
    models = {}
    for arm in ARMS:
        torch.manual_seed(seed)
        model = AdapterStudyJEV(AlignmentAdapterConfig(pre_merger_dim=input_dim,
                    teacher_dim=source.decision.config.vision_dim, hidden_dim=hidden_dim, dropout=0.0),
                    source.decision.config, arm)
        model.decision.load_state_dict(source.decision.state_dict())
        model.decision.eval().requires_grad_(False); models[arm] = model
    return models
