import torch
import torch.nn as nn

# 定义模型接口，承接qwen 过来的tensor，此处算法暂时用取mean，并假设jev为2560，vm送俩的为1280.
class VisualAlgorithm(nn.Module):
    def __init__(self, vision_dim, hidden_dim, debug=False):
        super().__init__()
        self.debug = bool(debug)

        self.projector = nn.Sequential(
            nn.Linear(vision_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, visual_token):
        if visual_token.ndim != 3:
            raise ValueError("visual_token must have shape [batch, tokens, vision_dim]")
        if self.debug:
            print("visual_tokens:", visual_token.shape)

        visual_feature = visual_token.mean(dim=1)
        if self.debug:
            print("after mean:", visual_feature.shape)

        output = self.projector(visual_feature)
        if self.debug:
            print("after projector:", output.shape)

        return output


if __name__ == "__main__":
    model = VisualAlgorithm(
        vision_dim=1280,
        hidden_dim=2560
    )

    test_token = torch.rand(
        1,
        729,
        1280
    )

    out = model(test_token)

    print("final:", out.shape)
