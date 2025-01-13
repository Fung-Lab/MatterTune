from __future__ import annotations

import nshutils as nu
from jmp.models.gemnet import GemNetOCBackbone

nu.pretty()

model = GemNetOCBackbone.from_pretrained_ckpt(
    "/net/csefiles/coc-fung-cluster/lingyu/checkpoints/jmp-s.pt"
)

base_lr = 8e-5  # 基础学习率
lr_factors = {
    "embedding": 0.3,
    "blocks_0": 0.30,
    "blocks_1": 0.40,
    "blocks_2": 0.55,
    "blocks_3": 0.625,
}


def generate_per_parameter_hparams(
    named_parameters,
    base_lr: float,
    lr_factors: dict[str, float],
):
    per_param_hparams = []

    for block_name, factor in lr_factors.items():
        matching_patterns = [name for name, _ in named_parameters if block_name in name]
        if matching_patterns:
            per_param_hparams.append(
                {
                    "patterns": matching_patterns,
                    "hparams": {"lr": base_lr * factor},
                }
            )

    return per_param_hparams


named_parameters = model.named_parameters()
print(named_parameters)

# 生成 per_parameter_hparams
per_parameter_hparams = generate_per_parameter_hparams(
    named_parameters, base_lr, lr_factors
)

print(per_parameter_hparams)
