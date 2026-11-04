"""固定形状的 3B 视觉导出；沿用 Qualcomm AI Hub 的视觉适配。"""
from pathlib import Path
import json

import torch
from safetensors import safe_open
from transformers import AutoConfig
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VisionTransformerPretrainedModel

from .qnn_vendor import (Conv2dInplaceConv3d, replace_linears_with_convs,
                         replace_visual_attention_with_adaptation)


def load_visual(checkpoint: Path, device: str):
    config = AutoConfig.from_pretrained(checkpoint, local_files_only=True)
    config.vision_config._attn_implementation = 'eager'
    with torch.device('meta'):
        visual = Qwen2_5_VisionTransformerPretrainedModel(config.vision_config)
    index = json.loads((checkpoint / 'model.safetensors.index.json').read_text())['weight_map']
    keys = [k for k in index if k.startswith(('visual.', 'model.visual.'))]
    if not keys:
        raise ValueError('checkpoint has no visual tower')
    state = {}
    for shard in sorted({index[k] for k in keys}):
        with safe_open(str(checkpoint / shard), framework='pt', device='cpu') as f:
            for k in keys:
                if index[k] == shard:
                    state[k.split('visual.', 1)[1]] = f.get_tensor(k)
    visual.load_state_dict(state, strict=True, assign=True)
    # Some Transformers releases construct the non-persistent RoPE buffer on meta.
    head_dim = config.vision_config.hidden_size // config.vision_config.num_heads
    inv_freq = 1.0 / (10000 ** (torch.arange(0, head_dim // 2, 2).float() / (head_dim // 2)))
    visual.rotary_pos_emb.inv_freq = inv_freq
    return visual.to(device=device, dtype=torch.float32).eval(), config


class FixedQwenVision(torch.nn.Module):
    """与官方 VEG 相同的静态位置/mask接口，输出为恢复空间顺序的 embedding。"""
    input_names = ['pixel_values', 'position_ids_cos', 'position_ids_sin',
                   'window_attention_mask', 'full_attention_mask']

    def __init__(self, visual, grid):
        super().__init__()
        self.seq_len = int(grid.prod(dim=1).sum().item())
        self.merge = visual.spatial_merge_unit
        self.full_layers = visual.fullatt_block_indexes
        window, cu_window = visual.get_window_index(grid)
        self.register_buffer('window_index', window)
        self.register_buffer('reverse_index', window.argsort())
        rope = visual.rot_pos_emb(grid).reshape(self.seq_len // self.merge, self.merge, -1)
        rope = rope[window].reshape(self.seq_len, -1)
        self.register_buffer('rope_cos', rope.cos())
        self.register_buffer('rope_sin', rope.sin())
        full_cu = torch.nn.functional.pad(grid[:, 1:].prod(dim=1).cumsum(0), (1, 0))
        win_cu = torch.unique_consecutive(torch.as_tensor(cu_window, device=grid.device))
        for name, cu in [('window_mask', win_cu), ('full_mask', full_cu)]:
            mask = torch.full((1, self.seq_len, self.seq_len), -1000., device=grid.device)
            for lo, hi in zip(cu[:-1].tolist(), cu[1:].tolist()):
                mask[:, lo:hi, lo:hi] = 0
            self.register_buffer(name, mask)
        # Keep cu arrays for the upstream signature; the explicit masks drive attention.
        self.register_buffer('cu_window', win_cu)
        self.register_buffer('cu_full', full_cu)
        with torch.device(grid.device):
            visual = replace_visual_attention_with_adaptation(visual)
            visual = replace_linears_with_convs(visual)
            visual.patch_embed.proj = Conv2dInplaceConv3d(visual.patch_embed.proj)
        self.patch_embed = visual.patch_embed
        self.blocks = visual.blocks
        self.projector = visual.merger

    def inputs(self, pixels):
        return (pixels, self.rope_cos, self.rope_sin, self.window_mask, self.full_mask)

    def forward(self, pixel_values, position_ids_cos, position_ids_sin,
                window_attention_mask, full_attention_mask):
        x = self.patch_embed(pixel_values).reshape(self.seq_len // self.merge, self.merge, -1)
        x = torch.index_select(x, 0, self.window_index).reshape(self.seq_len, -1)
        for i, block in enumerate(self.blocks):
            full = i in self.full_layers
            x = block(x, cu_seqlens=self.cu_full if full else self.cu_window,
                      rotary_pos_emb=(position_ids_cos, position_ids_sin),
                      attention_mask=full_attention_mask if full else window_attention_mask)
        return torch.index_select(self.projector(x), 0, self.reverse_index)


def compare_embeddings(reference, candidate):
    a, b = reference.float().flatten(), candidate.float().flatten()
    finite = bool(torch.isfinite(a).all() and torch.isfinite(b).all())
    if not finite:
        return {'finite': False, 'passed': False}
    delta = a - b
    nrmse = float(delta.square().mean().sqrt() / a.square().mean().sqrt().clamp_min(1e-12))
    cosine = float(torch.nn.functional.cosine_similarity(a[None], b[None]))
    return {'finite': True, 'max_abs': float(delta.abs().max()), 'nrmse': nrmse,
            'cosine': cosine, 'passed': nrmse < 0.005 and cosine > 0.9999}
