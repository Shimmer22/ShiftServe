#!/usr/bin/env python3
"""用已有 3B 权重验证官方视觉适配，并导出固定形状 FP32 ONNX。"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import numpy as np
from PIL import Image
import torch
from transformers import Qwen2VLImageProcessor
from a2o.qnn_vision import FixedQwenVision, compare_embeddings, load_visual


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--image', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--height', type=int, default=980)
    p.add_argument('--width', type=int, default=448)
    p.add_argument('--device', choices=['cuda', 'mps'], required=True)
    p.add_argument('--validate-only', action='store_true')
    a = p.parse_args()
    if a.height % 28 or a.width % 28:
        p.error('image dimensions must be multiples of 28')
    if a.device == 'cuda' and not torch.cuda.is_available():
        p.error('CUDA unavailable')
    if a.device == 'mps' and not torch.backends.mps.is_available():
        p.error('MPS unavailable')
    a.output.mkdir(parents=True, exist_ok=False)
    report = {'device': a.device, 'checkpoint': str(a.checkpoint.resolve()),
              'height': a.height, 'width': a.width, 'torch': torch.__version__,
              'image_sha256': hashlib.sha256(a.image.read_bytes()).hexdigest(),
              'adaptation': 'Qualcomm ai-hub-models 2565a5b11b8ab98a6079477294d3930b6fdbb18d',
              'output_order': 'natural_spatial_order; inverse window permutation already applied',
              'dtype': 'float32', 'passed': False}
    def save():
        (a.output / 'validation.json').write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
    save()
    try:
        if a.device == 'cuda':
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        visual, config = load_visual(a.checkpoint, a.device)
        proc = Qwen2VLImageProcessor.from_pretrained(a.checkpoint, local_files_only=True)
        image = Image.open(a.image).convert('RGB').resize((a.width, a.height), Image.Resampling.BICUBIC)
        image.save(a.output / 'input.png')
        data = proc(images=[image], return_tensors='pt', do_resize=False)
        pixels = data['pixel_values'].to(a.device, torch.float32)
        grid = data['image_grid_thw'].to(a.device)
        report['grid_thw'] = grid.tolist()
        with torch.inference_mode():
            start = time.perf_counter()
            ref = visual(pixels, grid_thw=grid)
            if hasattr(ref, 'pooler_output'):
                ref = ref.pooler_output
            # Reference is already materialized. Adapt the same tower in place,
            # avoiding a second 2.5 GiB FP32 copy on a shared GPU.
            fixed = FixedQwenVision(visual, grid).to(a.device).eval()
            adapted = fixed(*fixed.inputs(pixels))
            report['fp32_comparison'] = compare_embeddings(ref, adapted)
            report['output_shape'] = list(adapted.shape)
            if not report['fp32_comparison']['passed']:
                raise ValueError('official adaptation failed FP32 comparison')
            inputs = fixed.inputs(pixels)
            if adapted.shape[-1] != config.vision_config.out_hidden_size:
                raise ValueError('vision output width mismatch')
            report['validation_wall_s'] = time.perf_counter() - start
            ref.float().cpu().numpy().tofile(a.output / 'reference.f32')
            adapted.float().cpu().numpy().tofile(a.output / 'adapted.f32')
            input_specs = {}
            for name, value in zip(fixed.input_names, inputs):
                array = value.cpu().numpy()
                array.tofile(a.output / (name + '.raw'))
                input_specs[name] = {'shape': list(array.shape), 'dtype': str(array.dtype)}
            report['inputs'] = input_specs
            if not a.validate_only:
                print('Exporting FP32 ONNX (external data for large weights)', flush=True)
                torch.onnx.export(fixed, inputs, str(a.output / 'vision_encoder.onnx'),
                                  input_names=fixed.input_names, output_names=['image_features'],
                                  opset_version=17, do_constant_folding=True, dynamo=False)
                import onnx
                onnx.checker.check_model(str(a.output / 'vision_encoder.onnx'))
                report['onnx_checked'] = True
            report['passed'] = True
        save()
        print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
    except Exception as exc:
        report['failure'] = repr(exc)
        save()
        raise


if __name__ == '__main__':
    main()
