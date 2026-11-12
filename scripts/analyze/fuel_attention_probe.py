"""单条已有轨迹的 full-attention 诊断；不执行生成动作。"""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--case', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--model', required=True)
    a = p.parse_args()
    import numpy as np
    import torch
    import transformers
    from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    from transformers.models.qwen3_5.modeling_qwen3_5 import repeat_kv
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required; no fallback')
    torch.set_num_threads(4)
    a.output.mkdir(parents=True, exist_ok=True)
    case = json.loads(a.case.read_text())
    ids, gen, h = case['input_ids'], case['generated_ids'], case['H']
    # 预测第 i 个输出的 query 在 len(ids)-1+i；保留全部 P queries。
    selected = sorted({round(i*(len(gen)-1)/15) for i in range(min(16,len(gen)))})
    positions = sorted(set(range(h,len(ids))) | {len(ids)-1+i for i in selected})
    sequence = ids+gen[:-1]
    captured = {}
    state = {'offset': 0, 'local': []}
    base = ALL_ATTENTION_FUNCTIONS['sdpa']

    def measured(module, query, key, value, mask, **kw):
        out = base(module, query, key, value, mask, **kw)
        if state['local']:
            q = query[:,:,state['local'],:].float()
            k = repeat_kv(key,module.num_key_value_groups).float()
            logits = (q @ k.transpose(2,3))*kw['scaling']
            absolute = torch.tensor(state['local'],device=q.device)+state['offset']
            causal = torch.arange(k.shape[-2],device=q.device)[None,:] > absolute[:,None]
            logits.masked_fill_(causal[None,None],float('-inf'))
            w = logits.softmax(-1)[0].mean(0).cpu().numpy()
            target = captured.setdefault(module.layer_idx,np.zeros((len(positions),len(sequence)),np.float32))
            for j,pos in enumerate(absolute.tolist()):
                target[positions.index(pos),:w.shape[-1]] = w[j]
        return out

    ALL_ATTENTION_FUNCTIONS.register('sdpa', measured)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(a.model,dtype=torch.bfloat16,
        device_map='cuda:0',attn_implementation='sdpa',local_files_only=True).eval()
    model.model.visual = None
    tokenizer = AutoTokenizer.from_pretrained(a.model,local_files_only=True)
    cache, predicted = None, {}
    for offset in range(0,len(sequence),1024):
        chunk = sequence[offset:offset+1024]
        local = [x-offset for x in positions if offset<=x<offset+len(chunk)]
        state.update(offset=offset,local=local)
        with torch.inference_mode():
            out = model(input_ids=torch.tensor([chunk],device='cuda'),past_key_values=cache,
                use_cache=True,logits_to_keep=torch.tensor(local,device='cuda') if local else 1)
        cache = out.past_key_values
        if local:
            predicted.update(zip([x+offset for x in local],out.logits[0].argmax(-1).tolist()))
        del out
    layers = sorted(captured)
    weights = np.stack([captured[x] for x in layers])
    assert len(layers)==8 and np.allclose(weights.sum(-1),1,atol=1e-5)
    assert all(np.all(weights[:,i,pos+1:]==0) for i,pos in enumerate(positions))
    np.savez_compressed(a.output/'attention.npz',weights=weights,positions=positions,layers=layers)
    meta = dict(case,query_positions=positions,full_attention_layers=layers,
        heads_reduction='arithmetic mean of all query heads',decode_sample_indices=selected,
        greedy_matches=sum(predicted[len(ids)-1+i]==gen[i] for i in selected),
        greedy_checked=len(selected),torch=torch.__version__,transformers=transformers.__version__,
        model=a.model,model_config_sha256=hashlib.sha256(Path(a.model,'config.json').read_bytes()).hexdigest(),
        tokens=[tokenizer.decode([t]) for t in sequence],generated_text=tokenizer.decode(gen),
        measurement='HF teacher-forced replay; float32 QK softmax; full visible-key denominator',
        normalization_max_error=float(abs(weights.sum(-1)-1).max()))
    (a.output/'metadata.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:meta[k] for k in ['greedy_matches','greedy_checked','full_attention_layers','normalization_max_error']}))


if __name__ == '__main__':
    main()
