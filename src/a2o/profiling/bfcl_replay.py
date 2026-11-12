"""对 vLLM 原始轨迹做 teacher-forced 注意力采样；不改变原轨迹。"""
import argparse
import hashlib
import json
from pathlib import Path
import time
import traceback

from a2o.profiling.bfcl_attention import save


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root',required=True,type=Path)
    p.add_argument('--samples',type=int,default=16)
    p.add_argument('--chunk-size',type=int,default=2048)
    args = p.parse_args()
    root = args.root
    cfg = json.loads((root/'manifest.json').read_text())
    import torch
    import transformers
    from transformers import Qwen3_5ForConditionalGeneration
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    from transformers.models.qwen3_5.modeling_qwen3_5 import repeat_kv
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required; no fallback')
    torch.set_num_threads(4)
    base = ALL_ATTENTION_FUNCTIONS['sdpa']
    probe = {'positions':None,'sum':None,'count':0,'length':0,'offset':0}

    def measured(module, query, key, value, mask, **kw):
        output = base(module,query,key,value,mask,**kw)
        if probe['positions'] is not None:
            positions = probe['positions']
            q = query.index_select(2,positions)
            k = repeat_kv(key,module.num_key_value_groups)
            logits = (q @ k.transpose(2,3)).float()*kw['scaling']
            causal = torch.arange(k.shape[-2],device=k.device)[None,:]>(positions+probe['offset'])[:,None]
            logits.masked_fill_(causal[None,None,:,:],float('-inf'))
            weights = logits.softmax(-1)
            scores = weights[0,:,:,:probe['length']].mean(0).sum(0)
            probe['sum'] = scores if probe['sum'] is None else probe['sum']+scores
            probe['count'] += len(positions)
        return output

    ALL_ATTENTION_FUNCTIONS.register('sdpa',measured)
    t0 = time.time()
    model = Qwen3_5ForConditionalGeneration.from_pretrained(cfg['model'],dtype=torch.bfloat16,
        device_map='cuda:0',attn_implementation='sdpa',local_files_only=True).eval()
    model.model.visual = None
    torch.cuda.empty_cache()
    save(root/'replay-manifest.json',{'backend':'CUDA0','engine':'Transformers teacher-forced replay',
        'torch':torch.__version__,'transformers':transformers.__version__,'model':cfg['model'],
        'model_config_sha256':hashlib.sha256(Path(cfg['model'],'config.json').read_bytes()).hexdigest(),
        'samples_per_call':args.samples,'sample_rule':'evenly spaced prediction queries across all generated tokens, including first token',
        'prefill_chunk_size':args.chunk_size,
        'layers':'all 8 full-attention layers; all 16 query heads; causal mask',
        'kv_policy':'none','model_init_s':time.time()-t0,
        'limitation':'attention is measured in HF replay of vLLM outputs; kernel/cache numerical differences are checked by sampled greedy-token agreement'})
    folders = sorted(root.glob('multi_turn_long_context_*'))
    started = time.time()
    failed = []
    done = 0
    for folder in folders:
        if (folder/'probe-complete.json').exists():
            done += 1
            continue
        observations, checks = [], []
        events = list(map(json.loads,(folder/'events.jsonl').read_text().splitlines()))
        for event in events:
            key = event['step']
            inputs = json.loads((folder/(key+'.input.json')).read_text())
            if not inputs['tool_spans']:
                continue
            try:
                ids, generated = inputs['input_ids'], event['generated_ids']
                n = min(args.samples,len(generated))
                selected = sorted({round(i*(len(generated)-1)/max(n-1,1)) for i in range(n)})
                positions = [len(ids)-1+i for i in selected]
                probe.update(positions=None,sum=None,count=0,length=len(ids),offset=0)
                sequence = ids+generated[:-1]
                cache = None
                actual = []
                for offset in range(0,len(sequence),args.chunk_size):
                    chunk = sequence[offset:offset+args.chunk_size]
                    local = [i-offset for i in positions if offset<=i<offset+len(chunk)]
                    local_tensor = torch.tensor(local,device='cuda',dtype=torch.long) if local else None
                    probe.update(positions=local_tensor,offset=offset)
                    with torch.inference_mode():
                        output = model(input_ids=torch.tensor([chunk],device='cuda'),
                            past_key_values=cache,use_cache=True,
                            logits_to_keep=local_tensor if local else 1)
                    cache = output.past_key_values
                    if local:
                        actual.extend(output.logits[0].argmax(-1).tolist())
                    del output
                if probe['count']!=8*len(selected):
                    raise RuntimeError(f'expected {8*len(selected)} layer queries, got {probe["count"]}')
                means = (probe['sum']/probe['count']).cpu().tolist()
                check = {'step':key,'sampled_queries':len(selected),'sample_indices':selected,
                    'greedy_matches':sum(a==generated[i] for a,i in zip(actual,selected)),
                    'greedy_match_fraction':sum(a==generated[i] for a,i in zip(actual,selected))/len(selected)}
                checks.append(check)
                save(folder/(key+'.attention.json'),means)
                for span in inputs['tool_spans']:
                    a,b = span['start'],span['end']
                    observations.append({'tool_id':span['id'],'step':key,'tool_turn':span['turn'],
                        'turn':event['turn'],'tokens':b-a,'prompt_tokens':len(ids),
                        'history_fraction':(b-a)/len(ids),'attention_mass':sum(means[a:b]),
                        'attention_per_token':sum(means[a:b])/(b-a),
                        'max_token_mean_attention':max(means[a:b]),'layer_query_count':8*len(selected)})
                del cache
            except Exception:
                failure = {'task':folder.name,'step':key,'error':traceback.format_exc()}
                failed.append(failure)
                save(folder/(key+'.probe-failure.json'),failure)
            finally:
                probe.update(positions=None,sum=None,count=0,length=0,offset=0)
        save(folder/'observations.json',observations)
        save(folder/'replay-checks.json',checks)
        save(folder/'probe-complete.json',{'observations':len(observations),'checked_calls':len(checks)})
        done += 1
        elapsed = time.time()-started
        status = {'stage':'attention_replay','finished':done,'total':len(folders),
                  'elapsed_s':elapsed,'eta_seconds':elapsed/max(done,1)*(len(folders)-done),'failed_calls':len(failed)}
        save(root/'probe-status.json',status)
        print(json.dumps(status),flush=True)
    save(root/'probe-completion.json',{'tasks':done,'failed_calls':failed,'elapsed_s':time.time()-started})


if __name__=='__main__':
    main()
