#!/usr/bin/env python3
"""在现有 wins 容器内运行：冻结边界 prompt 重排、真实缓存计数与响应检查。"""
import argparse,collections,json,re,sys,time,urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from a2o.profiling.prompt_layout import MODES,rearrange,parts
from a2o.profiling.bfcl_attention import native_calls
from a2o.workloads.bfcl import parse_calls,common_prefix


def save(path,value):
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');temp.replace(path)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--model',required=True);p.add_argument('--endpoint',default='http://127.0.0.1:18769');p.add_argument('--preflight-only',action='store_true');a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(a.model,local_files_only=True)
    encode=lambda text:tok.encode(text,add_special_tokens=False)
    cases=[];stats=[]
    for folder in sorted(a.source.glob('multi_turn_miss_func_*')):
        if not (folder/'events.jsonl').exists():continue
        entry=json.loads((folder/'entry.json').read_text());events=[json.loads(s) for s in (folder/'events.jsonl').read_text().splitlines()];initial=(folder/'t00-s00.prompt.txt').read_text();docs={d['name']:d for d in entry['function']}
        for d in entry.get('held_docs',{}).values():docs.update({x['name']:x for x in d})
        for i,event in enumerate(events):
            turn=event['turn']
            if not i or not event['step'].endswith('s00') or str(turn) not in entry.get('held_docs',{}):continue
            previous=events[i-1];old=(folder/(previous['step']+'.prompt.txt')).read_text();current=(folder/(event['step']+'.prompt.txt')).read_text()
            assert encode(old)==json.loads((folder/(previous['step']+'.input.json')).read_text())['input_ids']
            assert encode(current)==json.loads((folder/(event['step']+'.input.json')).read_text())['input_ids']
            variants={}
            for mode in MODES:
                prev=rearrange(old,initial,mode);now=rearrange(current,initial,mode);pi=encode(prev)+previous['generated_ids'];ids=encode(now);h=common_prefix(pi,ids)
                dest=a.output/folder.name/event['step']/mode;dest.mkdir(parents=True,exist_ok=True);(dest/'previous.prompt.txt').write_text(prev);(dest/'current.prompt.txt').write_text(now)
                # Every release document remains in the appended conversation;
                # old tool definitions are exactly the unchanged initial header.
                if mode.startswith('append_docs'):
                    original_history=current[current.index('<|im_end|>\n')+len('<|im_end|>\n'):]
                    modified_history=now[now.index('<|im_end|>\n')+len('<|im_end|>\n'):]
                    if mode=='append_docs':assert original_history==modified_history
                    assert all(d['name'] in modified_history for d in entry['held_docs'][str(turn)])
                row=dict(task=folder.name,step=event['step'],mode=mode,H=h,P=len(ids)-h,total_tokens=len(ids),previous_tokens=len(pi),newly_released=[d['name'] for d in entry['held_docs'][str(turn)]]);stats.append(row)
                variants[mode]=dict(row=row,old_ids=pi,ids=ids,path=dest)
            assert variants['native']['row']['H']==event['H_common_prefix']
            active={d['name']:d for d in entry['function']}
            for release,added in entry.get('held_docs',{}).items():
                if int(release)<=turn:active.update({d['name']:d for d in added})
            cases.append(dict(task=folder.name,step=event['step'],docs=active,expected=event['response'],variants=variants))
    save(a.output/'preflight.json',dict(cases=len(cases),rows=stats,scope='matched frozen histories; new docs disclosed only at original release; not full rollout'))
    if a.preflight_only:return
    def post(path,body):
        req=urllib.request.Request(a.endpoint+path,json.dumps(body).encode(),{'Content-Type':'application/json'})
        with urllib.request.urlopen(req,timeout=1200) as response:
            data=response.read();return json.loads(data) if data.strip() else None
    def metrics():
        with urllib.request.urlopen(a.endpoint+'/metrics',timeout=30) as response:s=response.read().decode()
        value=collections.defaultdict(float)
        for line in s.splitlines():
            if not line or line.startswith('#'):continue
            name=line.split('{',1)[0].split(' ',1)[0]
            if any(key in name for key in ['prefix_cache','prompt_tokens','request_prefill_time','time_to_first_token']):
                try:value[name]+=float(line.rsplit(' ',1)[1])
                except ValueError:pass
        return dict(value)
    stop=[tok.eos_token_id,tok.convert_tokens_to_ids('<|im_end|>')]
    results=[]
    for i,case in enumerate(cases):
        shift=i%len(MODES);order=MODES[shift:]+MODES[:shift]
        for mode in order:
            v=case['variants'][mode];path=v['path'];save(a.output/'status.json',dict(phase='running',completed=len(results),total=len(cases)*len(MODES),task=case['task'],step=case['step'],mode=mode))
            # Clear only this private engine's prefix cache, then prime exactly
            # the same prior frozen history outside target-request measurement.
            reset=post('/reset_prefix_cache',{})
            assert reset.get('success') is True,'prefix cache reset did not succeed'
            prime=post('/v1/completions',dict(model='layout-qwen35',prompt=v['old_ids'],max_tokens=1,temperature=0,seed=20261108,return_token_ids=True,skip_special_tokens=False,stop_token_ids=stop))
            save(path/'prime.json',prime);before=metrics();t0=time.monotonic()
            response=post('/v1/completions',dict(model='layout-qwen35',prompt=v['ids'],max_tokens=1024,temperature=0,seed=20261108,return_token_ids=True,skip_special_tokens=False,stop_token_ids=stop))
            wall=time.monotonic()-t0;after=metrics();save(path/'response.json',response)
            choice=response['choices'][0];text=choice['text'];actual=expected=None;error=None
            try:
                actual=[c['canonical'] for c in native_calls(text,case['docs'],parse_calls)]
                expected=[c['canonical'] for c in native_calls(case['expected'],case['docs'],parse_calls)]
            except Exception as exc:error=repr(exc)
            row=dict(**v['row'],wall_s=wall,D=len(choice.get('token_ids') or []),finish_reason=choice['finish_reason'],actual_calls=actual,expected_calls=expected,calls_match=actual==expected if error is None else False,exact_text_match=text==case['expected'],parser_error=error,usage=response.get('usage'),metric_delta={k:after.get(k,0)-before.get(k,0) for k in set(after)|set(before)})
            save(path/'measurement.json',row);results.append(row);save(a.output/'measurements.json',results)
    save(a.output/'status.json',dict(phase='complete',completed=len(results),total=len(results)))


if __name__=='__main__':main()
