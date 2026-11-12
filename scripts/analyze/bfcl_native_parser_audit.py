"""在已有 vLLM 容器内将采集调用与官方 qwen3_coder 解析器逐项比对。"""
import ast
import argparse,json,pathlib,sys
sys.path.insert(0,'/workspace/private/a2o-text-pilot/src')
from transformers import AutoTokenizer
from vllm.tool_parsers import ToolParserManager
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from a2o.profiling.bfcl_vllm import schema
arg=argparse.ArgumentParser();arg.add_argument('directory',type=pathlib.Path);args=arg.parse_args()
root=args.directory
tok=AutoTokenizer.from_pretrained('/workspace/cache/pvst-vllm/qwen35-9b-bf16-20261016',local_files_only=True)
rows=[]
for f in root.glob('multi_turn_*/result.json'):
 entry=json.loads((f.parent/'entry.json').read_text());docs=entry['function']+sum(entry.get('held_docs',{}).values(),[])
 tools=[{'type':'function','function':schema({k:d[k] for k in ['name','description','parameters']})} for d in docs]
 request=ChatCompletionRequest(model='qwen35-9b',messages=[{'role':'user','content':'audit'}],tools=tools)
 parser=ToolParserManager.get_tool_parser('qwen3_coder')(tok,tools=request.tools)
 decoded=json.loads((f.parent/'decoded.json').read_text())
 for ev in map(json.loads,(f.parent/'events.jsonl').read_text().splitlines()):
  parsed=parser.extract_tool_calls(ev['response'],request)
  turn,step=int(ev['step'][1:3]),int(ev['step'][5:7])
  ours=decoded[turn][step] if turn<len(decoded) and step<len(decoded[turn]) else None
  canonical=[{'name':n.func.id,'arguments':{k.arg:ast.literal_eval(k.value) for k in n.keywords}} for n in [ast.parse(x,mode='eval').body for x in (ours or [])]]
  expected=[{'name':t.function.name,'arguments':json.loads(t.function.arguments)} for t in parsed.tool_calls]
  rows.append(dict(arguments_match=canonical==expected,task=f.parent.name,step=ev['step'],official_calls=[{'name':t.function.name,'arguments':t.function.arguments} for t in parsed.tool_calls],collected_calls=ours,parsed_by_collector=ours is not None,count_match=None if ours is None else len(ours)==len(parsed.tool_calls)))
(root/'official-parser-audit.json').write_text(json.dumps({'calls_checked':len(rows),'unparsed_limit_responses':sum(not r['parsed_by_collector'] for r in rows),'argument_mismatches':sum(not r['arguments_match'] for r in rows if r['parsed_by_collector']),'count_mismatches':sum(r['count_match'] is False for r in rows),'rows':rows},indent=2))
print(len(rows),sum(not r['count_match'] for r in rows))
