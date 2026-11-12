#!/usr/bin/env python3
"""主机 HF tokenizer 解码所选精确序列，供手机逐序列重编码核对。"""
import argparse
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--requests',type=Path,required=True)
    p.add_argument('--model',required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(a.model,local_files_only=True)
    fixtures=[]
    for request in json.loads(a.requests.read_text()):
        row=dict(boundary=request['boundary'])
        for key,ids in [('prompt',request['prefix_ids']+request['input_ids']),('generated',request['generated_ids'])]:
            text=tokenizer.decode(ids,skip_special_tokens=False,clean_up_tokenization_spaces=False)
            assert tokenizer.encode(text,add_special_tokens=False)==ids, (request['boundary'],key)
            row[key+'_text']=text
        fixtures.append(row)
    a.output.write_text(json.dumps(fixtures,ensure_ascii=False,indent=2)+'\n')
    print(len(fixtures))


if __name__=='__main__':
    main()
