"""私有 prompt 布局实验；工具公开时点与已保存消息正文保持不变。"""
import re

MODES=('native','rules_first','append_docs','append_docs_stable_history')
SYSTEM='<|im_start|>system\n'
END='<|im_end|>\n'


def parts(prompt):
    if not prompt.startswith(SYSTEM):raise ValueError('expected native tool system header')
    end=prompt.index(END)
    left=prompt.index('<tools>');right=prompt.index('\n</tools>',left)+len('\n</tools>')
    if right>end:raise ValueError('tools outside initial system header')
    return left,right,end


def rearrange(prompt,initial_prompt,mode):
    if mode not in MODES:raise ValueError(mode)
    left,right,end=parts(prompt)
    if mode=='rules_first':
        # Fixed instructions/system text precede the variable catalogue. No
        # function is added and no document appears before its release turn.
        return SYSTEM+prompt[right:end].strip()+'\n\n'+prompt[len(SYSTEM):right]+END+prompt[end+len(END):]
    if mode.startswith('append_docs'):
        il,ir,_=parts(initial_prompt)
        prompt=prompt[:left]+initial_prompt[il:ir]+prompt[right:]
        # Newly released descriptions already occur verbatim in the official
        # release user message. Keep that message; remove only header duplication.
        if mode.endswith('stable_history'):
            prompt=re.sub(r'(<\|im_start\|>assistant\n)(?!<think>)',r'\1<think>\n\n</think>\n\n',prompt)
    return prompt
