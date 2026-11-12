"""私有DSP布局v2解析；PMU槽是路由元数据，不是硬件计数。"""
from a2o.validation.kernel_probe import PATTERN, PATHS, REASONS


def parse_layout_line(line):
    match = PATTERN.search(line)
    if not match:
        return None
    values = [int(v.strip()) for v in match['pmu'].split(',')]
    if len(values) != 8 or values[0] & 0xFFFF0000 != 0xA2100000:
        return None
    path, reason = values[0] & 255, (values[0] >> 8) & 255
    if path not in PATHS or reason not in REASONS:
        raise ValueError('未知布局探针编码')
    return dict(path=PATHS[path], reason=REASONS[reason], M=values[1], N=values[2], K=values[3],
                weight_type=values[4], mc=values[5], nc=values[6], layout=values[7],
                names=match['names'], host_types=match['types'], usec=int(match['usec']), cycles=int(match['cycles']))
