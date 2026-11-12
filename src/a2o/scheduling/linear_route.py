"""独立校准得到非负仿射成本；按H/P连续预测，不要求精确命中。"""
import hashlib
import json
import math
from statistics import median


def fit_line(points):
    """二维非负最小二乘：枚举内部解及两条边界。"""
    n = len(points)
    if n < 3 or len({x for x, y in points}) < 2:
        raise ValueError('线性拟合至少需要三个样本及两个不同长度')
    if any(not math.isfinite(v) or v < 0 for point in points for v in point):
        raise ValueError('成本样本必须有限且非负')
    sx = sum(x for x, y in points); sy = sum(y for x, y in points)
    sxx = sum(x*x for x, y in points); sxy = sum(x*y for x, y in points)
    candidates = [(sy/n, 0), (0, sxy/sxx)]
    slope = (n*sxy-sx*sy)/(n*sxx-sx*sx)
    intercept = (sy-slope*sx)/n
    if intercept >= 0 and slope >= 0:
        candidates.append((intercept, slope))
    def sse(pair):
        a, b = pair
        return sum((a+b*x-y)**2 for x, y in points)
    a, b = min(candidates, key=sse)
    residuals = [a+b*x-y for x, y in points]
    total = sum((y-sy/n)**2 for x, y in points)
    return dict(intercept_ms=a, slope_ms_per_token=b, n=n,
                rmse_ms=math.sqrt(sse((a,b))/n),
                r2=1-sse((a,b))/total if total else None,
                max_abs_error_ms=max(abs(e) for e in residuals))


FIELDS = {'cpu_restore': ('H', 'cc_restore_ms'),
          'cpu_to_htp': ('H', 'hc_prepare_ms'),
          'cpu_prefill': ('P', 'cpu_prefill_ms'),
          'htp_prefill': ('P', 'htp_prefill_ms'),
          'htp_to_cpu': ('HP', 'htp_to_cpu_ms')}


def fit_model(responses):
    # 每个形状取中位数，数值差异不用于筛选性能数据；质量另外报告。
    model = dict(kind='linear_hp_v1', routing_p_min=32,
                 h_min=min(r['H'] for r in responses), h_max=max(r['H'] for r in responses),
                 p_min=min(r['P'] for r in responses), p_max=max(r['P'] for r in responses),
                 formula='CC=R(H)+C(P); HC=N(P)+M(H+P); common CPU decode excluded',
                 preparation='always prepare; ready only; CPU-to-HTP preparation hidden by actual tool gap',
                 extrapolation='linear allowed; every extrapolated decision logged')
    for key, (coordinate, metric) in FIELDS.items():
        points = [(r['H']+r['P'] if coordinate == 'HP' else r[coordinate],
                   median(row[metric] for row in r['rows'])) for r in responses]
        model[key] = fit_line(points)
    model['model_id'] = hashlib.sha256(json.dumps(model,sort_keys=True).encode()).hexdigest()
    return model


def predict(model, h, p):
    def line(key, x):
        fit = model[key]
        return max(0, fit['intercept_ms']+fit['slope_ms_per_token']*x)
    cc = line('cpu_restore', h)+line('cpu_prefill', p)
    hc = line('htp_prefill', p)+line('htp_to_cpu', h+p)
    return dict(cc_ms=cc, hc_ms=hc, route='htp' if p >= model.get('routing_p_min',32) and hc < cc else 'cpu')


def threshold(model, h):
    # HC-CC = a(H) - b*P；b>0时P超过阈值选择HTP。
    r,c,n,m = (model[k] for k in ('cpu_restore','cpu_prefill','htp_prefill','htp_to_cpu'))
    a = n['intercept_ms']+m['intercept_ms']-r['intercept_ms']-c['intercept_ms']+(m['slope_ms_per_token']-r['slope_ms_per_token'])*h
    b = c['slope_ms_per_token']-n['slope_ms_per_token']-m['slope_ms_per_token']
    return dict(H=h, numerator_ms=a, savings_slope_ms_per_token=b,
                p_crossover=a/b if b else None,
                rule='P > crossover selects HTP' if b>0 else 'P < crossover selects HTP' if b<0 else 'constant choice')
