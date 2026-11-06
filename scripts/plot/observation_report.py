#!/usr/bin/env python3
"""汇总主矩阵与P31配对复测矢量图；仅读取结果。依赖Typst与pypdf。"""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess
from zoneinfo import ZoneInfo
from pypdf import PdfReader, PdfWriter, Transformation

MM=72/25.4
NAMES=['01_backend_boundary','02_cost_breakdown','03_actual_gap_overlap','04_end_to_end_policies','05_oracle_gap','06_p31_paired_retest']


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--campaign',type=Path,required=True);ap.add_argument('--output',type=Path)
    a=ap.parse_args();base=a.campaign.resolve();fig=base/'figures';work=base/'report';work.mkdir(exist_ok=True)
    out=a.output or base/'xiaomi14pro-observation-report.pdf'
    retest_path=base.parent/'20261105-post-tool-p31-retest/summary.json'
    retest=json.loads(retest_path.read_text())
    assert len(retest)==2 and all(r['pairs']==20 and r['routes']=={'CC':20} for r in retest)
    d=json.loads((fig/'plot-data.json').read_text());c=json.loads((fig/'comparisons.json').read_text());accept=json.loads((base/'acceptance.json').read_text())
    assert accept['data_audit_passed'] and accept['visual_qa_passed']
    assert d['constants']['device_identity']['ro.soc.model']=='SM8650'
    rows=d['rows'];assert len(rows)==762
    q=lambda x:json.dumps(str(x),ensure_ascii=False)
    t=lambda x:'text('+q(x)+')'
    def table(headers,body,widths=None,size=9):
        cols=widths or '('+','.join('1fr' for _ in headers)+')'
        parts=[f'#text(size:{size}pt)[#table(columns:{cols},inset:(x:2.5mm,y:1.8mm),stroke:0.35pt+rgb("#d8e0e8"),align:(x,y)=>if x==0 {{left}} else {{right}},fill:(x,y)=>if y==0 {{rgb("#eef3f8")}} else {{none}},',
               'table.header('+','.join('text('+q(x)+',weight:"bold")' for x in headers)+'),']
        parts += [','.join(t(x) for x in row)+',' for row in body]
        return '\n'.join(parts)+')]'
    def header(title,subtitle):
        return '#block(height:10mm,width:100%)[#text(size:16pt,weight:"bold",fill:ink)['+title+']\\\n#text(size:8pt,fill:muted)['+subtitle+']]\n#v(2mm)\n'
    def paragraph(text,size=10):return '#text(size:'+str(size)+'pt)['+text+']\n'
    def figure_page(number,title,subtitle,height,caption,notes):
        return header(f'{number:02d} / {title}',subtitle)+f'#block(height:{height}mm,width:100%)[]\n#v(3mm)\n'+paragraph(caption,9)+'#v(3mm)\n'+notes
    lookup={(r['H'],r['P'],r['policy']):r for r in c['policies']}
    policy_rows=[]
    for h in [2048,4096]:
        for p in [24,31,32]:
            vals=[lookup[(h,p,k)]['median_total_ms'] for k in ['CC','HC','Post-tool','Always prep.','Cost-aware']]
            policy_rows.append([f'{h} / {p}',*[f'{v:.1f}' for v in vals],f'{100*(1-vals[-1]/vals[0]):+.1f}%'])
    maxjoint=max(c['regret'],key=lambda x:x['joint_empirical_gap_ms'])
    matched=sum(r['matched'] for r in c['regret'])
    prefix='''#set document(title:"Xiaomi 14 Pro：Observation与系统收益实验报告",author:"A2O 实验记录")
#set page(paper:"a4",flipped:true,margin:(x:10mm,top:10mm,bottom:12mm),footer:context [#set text(size:8pt,fill:rgb("#64748b")); #grid(columns:(1fr,auto),[A2O · Xiaomi 14 Pro · 2026-11-05 · P31 复测更新],[#counter(page).display("1") / 8])])
#set text(font:("PingFang SC","Arial"),size:10pt,lang:"zh",fill:rgb("#172b45"))
#set par(spacing:0pt,leading:4pt,justify:false)
#set block(spacing:0pt)
#let ink=rgb("#172b45")
#let muted=rgb("#526579")
#let blue=rgb("#0072b2")
'''
    cover=header('Xiaomi 14 Pro｜Observation 与系统收益','2026-11-05 · SM8650 / Hexagon v75 · 原矩阵每条件 3 次；新增 P31 每 H 20 组配对复测')
    cover+='''#v(2mm)
#grid(columns:(1fr,1fr,1fr),gutter:4mm,
  block(fill:rgb("#eef4fa"),inset:4mm,width:100%)[#text(size:20pt,weight:"bold")[762] 条主矩阵记录],
  block(fill:rgb("#eef4fa"),inset:4mm,width:100%)[#text(size:20pt,weight:"bold")[26 / 26] 组 HC 数值准入],
  block(fill:rgb("#eef4fa"),inset:4mm,width:100%)[#text(size:20pt,weight:"bold")[40] 组新增配对复测])
#v(5mm)
#text(weight:"bold",size:12pt)[核心判断]
#v(2mm)
'''
    cover+=paragraph('CPU/HTP 的优势区域及 P32 阶跃清晰。提前状态准备能够与真实 Tool 等待重叠，但当前 Cost-aware 尚未稳定优于 Always Prepare 或 Post-tool Routing；本轮支持性能表征，不能直接宣称新的稳定系统增益。')
    cover+='#v(2mm)\n'+paragraph('P31 更新：原图 Post-tool 的 374–442ms 优势未在 20 组配对中复现；40 次 Post-tool 全部选 CPU，配对差区间均跨 0。原始三次数据保留，第 8 页单列复测，不混合批次。',9)
    cover+='#v(4mm)\n#text(weight:"bold",size:11pt)[原始三次测量：策略完整响应中位数（ms；G 目标 200ms）]\n#v(2mm)\n'
    cover+=table(['H / P','CC','HC','Post-tool','Always Prepare','Cost-aware','相对 CC 缩短'],policy_rows,'(28mm,1fr,1fr,1fr,1fr,1fr,1fr)',9)
    cover+='#v(3mm)\n'+paragraph('CC = CPU prefill + CPU decode；HC = HTP prefill + CPU decode。所有策略从设备发起 Tool 到完整 JSON 发布统一计时，使用相同输入与内存分配预算。负百分比表示 Cost-aware 更慢；组间实际 G 有波动，非同轮严格配对。',9)
    cover+='#v(4mm)\n'+paragraph('矩阵：H = 2048 / 4096；P = 16、24、31、32、33、40、48、56、64、128、256、512、1024。156 次独立校准 + 606 次正式测量；另有 18 次新平台预检，不进入五张主图。',9)
    cover+='#v(3mm)\n'+paragraph('阅读顺序：第 2-4 页为 Observation，第 5 页为原始策略比较，第 6 页为 Oracle Gap，第 7 页列出配置与验收，第 8 页为 P31 复测。原矩阵每条件 3 次，不能以小幅差异替代统计证据。',9)
    p1=figure_page(1,'CPU/HTP 性能交界','全域 P 扫描及 31 / 32 / 33 细节；横轴 P，纵轴为 Tool 返回后的完整响应。',111,
        '固定服务器 G 目标 200ms；线为各点中位数，浅色点为三次原始测量。实线为 CC，虚线为提前准备的 HC。H 仅取两个历史长度；P 为 13 点稀疏扫描，并非每个整数。',
        paragraph('可观察到 P32 附近的非平滑阶跃；同时，全域扫描显示延迟继续随大 P 增长，不能把 31/32 的核行为当作整个性能曲面的规律。H2048/P31 两路径接近，不应将小差异夸大为稳定边界翻转。',10))
    prep=[]
    for h in [2048,4096]:
        for name in ['Late HC','Early HC']:
            vals=[x['prepare_ms'] for x in c['preparation'] if x['H']==h and x['policy']==name]
            prep.append([h,'返回后准备' if name=='Late HC' else '提前准备',f'{min(vals):.1f} - {max(vals):.1f}'])
    p2=figure_page(2,'Cost Breakdown：优势区域为何不同','阶段墙钟分解；每根柱采用阶段均值，黑点为三次完整响应。',105,
        'Observation 为观察处理；Blocking preparation 只计 Tool 返回后的阻塞准备；State handoff 包含状态迁移与检查。后台总准备不能再次加到响应堆叠中。Other 为同次完整时间减已列阶段的余额。',
        table(['H','准备方式','P24 / 31 / 32 的总准备中位数范围（ms）'],prep,'(22mm,55mm,1fr)',9)+'#v(3mm)\n'+paragraph('Prefill 主导大部分后端差异。上表总准备可被等待窗口隐藏，但隐藏成功不等于把准备耗时全部转化为完整响应收益；初始化和共同历史构建另列，不计入每轮 Tool 延迟。',9))
    p3=figure_page(3,'Tool Gap 重叠收益','横轴使用设备 CLOCK_BOOTTIME 测得的实际 G，绝不使用 HTTP 服务端目标代替。',114,
        '固定 P32、HC 路径；灰色在 Tool 返回后准备，蓝色在 Tool 等待期间准备。点为每次实际 G 与实际返回后响应；线连接各目标组的实测中位数。两种策略的实际 G 并不完全相同。',
        paragraph('提前准备在这些样本中降低了部分返回后等待，但幅度随 H / G 条件波动。仅有 0 / 50 / 200ms 三个服务端目标组、每组 3 次，不能据此宣称精确的隐藏阈值或稳定的单调关系。',10))
    p4=figure_page(4,'End-to-End 策略比较｜原始三次测量','原始数据保留；P31 低点不能作为稳定策略收益，独立配对复测见第 8 页。',140,
        '服务器 G 目标为 200ms；点为三次原始测量，横线为中位数。Post-tool 按包含准备的独立校准选路；Always Prepare 与 Cost-aware 均使用实际 Router。',
        paragraph('P31 的 Post-tool 实际均走 CPU。两个原批次使用相同随机种子与条件顺序；前两次 CPU prefill 偏快抬高了表观优势。新配对复测未复现该优势，具体运行状态原因尚未定位。',9))
    oracle_rows=[['严格选路 Oracle',f'{matched} / {len(c["regret"])} 个条件状态匹配','Tool 返回至完整响应'],['联合经验 Oracle（可等待）',f'最大观测差距 {maxjoint["joint_empirical_gap_ms"]:.1f} ms','Tool 发起至完整响应']]
    p5=figure_page(5,'Oracle Gap：仍有多少调度空间','左：同准备状态下的严格选路；右：允许等待的完整联合重放。两幅纵轴计时起点不同。',108,
        '严格候选只包含状态匹配且可用的路径；灰色叉号表示无法识别。联合候选包括全部九路实测重放及实际策略本身，保留候选与所选 trial ID，不拼接阶段最小值。',
        table(['Oracle 口径','本轮范围','计时起点'],oracle_rows,'(65mm,95mm,1fr)',9)+'#v(3mm)\n'+paragraph(f'联合差距最大出现在 H{maxjoint["H"]}/P{maxjoint["P"]}、G 目标 {maxjoint["gap_target_ms"]}ms。该条件的 Cost-aware Router 三次返回时均未 ready，因而走 CPU；允许等待的 HC 重放更快。这是后续核实短 Gap / pending 候选准入的线索，不是已经获得的加速；实际 G 波动与三次中位数取最小带来选择偏差。',9))
    setting_rows=[['模型 / KV','Qwen2.5-VL-3B-Instruct Q4_0 / F16 KV'],['设备 / 后端','Xiaomi 14 Pro，SM8650，Android 14，Hexagon v75；显式 hybrid'],['共同预算','context 6144；batch 128；HTP queue / opbatch 均为 1'],['策略采样域','H2048 / 4096，P24 / 31 / 32；G 目标 0 / 50 / 200ms；三策略 × CC / HC / Router'],['关闭的机制','KV 整理、通用预搬移、CPU repack；无 tiles、执行计划、自预热或在线自校准'],['响应与资格','相同完整 JSON 与 6 个生成 ID（含 EOG）；cosine 至少 0.99，NRMSE 仅记录'],['统一主时钟','CLOCK_BOOTTIME，包含系统挂起；steady_clock 仅作为诊断'],['双时钟读取抖动','允许 0.1ms 非原子读取差；主时钟恒等式仍为 0.01ms，原值不改写']]
    appendix=header('方法、验收与复现边界','本报告为固定文本微基准结果，不包含新增图像任务、外网 Tool、能耗或通用 Agent 正确率结论。')
    appendix+=table(['项目','设置或验收标准'],setting_rows,'(44mm,1fr)',9)
    appendix+='#v(4mm)\n'
    init_rows=[[x['H'],f'{x["model_context_init_ms"]/1000:.2f}',f'{x["history_build_ms"]/1000:.2f}'] for x in d['initialization']]
    appendix+=table(['H','模型 / 上下文初始化（s）','共同历史构建（s）'],init_rows,'(35mm,1fr,1fr)',9)
    appendix+='#v(3mm)\n'+paragraph('共同初始化、历史建立和 Tool 前恢复不计入每轮完整响应；观察处理、准备、prefill、交接、decode 与响应后回收分别记录。两批在同一设备串行运行，逐条件完整重放；不执行模型输出动作，不改变手机设置、CPU 频率或用户数据。',9)
    appendix+='#v(3mm)\n'+paragraph('数据验收：完整矩阵、原始记录一致性、HTTP 正文 hash、JSON / 完整 ID / EOG、迁移字节核验、发布顺序、主时钟恒等式和二进制来源检查均通过。26/26 个 H/P 条件接纳 HC；没有挑选成功子集。图形几何与视觉检查已完成。',9)
    appendix+='#v(3mm)\n'+paragraph('本地复现链：两份原始运行目录 → figures/plot-data.json、comparisons.json、measurements.csv → 五张 PNG/PDF → 本汇总。原始目录：20261105-165933-954653-http-oracle-grid；20261105-173045-235354-http-oracle-grid。原始行、失败出图日志、验收记录和本报告源文件均保留。',8)
    retest_rows=[]
    for run in sorted(retest,key=lambda r:r['H']):
        stats=run['statistics']['tool_to_response_ms'];lo,hi=stats['paired_delta_bootstrap_95_ci']
        retest_rows.append([run['H'],f'{stats["cpu_median"]:.1f}',f'{stats["post_tool_median"]:.1f}',
                           f'{stats["paired_delta_median"]:+.1f}',f'[{lo:.1f}, {hi:.1f}]'])
    p6=figure_page(6,'P31 配对复测｜原优势未复现','92 条记录全部验收通过；两个 H 各 20 组相邻配对，AB/BA 各半，使用不同随机种子。',110,
        '左：同一配对的完整响应以细线连接，横线为各策略中位数，所有慢点保留。右：Post-tool 减 CPU 的配对差中位数及 95% bootstrap 区间；负数表示 Post-tool 更快，三项区间均跨 0。',
        table(['H','CPU 中位数 ms','Post-tool 中位数 ms','配对差中位数 ms','95% 区间 ms'],retest_rows,'(22mm,1fr,1fr,1fr,1fr)',9)
        +'#v(3mm)\n'+paragraph('同一设备、模型、运行时库、输入历史、ctx6144 / batch128 / F16 KV；CPU prefill 与 decode 均为 4 线程。每 H 同一进程，先独立校准，再交错重放；40 次 Post-tool 全部走 CPU。本次与原图二进制不同，依据本轮同二进制对照判断策略效应。',9)
        +'#v(2mm)\n'+paragraph('结论：本轮不支持 P31 存在 Post-tool 专用稳定加速机制。未复刻原 13 个 P 的混合前驱序列，尚不能区分调度、频率或缓存等具体原因；同会话配对区间不代表跨会话稳定性。原始复测数据：results/20261105-post-tool-p31-retest/。',9))
    source=prefix+'\n#pagebreak()\n'.join([cover,p1,p2,p3,p4,p5,appendix,p6])
    typ=work/'report.typ';typ.write_text(source)
    wrapper=work/'report-wrapper.pdf'
    subprocess.run(['typst','compile',str(typ),str(wrapper)],check=True)
    doc=PdfReader(wrapper);assert len(doc.pages)==8,f'报告排版异常：{len(doc.pages)}页'
    writer=PdfWriter();sources=[]
    slots=[111,105,114,140,108]
    for i,page in enumerate(doc.pages):
        writer.add_page(page)
        if 1<=i<=5 or i==7:
            path=fig/(NAMES[5 if i==7 else i-1]+'.pdf');reader=PdfReader(path);sources.append(reader);src=reader.pages[0]
            w=float(src.mediabox.width);h=float(src.mediabox.height);slot_h=(110 if i==7 else slots[i-1])*MM;slot_w=277*MM
            scale=min(slot_w/w,slot_h/h)
            x=10*MM+(slot_w-w*scale)/2-float(src.mediabox.left)*scale
            y=float(page.mediabox.height)-(22*MM+slot_h)+(slot_h-h*scale)/2-float(src.mediabox.bottom)*scale
            writer.pages[-1].merge_transformed_page(src,Transformation().scale(scale).translate(x,y),over=True)
    writer.add_metadata({'/Title':'Xiaomi 14 Pro：Observation 与系统收益实验报告','/Author':'A2O 实验记录','/Subject':'SM8650；762条主矩阵记录；92条P31配对复测；2026-11-05'})
    out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('wb') as f:writer.write(f)
    assert len(PdfReader(out).pages)==8
    manifest=dict(created_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),output=str(out.resolve()),pages=8,
                  figures='PDF矢量嵌入；原始数据与复测分开',source_files_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [fig/(n+'.pdf') for n in NAMES]+[fig/'comparisons.json',fig/'plot-data.json',base/'acceptance.json',retest_path,fig/'p31-update-qa.json']})
    (work/'report-manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(manifest,ensure_ascii=False))


if __name__=='__main__':main()
