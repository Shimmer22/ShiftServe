# 当前曲线与决策点

只读取已有H4096文本校准，无新增设备测试。

- cost-vs-decode-cpu.png / htp.png：初始CPU/HTP驻留，P16和64，Gap0；横轴为同一输出轨迹的前向评估次数。实线中位数与范围只包含通过校验的记录（每点3次），未通过的HH逐次记录为虚线空心，不进入统计或选路。
- sampled-decision-map.png：完整短动作含EOG提交，按实测组件模拟G0–200ms。在当前合格的CC/HC/CH候选中，P16选择CC、P64选择HC，两种驻留位置均如此。HH未通过数值校验，因此不是四路径全局最优图。
- P17–63未测，不插值、不外推；色条宽度只用于显示离散采样点。暂无可信的连续切换阈值或迟滞带。
- 六个前向点不是六种实际生成长度，无法据此宣称已经定位不同真实输出长度下的切换阈值。Gap未测真实Tool并发。
- curve-data.csv仅含合格试验汇总；decision-data.csv记录候选集合和理想Gap下的逐点成本。HH原始失败数据保留在上级H*.jsonl。
- PNG、PDF、SVG同时导出，qa.json记录文字重叠/越界检查；figure-plan.json记录图意和来源。

复现：conda run -n visiontrim-qwen25vl python scripts/plot_decision_regions.py results/20261103-213658-342299-four-path-structured-grid
