# H/P 线性成本路由

2026-11-07 接替精确 H/P 白名单。首轮原 CPU 的73题保留；后续未运行条件和重复轮次使用 `toolsandbox-full-*-linear.json`。旧十题和旧校准不解释为本策略结果。

返回后比较 `CC=R(H)+C(P)` 与 `HC=N(P)+M(H+P)`。R是CPU锚点恢复，C/N分别是CPU/HTP prefill，M是HTP更新状态迁回CPU（包括完整状态字节核验）。每项使用非负截距和斜率的最小二乘拟合。迁回状态长度是H+P，因此迁移成本同时影响H项和P项。后续均在CPU decode，不用固定D预测未来生成长度。

Tool Gap内always prepare HTP锚点；只有实际ready且版本和前缀匹配时比较成本。准备未完成仍使用CPU，不阻塞等待；CPU→HTP成本单独拟合并记录，已被gap隐藏的部分不重复收费。旧任务仍占用lane时记录busy，不伪报ready。

独立校准网格H={128,768,1536}，P={1,8,32,128,384}，各三次，中位数参与拟合。样本由既有真实token片段截取或重复构建，属于受控长度校准，不是自然任务质量评测；固定随机顺序，性能样本不按greedy一致性筛掉，质量差异独立保存。拟合输出RMSE、R²与最大误差。范围外仍线性预测并记录extrapolated，不恢复精确白名单。

线性假设可能不能捕获batch128附近台阶、上下文注意力和内容影响。拟合残差必须和正式预测误差一起检查。若实测边界落在P<1，可能在已ready条件下全部P选择HTP；不人为制造CPU/NPU阈值。

后台接续入口 `scripts/collect/route_linear_handoff.py`：等既有CPU完整结束、同步结果、停止原服务、校准新服务、关闭校准服务，再接原Full的未运行条件及两轮重复。每轮模型设备串行持锁；当前切换状态在 `results/20261107-toolsandbox-full/linear-routing/progress.json`，校准细节在其 `calibration/progress.json`，正式采集状态仍在 `repeats/progress.json`。完整日志和首轮二进制差异保留。连接故障无限重试，真实计算失败停在现场。

用户指定下界保护：模型字段 `routing_p_min=32`，P≤31无条件选择CC；P≥32才比较预测成本。即使拟合预测小P的HC更快，也由下界保护覆盖；记录原因 `below_routing_p_min`。此阈值是配置规则，不是拟合得到的性能交点。
