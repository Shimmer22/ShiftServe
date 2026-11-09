# 主机 prompt 布局与共享前缀验证

入口 `scripts/collect/prompt_layout_driver.py` 在 wins 上运行，复用既有 BFCL 容器、Qwen3.5-9B BF16 与 tokenizer，独占主机 CUDA0。配置与范围记录在 `configs/prompt-layout-host.json`，不改变手机实验，不比较手机 GPU 后端。

输入为原 BFCL 主机轨迹的全部 18 个新增函数公开边界。四个布局：原模板、固定规则前移、保持初始目录且新增说明只在公开用户消息中追加、追加说明并保持历史空 think 标记。原开放用户消息保留，未提前公开未来函数；解析只接受当前已经开放的工具。后两种布局改变了工具说明所在的消息层级，不能仅由字段保留就宣称语义等价，需核对响应并另做完整任务评分。

每个条件先清空本私有服务的 prefix cache，再用变换后的上一请求及相同的冻结生成 token 预热；这部分不计目标请求时间。随后提交变换后的当前请求，记录服务端缓存计数、prefill 指标和完整输出。四条件顺序按边界轮换，当前只做单次机制验证，没有重复测量置信区间。新增函数时点与历史来源相同，但输出属于冻结历史下的单边界响应，不自动执行生成动作，不是完整任务 E2E 或官方准确率。

服务使用私有 localhost:18769，显式开启 prefix caching。为在条件间清空缓存，按已安装 vLLM 源码启用本私有服务的开发缓存接口。首个启动尝试因未启用该接口而在首个测量前返回 404；原错误日志保留，正式采集在修正接口配置后运行。退出只停止该端口对应的服务并释放主机进程锁。

结果：`results/20261109-prompt-layout-host/`。`preflight.json` 给出逻辑 H/P，`measurements.json` 保存真实缓存与调用检查；`summary.json` 由 `scripts/analyze/prompt_layout_host.py` 生成。逻辑 H 不等同实际缓存命中，Mamba/GDN 缓存检查点的粒度和位置可能阻止中间前缀复用。prefill 加速不能直接解释为完整请求加速，输出 token 数变化也要单列。

已完成结果与限制见 [18 个边界的四条件验证](../validation/prompt-layout-host.md)。
