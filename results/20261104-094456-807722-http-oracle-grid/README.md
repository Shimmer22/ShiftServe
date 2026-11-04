# 启动验收失败：HTTP 正文 P 不精确

本轮在 CPU/HTP 初始化及 H4096 建立后，生成 HTTP 正文夹具时触发 `HTTP fixture retokenization mismatch`，未执行任何测量。直接拼接 token piece 的正文重新 tokenize 后不保证等于原 token 序列，因此拒绝继续。

原始 `runtime.log`、`failure.json`、协议日志与设备文件均保留。后续实现改为 tokenize 完整候选正文并寻找精确 P=16/64，使用新具名目录重新运行；不把本轮计入通过组，不推导性能结果。
