# 中间 P 扫描启动记录

旧截断填充导致 P24 观察中出现 The cWait，HC 自由生成 cWait；cosine=0.987383，响应错误且未过 0.99。原始失败记录保留，不进入性能结论。

后续固定使用 neutral_padding_v1：完整中性 note 词＋换行＋完整 Wait 指令；P 由设备真实 tokenizer 校验。未改写本轮原始日志。
