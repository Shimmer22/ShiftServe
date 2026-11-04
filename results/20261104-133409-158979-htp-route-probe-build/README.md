# 私有 v79 路由探针构建

已复用 wins（10.29.243.183）缓存的 Snapdragon Android 工具链容器，实际编译验证 SDK6.4.0.2 / Hexagon Clang19.0.04。

只构建 v79 DSP skel；使用原源码的符号链接 overlay，3个待修改文件先转为私有副本，再应用仓库中的 `src/a2o/native/profiling/htp-route-probe.patch`。Release、量化group128及原计算内核/门槛/返回逻辑保留；没有开启会大量输出的FARF HIGH。原远端3个源码文件SHA复核未变。

探针在原dispatch分支写入实际路径、原因、M/N/K、DSP类型、HMX返回值和batch维，通过已有profile=2的八个PMU槽回传。该运行的PMU槽不是硬件性能计数。没有更改RPC描述符布局，也不需更换host bridge。完整构建命令、输入hash、补丁、CMakeCache和编译日志在本目录。

只在本次手机进程私有 ADSP_LIBRARY_PATH 下加载新库；主安装的host/v79库均保留且验后SHA未变。正式性能数据不使用探针库。设备结果见 `results/20261104-133528-960312-http-oracle-grid/`。
