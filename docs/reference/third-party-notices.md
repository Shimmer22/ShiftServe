# 第三方代码

`src/a2o/vision/qnn_vendor.py` 复用 Qualcomm AI Hub Models 的 `vision_encoder_adaptations.py`，以及 `model_adaptations.py` 中的 `ConvInplaceLinear`。上游 commit：`2565a5b11b8ab98a6079477294d3930b6fdbb18d`。仅将该类的导入改为同文件定义，避免引入与视觉导出无关的完整依赖链。

来源：[视觉适配](https://github.com/qualcomm/ai-hub-models/blob/2565a5b11b8ab98a6079477294d3930b6fdbb18d/src/qai_hub_models/models/templates/qwen2_vl/vision_encoder_adaptations.py)、[线性层转换](https://github.com/qualcomm/ai-hub-models/blob/2565a5b11b8ab98a6079477294d3930b6fdbb18d/src/qai_hub_models/models/templates/llm/model_adaptations.py)。固定位置/mask 包装参考同版本的 `vision_encoder.py`。

上游 BSD-3-Clause 许可：

Copyright 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS “AS IS” AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

