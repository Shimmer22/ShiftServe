ARG BASE_IMAGE
FROM ${BASE_IMAGE}
# Reuse an existing CUDA / PyTorch 2.6 image; SDK and models are bind mounts.
RUN apt-get update && apt-get install -y libc++1 libc++abi1
RUN python -m pip install \
    transformers==4.56.2 numpy==1.26.4 onnx==1.17.0 onnxruntime==1.22.1 \
    onnxsim==0.4.36 pandas==2.2.3 matplotlib==3.10.8 \
    aenum==3.1.17 pydantic==2.14.2 attrs==26.1.0 jsonschema==4.26.0 \
    mako==1.4.3 lxml==6.1.3 pyyaml==6.0.3 packaging==24.0
WORKDIR /workspace/projects/a2o-gap-profiling
