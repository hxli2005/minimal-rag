#!/usr/bin/env bash
# Run after mounting saved /data on FunHPC's PyTorch 2.6 / Python 3.10 / CUDA 12.4 image.
# Installs small offline dependencies only; does not run the benchmark.
set -euo pipefail
export PATH="/data/miniconda/envs/torch/bin:/data/miniconda/bin:$PATH"
cd /data/minimal-rag/qwen-bm25-20261015
python -c 'import sys, torch; assert sys.version_info[:2] == (3, 10); assert torch.__version__.split("+")[0] == "2.6.0"; print(torch.__version__, torch.version.cuda)'
python -m venv --system-site-packages .gpu-venv
.gpu-venv/bin/python -m pip install --no-index --find-links funhpc-wheels numpy==1.26.4 transformers==4.53.2
.gpu-venv/bin/python -c 'import inspect, numpy, torch, transformers; from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM; assert "logits_to_keep" in inspect.signature(Qwen3ForCausalLM.forward).parameters; print({"numpy":numpy.__version__, "transformers":transformers.__version__, "cuda_available":torch.cuda.is_available()})'
