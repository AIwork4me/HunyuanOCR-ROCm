# HunyuanOCR-ROCm

腾讯混元 OCR 模型的 AMD ROCm 移植与评估仓库。

在 AMD gfx1100 (RDNA3) 上运行 HunyuanOCR，并报告 OmniDocBench v1.6 评测结果。

## Install（安装）

```bash
git clone https://github.com/AIwork4me/HunyuanOCR-ROCm.git
cd HunyuanOCR-ROCm
pip install -e ".[dev]"
pip install -e ".[platform]"    # 可选：平台集成
```

## Demo（演示）

无需 GPU 的 smoke 后端：

```bash
python adapter/run_adapter.py --img-dir examples --out-dir /tmp/out --platform linux-rocm --backend smoke
```

## Evaluation（评测）

```bash
omnidocbench-rocm run \
  --stage all \
  --platform linux-rocm \
  --version v16 \
  --revision 2b161d0 \
  --adapter adapter/run_adapter.py \
  --model-id hunyuan-ocr \
  --backend vllm \
  --server-url http://127.0.0.1:8000/v1 \
  --git-commit "$(git rev-parse HEAD)" \
  --results-dir results/omnidocbench/v16/linux-rocm \
  --cdm
```

## Reproducibility（可复现性）

硬件：AMD gfx1100 (Radeon PRO W7900)，48 GB VRAM，ROCm 7.2。
评测数据与配置锁定于 `reproducibility.lock.yaml`。

## Known Gaps（已知限制）

- 平台标准 artifacts 尚未生成（等待 score/publish 执行）
- windows-hip 仍为 community-wanted
- 完整列表见 [README.md](README.md)
