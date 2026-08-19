#!/bin/bash
# 用 ncu 看单个 kernel 的详细指标
set -e

PROJECT_ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
cd "$PROJECT_ROOT/pytorch_binding"

mkdir -p profile/results

# 生成带时间戳的后缀，格式如: 20260818_113520
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# 确保 nvtx 装了
python3 -c "import nvtx" 2>/dev/null || {
    echo "Installing nvtx..."
    pip install --user nvtx
}

echo "Running ncu..."
ncu --set full \
    --target-processes all \
    --nvtx \
    --nvtx-include "FLASH_PROFILE/" \
    --cache-control all \
    -o "profile/results/kernel_profile_${TIMESTAMP}" \
    python3 -m tests.profile_kernel

echo ""
echo "Done. Report: profile/results/kernel_profile_${TIMESTAMP}.ncu-rep"
echo "View with: ncu-ui profile/results/kernel_profile_${TIMESTAMP}.ncu-rep"