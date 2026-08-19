#!/bin/bash
# 快速检查 kernel 编译产物
set -e

PROJECT_ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
cd "$PROJECT_ROOT/pytorch_binding"

SO_FILE=$(ls flash_attn/_C*.so 2>/dev/null | head -1)

if [ -z "$SO_FILE" ]; then
    echo "Error: .so not found. Run 'pip install -e .' first."
    exit 1
fi

echo "Analyzing: $SO_FILE"
echo ""
echo "=== Instruction counts ==="
cuobjdump --dump-sass "$SO_FILE" 2>&1 | awk '{print $2}' | sort | uniq -c | sort -rn | head -15

echo ""
echo "=== Tensor Core (HMMA) check ==="
HMMA_COUNT=$(cuobjdump --dump-sass "$SO_FILE" 2>&1 | grep -c "HMMA")
echo "HMMA instructions: $HMMA_COUNT"

if [ "$HMMA_COUNT" -gt 0 ]; then
    echo "✅ Tensor Core ENABLED"
else
    echo "❌ Tensor Core NOT enabled"
fi

echo ""
echo "=== Target architecture ==="
cuobjdump --dump-elf-symbols "$SO_FILE" 2>&1 | grep -i "arch =" | sort -u
