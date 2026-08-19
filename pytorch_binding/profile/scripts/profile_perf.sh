#!/bin/bash
# 用 nsys 看时间线
set -e

PROJECT_ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
cd "$PROJECT_ROOT/pytorch_binding"

mkdir -p profile/results

echo "Running nsys..."
nsys profile \
    --stats=true \
    --output=profile/results/perf_trace \
    python3 -m tests.profile_perf_trace

echo ""
echo "Done. Report: profile/results/perf_trace.nsys-rep"
echo "View with: nsys-ui profile/results/perf_trace.nsys-rep"
