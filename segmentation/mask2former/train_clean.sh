#!/usr/bin/env bash
# 已弃用复杂 awk 管道；终端一行式进度请直接用 train.py 的 --clean-console（默认开启）。
set -euo pipefail
cd "$(dirname "$0")"
exec python -u train.py "$@"
