#!/bin/bash
# 用黄小倩真 KB 重跑全部自动实验 → 写入 *_realkb 文件夹(原 bootstrap 数据不动)
# 网络恢复后运行:  bash code/run_realkb.sh
set -e
cd "$(dirname "$0")/.."
ROOT="$PWD"
export KB_DIR="$ROOT/results/knowledge_base_real"

echo "==== ① RAG 消融 (real KB) ===="
OUT_DIR="$ROOT/results/rag_ablation_realkb" python3 code/rag_ablation.py

echo "==== ② Agentic (real KB) ===="
OUT_DIR="$ROOT/results/agentic_realkb" python3 code/agentic_workflow.py

echo "==== ③ Agentic 高召回 (real KB) ===="
OUT_DIR="$ROOT/results/agentic_highrecall_realkb" python3 code/agentic_workflow_highrecall.py

echo "==== ④ 数据飞轮 (real KB) ===="
OUT_DIR="$ROOT/results/flywheel_realkb" python3 code/data_flywheel.py

echo "==== 全部完成。官方数字在 *_realkb 文件夹 ===="
