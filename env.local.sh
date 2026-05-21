#!/usr/bin/env bash
# 在本仓库根目录 source 此文件，使 OmniGibson 使用本地最小 datasets（不依赖 BEHAVIOR-1K 路径）。
# 用法: cd "$(dirname "$0")" && source env.local.sh

_LESSON_PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
export OMNIGIBSON_DATA_PATH="${_LESSON_PROJECT_ROOT}/datasets"
export OMNIGIBSON_APPDATA_PATH="${_LESSON_PROJECT_ROOT}/OmniGibson/appdata"
export PYTHONPATH="${_LESSON_PROJECT_ROOT}:${_LESSON_PROJECT_ROOT}/OmniGibson:${PYTHONPATH:-}"

# 导航 / 仿真 API 默认端口（可按需覆盖）
export BEHAVIOR_ENV_PORT="${BEHAVIOR_ENV_PORT:-5001}"
export BEHAVIOR_NAV_SUPERVISOR_PORT="${BEHAVIOR_NAV_SUPERVISOR_PORT:-5002}"

echo "[env.local] OMNIGIBSON_DATA_PATH=${OMNIGIBSON_DATA_PATH}"
echo "[env.local] OMNIGIBSON_APPDATA_PATH=${OMNIGIBSON_APPDATA_PATH}"
echo "[env.local] PYTHONPATH includes repo root + OmniGibson"

# 清理运行缓存（可选，打包前执行）:
# rm -rf "${_LESSON_PROJECT_ROOT}/OmniGibson/appdata/local" "${_LESSON_PROJECT_ROOT}/OmniGibson/appdata/global"
