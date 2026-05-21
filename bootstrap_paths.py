"""
在导入 omnigibson 之前设置本仓库的数据与 Python 路径。
入口脚本应首行: import bootstrap_paths  # noqa: F401
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

if "OMNIGIBSON_DATA_PATH" not in os.environ:
    os.environ["OMNIGIBSON_DATA_PATH"] = str(PROJECT_ROOT / "datasets")

if "OMNIGIBSON_APPDATA_PATH" not in os.environ:
    os.environ["OMNIGIBSON_APPDATA_PATH"] = str(PROJECT_ROOT / "OmniGibson" / "appdata")

for p in (str(PROJECT_ROOT), str(PROJECT_ROOT / "OmniGibson")):
    if p not in sys.path:
        sys.path.insert(0, p)
