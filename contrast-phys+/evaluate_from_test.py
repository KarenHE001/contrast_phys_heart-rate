#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
兼容入口：保留旧路径 `evaluate_from_test.py`。

实际实现已移动到 `evaluation/evaluate_from_test.py`，以便把评估相关代码集中管理。
"""

import os
import runpy

_HERE = os.path.dirname(os.path.abspath(__file__))
_TARGET = os.path.join(_HERE, "evaluation", "evaluate_from_test.py")

if __name__ == "__main__":
    runpy.run_path(_TARGET, run_name="__main__")

