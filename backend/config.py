# -*- coding: utf-8 -*-
"""
TeaVision V13 | 项目配置模块

集中管理所有后端配置项：
- 项目路径
- 模型注册表（3 规模 × 5 方案 = 15 模型）
- API 元数据
- 标签分类

=== V12 → V13 变更 ===
路径结构调整：
- V12: PROJECT_ROOT/backend/models/       → V13: PROJECT_ROOT/backend/models/
- V12: PROJECT_ROOT/backend/training_results/ → V13: PROJECT_ROOT/backend/training_results/
- 模型和训练结果仍在 backend/ 下，但 config.py 也移入 backend/
"""

import sys
from pathlib import Path
from typing import Dict, Any

# ==================== 路径配置 ====================

# 项目根目录（即 ultralytics-8.4.5_V13/）
# config.py 现在在 backend/ 目录下，所以 parent.parent = 项目根
PROJECT_ROOT = Path(__file__).parent.parent

# 后端目录
BACKEND_DIR = Path(__file__).parent

# 将 backend/ultralytics 目录加入 Python 搜索路径
sys.path.insert(0, str(BACKEND_DIR))

# 默认模型文件路径
DEFAULT_MODEL_PATH = BACKEND_DIR / "models" / "tea_best.pt"

# 训练结果存放目录
TRAINING_RESULTS_DIR = BACKEND_DIR / "training_results"


# ==================== 模型注册表 ====================

# 所有可用的训练模型配置
# key: 模型唯一标识符, value: 包含名称、路径和描述的字典
# 命名规范: yolo26{s/m/l}_scheme_{O/A/B/C/D}
# 方案说明:
#   O = 原始方案 (Original Baseline)
#   A = TBD (Triple-Branch Attention + Self-Correction Group Convolution)
#   B = BiFormer (Bi-Level Routing Attention)
#   C = Fusion (Multi-Scale Feature Fusion)
#   D = SPD-CARAFE (SPD-Conv + CARAFE Lightweight Upsampling)

AVAILABLE_MODELS: Dict[str, Dict[str, Any]] = {
    # ── 默认模型 ──
    "default": {
        "name": "YOLO-TBD 默认模型 (L)",
        "path": BACKEND_DIR / "models" / "tea_best.pt",
        "description": "当前综合性能最优模型 — YOLO26-L 方案A·TBD",
    },

    # ── Small 规模 ──
    "yolo26s_scheme_O": {
        "name": "YOLO26-S 方案O · 原始基线",
        "path": BACKEND_DIR / "models" / "yolo26s_scheme_O.pt",
        "description": "小模型 — 原始 YOLO26 基线方案",
    },
    "yolo26s_scheme_A": {
        "name": "YOLO26-S 方案A · TBD",
        "path": BACKEND_DIR / "models" / "yolo26s_scheme_A.pt",
        "description": "小模型 — 三分支注意力 + 自校准分组卷积改进方案",
    },
    "yolo26s_scheme_B": {
        "name": "YOLO26-S 方案B · BiFormer",
        "path": BACKEND_DIR / "models" / "yolo26s_scheme_B.pt",
        "description": "小模型 — BiFormer 双向路由注意力机制改进方案",
    },
    "yolo26s_scheme_C": {
        "name": "YOLO26-S 方案C · Fusion",
        "path": BACKEND_DIR / "models" / "yolo26s_scheme_C.pt",
        "description": "小模型 — 多尺度特征融合改进方案",
    },
    "yolo26s_scheme_D": {
        "name": "YOLO26-S 方案D · SPD-CARAFE",
        "path": BACKEND_DIR / "models" / "yolo26s_scheme_D.pt",
        "description": "小模型 — SPD-Conv + CARAFE 轻量上采样改进方案",
    },

    # ── Medium 规模 ──
    "yolo26m_scheme_O": {
        "name": "YOLO26-M 方案O · 原始基线",
        "path": BACKEND_DIR / "models" / "yolo26m_scheme_O.pt",
        "description": "中等模型 — 原始 YOLO26 基线方案",
    },
    "yolo26m_scheme_A": {
        "name": "YOLO26-M 方案A · TBD",
        "path": BACKEND_DIR / "models" / "yolo26m_scheme_A.pt",
        "description": "中等模型 — 三分支注意力 + 自校准分组卷积改进方案",
    },
    "yolo26m_scheme_B": {
        "name": "YOLO26-M 方案B · BiFormer",
        "path": BACKEND_DIR / "models" / "yolo26m_scheme_B.pt",
        "description": "中等模型 — BiFormer 双向路由注意力机制改进方案",
    },
    "yolo26m_scheme_C": {
        "name": "YOLO26-M 方案C · Fusion",
        "path": BACKEND_DIR / "models" / "yolo26m_scheme_C.pt",
        "description": "中等模型 — 多尺度特征融合改进方案",
    },
    "yolo26m_scheme_D": {
        "name": "YOLO26-M 方案D · SPD-CARAFE",
        "path": BACKEND_DIR / "models" / "yolo26m_scheme_D.pt",
        "description": "中等模型 — SPD-Conv + CARAFE 轻量上采样改进方案",
    },

    # ── Large 规模 ──
    "yolo26l_scheme_O": {
        "name": "YOLO26-L 方案O · 原始基线",
        "path": BACKEND_DIR / "models" / "yolo26l_scheme_O.pt",
        "description": "大模型 — 原始 YOLO26 基线方案",
    },
    "yolo26l_scheme_A": {
        "name": "YOLO26-L 方案A · TBD",
        "path": BACKEND_DIR / "models" / "yolo26l_scheme_A.pt",
        "description": "大模型 — 三分支注意力 + 自校准分组卷积改进方案",
    },
    "yolo26l_scheme_B": {
        "name": "YOLO26-L 方案B · BiFormer",
        "path": BACKEND_DIR / "models" / "yolo26l_scheme_B.pt",
        "description": "大模型 — BiFormer 双向路由注意力机制改进方案",
    },
    "yolo26l_scheme_C": {
        "name": "YOLO26-L 方案C · Fusion",
        "path": BACKEND_DIR / "models" / "yolo26l_scheme_C.pt",
        "description": "大模型 — 多尺度特征融合改进方案",
    },
    "yolo26l_scheme_D": {
        "name": "YOLO26-L 方案D · SPD-CARAFE",
        "path": BACKEND_DIR / "models" / "yolo26l_scheme_D.pt",
        "description": "大模型 — SPD-Conv + CARAFE 轻量上采样改进方案",
    },
}

# ==================== 训练运行显示名称 ====================

# 将训练结果目录名映射为规范的中文显示名称
# 按 S → M → L 排列，每个规模下按 O → A → B → C → D 排列
TRAINING_RUN_DISPLAY_NAMES: Dict[str, str] = {
    # Small
    "yolo26s_scheme_O":  "YOLO26-S 方案O · 原始基线",
    "yolo26s_scheme_A":  "YOLO26-S 方案A · TBD",
    "yolo26s_scheme_B":  "YOLO26-S 方案B · BiFormer",
    "yolo26s_scheme_C":  "YOLO26-S 方案C · Fusion",
    "yolo26s_scheme_D":  "YOLO26-S 方案D · SPD-CARAFE",
    # Medium
    "yolo26m_scheme_O":  "YOLO26-M 方案O · 原始基线",
    "yolo26m_scheme_A":  "YOLO26-M 方案A · TBD",
    "yolo26m_scheme_B":  "YOLO26-M 方案B · BiFormer",
    "yolo26m_scheme_C":  "YOLO26-M 方案C · Fusion",
    "yolo26m_scheme_D":  "YOLO26-M 方案D · SPD-CARAFE",
    # Large
    "yolo26l_scheme_O":  "YOLO26-L 方案O · 原始基线",
    "yolo26l_scheme_A":  "YOLO26-L 方案A · TBD",
    "yolo26l_scheme_B":  "YOLO26-L 方案B · BiFormer",
    "yolo26l_scheme_C":  "YOLO26-L 方案C · Fusion",
    "yolo26l_scheme_D":  "YOLO26-L 方案D · SPD-CARAFE",
}


# ==================== API 元数据 ====================

# FastAPI 应用信息
APP_TITLE = "TeaVision | 智能茶叶形态检测系统 API"
APP_VERSION = "4.0.0"

# API 接口说明文档（ Markdown 格式，展示在 Swagger UI 顶部）
API_DESCRIPTION = """
<img src="https://img.shields.io/badge/TeaVision-v4.0-2D4033?style=for-the-badge&logo=leaf" alt="TeaVision Banner">

## 系统简介

**TeaVision** 是一个基于 **YOLO-TBD** 架构的高精度茶叶嫩芽检测系统。
专为解决复杂背景与不确定性环境下的微小目标检测难题而设计。

### 系统架构 (V13 · 规范分层)

```
后端 (Backend)
├── 接口层 (Controller)  → 接收请求、返回响应
├── 业务层 (Service)     → YOLO 模型推理、茶叶检测核心
├── 数据层 (DAO)         → 数据库增删改查
└── 工具层 (Utils)       → 图片处理、文件管理
```

### 核心架构 (YOLO-TBD)

*   **TBAM (Triple-Branch Attention)**: 三分支注意力机制，实现空间与通道特征的深度交互。
*   **SCGC (Self-Correction Group Convolution)**: 自校准分组卷积，动态适应不同尺度的茶叶特征。
*   **Enhanced PAFPN**: 改进的多尺度特征融合网络，提升微小目标的召回率。

### 多规模模型支持

本版本支持 **15 个模型** (3 规模 × 5 方案):
*   **Small (S)**: 轻量级推理，适合边缘设备
*   **Medium (M)**: 平衡性能与速度
*   **Large (L)**: 最高精度，适合服务器部署

### 接口功能

*   **检测引擎**: 支持高精度茶叶嫩芽检测、置信度过滤及可视化标注。
*   **数据增强**: 提供 HSV 变换、MixUp、Mosaic 等多种增强策略的实时演示。
*   **智能统计**: 实时监控系统检测量、推理延迟及品种分布数据。
*   **多模型对比**: 支持同时使用多个训练模型进行检测对比。

---

*由合肥工业大学计算机与信息学院开发 | Powered by Ultralytics YOLO*
"""

# API 标签分组（用于 Swagger 文档中的功能分类）
TAGS_METADATA = [
    {
        "name": "Detection",
        "description": "核心检测功能。上传图片进行茶叶嫩芽的实时检测与分析。"
                       "支持单图检测、批量检测和多模型对比。",
    },
    {
        "name": "Augmentation",
        "description": "数据增强演示。提供多种图像处理与增强算法的可视化展示。",
    },
    {
        "name": "Training",
        "description": "训练结果查询。获取模型训练过程中的指标数据和可视化图片。",
    },
    {
        "name": "System",
        "description": "系统信息与遥测。获取模型状态、统计数据及系统运行指标。",
    },
]

# 联系信息
CONTACT_INFO = {
    "name": "TeaVision Team",
    "url": "https://github.com/ultralytics/ultralytics",
    "email": "contact@teavision.ai",
}

# 许可证
LICENSE_INFO = {
    "name": "MIT License",
    "url": "https://opensource.org/licenses/MIT",
}
