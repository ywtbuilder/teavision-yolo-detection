# -*- coding: utf-8 -*-
"""
TeaVision V13 | 训练结果接口 (Controller)

提供训练过程数据查询的 API 接口：
- GET /training/runs                           → 获取训练运行列表
- GET /training/run/{run_id}/metrics           → 获取训练指标数据
- GET /training/run/{run_id}/image/{image_name} → 获取训练结果图片
"""

import numpy as np
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from backend.config import TRAINING_RESULTS_DIR, TRAINING_RUN_DISPLAY_NAMES

# 创建路由器
router = APIRouter(tags=["Training"])


def _get_display_name(dir_name: str) -> str:
    """将目录名转换为规范的显示名称"""
    return TRAINING_RUN_DISPLAY_NAMES.get(dir_name, dir_name)


# ==================== 训练运行列表 ====================

@router.get(
    "/training/runs",
    summary="获取训练运行列表",
)
async def get_training_runs():
    """
    获取所有可用的训练运行记录

    遍历训练结果目录，查找包含 results.csv 的子目录。
    支持嵌套目录结构（如 yolo26s_tbd/train4）。
    """
    if not TRAINING_RESULTS_DIR.exists():
        return {"success": True, "runs": []}

    runs = []

    for item in TRAINING_RESULTS_DIR.iterdir():
        if not item.is_dir():
            continue

        display_name = _get_display_name(item.name)

        run_info = {
            "id": item.name,
            "name": display_name,
            "path": str(item.relative_to(TRAINING_RESULTS_DIR)),
            "has_results": (item / "results.csv").exists(),
        }

        # 查找嵌套的训练运行
        sub_runs = []
        for sub in item.iterdir():
            if sub.is_dir() and (sub / "results.csv").exists():
                sub_runs.append({
                    "id": f"{item.name}/{sub.name}",
                    "name": f"{display_name} / {sub.name}",
                    "parent": item.name,
                })

        if sub_runs:
            runs.extend(sub_runs)
        elif run_info["has_results"]:
            runs.append(run_info)

    # 按配置中的定义顺序排序
    display_order = list(TRAINING_RUN_DISPLAY_NAMES.keys())

    def _sort_key(run):
        parent = run.get("parent", run["id"].split("/")[0])
        try:
            idx = display_order.index(parent)
        except ValueError:
            idx = len(display_order)
        return (idx, run.get("name", ""))

    runs.sort(key=_sort_key)

    return {"success": True, "runs": runs}


# ==================== 训练指标数据 ====================

@router.get(
    "/training/run/{run_id:path}/metrics",
    summary="获取训练指标数据",
)
async def get_training_metrics(run_id: str):
    """
    解析 results.csv 并返回 JSON 格式的训练指标

    包含 loss 曲线、mAP 指标等完整训练历史数据。
    """
    try:
        # 安全路径检查
        run_path = (TRAINING_RESULTS_DIR / run_id).resolve()
        if not str(run_path).startswith(str(TRAINING_RESULTS_DIR.resolve())):
            raise HTTPException(status_code=403, detail="非法路径访问")

        csv_path = run_path / "results.csv"
        if not csv_path.exists():
            raise HTTPException(status_code=404, detail="未找到结果文件")

        import pandas as pd

        df = pd.read_csv(csv_path)
        df.columns = [c.strip() for c in df.columns]

        df = df.replace([np.inf, -np.inf], np.nan)
        data_dict = df.to_dict(orient="list")

        for key, values in data_dict.items():
            data_dict[key] = [None if pd.isna(x) else x for x in values]

        return {
            "success": True,
            "columns": df.columns.tolist(),
            "data": data_dict,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"读取训练指标失败: {str(e)}",
        )


# ==================== 训练结果图片 ====================

@router.get(
    "/training/run/{run_id:path}/image/{image_name}",
    summary="获取训练结果图片",
)
async def get_training_image(run_id: str, image_name: str):
    """
    获取指定的训练结果图片

    支持 PNG、JPG、JPEG 格式的训练可视化图片。
    """
    try:
        run_path = (TRAINING_RESULTS_DIR / run_id).resolve()
        if not str(run_path).startswith(str(TRAINING_RESULTS_DIR.resolve())):
            raise HTTPException(status_code=403, detail="非法路径访问")

        img_path = run_path / image_name

        allowed_extensions = {".png", ".jpg", ".jpeg"}
        if img_path.suffix.lower() not in allowed_extensions:
            raise HTTPException(status_code=400, detail="不支持的文件类型")

        if not img_path.exists():
            raise HTTPException(status_code=404, detail="图片不存在")

        return FileResponse(img_path)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"获取训练图片失败: {str(e)}",
        )
