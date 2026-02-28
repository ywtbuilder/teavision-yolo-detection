# -*- coding: utf-8 -*-
"""
TeaVision V13 | 统计数据接口 (Controller)

提供检测系统遥测与统计的 API 接口：
- GET /stats              → 获取系统总览统计
- GET /stats/trend        → 获取检测趋势数据
- GET /stats/distribution → 获取品种分布数据

=== 接口层职责 ===
接收请求 → 调用数据层查询 → 格式化并返回响应
"""

from fastapi import APIRouter, Query

# 创建路由器
router = APIRouter(tags=["System"])


# ==================== 系统总览统计 ====================

@router.get(
    "/stats",
    summary="获取系统统计信息",
)
async def get_stats():
    """
    获取系统实时统计数据

    从数据库读取累计检测次数、今日检测量、平均置信度等关键指标。
    用于前端仪表盘展示。
    """
    try:
        from backend.dao.database import get_total_stats

        stats = get_total_stats()
        return {
            "success": True,
            "total_detections": stats["total_detections"],
            "today_detections": stats["today_detections"],
            "avg_confidence": (
                round(stats["avg_confidence"] * 100, 1)
                if stats["avg_confidence"]
                else 0
            ),
            "varieties_count": stats["varieties_count"],
            "total_objects": stats["total_objects"],
            "avg_inference_time": (
                round(stats["avg_inference_time"], 1)
                if stats["avg_inference_time"]
                else 0
            ),
        }
    except Exception as e:
        # 数据库不可用时返回默认值，不阻塞前端
        return {
            "success": False,
            "total_detections": 0,
            "today_detections": 0,
            "avg_confidence": 0,
            "varieties_count": 0,
            "total_objects": 0,
            "avg_inference_time": 0,
            "error": str(e),
        }


# ==================== 检测趋势 ====================

@router.get(
    "/stats/trend",
    summary="获取检测趋势数据",
)
async def get_stats_trend(
    days: int = Query(7, le=30, description="查询天数，最多30天"),
):
    """
    获取近期的检测趋势数据

    按天统计检测次数，用于前端趋势图表。
    """
    try:
        from backend.dao.database import get_daily_trend

        trend_data = get_daily_trend(days)
        return {"success": True, "data": trend_data}
    except Exception as e:
        return {"success": False, "error": str(e)}


# ==================== 品种分布 ====================

@router.get(
    "/stats/distribution",
    summary="获取品种分布数据",
)
async def get_stats_distribution():
    """
    获取检测品种的分布数据

    统计各茶叶品种的检测次数，用于前端饼图/条形图。
    """
    try:
        from backend.dao.database import get_variety_distribution

        dist_data = get_variety_distribution()
        return {"success": True, "data": dist_data}
    except Exception as e:
        return {"success": False, "error": str(e)}
