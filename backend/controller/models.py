# -*- coding: utf-8 -*-
"""
TeaVision V13 | 模型管理接口 (Controller)

提供模型查询相关的 API 接口：
- GET /models      → 获取可用模型列表
- GET /model/info  → 获取当前模型元数据

=== 接口层职责 ===
只做「接收请求 → 调用业务层 → 返回响应」
"""

from fastapi import APIRouter, HTTPException

from backend.config import AVAILABLE_MODELS
from backend.schemas import ModelInfo
from backend.service.model_service import get_model

# 创建路由器
router = APIRouter(tags=["System"])


# ==================== 模型列表 ====================

@router.get(
    "/models",
    summary="获取可用模型列表",
)
async def get_available_models():
    """
    获取所有可用的训练模型

    返回系统中注册的所有模型信息，包括名称、描述和可用状态。
    用于前端模型选择器。
    """
    models_list = []
    for key, info in AVAILABLE_MODELS.items():
        models_list.append({
            "key": key,
            "name": info["name"],
            "description": info["description"],
            "available": info["path"].exists(),
        })

    return {
        "success": True,
        "models": models_list,
    }


# ==================== 模型元数据 ====================

@router.get(
    "/model/info",
    response_model=ModelInfo,
    summary="获取模型元数据",
)
async def get_model_info():
    """
    获取当前加载的模型信息

    返回模型的名称、类型、支持的茶叶品种列表及输入尺寸配置。
    """
    try:
        model = get_model()
        return ModelInfo(
            model_name=(
                model.model_name if hasattr(model, "model_name") else "YOLO-TBD"
            ),
            model_type=model.task if hasattr(model, "task") else "detect",
            num_classes=len(model.names),
            class_names=model.names,
            input_size=640,
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"获取模型信息失败: {str(e)}",
        )
