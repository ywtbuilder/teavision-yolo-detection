# -*- coding: utf-8 -*-
"""
TeaVision V13 | Pydantic 数据模型

定义 API 请求与响应的数据结构：
- DetectionResult   → 单目标检测结果
- DetectionResponse → 检测接口响应体
- ModelInfo         → 模型元数据信息
"""

from typing import Optional, List
from pydantic import BaseModel, Field


class DetectionResult(BaseModel):
    """单目标检测结果详情"""

    class_id: int = Field(
        ...,
        description="类别 ID",
        examples=[0],
    )
    class_name: str = Field(
        ...,
        description="茶叶品种名称",
        examples=["黄山毛峰"],
    )
    confidence: float = Field(
        ...,
        description="检测置信度 (0.0 ~ 1.0)",
        examples=[0.95],
    )
    bbox: List[float] = Field(
        ...,
        description="边界框坐标 [x1, y1, x2, y2]",
        examples=[[100.5, 200.0, 350.5, 400.0]],
    )


class DetectionResponse(BaseModel):
    """检测请求的完整响应体"""

    success: bool = Field(
        ...,
        description="请求是否成功",
    )
    message: str = Field(
        ...,
        description="状态消息",
        examples=["检测完成"],
    )
    inference_time: float = Field(
        ...,
        description="模型推理耗时 (毫秒)",
        examples=[24.5],
    )
    detections: List[DetectionResult] = Field(
        ...,
        description="检测到的目标列表",
    )
    image_base64: Optional[str] = Field(
        None,
        description="标注后的图片 (Base64 编码的 JPEG)",
    )


class ModelInfo(BaseModel):
    """模型元数据信息"""

    model_name: str = Field(
        ...,
        description="模型名称",
        examples=["YOLO-TBD-v1"],
    )
    model_type: str = Field(
        ...,
        description="任务类型",
        examples=["detect"],
    )
    num_classes: int = Field(
        ...,
        description="支持的类别数量",
        examples=[10],
    )
    class_names: dict = Field(
        ...,
        description="类别 ID 到名称的映射",
        examples=[{0: "黄山毛峰", 1: "信阳毛尖"}],
    )
    input_size: int = Field(
        ...,
        description="模型输入尺寸 (像素)",
        examples=[640],
    )
