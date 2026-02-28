# -*- coding: utf-8 -*-
"""
TeaVision V13 | 检测业务服务

从 V12 的 detection 路由中提取的业务逻辑：
- 检测结果持久化到数据库
- 保持接口层（Controller）的轻量化

职责：
- 将检测结果保存到数据库（调用 DAO 层）
- 封装数据转换逻辑
"""

import time
from typing import List

from fastapi import UploadFile
from PIL import Image

from backend.schemas import DetectionResult


def save_detection_to_db(
    file: UploadFile,
    image: Image.Image,
    model,
    detections: List[DetectionResult],
    inference_time_ms: float,
    conf: float,
    iou: float,
) -> None:
    """
    将检测结果保存到数据库

    此函数不会抛出异常——数据库写入失败时仅打印警告，
    不影响 API 正常返回检测结果。

    Args:
        file:              上传的文件对象
        image:             PIL 图像对象
        model:             YOLO 模型实例
        detections:        检测结果列表
        inference_time_ms: 推理耗时（毫秒）
        conf:              置信度阈值
        iou:               IoU 阈值
    """
    try:
        from backend.dao.database import save_detection_record

        avg_confidence = (
            sum(d.confidence for d in detections) / len(detections)
            if detections
            else 0.0
        )

        record_model_name = (
            model.model_name if hasattr(model, "model_name") else "YOLO-TBD"
        )
        img_width, img_height = image.size

        # DetectionResult → dict，用于数据库存储
        detect_dicts = [
            {
                "class_id": d.class_id,
                "class_name": d.class_name,
                "confidence": d.confidence,
                "bbox": d.bbox,
            }
            for d in detections
        ]

        save_detection_record(
            image_name=file.filename or f"upload_{int(time.time())}.jpg",
            image_size=(img_width, img_height),
            model_name=record_model_name,
            total_objects=len(detections),
            avg_confidence=avg_confidence,
            inference_time_ms=inference_time_ms,
            detection_type="image",
            conf_threshold=conf,
            iou_threshold=iou,
            objects=detect_dicts,
        )
    except Exception as db_err:
        print(f"[警告] 数据库记录保存失败: {db_err}")
