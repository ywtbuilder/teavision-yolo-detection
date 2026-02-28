# -*- coding: utf-8 -*-
"""
TeaVision V13 | 模型服务 — 业务层核心 ⭐

这是整个项目最核心、最有价值的一层。

职责：
- YOLO 模型的加载与缓存管理
- 图片推理执行与结果解析
- 视频推理执行与结果生成
- 检测结果标注图片的编码

=== V12 → V13 变更 ===
- 图片预处理函数 → 迁移到 utils/image_utils.py（工具层）
- 模型加载/推理/视频处理 → 保留在此（业务层核心）
- 导入路径更新：interface.xxx → backend.xxx
"""

import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import cv2
import numpy as np

from ultralytics import YOLO

from backend.config import AVAILABLE_MODELS, DEFAULT_MODEL_PATH
from backend.schemas import DetectionResult
from backend.utils.image_utils import encode_image_base64


# ==================== 全局模型缓存 ====================

# 已加载的模型实例字典，key 为模型标识符
_model_cache: Dict[str, YOLO] = {}


def get_model(model_key: str = "default") -> YOLO:
    """
    获取或加载模型实例

    采用延迟加载策略：首次请求时加载模型并缓存，
    后续请求直接从缓存返回。

    Args:
        model_key: 模型标识符，对应 AVAILABLE_MODELS 中的 key

    Returns:
        已加载的 YOLO 模型实例
    """
    global _model_cache

    # 命中缓存，直接返回
    if model_key in _model_cache:
        return _model_cache[model_key]

    # 查找模型路径
    if model_key in AVAILABLE_MODELS:
        model_path = AVAILABLE_MODELS[model_key]["path"]
    else:
        # 尝试作为直接文件路径使用
        model_path = Path(model_key)

    # 模型文件不存在时回退到默认模型
    if not model_path.exists():
        print(f"[警告] 模型文件不存在: {model_path}，回退到默认模型")
        if "default" in _model_cache:
            return _model_cache["default"]
        model_path = AVAILABLE_MODELS["default"]["path"]
        if not model_path.exists():
            model_path = DEFAULT_MODEL_PATH
            if not model_path.exists():
                print(f"[警告] 默认模型也未找到，将使用标准 YOLO11n 模型")
                model_path = Path("yolo11n.pt")

    # 加载并缓存模型
    print(f"[系统] 正在加载模型: {model_path}")
    loaded_model = YOLO(str(model_path))
    _model_cache[model_key] = loaded_model
    return loaded_model


# ==================== 推理执行 ====================

def run_inference(
    model: YOLO,
    img_bgr: np.ndarray,
    conf: float = 0.25,
    iou: float = 0.45,
) -> Tuple[List[DetectionResult], float, Any]:
    """
    执行模型推理

    Args:
        model:   YOLO 模型实例
        img_bgr: BGR 格式的输入图片
        conf:    置信度阈值
        iou:     IoU 阈值（NMS）

    Returns:
        (检测结果列表, 推理耗时毫秒, 原始结果对象) 元组
    """
    start_time = time.time()
    results = model.predict(
        source=img_bgr,
        conf=conf,
        iou=iou,
        verbose=False,
    )
    inference_time_ms = (time.time() - start_time) * 1000

    # 解析检测框
    detections: List[DetectionResult] = []
    result = results[0]

    if result.boxes is not None:
        for box in result.boxes:
            cls_id = int(box.cls.item())
            confidence = float(box.conf.item())
            detections.append(
                DetectionResult(
                    class_id=cls_id,
                    class_name=model.names[cls_id],
                    confidence=confidence,
                    bbox=box.xyxy[0].tolist(),
                )
            )

    return detections, inference_time_ms, result


def encode_annotated_image(result: Any, quality: int = 90) -> str:
    """
    将标注后的检测结果图编码为 Base64 字符串

    Args:
        result:  YOLO 推理返回的原始结果对象
        quality: JPEG 压缩质量 (1-100)

    Returns:
        Base64 编码的 JPEG 图片字符串
    """
    # YOLO result.plot() 返回 BGR 格式
    annotated_bgr = result.plot()
    return encode_image_base64(annotated_bgr, quality=quality, is_bgr=True)


# ==================== 视频推理 ====================

def run_video_inference(
    model: YOLO,
    video_path: str,
    conf: float = 0.25,
    iou: float = 0.45,
) -> Optional[str]:
    """
    执行视频推理并生成 H.264 MP4 视频

    为了支持浏览器在线预览，必须强制使用 H.264 编码 (avc1)。
    由于 OpenCV 默认的 .avi/MJPG 浏览器不支持，因此手动写入视频流。

    Args:
        model:      YOLO 模型实例
        video_path: 输入视频文件路径
        conf:       置信度阈值
        iou:        IoU 阈值

    Returns:
        生成视频的相对路径，失败返回 None
    """
    from datetime import datetime

    run_name = f"video_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    # 使用绝对路径
    project_dir = Path("runs/detect").resolve()
    save_dir = project_dir / run_name
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / "video.mp4"

    try:
        # 1. 获取视频 FPS
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        cap.release()

        # 2. 执行流式推理
        results = model.predict(
            source=video_path,
            conf=conf,
            iou=iou,
            save=False,
            stream=True,
            verbose=False
        )

        out = None

        # 3. 逐帧处理并写入
        print(f"[系统] 开始视频处理: {run_name}, FPS={fps}")

        for result in results:
            frame = result.plot()

            if out is None:
                h, w = frame.shape[:2]

                try:
                    fourcc = cv2.VideoWriter_fourcc(*'avc1')
                    out = cv2.VideoWriter(str(save_path), fourcc, fps, (w, h))
                    if not out.isOpened():
                        raise Exception("avc1 writer failed")
                except Exception:
                    print("[警告] H.264 编码失败，回退到 mp4v")
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    out = cv2.VideoWriter(str(save_path), fourcc, fps, (w, h))

            if out:
                out.write(frame)

        if out:
            out.release()
            print(f"[系统] 视频处理完成: {save_path}")

        # 4. 返回路径
        if save_path.exists() and save_path.stat().st_size > 0:
            try:
                rel_path = save_path.relative_to(Path.cwd())
                return str(rel_path).replace("\\", "/")
            except ValueError:
                return str(save_path)
        else:
            print("[错误] 生成的视频文件为空")
            return None

    except Exception as e:
        print(f"[错误] 视频推理失败: {e}")
        return None
