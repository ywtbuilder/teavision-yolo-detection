# -*- coding: utf-8 -*-
"""
TeaVision V13 | 检测接口 (Controller)

提供茶叶目标检测的 API 接口：
- POST /detect              → 单图检测（默认模型）
- POST /detect/with-model   → 指定模型检测
- POST /detect/batch        → 批量检测
- POST /detect/compare      → 多模型对比检测
- POST /detect/video        → 视频检测

=== 接口层职责 ===
✅ 接收请求参数、校验输入
✅ 调用业务层 (Service) 处理
✅ 组装并返回响应
❌ 不做模型推理
❌ 不做数据库操作
❌ 不做图片预处理
"""

import time
from typing import List

from fastapi import APIRouter, File, UploadFile, HTTPException, Query

from backend.config import AVAILABLE_MODELS
from backend.schemas import DetectionResult, DetectionResponse

# 业务层
from backend.service.model_service import (
    get_model,
    run_inference,
    encode_annotated_image,
    run_video_inference,
)
from backend.service.detection_service import save_detection_to_db

# 工具层
from backend.utils.image_utils import preprocess_image
from backend.utils.file_utils import save_temp_file, cleanup_temp_file

# 创建路由器，指定标签前缀
router = APIRouter(tags=["Detection"])


# ==================== 单图检测 ====================

@router.post(
    "/detect",
    response_model=DetectionResponse,
    summary="执行茶叶检测",
)
async def detect_image(
    file: UploadFile = File(..., description="上传待检测的茶叶图片 (JPG/PNG)"),
    conf: float = Query(0.25, ge=0.0, le=1.0, description="置信度阈值"),
    iou: float = Query(0.45, ge=0.0, le=1.0, description="IoU 阈值 (NMS)"),
    return_image: bool = Query(True, description="是否返回带检测框的标注图片"),
):
    """
    上传图片并执行 YOLO-TBD 检测推理

    - **file**: 图片文件，建议分辨率 > 640×640
    - **conf**: 过滤低置信度目标，默认 0.25
    - **iou**: 非极大值抑制阈值，默认 0.45
    - **return_image**: 若为 True，响应中包含 Base64 编码的检测结果图
    """
    try:
        # 1. 读取并预处理图片（调用工具层）
        contents = await file.read()
        image, img_bgr = preprocess_image(contents)

        # 2. 执行推理（调用业务层）
        model = get_model()
        detections, inference_time_ms, result = run_inference(
            model, img_bgr, conf=conf, iou=iou
        )

        # 3. 保存检测记录（调用业务层 → 数据层）
        save_detection_to_db(
            file=file,
            image=image,
            model=model,
            detections=detections,
            inference_time_ms=inference_time_ms,
            conf=conf,
            iou=iou,
        )

        # 4. 生成标注图片（可选）
        image_base64 = None
        if return_image:
            image_base64 = encode_annotated_image(result)

        # 5. 组装响应
        return DetectionResponse(
            success=True,
            message=f"检测完成，发现 {len(detections)} 个目标",
            inference_time=round(inference_time_ms, 2),
            detections=detections,
            image_base64=image_base64,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"检测失败: {str(e)}")


# ==================== 指定模型检测 ====================

@router.post(
    "/detect/with-model",
    summary="使用指定模型检测",
)
async def detect_with_model(
    file: UploadFile = File(..., description="上传待检测的茶叶图片"),
    model_key: str = Query("default", description="模型标识符"),
    conf: float = Query(0.25, ge=0.0, le=1.0, description="置信度阈值"),
    iou: float = Query(0.45, ge=0.0, le=1.0, description="IoU 阈值"),
    return_image: bool = Query(True, description="是否返回标注图片"),
):
    """
    使用指定模型进行检测

    - **model_key**: 模型标识符，可通过 GET /models 接口获取
    - 其他参数同 /detect 接口
    """
    try:
        contents = await file.read()
        image, img_bgr = preprocess_image(contents)

        model = get_model(model_key)
        detections, inference_time_ms, result = run_inference(
            model, img_bgr, conf=conf, iou=iou
        )

        # Save detection record
        save_detection_to_db(
            file=file,
            image=image,
            model=model,
            detections=detections,
            inference_time_ms=inference_time_ms,
            conf=conf,
            iou=iou,
        )

        image_base64 = None
        if return_image:
            image_base64 = encode_annotated_image(result)

        model_name = AVAILABLE_MODELS.get(model_key, {}).get("name", model_key)

        return {
            "success": True,
            "model_key": model_key,
            "model_name": model_name,
            "message": f"检测完成，发现 {len(detections)} 个目标",
            "inference_time": round(inference_time_ms, 2),
            "detections": [d.model_dump() for d in detections],
            "image_base64": image_base64,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"检测失败: {str(e)}")


# ==================== 批量检测 ====================

@router.post(
    "/detect/batch",
    summary="批量检测多张图片",
)
async def detect_batch(
    files: List[UploadFile] = File(..., description="上传多张待检测图片"),
    model_key: str = Query("default", description="模型标识符"),
    conf: float = Query(0.25, ge=0.0, le=1.0, description="置信度阈值"),
    iou: float = Query(0.45, ge=0.0, le=1.0, description="IoU 阈值"),
    return_image: bool = Query(True, description="是否返回标注图片"),
):
    """
    批量检测多张图片

    一次上传多张图片，使用同一模型进行检测。返回每张图片的独立检测结果。
    """
    results_list = []
    total_start_time = time.time()

    model = get_model(model_key)
    model_name = AVAILABLE_MODELS.get(model_key, {}).get("name", model_key)

    for idx, file in enumerate(files):
        try:
            contents = await file.read()
            image_pil, img_bgr = preprocess_image(contents)

            detections, inference_time_ms, result = run_inference(
                model, img_bgr, conf=conf, iou=iou
            )

            # Save detection record
            save_detection_to_db(
                file=file,
                image=image_pil,
                model=model,
                detections=detections,
                inference_time_ms=inference_time_ms,
                conf=conf,
                iou=iou,
            )

            image_base64 = None
            if return_image:
                image_base64 = encode_annotated_image(result)

            results_list.append({
                "index": idx,
                "filename": file.filename,
                "success": True,
                "inference_time": round(inference_time_ms, 2),
                "detection_count": len(detections),
                "detections": [d.model_dump() for d in detections],
                "image_base64": image_base64,
            })

        except Exception as e:
            results_list.append({
                "index": idx,
                "filename": file.filename,
                "success": False,
                "error": str(e),
            })

    total_time_ms = (time.time() - total_start_time) * 1000

    return {
        "success": True,
        "model_key": model_key,
        "model_name": model_name,
        "total_images": len(files),
        "total_time": round(total_time_ms, 2),
        "results": results_list,
    }


# ==================== 多模型对比检测 ====================

@router.post(
    "/detect/compare",
    summary="多模型对比检测",
)
async def detect_compare(
    file: UploadFile = File(..., description="上传待检测图片"),
    model_keys: str = Query(
        "default,yolo26s_scheme_A,yolo26s_scheme_B",
        description="模型标识符列表，多个用逗号分隔",
    ),
    conf: float = Query(0.25, ge=0.0, le=1.0, description="置信度阈值"),
    iou: float = Query(0.45, ge=0.0, le=1.0, description="IoU 阈值"),
):
    """
    使用多个模型对比检测同一张图片

    上传一张图片，同时使用多个模型进行检测，返回各模型的检测结果以便对比。

    - **model_keys**: 模型标识符列表，用逗号分隔
    """
    model_key_list = [k.strip() for k in model_keys.split(",") if k.strip()]

    if not model_key_list:
        raise HTTPException(status_code=400, detail="请至少选择一个模型")

    try:
        contents = await file.read()
        image_pil, img_bgr = preprocess_image(contents)

        comparison_results = []

        for model_key in model_key_list:
            try:
                model = get_model(model_key)
                model_name = AVAILABLE_MODELS.get(model_key, {}).get("name", model_key)

                detections, inference_time_ms, result = run_inference(
                    model, img_bgr, conf=conf, iou=iou
                )

                # Save detection record
                save_detection_to_db(
                    file=file,
                    image=image_pil,
                    model=model,
                    detections=detections,
                    inference_time_ms=inference_time_ms,
                    conf=conf,
                    iou=iou,
                )

                image_base64 = encode_annotated_image(result)

                det_dicts = [d.model_dump() for d in detections]
                avg_conf = (
                    sum(d["confidence"] for d in det_dicts) / len(det_dicts)
                    if det_dicts
                    else 0
                )

                comparison_results.append({
                    "model_key": model_key,
                    "model_name": model_name,
                    "success": True,
                    "inference_time": round(inference_time_ms, 2),
                    "detection_count": len(detections),
                    "avg_confidence": round(avg_conf, 4),
                    "detections": det_dicts,
                    "image_base64": image_base64,
                })

            except Exception as e:
                comparison_results.append({
                    "model_key": model_key,
                    "model_name": AVAILABLE_MODELS.get(model_key, {}).get(
                        "name", model_key
                    ),
                    "success": False,
                    "error": str(e),
                })

        return {
            "success": True,
            "filename": file.filename,
            "models_compared": len(model_key_list),
            "results": comparison_results,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"对比检测失败: {str(e)}")


# ==================== 视频检测 ====================

@router.post(
    "/detect/video",
    summary="视频检测",
)
async def detect_video(
    file: UploadFile = File(..., description="上传待检测视频"),
    model_key: str = Query("default", description="模型标识符"),
    conf: float = Query(0.25, ge=0.0, le=1.0, description="置信度阈值"),
    iou: float = Query(0.45, ge=0.0, le=1.0, description="IoU 阈值"),
):
    """
    上传视频并执行 YOLO-TBD 检测推理

    返回处理后的视频访问 URL。
    """
    temp_path = None

    try:
        # 1. 保存上传的视频（调用工具层）
        temp_path = save_temp_file(file, prefix="video")

        # 2. 获取模型并执行推理（调用业务层）
        model = get_model(model_key)
        output_path = run_video_inference(model, str(temp_path), conf, iou)

        # 3. 组装响应
        if output_path:
            url_path = output_path.replace("\\", "/")
            if not url_path.startswith("/"):
                url_path = "/" + url_path

            return {
                "success": True,
                "message": "视频检测完成",
                "video_url": url_path
            }
        else:
            return {"success": False, "message": "视频处理失败，未能生成结果文件"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"视频检测失败: {str(e)}")
    finally:
        # 清理临时文件（调用工具层）
        if temp_path:
            cleanup_temp_file(temp_path)
