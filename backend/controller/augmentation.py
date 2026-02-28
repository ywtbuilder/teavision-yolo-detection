# -*- coding: utf-8 -*-
"""
TeaVision V13 | 数据增强接口 (Controller)

提供图像增强效果预览的 API 接口：
- POST /augment → 上传图片并应用多种增强效果

=== 接口层职责 ===
接收图片和增强参数 → 调用工具层处理 → 返回增强结果
"""

import io

import cv2
import numpy as np
from PIL import Image
from fastapi import APIRouter, File, UploadFile, HTTPException, Query
from fastapi.responses import JSONResponse

# 工具层
from backend.utils.image_utils import preprocess_image, encode_image_base64

# 创建路由器
router = APIRouter(tags=["Augmentation"])


# ==================== 图像增强 ====================

@router.post(
    "/augment",
    summary="图像增强演示",
)
async def augment_image(
    file: UploadFile = File(..., description="上传待处理的图片"),
    hsv_h: float = Query(0.015, ge=0.0, le=1.0, description="HSV 色调增强 (Hue)"),
    hsv_s: float = Query(0.7, ge=0.0, le=1.0, description="HSV 饱和度增强 (Saturation)"),
    hsv_v: float = Query(0.4, ge=0.0, le=1.0, description="HSV 亮度增强 (Value)"),
    flip_h: bool = Query(False, description="启用水平翻转"),
    flip_v: bool = Query(False, description="启用垂直翻转"),
    rotate: int = Query(0, ge=-180, le=180, description="旋转角度 (-180° ~ 180°)"),
    blur: int = Query(0, ge=0, le=10, description="高斯模糊半径 (0=关闭)"),
):
    """
    数据增强效果预览

    实时演示多种图像增强策略的效果，支持组合叠加：

    - **色彩空间变换 (HSV)**: 色调、饱和度、亮度调节
    - **几何变换**: 水平翻转、垂直翻转、旋转
    - **像素级变换**: 高斯模糊
    """
    try:
        # 读取并预处理图片（调用工具层）
        contents = await file.read()
        _, img_bgr = preprocess_image(contents)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # 存储增强结果
        augmented_images = {"original": img_rgb}

        # HSV 色彩空间变换
        if hsv_h > 0 or hsv_s > 0 or hsv_v > 0:
            hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
            hsv[:, :, 0] = (hsv[:, :, 0] + hsv_h * 180) % 180
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * (1 + hsv_s), 0, 255)
            hsv[:, :, 2] = np.clip(hsv[:, :, 2] * (1 + hsv_v), 0, 255)
            hsv_img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
            augmented_images["hsv"] = hsv_img

        # 水平翻转
        if flip_h:
            augmented_images["flip_horizontal"] = cv2.flip(img_rgb, 1)

        # 垂直翻转
        if flip_v:
            augmented_images["flip_vertical"] = cv2.flip(img_rgb, 0)

        # 旋转变换
        if rotate != 0:
            h, w = img_rgb.shape[:2]
            center = (w // 2, h // 2)
            matrix = cv2.getRotationMatrix2D(center, rotate, 1.0)
            rotated = cv2.warpAffine(img_rgb, matrix, (w, h))
            augmented_images["rotate"] = rotated

        # 高斯模糊
        if blur > 0:
            kernel_size = blur * 2 + 1
            blurred = cv2.GaussianBlur(img_rgb, (kernel_size, kernel_size), 0)
            augmented_images["blur"] = blurred

        # 批量编码为 Base64（调用工具层）
        result = {}
        for name, img in augmented_images.items():
            result[name] = encode_image_base64(img, quality=90, is_bgr=False)

        return JSONResponse(content={
            "success": True,
            "message": f"生成 {len(result)} 种增强效果",
            "images": result,
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"增强失败: {str(e)}")
