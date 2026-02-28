# -*- coding: utf-8 -*-
"""
TeaVision V13 | 图片处理工具

从 V12 的 model_service.py 和 augmentation.py 中提取的通用图片处理函数：
- preprocess_image   → 上传图片预处理（PIL + OpenCV 双格式）
- encode_image_base64 → numpy/PIL 图片编码为 Base64 JPEG
- pil_to_bgr         → PIL → OpenCV BGR 转换
"""

import io
import base64
from typing import Tuple

import cv2
import numpy as np
from PIL import Image


def preprocess_image(contents: bytes) -> Tuple[Image.Image, np.ndarray]:
    """
    预处理上传的图片数据

    将原始字节流转换为 PIL 图像和 OpenCV BGR 格式的 numpy 数组。

    Args:
        contents: 图片的原始二进制数据

    Returns:
        (PIL.Image, BGR格式的numpy数组) 元组
    """
    image = Image.open(io.BytesIO(contents))

    # 统一转换为 RGB 模式
    if image.mode != "RGB":
        image = image.convert("RGB")

    # PIL (RGB) → numpy (RGB) → OpenCV (BGR)
    img_rgb = np.array(image)
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    return image, img_bgr


def encode_image_base64(
    image: np.ndarray,
    quality: int = 90,
    is_bgr: bool = True,
) -> str:
    """
    将图片编码为 Base64 JPEG 字符串

    Args:
        image:   numpy 格式的图片数据
        quality: JPEG 压缩质量 (1-100)
        is_bgr:  输入是否为 BGR 格式（OpenCV 默认），True 时自动转 RGB

    Returns:
        Base64 编码的 JPEG 图片字符串
    """
    if is_bgr:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    pil_img = Image.fromarray(image)
    buffer = io.BytesIO()
    pil_img.save(buffer, format="JPEG", quality=quality)

    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def pil_to_bgr(image: Image.Image) -> np.ndarray:
    """
    PIL 图像转 OpenCV BGR 格式

    Args:
        image: PIL 图像对象

    Returns:
        BGR 格式的 numpy 数组
    """
    if image.mode != "RGB":
        image = image.convert("RGB")
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
