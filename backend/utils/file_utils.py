# -*- coding: utf-8 -*-
"""
TeaVision V13 | 文件处理工具

通用文件操作函数：
- save_temp_file    → 保存上传的临时文件
- cleanup_temp_file → 清理临时文件
- ensure_dir        → 确保目录存在
- get_relative_url  → 将文件路径转为 URL 格式
"""

import os
import time
import shutil
from pathlib import Path
from typing import Optional

from fastapi import UploadFile


# 临时文件存放目录
TEMP_DIR = Path("temp_videos")


def save_temp_file(upload_file: UploadFile, prefix: str = "upload") -> Path:
    """
    将上传文件保存为临时文件

    Args:
        upload_file: FastAPI 上传文件对象
        prefix:      文件名前缀

    Returns:
        临时文件的 Path 对象
    """
    TEMP_DIR.mkdir(exist_ok=True)
    filename = upload_file.filename or "unknown"
    temp_path = TEMP_DIR / f"{prefix}_{int(time.time())}_{filename}"

    with temp_path.open("wb") as buffer:
        shutil.copyfileobj(upload_file.file, buffer)

    return temp_path


def cleanup_temp_file(path: Path) -> None:
    """
    安全删除临时文件

    Args:
        path: 待删除文件的路径
    """
    if path and path.exists():
        try:
            os.remove(path)
        except OSError:
            pass


def ensure_dir(directory: Path) -> Path:
    """
    确保目录存在，不存在则创建

    Args:
        directory: 目标目录路径

    Returns:
        目录路径
    """
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def get_relative_url(file_path: Path) -> Optional[str]:
    """
    将文件路径转为前端可访问的 URL 路径

    Args:
        file_path: 文件的绝对/相对路径

    Returns:
        URL 格式的路径字符串，失败返回 None
    """
    if not file_path or not file_path.exists():
        return None

    try:
        rel_path = file_path.relative_to(Path.cwd())
        url_path = str(rel_path).replace("\\", "/")
        if not url_path.startswith("/"):
            url_path = "/" + url_path
        return url_path
    except ValueError:
        return "/" + str(file_path).replace("\\", "/")
