# -*- coding: utf-8 -*-
"""
TeaVision V13 | FastAPI 应用入口

=== 架构说明 ===
本文件仅负责：
1. 创建 FastAPI 应用实例
2. 配置 CORS 跨域中间件
3. 注册各功能域路由（接口层 Controller）
4. 提供根路由和启动入口

=== V13 规范分层 ===
前端 (frontend/)
└── 后端 (backend/)
    ├── 接口层 (controller/) → 收请求、返响应
    ├── 业务层 (service/)    → YOLO 模型推理核心
    ├── 数据层 (dao/)        → 数据库增删改查
    └── 工具层 (utils/)      → 图片处理、文件管理
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# 导入配置
from backend.config import (
    APP_TITLE,
    APP_VERSION,
    API_DESCRIPTION,
    TAGS_METADATA,
    CONTACT_INFO,
    LICENSE_INFO,
)

# 导入接口层 (Controller) 路由模块
from backend.controller import detection, models, stats, training, augmentation

# 导入自定义 Swagger UI
from backend.swagger_ui import register_swagger_ui


# ==================== 创建应用 ====================

app = FastAPI(
    title=APP_TITLE,
    description=API_DESCRIPTION,
    version=APP_VERSION,
    contact=CONTACT_INFO,
    license_info=LICENSE_INFO,
    openapi_tags=TAGS_METADATA,
    docs_url=None,      # 禁用默认文档，使用自定义版本
    redoc_url="/redoc",  # 保留 ReDoc 备用
)


# ==================== 中间件配置 ====================

# CORS 跨域配置（允许前端从不同端口/域名访问）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # 允许所有来源（本地开发和服务器部署通用）
    allow_credentials=True,
    allow_methods=["*"],       # 允许所有 HTTP 方法
    allow_headers=["*"],       # 允许所有请求头
    expose_headers=["*"],      # 暴露所有响应头给前端
)


# ==================== 注册路由 (接口层) ====================

# 按功能域注册各 Controller 模块
app.include_router(detection.router)     # 检测接口
app.include_router(models.router)        # 模型管理
app.include_router(stats.router)         # 统计数据
app.include_router(training.router)      # 训练结果
app.include_router(augmentation.router)  # 数据增强

# 注册自定义 Swagger UI 文档
register_swagger_ui(app)


# ==================== 静态文件服务 ====================

from fastapi.staticfiles import StaticFiles
import os

# 确保 runs 目录存在
os.makedirs("runs", exist_ok=True)

# 挂载 runs 目录，用于访问生成的视频/图片结果
app.mount("/runs", StaticFiles(directory="runs"), name="runs")


# ==================== 根路由 ====================

@app.get("/", tags=["System"])
async def root():
    """
    API 根路由

    返回系统基本信息、版本号及硬件设备状态。
    """
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    device_name = torch.cuda.get_device_name(0) if device == "cuda" else "CPU"

    return {
        "message": "TeaVision | Precision Engine API",
        "version": APP_VERSION,
        "docs": "/docs",
        "device": device,
        "device_name": device_name,
    }


# ==================== 启动入口 ====================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "backend.app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # 开发模式启用热重载
    )
