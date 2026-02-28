# -*- coding: utf-8 -*-
"""
TeaVision V13 | 后端模块

规范分层架构：
├── app.py           → FastAPI 应用入口
├── config.py        → 项目配置与模型注册
├── schemas.py       → Pydantic 数据模型
├── swagger_ui.py    → 自定义 API 文档界面
├── controller/      → 接口层（接收请求、返回响应）
├── service/         → 业务层（YOLO 模型推理核心）
├── dao/             → 数据层（数据库增删改查）
├── utils/           → 工具层（图片处理、文件管理）
├── models/          → 训练好的模型文件 (.pt)
├── training_results/→ 训练结果数据
└── ultralytics/     → YOLO 核心库
"""
