# -*- coding: utf-8 -*-
"""
TeaVision V13 | 业务层 (Service)

系统核心层，封装所有业务逻辑：
- model_service.py     → YOLO 模型加载、缓存、推理（项目灵魂）
- detection_service.py → 检测结果处理、数据库持久化

设计原则：
- 业务层只做「处理」，不做「接收请求」和「返回响应」
- 业务层调用 DAO 层存数据、调用 Utils 层处理图片
- 接口层 (Controller) 只负责调用业务层
"""
