# -*- coding: utf-8 -*-
"""
TeaVision V13 | 接口层 (Controller)

只干一件事：接收前端请求，调用业务层处理，返回响应。
不跑模型、不存数据、不处理逻辑 — 就是个传声筒。

按功能域拆分：
- detection.py    → 检测相关接口（单图/批量/对比/视频）
- models.py       → 模型管理接口
- stats.py        → 统计数据接口
- training.py     → 训练结果接口
- augmentation.py → 数据增强接口
"""
