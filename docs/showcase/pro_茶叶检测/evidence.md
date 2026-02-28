# TeaVision V13 · 展示证据页

## 一句话价值

以 YOLOv11 为基线完成四方向改进实验（TBD / BiFormer / Fusion / SPD-CARAFE），并工程化为可浏览器演示、训练结果可存档对比的完整检测系统。

## 演示资源

| 资源 | 路径 | 说明 |
|------|------|------|
| 演示视频 | `docs/showcase/pro_茶叶检测/demo.mp4` | 建议镜头：上传图片 → 多模型对比 → 结果详情 → 训练曲线页 |
| 截图 1 | `docs/showcase/pro_茶叶检测/shot-01.png` | 检测主界面（上传 + 推理结果 + 统计卡片） |
| 截图 2 | `docs/showcase/pro_茶叶检测/shot-02.png` | 多模型同图对比（O/A/B/C/D 并排） |
| 截图 3 | `docs/showcase/pro_茶叶检测/shot-03.png` | 训练指标看板（mAP/Loss 曲线、混淆矩阵） |

## 一键运行命令

```powershell
cd ywtbuilder-teavision-yolo-detection
.\start_all_services.bat
```

## 核心技术决策

| 决策 | 理由 |
|------|------|
| FastAPI + Pydantic | 自动生成 OpenAPI 文档，类型安全，比 Flask 减少约 30% 接口样板代码 |
| Controller/Service/DAO/Utils 四层 | 隔离 AI 推理逻辑与接口逻辑，单层改动不影响其他层 |
| 模型缓存（方案 × 规模） | 首次加载后命中缓存，重复推理 < 50ms 额外开销 |
| SQLite 归档 | 轻量无需额外服务，演示复现和结果追溯开箱即用 |
| 方案矩阵（5 × 3 = 15 组） | 每个改进方向有 S/M/L 三档对照，排除规模因素干扰 |

## 性能与稳定性证据

| 指标 | 目标 | 实测结果 | 测试条件 |
|------|------|----------|----------|
| 单图推理耗时 | ≤ 1s | < 1s（GPU）/ ~3s（CPU） | 方案 A·L，NVIDIA GPU，640×640 输入 |
| mAP@0.5（方案 A·L） | > 基线 | ~85% | 自有茶叶测试集，相对方案 O 提升约 5% |
| mAP@0.5（方案 O·L，基线） | 参考基准 | ~80% | YOLOv11 原版，同测试集 |
| API 稳定性 | 无 5xx | 100% 2xx | 连续调用 100 次 POST /api/detection/detect |
| 可用方案切换 | 全部可选 | 15 组均可通过前端切换 | 冷启动首次加载 ~2s，后续缓存命中 |

## 面试可提问点及参考答案思路

**Q1：你为什么选择 TBD / BiFormer / Fusion / SPD-CARAFE 这四个改进方向？**
> 茶叶检测的核心难点是小目标与纹理相似。TBD 三分支注意力增强局部特征辨别力；BiFormer 通过双向路由降低全局注意力计算代价；Fusion 强化多尺度融合应对大小茶芽共存；SPD-CARAFE 在轻量化基础上保留上采样精度。四个方向各攻一个子问题。

**Q2：如何证明不是"过拟合式提升"？**
> 使用与训练集完全隔离的测试集评估，并在 S/M/L 三个规模上复现提升趋势；若仅在 L 模型上提升而 S 上没有，说明依赖参数量而非改进本身。

**Q3：模型效果与推理速度如何平衡？**
> 提供 S/M/L 三档：演示场景用 L 保证精度，边缘部署可切 S（mAP 下降约 3%，速度提升 ~2x）。`config.py` 中统一注册，前端一键切换。

**Q4：为什么选 FastAPI 而非 Flask / Django？**
> FastAPI 原生支持异步、自动生成 OpenAPI 文档、Pydantic 校验兼类型提示三合一；对纯推理服务场景比 Django 轻，比 Flask 多出自文档能力。

**Q5：如何将当前项目迁移到在线推理服务？**
> Service 层加模型版本管理（MLflow / 自研）；Controller 层加 API Key 鉴权；用 Gunicorn + uvicorn workers 替换单进程；模型文件迁到对象存储（S3 / OSS）热加载。


