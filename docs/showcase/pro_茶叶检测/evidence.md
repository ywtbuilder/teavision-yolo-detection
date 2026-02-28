# pro_茶叶检测 展示证据页

## 一句话价值

以 YOLOv11 为基线完成多方案改进与工程化重构，提供可视化检测演示与训练结果对比闭环。

## 1 分钟演示视频

- 文件：`docs/showcase/pro_茶叶检测/demo.mp4`
- 建议镜头：上传图片 -> 多模型对比 -> 结果详情 -> 训练曲线页

## 3 张关键截图

1. `shot-01.png`：检测主界面（上传 + 推理结果）
2. `shot-02.png`：多模型同图对比
3. `shot-03.png`：训练指标曲线/混淆矩阵

## 一键运行命令

```powershell
cd pro_茶叶检测
pwsh -NoLogo -NoProfile -ExecutionPolicy Bypass -Command '.\start_all_services.bat'
```

## 核心技术决策

1. 四层后端架构：Controller/Service/DAO/Utils 分层，降低耦合。
2. 模型方案矩阵：A/B/C/D 与 S/M/L 组合，保证选型可解释。
3. SQLite 归档：推理记录与对比结果可追溯。

## 性能/稳定性证据

| 指标 | 目标 | 当前结果 | 说明 |
|---|---:|---:|---|
| 单图推理耗时 | <= 1s | 待填充 | 指定模型与硬件配置 |
| mAP@0.5 | 持续提升 | 待填充 | 相对 O 方案提升 |
| API 稳定性 | 无 5xx | 待填充 | 连续调用 100 次 |

## 面试可提问点

1. 你为什么选择这些改进模块（TBD/BiFormer/Fusion/SPD-CARAFE）？
2. 如何证明不是“过拟合式提升”？
3. 模型效果与推理速度如何平衡？
4. 为什么选择 FastAPI 而不是 Flask/Django？
5. 如何将当前项目迁移到在线推理服务？

