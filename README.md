# 🍃 TeaVision V13 - 茶叶形态智能检测系统

基于 YOLO-TBD 架构的茶叶形态智能检测系统，专注于不确定性环境下的精准目标检测。

## Showcase

### 一句话价值

把 YOLO 改进实验工程化为可演示、可复盘、可对比的视觉检测系统。

### 1分钟演示视频

- [demo.mp4](docs/showcase/pro_茶叶检测/demo.mp4)

### 3张关键截图

1. [shot-01.png（单图检测）](docs/showcase/pro_茶叶检测/shot-01.png)
2. [shot-02.png（多模型对比）](docs/showcase/pro_茶叶检测/shot-02.png)
3. [shot-03.png（训练曲线/混淆矩阵）](docs/showcase/pro_茶叶检测/shot-03.png)

### 一键运行命令

```powershell
cd pro_茶叶检测
pwsh -NoLogo -NoProfile -ExecutionPolicy Bypass -Command '.\start_all_services.bat'
```

### 核心技术决策

1. 后端采用 Controller/Service/DAO/Utils 四层架构，隔离模型逻辑与接口逻辑。
2. 采用 A/B/C/D + S/M/L 方案矩阵，提升实验可解释性。
3. 使用 SQLite 归档检测记录，保证演示复现能力。

### 性能/稳定性证据

- 证据页：[evidence.md](docs/showcase/pro_茶叶检测/evidence.md)
- 建议展示指标：单图推理耗时、mAP/F1 提升幅度、连续请求错误率。

### 面试可提问点

1. 模型改进如何保证公平对比？
2. 精度提升与推理时延如何权衡？
3. 若上线推理服务，如何做版本管理与回滚？

## 📋 项目简介

TeaVision 是一个完整的目标检测解决方案，集成了：
- 🎯 **高精度检测模型** - 基于 YOLOv11 改进的多尺度检测网络
- 🖥️ **Web 可视化界面** - 直观的前端界面用于模型演示
- 🔌 **RESTful API** - 基于 FastAPI 的高性能推理接口
- 📊 **训练结果管理** - 多方案训练结果对比与分析

## 🏗️ 系统架构（V13 · 规范分层）

> V13 相比 V12 的核心改进：**将原来混乱平级的 `interface/` + `database/` 统一归入 `backend/` 内部，按规范分为 4 层**

```
ultralytics-8.4.5_V13/
├── frontend/                    # ① 前端模块（独立）
│   ├── index.html              #    主页面
│   ├── src/                    #    源代码 (JS/CSS)
│   │   ├── pages/              #    页面组件
│   │   ├── components/         #    公共组件
│   │   ├── hooks/              #    工具钩子
│   │   └── styles/             #    样式系统
│   ├── assets/                 #    静态资源
│   └── static/                 #    茶叶数据 & 图片
│
└── backend/                     # ② 后端模块（内部 4 层）
    ├── app.py                  #    FastAPI 应用入口
    ├── config.py               #    项目配置 & 模型注册
    ├── schemas.py              #    Pydantic 数据模型
    ├── swagger_ui.py           #    自定义 API 文档界面
    │
    ├── controller/             #    2.1 接口层 — 收请求、返响应
    │   ├── detection.py        #         检测接口
    │   ├── models.py           #         模型管理
    │   ├── stats.py            #         统计数据
    │   ├── training.py         #         训练结果
    │   └── augmentation.py     #         数据增强
    │
    ├── service/                #    2.2 业务层 — YOLO 推理核心 ⭐
    │   ├── model_service.py    #         模型加载/缓存/推理
    │   └── detection_service.py#         检测结果持久化
    │
    ├── dao/                    #    2.3 数据层 — 数据库增删改查
    │   └── database.py         #         SQLite 数据访问
    │
    ├── utils/                  #    2.4 工具层 — 通用函数
    │   ├── image_utils.py      #         图片预处理/编码
    │   └── file_utils.py       #         文件管理/路径处理
    │
    ├── models/                 #    训练好的模型文件 (.pt)
    ├── training_results/       #    训练结果数据
    └── ultralytics/            #    YOLO 核心库
```

### 为什么这样分层？

| 层级 | 职责 | 原则 |
|------|------|------|
| **接口层** (Controller) | 接收前端请求，返回响应 | 不跑模型、不存数据 — 就是个传声筒 |
| **业务层** (Service) | YOLO 模型推理、茶叶检测 | 项目核心！所有 AI 检测逻辑只放这里 |
| **数据层** (DAO) | 数据库读写操作 | 不碰模型、不写业务 |
| **工具层** (Utils) | 图片处理、文件管理 | 通用工具，到处复用 |

## ✨ 主要特性

### 多方案模型支持
- **Scheme O**: 原始基准模型
- **Scheme A**: TBD 三分支注意力 + 自校准分组卷积
- **Scheme B**: BiFormer 双向路由注意力机制
- **Scheme C**: Fusion 多尺度特征融合
- **Scheme D**: SPD-CARAFE 轻量化设计

支持三种模型规模：**S** (小型) / **M** (中型) / **L** (大型)

### 功能模块
- 📸 **实时检测** - 上传图片即时返回检测结果
- 🎬 **视频检测** - 上传视频进行逐帧检测
- 🔄 **数据增强** - 多种增强策略可视化
- 📈 **训练分析** - 训练曲线、性能对比
- 🔀 **多模型对比** - 同图多模型并行检测
- 🎨 **主题切换** - 多套精美 UI 主题

## 🚀 快速开始

### 环境要求
```
Python >= 3.8
```

### 安装依赖
```bash
pip install -r requirements.txt
```

### 启动服务
```bash
# Windows 一键启动
start_all_services.bat

# 或手动启动后端
python -m uvicorn backend.app:app --host 0.0.0.0 --port 8000

# 手动启动前端（另一个终端）
cd frontend && python -m http.server 3000
```

### 访问界面
打开浏览器访问：
- 前端界面: http://localhost:3000
- API 文档: http://localhost:8000/docs

## 📊 V12 → V13 架构对比

### V12（有问题的）
```
├── backend/     ← 只放模型文件和库
├── frontend/    ← 前端
├── interface/   ← ❌ 和后端平级（但实际是后端的一部分！）
└── database/    ← ❌ 和后端平级（但实际是后端的一部分！）
```
**问题**：把「儿子」和「爸爸」放在同一辈，接口层和数据层不应该和后端平级。

### V13（规范的）
```
├── frontend/    ← 前端（独立模块）
└── backend/     ← 后端（内部 4 层）
    ├── controller/  ← 接口层
    ├── service/     ← 业务层（核心！）
    ├── dao/         ← 数据层
    └── utils/       ← 工具层
```
**改进**：后端内部按职责分层，每层只干自己的事。

## 🛠️ 技术栈

- **后端**: FastAPI, Ultralytics YOLOv11, PyTorch
- **前端**: Vanilla JavaScript, Chart.js, Lucide Icons
- **数据处理**: NumPy, Pandas, OpenCV
- **API 文档**: Swagger UI (OpenAPI 3.0)
- **数据库**: SQLite

## 📄 许可证

本项目仅供学习和研究使用。

## 👨‍💻 作者

项目展示 - 2026

## 🙏 致谢

- [Ultralytics YOLOv11](https://github.com/ultralytics/ultralytics)
- FastAPI Framework
- PyTorch Team

---

⭐ 如果这个项目对你有帮助，欢迎 Star！

