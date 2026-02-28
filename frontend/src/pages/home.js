/**
 * TeaVision V12 | 首页模块
 *
 * 包含：
 * - Hero 区域（标题、描述、状态徽章）
 * - 系统统计数据（从 API 实时获取）
 * - 功能导航卡片（Bento Grid 布局）
 */

const HomePage = {
    /**
     * 渲染首页
     * @param {HTMLElement} container - 页面容器
     */
    render(container) {
        container.innerHTML = `
            <div class="tv-fade-in">
                <!-- Hero 区 -->
                <section class="tv-hero">
                    <h1 class="tv-hero__title">
                        茶芽智识
                        <span class="tv-hero__title-accent">TeaVision</span>
                    </h1>
                    <p class="tv-hero__desc">
                        基于 YOLO-TBD 架构的高精度茶叶嫩芽检测系统。
                        融合三分支注意力机制与自校准分组卷积，
                        精准应对复杂背景下的微小目标识别挑战。
                    </p>
                    <div class="tv-hero__badges">
                        <span class="tv-hero__badge">
                            ${utils.icon('cpu', 14)} YOLO-TBD 架构
                        </span>
                        <span class="tv-hero__badge">
                            ${utils.icon('zap', 14)} 实时推理
                        </span>
                        <span class="tv-hero__badge">
                            ${utils.icon('layers', 14)} 多模型对比
                        </span>
                    </div>
                </section>

                <!-- 统计数据 -->
                <section class="tv-section">
                    <h2 class="tv-section__title">系统概览</h2>
                    <div class="tv-bento--4 tv-bento" id="statsGrid">
                        <div class="tv-stat-card">
                            <div class="tv-stat-card__value" id="statTotal">--</div>
                            <div class="tv-stat-card__label">累计检测</div>
                        </div>
                        <div class="tv-stat-card">
                            <div class="tv-stat-card__value" id="statToday">--</div>
                            <div class="tv-stat-card__label">今日检测</div>
                        </div>
                        <div class="tv-stat-card">
                            <div class="tv-stat-card__value" id="statConf">--</div>
                            <div class="tv-stat-card__label">平均置信度</div>
                        </div>
                        <div class="tv-stat-card">
                            <div class="tv-stat-card__value" id="statTime">--</div>
                            <div class="tv-stat-card__label">平均耗时</div>
                        </div>
                    </div>
                </section>

                <!-- 功能导航 -->
                <section class="tv-section">
                    <h2 class="tv-section__title">核心功能</h2>
                    <div class="tv-bento tv-bento--3">
                        ${HomePage._featureCard('scan-search', '检测引擎', '上传茶叶图片，实时获取高精度检测结果。支持单图、批量和多模型对比三种模式。', '#/detection')}
                        ${HomePage._featureCard('chart-line', '训练成果', '可视化模型训练过程中的 Loss 曲线、mAP 指标和混淆矩阵等关键数据。', '#/training')}
                        ${HomePage._featureCard('columns-3', '模型对比', '横向对比不同训练方案的性能指标，辅助模型选型决策。', '#/comparison')}
                        ${HomePage._featureCard('palette', '数据增强', '实时预览 HSV 变换、翻转、旋转等增强策略对图像的影响效果。', '#/augmentation')}
                        ${HomePage._featureCard('book-open', '茶叶档案', '详尽的茶叶品种知识库，涵盖产地、工艺和品鉴要点。', '#/knowledge')}
                        ${HomePage._featureCard('bar-chart-3', '智能统计', '多维度检测数据统计看板，追踪检测趋势与品种分布。', '#/statistics')}
                    </div>
                </section>
            </div>
        `;

        // 加载统计数据
        HomePage._loadStats();
    },

    /**
     * 生成功能卡片 HTML
     */
    _featureCard(icon, title, desc, href) {
        return `
            <a href="${href}" class="tv-card tv-feature-card" style="text-decoration:none;color:inherit">
                <i data-lucide="${icon}" class="tv-feature-card__icon"></i>
                <h3 class="tv-card__title">${title}</h3>
                <p class="tv-card__desc">${desc}</p>
                <i data-lucide="arrow-up-right" class="tv-feature-card__arrow"></i>
            </a>
        `;
    },

    /**
     * 从 API 加载统计数据
     */
    async _loadStats() {
        try {
            const data = await api.get('/stats');
            if (data.success) {
                utils.setText(document.getElementById('statTotal'), utils.formatNumber(data.total_detections));
                utils.setText(document.getElementById('statToday'), utils.formatNumber(data.today_detections));
                utils.setText(document.getElementById('statConf'), data.avg_confidence + '%');
                utils.setText(document.getElementById('statTime'), data.avg_inference_time + 'ms');
            }
        } catch (err) {
            console.warn('[首页] 统计数据加载失败:', err);
        }
    },
};
