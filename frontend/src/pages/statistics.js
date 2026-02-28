/**
 * TeaVision V12 | 智能统计页
 *
 * 多维度检测数据统计看板 + Chart.js 数据可视化
 * 移植自 V10 的完整统计功能。
 */

const StatisticsPage = {
    /** @type {Chart|null} 趋势折线图实例 */
    trendChart: null,
    /** @type {Chart|null} 品种甜甜圈图实例 */
    varietyChart: null,

    render(container) {
        // 销毁旧 Chart 实例（SPA 页面切换时避免内存泄漏）
        if (this.trendChart) { this.trendChart.destroy(); this.trendChart = null; }
        if (this.varietyChart) { this.varietyChart.destroy(); this.varietyChart = null; }

        container.innerHTML = `
            <div class="tv-fade-in">
                <!-- 页面标题 -->
                <div class="tv-page-header">
                    <h1 class="tv-page-title">智能统计</h1>
                    <p class="tv-page-subtitle">深度剖析检测准确率及群体分布情况，实时监控系统性能指标，提供详尽的数据可视化分析报告。</p>
                </div>

                <!-- 概览统计卡片 (4 列) -->
                <div class="tv-stats-grid" id="statsCards">
                    <div class="tv-stat-card">
                        <svg xmlns="http://www.w3.org/2000/svg" width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="var(--color-accent)" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="margin-bottom:8px"><path d="M3 7V5a2 2 0 0 1 2-2h2"/><path d="M17 3h2a2 2 0 0 1 2 2v2"/><path d="M21 17v2a2 2 0 0 1-2 2h-2"/><path d="M7 21H5a2 2 0 0 1-2-2v-2"/><line x1="7" x2="17" y1="12" y2="12"/></svg>
                        <div class="tv-stat-card__value" id="sTotal">--</div>
                        <div class="tv-stat-card__label">累计检测次数</div>
                    </div>
                    <div class="tv-stat-card">
                        <svg xmlns="http://www.w3.org/2000/svg" width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="var(--color-accent)" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="margin-bottom:8px"><rect width="18" height="18" x="3" y="4" rx="2" ry="2"/><line x1="16" x2="16" y1="2" y2="6"/><line x1="8" x2="8" y1="2" y2="6"/><line x1="3" x2="21" y1="10" y2="10"/></svg>
                        <div class="tv-stat-card__value" id="sToday">--</div>
                        <div class="tv-stat-card__label">今日检测量</div>
                    </div>
                    <div class="tv-stat-card">
                        <svg xmlns="http://www.w3.org/2000/svg" width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="var(--color-accent)" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="margin-bottom:8px"><circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2"/></svg>
                        <div class="tv-stat-card__value" id="sConf">--</div>
                        <div class="tv-stat-card__label">平均置信度</div>
                    </div>
                    <div class="tv-stat-card">
                        <svg xmlns="http://www.w3.org/2000/svg" width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="var(--color-accent)" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="margin-bottom:8px"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/></svg>
                        <div class="tv-stat-card__value" id="sTime">--</div>
                        <div class="tv-stat-card__label">平均推理时间</div>
                    </div>
                </div>

                <!-- 分割线 -->
                <hr style="border:none;border-top:1px solid var(--color-border);margin:2.5rem 0">

                <!-- 图表标题 -->
                <h2 style="font-family:var(--font-serif);font-size:2rem;font-weight:700;color:var(--color-primary);margin-bottom:1.5rem;letter-spacing:-0.02em">
                    数据可视化
                </h2>

                <!-- 图表区域 (2 列) -->
                <div class="tv-bento tv-bento--2">
                    <div class="tv-card">
                        <h3 class="tv-card__title">检测趋势 (近7天)</h3>
                        <div style="margin-top:16px;position:relative;height:260px">
                            <canvas id="trendChart"></canvas>
                        </div>
                    </div>
                    <div class="tv-card">
                        <h3 class="tv-card__title">档案收录分布</h3>
                        <div style="margin-top:16px;position:relative;height:260px">
                            <canvas id="varietyChart"></canvas>
                        </div>
                    </div>
                </div>

                <!-- 附加统计 (3 列) -->
                <div class="tv-bento tv-bento--3" style="margin-top:1.5rem">
                    <div class="tv-stat-card">
                        <svg xmlns="http://www.w3.org/2000/svg" width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="var(--color-accent)" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="margin-bottom:8px"><path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"/></svg>
                        <div class="tv-stat-card__value" id="sTotalObj" style="font-size:2rem">--</div>
                        <div class="tv-stat-card__label">总检测目标</div>
                        <p style="font-size:0.75rem;color:var(--color-text-light);margin-top:4px">累计检测到的茶叶目标数量</p>
                    </div>
                    <div class="tv-stat-card">
                        <svg xmlns="http://www.w3.org/2000/svg" width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="var(--color-accent)" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="margin-bottom:8px"><path d="M11 20A7 7 0 0 1 9.8 6.9C15.5 4.9 17 3.5 17 3.5s1 2.5-1 6c2-1 4-3.5 5-6 1 3.5-1 7.5-3.5 10 2 0 4-1 5.5-3-2 6.5-9.5 10.5-13 10Z"/></svg>
                        <div class="tv-stat-card__value" id="sVarieties" style="font-size:2rem">--</div>
                        <div class="tv-stat-card__label">品种数量</div>
                        <p style="font-size:0.75rem;color:var(--color-text-light);margin-top:4px">已识别的茶叶品种类型</p>
                    </div>
                    <div class="tv-stat-card">
                        <svg xmlns="http://www.w3.org/2000/svg" width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="var(--color-success)" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="margin-bottom:8px"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg>
                        <div class="tv-stat-card__value" id="sStatus" style="font-size:2rem;color:var(--color-success)">检测中...</div>
                        <div class="tv-stat-card__label">系统状态</div>
                        <p style="font-size:0.75rem;color:var(--color-text-light);margin-top:4px">API 服务运行状态</p>
                    </div>
                </div>

                <!-- 性能指标面板 -->
                <div class="tv-card" style="margin-top:1.5rem;background:linear-gradient(135deg, rgba(184,149,106,0.06) 0%, rgba(74,64,58,0.06) 100%)">
                    <h3 class="tv-card__title" style="margin-bottom:1rem">性能指标</h3>
                    <div class="tv-bento tv-bento--4">
                        <div>
                            <div style="font-size:0.75rem;color:var(--color-text-light);margin-bottom:4px">模型版本</div>
                            <div style="font-size:1.15rem;font-weight:700;color:var(--color-primary)" id="sModelName">YOLO-TBD</div>
                        </div>
                        <div>
                            <div style="font-size:0.75rem;color:var(--color-text-light);margin-bottom:4px">输入尺寸</div>
                            <div style="font-size:1.15rem;font-weight:700;color:var(--color-primary)">640×640</div>
                        </div>
                        <div>
                            <div style="font-size:0.75rem;color:var(--color-text-light);margin-bottom:4px">类别数量</div>
                            <div style="font-size:1.15rem;font-weight:700;color:var(--color-primary)" id="sNumClasses">--</div>
                        </div>
                        <div>
                            <div style="font-size:0.75rem;color:var(--color-text-light);margin-bottom:4px">推理设备</div>
                            <div style="font-size:1.15rem;font-weight:700;color:var(--color-primary)">GPU/CPU</div>
                        </div>
                    </div>
                </div>
            </div>`;

        // 加载各项数据
        this._loadStats();
        this._initCharts();
    },

    /**
     * 加载概览统计 + 模型信息
     */
    async _loadStats() {
        try {
            const d = await api.get('/stats');
            if (d.success) {
                utils.setText(document.getElementById('sTotal'), utils.formatNumber(d.total_detections));
                utils.setText(document.getElementById('sToday'), utils.formatNumber(d.today_detections));
                utils.setText(document.getElementById('sConf'), d.avg_confidence + '%');
                utils.setText(document.getElementById('sTime'), d.avg_inference_time + 'ms');
                utils.setText(document.getElementById('sTotalObj'), utils.formatNumber(d.total_objects || 0));
                utils.setText(document.getElementById('sVarieties'), d.varieties_count || '--');
            }
            // 系统在线
            const statusEl = document.getElementById('sStatus');
            if (statusEl) {
                statusEl.textContent = '在线';
                statusEl.style.color = 'var(--color-success)';
            }
        } catch (e) {
            console.warn('[统计] 概览加载失败:', e);
            const statusEl = document.getElementById('sStatus');
            if (statusEl) {
                statusEl.textContent = '离线';
                statusEl.style.color = 'var(--color-error)';
            }
        }

        // 加载模型信息
        try {
            const m = await api.get('/model/info');
            if (m) {
                let modelName = m.model_name || 'YOLO-TBD';
                // 提取文件名
                if (modelName.includes('\\') || modelName.includes('/')) {
                    modelName = modelName.split(/[\\/]/).pop();
                }
                modelName = modelName.replace('.pt', '');
                utils.setText(document.getElementById('sModelName'), modelName);
                utils.setText(document.getElementById('sNumClasses'), m.num_classes || '--');
            }
        } catch (e) { console.warn('[统计] 模型信息加载失败'); }
    },

    /**
     * 初始化 Chart.js 图表（趋势折线图 + 品种甜甜圈图）
     */
    async _initCharts() {
        // 获取当前主题颜色
        const style = getComputedStyle(document.documentElement);
        const accentColor = style.getPropertyValue('--color-accent').trim() || '#B8956A';
        const borderColor = style.getPropertyValue('--color-border').trim() || '#E8E2DA';
        const textColor = style.getPropertyValue('--color-text').trim() || '#2C2418';
        const textMuted = style.getPropertyValue('--color-text-light').trim() || '#8A7E72';

        try {
            // ========== 1. 检测趋势折线图 ==========
            const trendResp = await api.get('/stats/trend', { days: 7 });
            let trendLabels = [], trendData = [];

            if (trendResp && trendResp.success && trendResp.data) {
                trendLabels = trendResp.data.dates || [];
                trendData = trendResp.data.detections || [];
            } else if (trendResp && trendResp.success && trendResp.trend) {
                // 兼容旧版 API 格式
                trendLabels = trendResp.trend.map(t => t.date);
                trendData = trendResp.trend.map(t => t.count);
            }

            const trendCtx = document.getElementById('trendChart');
            if (trendCtx) {
                this.trendChart = new Chart(trendCtx, {
                    type: 'line',
                    data: {
                        labels: trendLabels,
                        datasets: [{
                            label: '检测次数',
                            data: trendData,
                            borderColor: accentColor,
                            backgroundColor: 'rgba(184, 149, 106, 0.1)',
                            fill: true,
                            tension: 0.4,
                            pointBackgroundColor: accentColor,
                            pointBorderColor: style.getPropertyValue('--color-surface').trim() || '#fff',
                            pointBorderWidth: 2,
                            pointRadius: 5
                        }]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        plugins: {
                            legend: { display: false }
                        },
                        scales: {
                            y: {
                                beginAtZero: true,
                                grid: { color: borderColor },
                                ticks: {
                                    stepSize: 1,
                                    color: textMuted,
                                    font: { family: "'Plus Jakarta Sans', sans-serif" }
                                }
                            },
                            x: {
                                grid: { display: false },
                                ticks: {
                                    color: textMuted,
                                    font: { family: "'Plus Jakarta Sans', sans-serif" }
                                }
                            }
                        }
                    }
                });
            }

            // ========== 2. 档案收录分布甜甜圈图 ==========
            // 从 knowledge.json 获取茶叶分类数据
            let distLabels = [], distData = [], bgColors = [];

            try {
                const knowledgeResp = await fetch('static/knowledge.json');
                if (knowledgeResp.ok) {
                    const teas = await knowledgeResp.json();
                    const categoryCounts = {};
                    teas.forEach(tea => {
                        if (tea.tags && tea.tags.length > 0) {
                            const tag = tea.tags[0]; // 使用主标签
                            categoryCounts[tag] = (categoryCounts[tag] || 0) + 1;
                        }
                    });
                    distLabels = Object.keys(categoryCounts);
                    distData = Object.values(categoryCounts);

                    // 茶叶大地色调色盘
                    const palette = [
                        '#4D5D53', // 绿茶
                        '#A8655F', // 红茶
                        '#D4C5B5', // 白茶
                        '#B8956A', // 黄茶
                        '#5D4037', // 黑茶
                        '#6F806D', // 青茶
                        '#8B6F47', // 备用
                        '#7B9E87'  // 备用
                    ];
                    bgColors = palette.slice(0, distLabels.length);
                }
            } catch (e) { console.warn('[统计] knowledge.json 加载失败'); }

            const varietyCtx = document.getElementById('varietyChart');
            if (varietyCtx) {
                this.varietyChart = new Chart(varietyCtx, {
                    type: 'doughnut',
                    data: {
                        labels: distLabels,
                        datasets: [{
                            data: distData,
                            backgroundColor: bgColors,
                            borderWidth: 0,
                            hoverOffset: 4
                        }]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        cutout: '65%',
                        plugins: {
                            legend: {
                                position: 'right',
                                labels: {
                                    padding: 16,
                                    usePointStyle: true,
                                    color: textColor,
                                    font: {
                                        family: "'Plus Jakarta Sans', sans-serif",
                                        size: 12
                                    }
                                }
                            },
                            tooltip: {
                                callbacks: {
                                    label: function(context) {
                                        let label = context.label || '';
                                        if (label) label += ': ';
                                        const value = context.raw;
                                        const total = context.chart._metasets[context.datasetIndex].total;
                                        const pct = Math.round((value / total) * 100) + '%';
                                        return label + value + ' 种 (' + pct + ')';
                                    }
                                }
                            }
                        }
                    }
                });
            }

        } catch (e) {
            console.error('[统计] 图表初始化失败:', e);
        }
    },
};
