/**
 * TeaVision V12 | 训练成果页
 *
 * 显示模型训练过程的关键指标：
 * - 训练运行列表选择
 * - Loss 曲线图表
 * - mAP 指标图表
 */

const TrainingPage = {
    /**
     * @type {Object<string, Chart>}
     */
    charts: {},

    render(container) {
        container.innerHTML = `
            <div class="tv-fade-in">
                <div class="tv-page-header">
                    <h1 class="tv-page-title">训练成果</h1>
                    <p class="tv-page-subtitle">可视化模型训练过程中的 Loss 曲线与 mAP 指标</p>
                </div>

                <!-- 训练运行选择 -->
                <div class="tv-card" style="margin-bottom:var(--space-xl)">
                    <div class="tv-param-row">
                        <label style="font-size:0.85rem;font-weight:600">选择训练运行</label>
                    </div>
                    <select class="tv-select" id="runSelect" style="width:100%;margin-top:var(--space-sm)">
                        <option value="">加载中...</option>
                    </select>
                </div>

                <!-- 图表区域 -->
                <div id="chartsArea">
                    <div class="tv-loading">
                        <div class="tv-spinner"></div>
                        <span>选择训练运行以加载数据</span>
                    </div>
                </div>
            </div>
        `;

        this._loadRuns();
    },

    /**
     * 加载训练运行列表
     */
    async _loadRuns() {
        try {
            const data = await api.get('/training/runs');
            if (data.success && data.runs.length > 0) {
                const select = document.getElementById('runSelect');
                select.innerHTML = '<option value="">请选择训练运行</option>' +
                    data.runs.map(r => `<option value="${r.id}">${r.name}</option>`).join('');

                select.addEventListener('change', () => {
                    if (select.value) this._loadMetrics(select.value);
                });

                // 自动选择第一个
                if (data.runs.length > 0) {
                    select.value = data.runs[0].id;
                    this._loadMetrics(data.runs[0].id);
                }
            }
        } catch (err) {
            console.warn('[训练] 运行列表加载失败:', err);
        }
    },

    /**
     * 加载并展示训练指标
     */
    async _loadMetrics(runId) {
        // 清理旧图表
        Object.values(this.charts).forEach(c => c.destroy());
        this.charts = {};

        const area = document.getElementById('chartsArea');
        area.innerHTML = '<div class="tv-loading"><div class="tv-spinner"></div><span>加载训练数据...</span></div>';

        try {
            const data = await api.get(`/training/run/${runId}/metrics`);
            if (!data.success) throw new Error('数据加载失败');

            const columns = data.columns;
            const metrics = data.data;

            // 查找关键列
            const epochs = metrics['epoch'] || metrics['Epoch'] || [];

            // 识别列名 (兼容 train/box_loss, val/box_loss 等)
            const getCol = (name) => {
                const key = columns.find(c => c.includes(name));
                return key ? metrics[key] : [];
            };

            const trainBox = getCol('train/box_loss') || getCol('train/box');
            const valBox = getCol('val/box_loss') || getCol('val/box');
            const map50 = getCol('mAP50');
            const map5095 = getCol('mAP50-95');

            area.innerHTML = `
                <div class="tv-bento tv-bento--2">
                    <!-- Loss 曲线 -->
                    <div class="tv-chart-container">
                        <h3 class="tv-card__title" style="margin-bottom:16px">Loss 曲线</h3>
                        <div style="height:300px;position:relative">
                            <canvas id="lossChart"></canvas>
                        </div>
                    </div>
                    <!-- mAP 指标 -->
                    <div class="tv-chart-container">
                        <h3 class="tv-card__title" style="margin-bottom:16px">性能指标</h3>
                        <div style="height:300px;position:relative">
                            <canvas id="mapChart"></canvas>
                        </div>
                    </div>
                </div>

                <!-- 训练结果图片 -->
                <div class="tv-section" style="margin-top:var(--space-xl)">
                    <h3 class="tv-section__title">训练可视化</h3>
                    <div class="tv-training-gallery" id="trainingImages">
                        ${[
                    { file: 'results.png', label: '训练结果总览', wide: true },
                    { file: 'confusion_matrix.png', label: '混淆矩阵' },
                    { file: 'confusion_matrix_normalized.png', label: '归一化混淆矩阵' },
                    { file: 'BoxF1_curve.png', label: 'F1 曲线' },
                    { file: 'BoxP_curve.png', label: 'Precision 曲线' },
                    { file: 'BoxR_curve.png', label: 'Recall 曲线' },
                    { file: 'BoxPR_curve.png', label: 'P-R 曲线' },
                ].map(img => `
                            <div class="tv-training-gallery__item ${img.wide ? 'tv-training-gallery__item--wide' : ''}">
                                <img src="${CONFIG.API_BASE_URL}/training/run/${runId}/image/${img.file}" 
                                     class="tv-training-gallery__img" 
                                     alt="${img.label}"
                                     loading="lazy"
                                     onclick="TrainingPage._openLightbox(this.src)"
                                     onerror="this.closest('.tv-training-gallery__item').style.display='none'">
                                <div class="tv-training-gallery__caption">${img.label}</div>
                            </div>
                        `).join('')}
                    </div>
                </div>

                <!-- 最终指标 -->
                <div class="tv-section" style="margin-top:var(--space-xl)">
                    <h3 class="tv-section__title">最终指标</h3>
                    <div class="tv-bento tv-bento--4" id="finalMetrics"></div>
                </div>
            `;

            // 绘制 Loss 图表
            this._initChart('lossChart', 'Loss 趋势', epochs, [
                { label: 'Train Box Loss', data: trainBox, borderColor: '#B8956A', backgroundColor: 'transparent' },
                { label: 'Val Box Loss', data: valBox, borderColor: '#2D4033', backgroundColor: 'transparent' }
            ]);

            // 绘制 mAP 图表
            this._initChart('mapChart', 'mAP 趋势', epochs, [
                { label: 'mAP@50', data: map50, borderColor: '#B8956A', backgroundColor: 'transparent' },
                { label: 'mAP@50-95', data: map5095, borderColor: '#A8655F', backgroundColor: 'transparent' }
            ]);

            // 渲染最终指标
            const lastIdx = epochs.length - 1;
            const finalHtml = [
                { label: 'Best mAP@50', value: map50 ? Math.max(...map50) : 0, format: v => (v * 100).toFixed(1) + '%' },
                { label: 'Best mAP@50-95', value: map5095 ? Math.max(...map5095) : 0, format: v => (v * 100).toFixed(1) + '%' },
                { label: 'Final Train Loss', value: trainBox && trainBox.length ? trainBox[lastIdx] : 0, format: v => (v || 0).toFixed(4) },
                { label: 'Final Val Loss', value: valBox && valBox.length ? valBox[lastIdx] : 0, format: v => (v || 0).toFixed(4) },
            ].map(item => `
                <div class="tv-stat-card">
                    <div class="tv-stat-card__value">${item.format(item.value)}</div>
                    <div class="tv-stat-card__label">${item.label}</div>
                </div>
            `).join('');

            document.getElementById('finalMetrics').innerHTML = finalHtml;

        } catch (err) {
            console.error(err);
            area.innerHTML = `<div class="tv-card" style="text-align:center;color:var(--color-error)">加载失败: ${err.message}</div>`;
        }
    },

    /**
     * 初始化 Chart.js 图表
     */
    _initChart(canvasId, title, labels, datasets) {
        if (this.charts[canvasId]) {
            this.charts[canvasId].destroy();
        }

        const ctx = document.getElementById(canvasId).getContext('2d');
        const style = getComputedStyle(document.documentElement);
        // Fallback or specific color
        const textColor = style.getPropertyValue('--color-text-light').trim() || '#8A7E72';

        this.charts[canvasId] = new Chart(ctx, {
            type: 'line',
            data: {
                labels: labels,
                datasets: datasets.map(ds => ({
                    ...ds,
                    borderWidth: 2,
                    tension: 0.3,
                    pointRadius: 0,
                    pointHoverRadius: 4,
                    fill: false
                }))
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'bottom',
                        labels: { usePointStyle: true, font: { family: "var(--font-sans)" } }
                    },
                    tooltip: {
                        mode: 'index',
                        intersect: false,
                        backgroundColor: 'rgba(255, 255, 255, 0.9)',
                        titleColor: '#2C2418',
                        bodyColor: '#5D5447',
                        borderColor: '#E8E4DE',
                        borderWidth: 1,
                        padding: 10
                    }
                },
                interaction: {
                    mode: 'nearest',
                    axis: 'x',
                    intersect: false
                },
                scales: {
                    x: {
                        grid: { display: false },
                        ticks: { color: textColor, maxTicksLimit: 10 }
                    },
                    y: {
                        grid: { color: 'rgba(0,0,0,0.04)' },
                        ticks: { color: textColor }
                    }
                }
            }
        });
    },

    /**
     * 全屏查看训练图片
     */
    _openLightbox(src) {
        const overlay = document.createElement('div');
        overlay.className = 'tv-lightbox';
        overlay.innerHTML = `<img src="${src}" alt="训练可视化">`;
        overlay.addEventListener('click', () => overlay.remove());
        document.addEventListener('keydown', function handler(e) {
            if (e.key === 'Escape') { overlay.remove(); document.removeEventListener('keydown', handler); }
        });
        document.body.appendChild(overlay);
    },
};
