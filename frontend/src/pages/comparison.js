/**
 * TeaVision V12 | 模型对比页
 *
 * 移植 V10 完整功能：
 * - 侧边栏复选框选择模型
 * - 核心指标表格（Best mAP@50, mAP@50-95, Epochs, Box Loss）
 * - Chart.js 训练趋势对比折线图
 * - 移动端卡片式布局
 */

const ComparisonPage = {
    /** @type {Object<string, Chart>} 活动 Chart.js 实例 */
    charts: {},
    /** @type {Set<string>} 当前选中的训练运行 ID */
    selectedIds: new Set(),
    /** @type {Array} 所有训练运行列表 */
    allRuns: [],

    /** 图表配色盘 */
    palette: [
        '#B8956A', '#2D4033', '#A8655F', '#5D7C85',
        '#D4C5B5', '#4A403A', '#8B7355', '#4D5D53',
    ],

    render(container) {
        // 清理旧 Chart 实例
        Object.values(this.charts).forEach(c => c.destroy());
        this.charts = {};
        this.selectedIds.clear();

        container.innerHTML = `
            <div class="tv-fade-in">
                <div class="tv-page-header">
                    <h1 class="tv-page-title">模型对比</h1>
                    <p class="tv-page-subtitle">勾选训练方案，实时对比 mAP、Loss 等关键训练指标</p>
                </div>

                <div class="tv-comparison-layout">
                    <!-- 侧边栏 -->
                    <aside class="tv-comparison-sidebar" id="cmpSidebar">
                        <div class="tv-comparison-sidebar__header">
                            <span>选择对比模型</span>
                            <button class="tv-comparison-sidebar__clear" id="cmpClear">清空</button>
                        </div>

                        <!-- 快捷预设按钮 -->
                        <div class="tv-comparison-presets" id="cmpPresets">
                            <div class="tv-comparison-presets__title">按规模对比</div>
                            <div class="tv-comparison-presets__group">
                                <button class="tv-preset-btn tv-preset-btn--scale" data-preset="scale-s">所有 S 模型</button>
                                <button class="tv-preset-btn tv-preset-btn--scale" data-preset="scale-m">所有 M 模型</button>
                                <button class="tv-preset-btn tv-preset-btn--scale" data-preset="scale-l">所有 L 模型</button>
                            </div>
                            <div class="tv-comparison-presets__title" style="margin-top:6px">按方案对比</div>
                            <div class="tv-comparison-presets__group">
                                <button class="tv-preset-btn tv-preset-btn--scheme" data-preset="scheme-O">方案O · 基线</button>
                                <button class="tv-preset-btn tv-preset-btn--scheme" data-preset="scheme-A">方案A · TBD</button>
                                <button class="tv-preset-btn tv-preset-btn--scheme" data-preset="scheme-B">方案B · BiFormer</button>
                                <button class="tv-preset-btn tv-preset-btn--scheme" data-preset="scheme-C">方案C · Fusion</button>
                                <button class="tv-preset-btn tv-preset-btn--scheme" data-preset="scheme-D">方案D · SPD</button>
                            </div>
                        </div>

                        <ul class="tv-comparison-sidebar__list" id="cmpRunList">
                            <li style="padding:12px;text-align:center;color:var(--color-text-light)">加载中...</li>
                        </ul>
                    </aside>

                    <!-- 主面板 -->
                    <main class="tv-comparison-main" id="cmpMain">
                        <!-- 空状态 -->
                        <div id="cmpEmpty" class="tv-comparison-empty">
                            <svg xmlns="http://www.w3.org/2000/svg" width="56" height="56" viewBox="0 0 24 24" fill="none" stroke="var(--color-text-light)" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" style="opacity:0.4;margin-bottom:1rem">
                                <line x1="18" x2="18" y1="20" y2="10"/><line x1="12" x2="12" y1="20" y2="4"/><line x1="6" x2="6" y1="20" y2="14"/>
                            </svg>
                            <h3 style="margin-bottom:0.5rem;color:var(--color-primary)">请选择至少两个模型进行对比</h3>
                            <p style="color:var(--color-text-light);font-size:0.9rem">在左侧列表中勾选模型，或点击上方快捷按钮批量选择</p>
                        </div>

                        <!-- 加载状态 -->
                        <div id="cmpLoading" class="tv-comparison-empty" style="display:none">
                            <div class="tv-spinner" style="margin-bottom:1rem"></div>
                            <p style="color:var(--color-text-light)">正在分析数据...</p>
                        </div>

                        <!-- 对比结果 -->
                        <div id="cmpView" style="display:none">
                            <!-- 核心指标表格 -->
                            <h3 style="font-family:var(--font-serif);font-size:1.3rem;font-weight:700;color:var(--color-primary);margin-bottom:var(--space-md)">核心指标概览</h3>
                            <div class="tv-card" style="overflow-x:auto;margin-bottom:var(--space-xl);padding:0">
                                <table class="tv-comparison-table" id="cmpMetricsTable">
                                    <thead>
                                        <tr>
                                            <th>模型名称</th>
                                            <th>Best mAP@50</th>
                                            <th>Best mAP@50-95</th>
                                            <th>总轮次</th>
                                            <th>最终 Box Loss</th>
                                        </tr>
                                    </thead>
                                    <tbody id="cmpTableBody"></tbody>
                                </table>
                            </div>

                            <!-- 趋势对比图 -->
                            <h3 style="font-family:var(--font-serif);font-size:1.3rem;font-weight:700;color:var(--color-primary);margin-bottom:var(--space-md)">训练趋势对比</h3>
                            <div class="tv-bento tv-bento--2">
                                <div class="tv-card" style="height:380px;padding:var(--space-lg)">
                                    <canvas id="cmpChart50"></canvas>
                                </div>
                                <div class="tv-card" style="height:380px;padding:var(--space-lg)">
                                    <canvas id="cmpChart5095"></canvas>
                                </div>
                            </div>
                        </div>
                    </main>
                </div>
            </div>`;

        // 绑定清空按钮
        document.getElementById('cmpClear').addEventListener('click', () => {
            this.selectedIds.clear();
            document.querySelectorAll('.tv-comparison-run').forEach(el => el.classList.remove('selected'));
            this._updateView();
        });

        // 绑定快捷预设按钮
        document.querySelectorAll('.tv-preset-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                this._applyPreset(btn.dataset.preset);
            });
        });

        this._loadRuns();
    },

    /**
     * 应用快捷预设选择
     * @param {string} preset - 预设类型，如 "scale-s", "scheme-A"
     */
    _applyPreset(preset) {
        if (this.allRuns.length === 0) return;

        // 清空当前选择
        this.selectedIds.clear();
        document.querySelectorAll('.tv-comparison-run').forEach(el => el.classList.remove('selected'));

        const [type, value] = preset.split('-');
        let matchingRuns = [];

        if (type === 'scale') {
            // 按规模筛选：模型名中包含 yolo26s / yolo26m / yolo26l
            const scaleKey = value.toLowerCase(); // 's', 'm', 'l'
            matchingRuns = this.allRuns.filter(run => {
                const name = run.id.toLowerCase();
                // 匹配 yolo26s_ 或 yolo26m_ 或 yolo26l_
                return name.includes(`yolo26${scaleKey}_`) || name.includes(`yolo26${scaleKey}-`);
            });
        } else if (type === 'scheme') {
            // 按方案筛选：模型名中包含 scheme_O / scheme_A 等
            matchingRuns = this.allRuns.filter(run => {
                const name = run.id.toLowerCase();
                return name.includes(`scheme_${value.toLowerCase()}`);
            });
        }

        // 限制最多 8 个
        matchingRuns = matchingRuns.slice(0, 8);

        matchingRuns.forEach(run => {
            this.selectedIds.add(run.id);
            const el = document.querySelector(`.tv-comparison-run[data-id="${run.id}"]`);
            if (el) el.classList.add('selected');
        });

        this._updateView();
    },

    /**
     * 加载训练运行列表
     */
    async _loadRuns() {
        try {
            const data = await api.get('/training/runs');
            this.allRuns = data.runs || [];
            this._renderRunList();
        } catch (e) {
            document.getElementById('cmpRunList').innerHTML =
                '<li style="padding:12px;color:var(--color-error)">加载失败</li>';
        }
    },

    _renderRunList() {
        const list = document.getElementById('cmpRunList');
        list.innerHTML = '';

        if (this.allRuns.length === 0) {
            list.innerHTML = '<li style="padding:12px;color:var(--color-text-light)">暂无训练记录</li>';
            return;
        }

        this.allRuns.forEach(run => {
            const li = document.createElement('li');
            li.className = 'tv-comparison-run';
            li.dataset.id = run.id;
            li.innerHTML = `
                <span class="tv-comparison-run__checkbox">
                    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"/></svg>
                </span>
                <span class="tv-comparison-run__name">${run.name}</span>`;

            li.addEventListener('click', () => {
                if (this.selectedIds.has(run.id)) {
                    this.selectedIds.delete(run.id);
                    li.classList.remove('selected');
                } else {
                    if (this.selectedIds.size >= 8) {
                        alert('最多只能同时对比 8 个模型');
                        return;
                    }
                    this.selectedIds.add(run.id);
                    li.classList.add('selected');
                }
                this._updateView();
            });

            list.appendChild(li);
        });
    },

    /**
     * 更新主面板视图
     */
    async _updateView() {
        const empty = document.getElementById('cmpEmpty');
        const loading = document.getElementById('cmpLoading');
        const view = document.getElementById('cmpView');

        if (this.selectedIds.size === 0) {
            empty.style.display = 'flex';
            loading.style.display = 'none';
            view.style.display = 'none';
            return;
        }

        empty.style.display = 'none';
        loading.style.display = 'flex';
        view.style.display = 'none';

        try {
            const promises = Array.from(this.selectedIds).map(id =>
                api.get(`/training/run/${encodeURIComponent(id)}/metrics`)
                    .then(data => ({ id, ...data }))
            );
            const results = await Promise.all(promises);
            this._renderComparison(results);
            loading.style.display = 'none';
            view.style.display = 'block';
        } catch (e) {
            loading.innerHTML = `<p style="color:var(--color-error)">数据加载失败: ${e.message}</p>`;
        }
    },

    /**
     * 渲染对比结果（表格 + 图表）
     */
    _renderComparison(results) {
        const tbody = document.getElementById('cmpTableBody');
        tbody.innerHTML = '';

        const map50Datasets = [];
        const map5095Datasets = [];

        // 按 allRuns 顺序排序
        const orderedIds = this.allRuns.map(r => r.id);
        results.sort((a, b) => orderedIds.indexOf(a.id) - orderedIds.indexOf(b.id));

        results.forEach((res, i) => {
            if (!res.success) return;

            const run = this.allRuns.find(r => r.id === res.id);
            const color = this.palette[i % this.palette.length];
            const name = run ? run.name : res.id;
            const data = res.data;
            const cols = res.columns;

            const getCol = (key) => {
                const found = cols.find(c => c.includes(key));
                return found ? data[found] : [];
            };

            const epochs = getCol('epoch');
            const map50 = getCol('mAP50');
            const map5095 = getCol('mAP50-95');
            const valBoxLoss = getCol('val/box_loss').length ? getCol('val/box_loss') : getCol('val/box');

            const best50 = map50.length ? Math.max(...map50.filter(v => v != null)) : 0;
            const best5095 = map5095.length ? Math.max(...map5095.filter(v => v != null)) : 0;
            const lastLoss = valBoxLoss.length ? (valBoxLoss[valBoxLoss.length - 1] || 0) : 0;

            // 表格行
            const tr = document.createElement('tr');
            tr.innerHTML = `
                <td data-label="模型">
                    <span style="display:inline-block;width:10px;height:10px;border-radius:50%;background:${color};margin-right:8px;vertical-align:middle"></span>
                    <strong>${name}</strong>
                </td>
                <td data-label="mAP@50"><strong>${(best50 * 100).toFixed(1)}%</strong></td>
                <td data-label="mAP@50-95">${(best5095 * 100).toFixed(1)}%</td>
                <td data-label="总轮次">${epochs.length}</td>
                <td data-label="Box Loss">${typeof lastLoss === 'number' ? lastLoss.toFixed(4) : '--'}</td>`;
            tbody.appendChild(tr);

            // 图表数据集
            map50Datasets.push({
                label: name,
                data: map50.map((v, j) => ({ x: epochs[j], y: v })),
                borderColor: color,
                backgroundColor: 'transparent',
                borderWidth: 2,
                tension: 0.3,
                pointRadius: 0,
            });
            map5095Datasets.push({
                label: name,
                data: map5095.map((v, j) => ({ x: epochs[j], y: v })),
                borderColor: color,
                backgroundColor: 'transparent',
                borderWidth: 2,
                tension: 0.3,
                pointRadius: 0,
            });
        });

        this._initChart('cmpChart50', 'mAP@50 对比趋势', map50Datasets);
        this._initChart('cmpChart5095', 'mAP@50-95 对比趋势', map5095Datasets);
    },

    /**
     * 初始化/重建 Chart.js 图表
     */
    _initChart(canvasId, title, datasets) {
        const ctx = document.getElementById(canvasId);
        if (!ctx) return;

        if (this.charts[canvasId]) this.charts[canvasId].destroy();

        const style = getComputedStyle(document.documentElement);
        const textColor = style.getPropertyValue('--color-text-light').trim() || '#8A7E72';

        this.charts[canvasId] = new Chart(ctx, {
            type: 'line',
            data: { datasets },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    title: {
                        display: true,
                        text: title,
                        font: { family: "'Plus Jakarta Sans', sans-serif", size: 15, weight: '600' },
                        color: style.getPropertyValue('--color-primary').trim() || '#2C2418',
                        padding: { bottom: 16 },
                    },
                    legend: {
                        position: 'bottom',
                        labels: {
                            usePointStyle: true,
                            padding: 16,
                            font: { family: "'Plus Jakarta Sans', sans-serif", size: 11 },
                        },
                    },
                    tooltip: { mode: 'index', intersect: false },
                },
                scales: {
                    x: {
                        type: 'linear',
                        title: { display: true, text: 'Epoch', color: textColor },
                        grid: { display: false },
                        ticks: { color: textColor },
                    },
                    y: {
                        beginAtZero: true,
                        max: 1.0,
                        title: { display: true, text: '指标值', color: textColor },
                        grid: { color: 'rgba(0,0,0,0.04)' },
                        ticks: { color: textColor },
                    },
                },
                interaction: { mode: 'nearest', axis: 'x', intersect: false },
            },
        });
    },
};
