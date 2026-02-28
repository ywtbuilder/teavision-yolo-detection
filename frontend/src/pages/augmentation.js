/**
 * TeaVision V12 | 数据增强页
 *
 * 实时预览图像增强效果：
 * - HSV 色彩空间变换
 * - 几何变换（翻转、旋转）
 * - 像素级变换（模糊）
 */

const AugmentationPage = {
    render(container) {
        container.innerHTML = `
            <div class="tv-fade-in">
                <div class="tv-page-header">
                    <h1 class="tv-page-title">数据增强</h1>
                    <p class="tv-page-subtitle">实时预览多种数据增强策略对茶叶图像的影响效果</p>
                </div>

                <div class="tv-detect-grid">
                    <!-- 左侧：上传和参数 -->
                    <div>
                        <div class="tv-dropzone" id="augDropzone">
                            <i data-lucide="image-plus" class="tv-dropzone__icon"></i>
                            <p class="tv-dropzone__text"><strong>上传图片</strong> 以预览增强效果</p>
                            <input type="file" id="augFileInput" accept="image/*" style="display:none">
                        </div>

                        <div class="tv-card tv-card--flat" style="margin-top:var(--space-lg)">
                            <h3 class="tv-card__title">增强参数</h3>

                            <div class="tv-detect-params" style="margin-top:var(--space-md)">
                                <div class="tv-param-row">
                                    <label>色调 (Hue)</label>
                                    <span class="tv-param-value" id="hueVal">0.015</span>
                                </div>
                                <input type="range" class="tv-slider" id="hueSlider" min="0" max="1" step="0.005" value="0.015">

                                <div class="tv-param-row">
                                    <label>饱和度 (Saturation)</label>
                                    <span class="tv-param-value" id="satVal">0.7</span>
                                </div>
                                <input type="range" class="tv-slider" id="satSlider" min="0" max="1" step="0.05" value="0.7">

                                <div class="tv-param-row">
                                    <label>亮度 (Value)</label>
                                    <span class="tv-param-value" id="valVal">0.4</span>
                                </div>
                                <input type="range" class="tv-slider" id="valSlider" min="0" max="1" step="0.05" value="0.4">

                                <div class="tv-param-row">
                                    <label>旋转角度</label>
                                    <span class="tv-param-value" id="rotVal">0°</span>
                                </div>
                                <input type="range" class="tv-slider" id="rotSlider" min="-180" max="180" step="5" value="0">

                                <div class="tv-param-row">
                                    <label>模糊强度</label>
                                    <span class="tv-param-value" id="blurVal">0</span>
                                </div>
                                <input type="range" class="tv-slider" id="blurSlider" min="0" max="10" step="1" value="0">

                                <div style="display:flex;gap:var(--space-md);margin-top:var(--space-sm)">
                                    <label style="display:flex;align-items:center;gap:6px;font-size:0.85rem;cursor:pointer">
                                        <input type="checkbox" id="flipH"> 水平翻转
                                    </label>
                                    <label style="display:flex;align-items:center;gap:6px;font-size:0.85rem;cursor:pointer">
                                        <input type="checkbox" id="flipV"> 垂直翻转
                                    </label>
                                </div>
                            </div>

                            <button class="tv-btn tv-btn--primary tv-btn--lg" id="augBtn" disabled
                                    style="width:100%;margin-top:var(--space-lg)">
                                ${utils.icon('sparkles', 18)} 生成增强效果
                            </button>
                        </div>
                    </div>

                    <!-- 右侧：结果展示 -->
                    <div>
                        <div id="augResults" class="tv-card" style="min-height:300px;display:flex;align-items:center;justify-content:center;color:var(--color-text-light)">
                            <div style="text-align:center">
                                <i data-lucide="sparkles" style="width:48px;height:48px;margin:0 auto 16px;opacity:0.3"></i>
                                <p>增强结果将在此展示</p>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        `;

        this._initControls();
    },

    /** @type {File|null} */
    _selectedFile: null,

    _initControls() {
        const dropzone = document.getElementById('augDropzone');
        const fileInput = document.getElementById('augFileInput');

        dropzone.addEventListener('click', () => fileInput.click());
        fileInput.addEventListener('change', (e) => {
            if (e.target.files[0]) {
                this._selectedFile = e.target.files[0];
                document.getElementById('augBtn').disabled = false;
                Toast.show('图片已就绪', 'info');
            }
        });

        // 滑块值同步
        const sliders = [
            { id: 'hueSlider', target: 'hueVal', suffix: '' },
            { id: 'satSlider', target: 'satVal', suffix: '' },
            { id: 'valSlider', target: 'valVal', suffix: '' },
            { id: 'rotSlider', target: 'rotVal', suffix: '°' },
            { id: 'blurSlider', target: 'blurVal', suffix: '' },
        ];

        sliders.forEach(s => {
            const slider = document.getElementById(s.id);
            slider.addEventListener('input', () => {
                document.getElementById(s.target).textContent = slider.value + s.suffix;
            });
        });

        document.getElementById('augBtn').addEventListener('click', () => this._runAugmentation());
    },

    async _runAugmentation() {
        if (!this._selectedFile) return;

        const btn = document.getElementById('augBtn');
        btn.disabled = true;
        btn.innerHTML = '<div class="tv-spinner" style="width:18px;height:18px;border-width:2px"></div> 生成中...';

        const formData = new FormData();
        formData.append('file', this._selectedFile);

        const params = {
            hsv_h: document.getElementById('hueSlider').value,
            hsv_s: document.getElementById('satSlider').value,
            hsv_v: document.getElementById('valSlider').value,
            rotate: document.getElementById('rotSlider').value,
            blur: document.getElementById('blurSlider').value,
            flip_h: document.getElementById('flipH').checked,
            flip_v: document.getElementById('flipV').checked,
        };

        try {
            const data = await api.upload('/augment', formData, params);
            if (data.success) {
                const results = document.getElementById('augResults');
                const labels = { original: '原图', hsv: 'HSV 变换', flip_horizontal: '水平翻转', flip_vertical: '垂直翻转', rotate: '旋转', blur: '高斯模糊' };

                results.innerHTML = `
                    <div class="tv-augment-grid" style="width:100%">
                        ${Object.entries(data.images).map(([key, base64]) => `
                            <div class="tv-augment-card">
                                <img src="data:image/jpeg;base64,${base64}" class="tv-augment-card__img" alt="${labels[key] || key}">
                                <div class="tv-augment-card__label">${labels[key] || key}</div>
                            </div>
                        `).join('')}
                    </div>
                `;

                Toast.show(`生成 ${Object.keys(data.images).length} 种效果`, 'success');
            }
        } catch (err) {
            Toast.show('增强失败: ' + err.message, 'error');
        } finally {
            btn.disabled = false;
            btn.innerHTML = `${utils.icon('sparkles', 18)} 生成增强效果`;
            if (window.lucide) lucide.createIcons({ nodes: [btn] });
        }
    },
};
