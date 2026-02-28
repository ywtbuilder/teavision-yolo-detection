/**
 * TeaVision V12 | 检测引擎页
 *
 * 三种检测模式：
 * 1. 单图检测 - 上传一张图片进行检测
 * 2. 批量检测 - 同时上传多张图片
 * 3. 模型对比 - 同一张图片使用多个模型检测
 */

const DetectionPage = {
    /** @type {string} 当前激活的模式 */
    _mode: 'single',

    /**
     * 渲染检测页
     * @param {HTMLElement} container - 页面容器
     */
    render(container) {
        container.innerHTML = `
            <div class="tv-fade-in">
                <div class="tv-page-header">
                    <h1 class="tv-page-title">检测引擎</h1>
                    <p class="tv-page-subtitle">上传茶叶图片，获取高精度 YOLO-TBD 检测结果</p>
                </div>

                <!-- 模式切换标签页 -->
                <div class="tv-tabs">
                    <button class="tv-tab active" data-mode="single">单图检测</button>
                    <button class="tv-tab" data-mode="batch">批量检测</button>
                    <button class="tv-tab" data-mode="video">视频检测</button>
                    <button class="tv-tab" data-mode="compare">模型对比</button>
                </div>

                <!-- 检测区域 -->
                <div class="tv-detect-grid">
                    <!-- 左侧：上传和参数 -->
                    <div>
                        <!-- 模型选择（对比模式时显示多选） -->
                        <div class="tv-card tv-card--flat" style="margin-bottom:var(--space-lg)">
                            <label style="display:block;font-size:0.85rem;font-weight:600;margin-bottom:var(--space-sm)">选择模型</label>
                            <select class="tv-select" id="modelSelect" style="width:100%">
                                <option value="default">加载中...</option>
                            </select>
                            <div id="compareModels" style="display:none;margin-top:var(--space-md)">
                                <label style="font-size:0.8rem;color:var(--color-text-light)">对比模型（多选）</label>
                                <div id="modelCheckboxes" style="margin-top:var(--space-sm);display:flex;flex-direction:column;gap:6px"></div>
                            </div>
                        </div>

                        <!-- 上传区域 -->
                        <div class="tv-dropzone" id="dropzone">
                            <i data-lucide="upload-cloud" class="tv-dropzone__icon"></i>
                            <p class="tv-dropzone__text">
                                <strong>点击上传</strong> 或拖拽图片至此处
                            </p>
                            <p style="font-size:0.75rem;color:var(--color-text-light);margin-top:8px" id="dropzoneHint">
                                支持 JPG、PNG 格式
                            </p>
                            <input type="file" id="fileInput" accept="image/*" style="display:none">
                            <input type="file" id="fileInputMulti" accept="image/*" multiple style="display:none">
                            <input type="file" id="fileInputVideo" accept="video/*" style="display:none">
                        </div>

                        <!-- 参数调节 -->
                        <div class="tv-detect-params" style="margin-top:var(--space-lg)">
                            <div class="tv-param-row">
                                <label>置信度阈值</label>
                                <span class="tv-param-value" id="confValue">0.25</span>
                            </div>
                            <input type="range" class="tv-slider" id="confSlider" min="0" max="1" step="0.05" value="0.25">

                            <div class="tv-param-row">
                                <label>IoU 阈值</label>
                                <span class="tv-param-value" id="iouValue">0.45</span>
                            </div>
                            <input type="range" class="tv-slider" id="iouSlider" min="0" max="1" step="0.05" value="0.45">
                        </div>

                        <!-- 检测按钮 -->
                        <button class="tv-btn tv-btn--primary tv-btn--lg" id="detectBtn" disabled
                                style="width:100%;margin-top:var(--space-lg)">
                            ${utils.icon('scan-search', 18)} 开始检测
                        </button>
                    </div>

                    <!-- 右侧：结果展示 -->
                    <div>
                        <div class="tv-card" id="resultArea" style="min-height:300px;display:flex;align-items:center;justify-content:center;color:var(--color-text-light)">
                            <div style="text-align:center">
                                <i data-lucide="image" style="width:48px;height:48px;margin:0 auto 16px;opacity:0.3"></i>
                                <p>上传后检测结果将在此展示</p>
                            </div>
                        </div>

                        <!-- 检测统计 -->
                        <div id="detectionStats" style="display:none;margin-top:var(--space-lg)">
                            <div class="tv-bento tv-bento--3">
                                <div class="tv-stat-card">
                                    <div class="tv-stat-card__value" id="rObjectCount">0</div>
                                    <div class="tv-stat-card__label">检测目标</div>
                                </div>
                                <div class="tv-stat-card">
                                    <div class="tv-stat-card__value" id="rInferTime">0</div>
                                    <div class="tv-stat-card__label">推理耗时</div>
                                </div>
                                <div class="tv-stat-card">
                                    <div class="tv-stat-card__value" id="rAvgConf">0</div>
                                    <div class="tv-stat-card__label">平均置信度</div>
                                </div>
                            </div>
                        </div>

                        <!-- 检测列表 -->
                        <div id="detectionList" class="tv-detection-list"></div>
                    </div>
                </div>
            </div>
        `;

        // 初始化交互
        this._initTabs();
        this._initDropzone();
        this._initSliders();
        this._loadModels();
    },

    /**
     * 初始化标签页切换
     */
    _initTabs() {
        document.querySelectorAll('.tv-tab').forEach(tab => {
            tab.addEventListener('click', () => {
                document.querySelectorAll('.tv-tab').forEach(t => t.classList.remove('active'));
                tab.classList.add('active');
                this._mode = tab.dataset.mode;

                // 切换上传模式
                const compareModels = document.getElementById('compareModels');
                compareModels.style.display = this._mode === 'compare' ? 'block' : 'none';

                // 更新提示文本
                const dropzoneText = document.querySelector('.tv-dropzone__text');
                const dropzoneHint = document.getElementById('dropzoneHint');

                if (this._mode === 'batch') {
                    dropzoneText.innerHTML = '<strong>点击上传</strong> 或拖拽多张图片';
                    dropzoneHint.textContent = '支持 JPG、PNG 格式';
                } else if (this._mode === 'video') {
                    dropzoneText.innerHTML = '<strong>点击上传</strong> 或拖拽视频文件';
                    dropzoneHint.textContent = '支持 MP4、AVI、MOV 格式';
                } else {
                    dropzoneText.innerHTML = '<strong>点击上传</strong> 或拖拽图片至此处';
                    dropzoneHint.textContent = '支持 JPG、PNG 格式';
                }

                // 重置
                document.getElementById('resultArea').innerHTML = `
                    <div style="text-align:center">
                        <i data-lucide="${this._mode === 'video' ? 'video' : 'image'}" style="width:48px;height:48px;margin:0 auto 16px;opacity:0.3"></i>
                        <p>上传后检测结果将在此展示</p>
                    </div>
                `;
                lucide.createIcons();
                this._pendingFiles = [];
                document.getElementById('detectBtn').disabled = true;
                document.getElementById('detectionStats').style.display = 'none';
                document.getElementById('detectionList').innerHTML = '';
            });
        });
    },

    /**
     * 初始化拖拽上传区域
     */
    _initDropzone() {
        const dropzone = document.getElementById('dropzone');
        const fileInput = document.getElementById('fileInput');
        const fileInputMulti = document.getElementById('fileInputMulti');
        const fileInputVideo = document.getElementById('fileInputVideo');

        // 点击触发文件选择
        dropzone.addEventListener('click', () => {
            if (this._mode === 'batch') {
                fileInputMulti.click();
            } else if (this._mode === 'video') {
                fileInputVideo.click();
            } else {
                fileInput.click();
            }
        });

        // 文件选择回调
        fileInput.addEventListener('change', (e) => this._handleInput(e));
        fileInputMulti.addEventListener('change', (e) => this._handleInput(e));
        fileInputVideo.addEventListener('change', (e) => this._handleInput(e));

        // 拖拽事件
        ['dragenter', 'dragover'].forEach(evt => {
            dropzone.addEventListener(evt, (e) => {
                e.preventDefault();
                dropzone.classList.add('dragover');
            });
        });

        ['dragleave', 'drop'].forEach(evt => {
            dropzone.addEventListener(evt, (e) => {
                e.preventDefault();
                dropzone.classList.remove('dragover');
            });
        });

        dropzone.addEventListener('drop', (e) => {
            e.preventDefault();
            const items = Array.from(e.dataTransfer.files);
            let files = [];

            if (this._mode === 'video') {
                files = items.filter(f => f.type.startsWith('video/'));
            } else {
                files = items.filter(f => f.type.startsWith('image/'));
            }

            if (files.length > 0) this._handleFiles(files);
            else Toast.show('文件格式不正确', 'warning');
        });

        // 快捷键：Ctrl + Enter 触发检测
        document.addEventListener('keydown', (e) => {
            if (e.ctrlKey && e.key === 'Enter') {
                document.getElementById('detectBtn')?.click();
            }
        });
    },

    _handleInput(e) {
        if (e.target.files.length > 0) {
            this._handleFiles(Array.from(e.target.files));
        }
        e.target.value = ''; // Reset
    },

    /**
     * 初始化滑块
     */
    _initSliders() {
        const confSlider = document.getElementById('confSlider');
        const iouSlider = document.getElementById('iouSlider');

        confSlider.addEventListener('input', () => {
            document.getElementById('confValue').textContent = confSlider.value;
        });

        iouSlider.addEventListener('input', () => {
            document.getElementById('iouValue').textContent = iouSlider.value;
        });
    },

    /**
     * 加载可用模型列表
     */
    async _loadModels() {
        try {
            const data = await api.get('/models');
            if (data.success) {
                const select = document.getElementById('modelSelect');
                const checkboxes = document.getElementById('modelCheckboxes');

                select.innerHTML = data.models.map(m =>
                    `<option value="${m.key}" ${!m.available ? 'disabled' : ''}>${m.name}${m.available ? '' : ' (不可用)'}</option>`
                ).join('');

                checkboxes.innerHTML = data.models.filter(m => m.available).map(m =>
                    `<label style="display:flex;align-items:center;gap:8px;font-size:0.85rem;cursor:pointer">
                        <input type="checkbox" value="${m.key}" ${m.key === 'default' ? 'checked' : ''}>
                        ${m.name}
                    </label>`
                ).join('');
            }
        } catch (err) {
            console.warn('[检测] 模型列表加载失败:', err);
        }
    },

    /** @type {File[]} 待检测文件 */
    _pendingFiles: [],

    /**
     * 处理选择的文件
     */
    _handleFiles(files) {
        this._pendingFiles = files;
        const btn = document.getElementById('detectBtn');
        btn.disabled = false;

        // 显示预览
        const resultArea = document.getElementById('resultArea');
        const file = files[0];

        if (file.type.startsWith('video/')) {
            const url = URL.createObjectURL(file);
            resultArea.innerHTML = `
                <video src="${url}" class="tv-result-image" controls style="max-height:400px;width:100%;object-fit:contain;background:#000"></video>
                <div style="margin:8px auto 0;font-size:0.85rem;color:var(--color-text-light);max-width:300px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;text-align:center" title="${file.name}">
                    ${file.name} <span style="opacity:0.7">(${(file.size / 1024 / 1024).toFixed(1)} MB)</span>
                </div>
            `;
        } else {
            const reader = new FileReader();
            reader.onload = (e) => {
                resultArea.innerHTML = `<img src="${e.target.result}" class="tv-result-image" alt="待检测图片">`;
            };
            reader.readAsDataURL(file);
        }

        // 绑定检测按钮
        btn.onclick = () => this._runDetection();

        Toast.show(`已选择 ${files.length} 个文件`, 'info');
    },

    /**
     * 执行检测
     */
    async _runDetection() {
        if (this._pendingFiles.length === 0) return;

        const btn = document.getElementById('detectBtn');
        btn.disabled = true;

        let loadingMsg = '检测中...';
        if (this._mode === 'video') {
            loadingMsg = '视频逐帧处理中，请耐心等待...';
        } else if (this._mode === 'batch') {
            loadingMsg = '批量处理中...';
        }

        btn.innerHTML = `<div class="tv-spinner" style="width:18px;height:18px;border-width:2px"></div> ${loadingMsg}`;

        const conf = document.getElementById('confSlider').value;
        const iou = document.getElementById('iouSlider').value;
        const modelKey = document.getElementById('modelSelect').value;

        try {
            let data;

            if (this._mode === 'single') {
                // 单图检测
                const formData = new FormData();
                formData.append('file', this._pendingFiles[0]);
                data = await api.upload('/detect/with-model', formData, { model_key: modelKey, conf, iou, return_image: true });
                this._showSingleResult(data);

            } else if (this._mode === 'batch') {
                // 批量检测
                const formData = new FormData();
                this._pendingFiles.forEach(f => formData.append('files', f));
                data = await api.upload('/detect/batch', formData, { model_key: modelKey, conf, iou, return_image: true });
                this._showBatchResult(data);

            } else if (this._mode === 'video') {
                // 视频检测
                const formData = new FormData();
                formData.append('file', this._pendingFiles[0]);
                // 视频检测可能耗时较长
                data = await api.upload('/detect/video', formData, { model_key: modelKey, conf, iou });
                this._showVideoResult(data);

            } else if (this._mode === 'compare') {
                // 多模型对比
                const checked = document.querySelectorAll('#modelCheckboxes input:checked');
                const modelKeys = Array.from(checked).map(c => c.value).join(',');
                const formData = new FormData();
                formData.append('file', this._pendingFiles[0]);
                data = await api.upload('/detect/compare', formData, { model_keys: modelKeys, conf, iou });
                this._showCompareResult(data);
            }

            Toast.show('检测完成', 'success');
        } catch (err) {
            Toast.show('检测失败: ' + err.message, 'error');
        } finally {
            btn.disabled = false;
            btn.innerHTML = `${utils.icon('scan-search', 18)} 开始检测`;
            if (window.lucide) lucide.createIcons({ nodes: [btn] });
        }
    },

    /**
     * 显示单图检测结果
     */
    _showSingleResult(data) {
        if (!data.success) return;

        const resultArea = document.getElementById('resultArea');
        resultArea.innerHTML = data.image_base64
            ? `<img src="data:image/jpeg;base64,${data.image_base64}" class="tv-result-image" alt="检测结果">`
            : '<p style="text-align:center;color:var(--color-text-light)">无标注图片</p>';

        // 更新统计
        const stats = document.getElementById('detectionStats');
        stats.style.display = 'block';
        utils.setText(document.getElementById('rObjectCount'), data.detections?.length || 0);
        utils.setText(document.getElementById('rInferTime'), data.inference_time + 'ms');
        const avgConf = data.detections?.length
            ? (data.detections.reduce((s, d) => s + d.confidence, 0) / data.detections.length * 100).toFixed(1) + '%'
            : '0%';
        utils.setText(document.getElementById('rAvgConf'), avgConf);

        // 检测列表
        const list = document.getElementById('detectionList');
        list.innerHTML = (data.detections || []).map(d => `
            <div class="tv-detection-item">
                <span class="tv-detection-item__name">${d.class_name}</span>
                <span class="tv-detection-item__conf">${(d.confidence * 100).toFixed(1)}%</span>
            </div>
        `).join('');
    },

    /**
     * 显示批量检测结果
     */
    _showBatchResult(data) {
        if (!data.success) return;

        const resultArea = document.getElementById('resultArea');
        resultArea.innerHTML = `
            <div style="width:100%">
                <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:16px">
                    <h3 style="font-size:1rem;font-weight:700">批量检测结果</h3>
                    <span class="tv-badge tv-badge--success">${data.total_images} 张 · ${data.total_time}ms</span>
                </div>
                <div class="tv-augment-grid">
                    ${data.results.map(r => r.success ? `
                        <div class="tv-augment-card">
                            <img src="data:image/jpeg;base64,${r.image_base64}" class="tv-augment-card__img" alt="${r.filename}">
                            <div class="tv-augment-card__label">${r.filename} · ${r.detection_count} 个目标 · ${r.inference_time}ms</div>
                        </div>
                    ` : `
                        <div class="tv-augment-card" style="border-color:var(--color-error)">
                            <div class="tv-augment-card__label" style="color:var(--color-error)">${r.filename} · 失败</div>
                        </div>
                    `).join('')}
                </div>
            </div>
        `;

        document.getElementById('detectionStats').style.display = 'none';
        document.getElementById('detectionList').innerHTML = '';
    },

    /**
     * 显示视频检测结果
     */
    _showVideoResult(data) {
        if (!data.success) return;

        const resultArea = document.getElementById('resultArea');
        const fullVideoUrl = `${CONFIG.API_BASE_URL}${data.video_url}`;

        resultArea.innerHTML = `
            <div style="text-align:center;width:100%">
                <h3 style="margin-bottom:10px;font-size:1rem;font-weight:700;color:var(--color-primary)">检测完成</h3>
                <video src="${fullVideoUrl}" class="tv-result-image" controls autoplay style="max-height:500px;width:100%;object-fit:contain;background:#000;border-radius:var(--radius-md)"></video>
                <div style="margin-top:16px">
                    <a href="${fullVideoUrl}" download="detection_result.mp4" class="tv-btn tv-btn--primary">
                        ${utils.icon('download', 18)} 下载检测结果视频
                    </a>
                </div>
            </div>
        `;

        document.getElementById('detectionStats').style.display = 'none';
        document.getElementById('detectionList').innerHTML = '';
        if (window.lucide) lucide.createIcons();
    },

    /**
     * 显示多模型对比结果
     */
    _showCompareResult(data) {
        if (!data.success) return;

        const resultArea = document.getElementById('resultArea');
        resultArea.innerHTML = `
            <div style="width:100%">
                <h3 style="font-size:1rem;font-weight:700;margin-bottom:16px">模型对比结果</h3>
                <div class="tv-augment-grid">
                    ${data.results.map(r => r.success ? `
                        <div class="tv-augment-card">
                            <img src="data:image/jpeg;base64,${r.image_base64}" class="tv-augment-card__img" alt="${r.model_name}">
                            <div class="tv-augment-card__label">
                                ${r.model_name}<br>
                                <span style="font-weight:400;font-size:0.75rem;color:var(--color-text-light)">
                                    ${r.detection_count} 目标 · ${r.inference_time}ms · 置信度 ${(r.avg_confidence * 100).toFixed(1)}%
                                </span>
                            </div>
                        </div>
                    ` : `
                        <div class="tv-augment-card" style="border-color:var(--color-error)">
                            <div class="tv-augment-card__label" style="color:var(--color-error)">${r.model_name} · 失败</div>
                        </div>
                    `).join('')}
                </div>
            </div>
        `;

        document.getElementById('detectionStats').style.display = 'none';
        document.getElementById('detectionList').innerHTML = '';
    },
};
