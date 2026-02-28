/**
 * TeaVision V12 | 主题控制器
 *
 * 管理 16 个手工定制主题的切换、持久化和 UI 面板。
 * 主题列表：8 个浅色 + 8 个深色。
 */

// ==================== 主题注册表 ====================

const THEMES = [
    // 浅色主题 (8)
    { id: 'warm-earth', name: '暖大地色', nameEn: 'Warm Earth', description: '温暖自然的咖啡大地色系', isDark: false },
    { id: 'jade-serenity', name: '翡翠禅意', nameEn: 'Jade Serenity', description: '东方美学清新翡翠绿调', isDark: false },
    { id: 'lavender-dusk', name: '薰衣草黄昏', nameEn: 'Lavender Dusk', description: '柔和浪漫的薰衣草紫调', isDark: false },
    { id: 'arctic-frost', name: '极地霜雪', nameEn: 'Arctic Frost', description: '纯净极简的北欧冷调', isDark: false },
    { id: 'terracotta-sunset', name: '陶土日落', nameEn: 'Terracotta Sunset', description: '温暖大胆的陶土橘红', isDark: false },
    { id: 'ink-wash', name: '水墨丹青', nameEn: 'Ink Wash', description: '高雅黑白灰调，极简东方美学', isDark: false },
    { id: 'sakura-breeze', name: '樱花微风', nameEn: 'Sakura Breeze', description: '优雅淡粉暖灰，日式赏樱氛围', isDark: false },
    { id: 'classic-manuscript', name: '古籍羊皮', nameEn: 'Classic Manuscript', description: '致敬经典，泛黄的纸张与墨迹', isDark: false },
    // 深色主题 (8)
    { id: 'obsidian-night', name: '深邃夜色', nameEn: 'Obsidian Night', description: '高对比度深色模式', isDark: true },
    { id: 'industrial-rust', name: '工业锈迹', nameEn: 'Industrial Rust', description: '硬朗深灰与锈色', isDark: true },
    { id: 'deep-ocean', name: '深海潜游', nameEn: 'Deep Ocean', description: '静谧深蓝与生物荧光', isDark: true },
    { id: 'midnight-forest', name: '深夜森林', nameEn: 'Midnight Forest', description: '深邃暗绿，神秘自然', isDark: true },
    { id: 'noir-cinema', name: '黑白电影', nameEn: 'Noir Cinema', description: '经典黑白，高对比度戏剧感', isDark: true },
    { id: 'volcanic-ember', name: '火山余烬', nameEn: 'Volcanic Ember', description: '深黑与熔岩红', isDark: true },
    { id: 'cosmic-void', name: '宇宙虚空', nameEn: 'Cosmic Void', description: '浩瀚宇宙，深紫与星云', isDark: true },
    { id: 'carbon-fiber', name: '碳纤质感', nameEn: 'Carbon Fiber', description: '科技碳纤维，银色金属质感', isDark: true },
];


// ==================== 主题控制器 ====================

const ThemeController = {
    /** @type {string} 当前激活的主题 ID */
    currentTheme: 'warm-earth',

    /** @type {boolean} 主题面板是否展开 */
    isPanelOpen: false,

    /**
     * 初始化主题控制器
     *
     * 从 localStorage 恢复上次选择的主题，生成主题选择面板。
     */
    init() {
        // 恢复已保存的主题
        const saved = this._getSavedTheme();
        if (saved) this.currentTheme = saved;
        this._applyTheme(this.currentTheme);

        // 生成主题选择面板
        this._renderPanel();

        // 绑定面板开关事件
        const toggleBtn = document.getElementById('themeToggle');
        const panel = document.getElementById('themePanel');
        const closeBtn = document.getElementById('themePanelClose');

        if (toggleBtn) {
            toggleBtn.addEventListener('click', (e) => {
                e.stopPropagation();
                this.isPanelOpen = !this.isPanelOpen;
                panel.classList.toggle('active', this.isPanelOpen);
            });
        }

        // 关闭按钮
        if (closeBtn) {
            closeBtn.addEventListener('click', () => {
                this.isPanelOpen = false;
                panel.classList.remove('active');
            });
        }

        // 点击空白处关闭面板
        document.addEventListener('click', (e) => {
            const fab = document.getElementById('themeFab');
            if (fab && !fab.contains(e.target)) {
                this.isPanelOpen = false;
                panel.classList.remove('active');
            }
        });

        // ESC 键关闭面板
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape' && this.isPanelOpen) {
                this.isPanelOpen = false;
                panel.classList.remove('active');
            }
        });
    },

    /**
     * 切换到指定主题
     * @param {string} themeId - 主题标识符
     */
    switchTheme(themeId) {
        this.currentTheme = themeId;
        this._applyTheme(themeId);
        this._saveTheme(themeId);
        this._updateActiveState();
        this._updateLogoInversion(themeId);

        // 显示切换成功提示
        const theme = THEMES.find(t => t.id === themeId);
        if (theme && window.Toast) {
            Toast.show(`已切换至「${theme.name}」主题`, 'success');
        }
    },

    // ---- 内部方法 ----

    /**
     * 应用主题到文档根元素
     */
    _applyTheme(themeId) {
        document.documentElement.setAttribute('data-theme', themeId);
        this._updateLogoInversion(themeId);
    },

    /**
     * 深色主题下反转 Logo 亮度
     */
    _updateLogoInversion(themeId) {
        const theme = THEMES.find(t => t.id === themeId);
        const isDark = theme && theme.isDark;
        const logos = document.querySelectorAll('.tv-navbar__school-logo, .tv-footer__school-logo');
        logos.forEach(logo => {
            logo.style.filter = isDark ? 'brightness(0) invert(1)' : 'none';
        });
    },

    /**
     * 渲染主题选择面板（采用 V10 色块预览样式）
     */
    _renderPanel() {
        const grid = document.getElementById('themeGrid');
        if (!grid) return;

        // 使用 V10 的 theme-grid 2列网格布局 + 色块预览
        grid.className = 'theme-grid';
        grid.innerHTML = THEMES.map(t => this._renderThemeItem(t)).join('');

        // 绑定点击事件
        grid.querySelectorAll('.theme-option').forEach(item => {
            item.addEventListener('click', () => {
                this.switchTheme(item.dataset.theme);
            });
        });

        this._updateActiveState();
    },

    /**
     * 渲染单个主题选项（带色块预览 + 中文名称）
     */
    _renderThemeItem(theme) {
        const isActive = theme.id === this.currentTheme;
        return `
            <div class="theme-option ${isActive ? 'active' : ''}" 
                 data-theme="${theme.id}" 
                 title="${theme.description}">
                <div class="theme-swatch"></div>
                <span class="theme-name">${theme.name}</span>
            </div>
        `;
    },

    _updateActiveState() {
        document.querySelectorAll('.theme-option').forEach(item => {
            item.classList.toggle('active', item.dataset.theme === this.currentTheme);
        });
    },

    _getSavedTheme() {
        try { return localStorage.getItem('teavision-theme'); }
        catch { return null; }
    },

    _saveTheme(themeId) {
        try { localStorage.setItem('teavision-theme', themeId); }
        catch { /* localStorage 不可用 */ }
    },
};
