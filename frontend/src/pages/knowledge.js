/**
 * TeaVision V12 | 茶叶档案页
 *
 * 茶叶品种知识库，数据来源：static/knowledge.json
 */

const KnowledgePage = {
    _data: [],
    _currentCategory: 'all',

    render(container) {
        container.innerHTML = `
            <div class="tv-fade-in">
                <div class="tv-page-header">
                    <h1 class="tv-page-title">茶叶档案</h1>
                    <p class="tv-page-subtitle">详尽的茶叶品种知识库</p>
                </div>
                
                <div class="tv-knowledge-controls">
                    <div class="tv-search-wrapper">
                        <i data-lucide="search" class="tv-search-icon"></i>
                        <input type="text" class="tv-input tv-search-input" id="teaSearch" placeholder="搜索茶叶品种...">
                    </div>
                    <div class="tv-category-tabs" id="categoryTabs">
                        <button class="tv-tab active" data-cat="all">全部</button>
                    </div>
                </div>

                <div id="teaList" class="tv-tea-grid">
                    <div class="tv-loading"><div class="tv-spinner"></div><span>加载中...</span></div>
                </div>
            </div>`;
        this._loadData();
    },

    async _loadData() {
        try {
            const res = await fetch('static/knowledge.json');
            const data = await res.json();
            this._data = data.teas || data;
            const categories = [...new Set(this._data.flatMap(t => t.tags || []))];
            const tabs = document.getElementById('categoryTabs');

            // Keep "All" tab and append others
            const allBtn = '<button class="tv-tab active" data-cat="all">全部</button>';
            tabs.innerHTML = allBtn + categories.map(c => `<button class="tv-tab" data-cat="${c}">${c}</button>`).join('');

            tabs.querySelectorAll('.tv-tab').forEach(tab => {
                tab.addEventListener('click', () => {
                    tabs.querySelectorAll('.tv-tab').forEach(t => t.classList.remove('active'));
                    tab.classList.add('active');
                    this._currentCategory = tab.dataset.cat;
                    this._renderList();
                });
            });

            document.getElementById('teaSearch').addEventListener('input', utils.debounce(() => this._renderList(), 300));
            this._renderList();
            if (window.lucide) lucide.createIcons();
        } catch (err) {
            document.getElementById('teaList').innerHTML = '<div class="tv-card" style="text-align:center;color:var(--color-error)">加载失败</div>';
        }
    },

    _renderList() {
        const search = (document.getElementById('teaSearch')?.value || '').trim().toLowerCase();
        const list = document.getElementById('teaList');
        let filtered = this._data;
        if (this._currentCategory !== 'all') filtered = filtered.filter(t => (t.tags || []).includes(this._currentCategory));
        if (search) filtered = filtered.filter(t => (t.title || t.name || '').toLowerCase().includes(search) || (t.description || '').toLowerCase().includes(search));

        if (!filtered.length) {
            list.innerHTML = `
                <div class="tv-empty-state">
                    <i data-lucide="coffee" style="width:48px;height:48px;opacity:0.5;margin-bottom:12px"></i>
                    <p>未找到匹配的茶叶品种</p>
                </div>`;
            if (window.lucide) lucide.createIcons();
            return;
        }

        list.innerHTML = filtered.map(tea => {
            const teaName = tea.title || tea.name || '未知';
            const imgSrc = tea.image ? tea.image.replace(/^\.\.\//, '') : '';
            return `
            <div class="tv-tea-card">
                <div class="tv-tea-card__image-wrapper">
                    ${imgSrc ? `<img src="${imgSrc}" alt="${teaName}" class="tv-tea-card__img" loading="lazy" onerror="this.style.display='none'">` : '<div class="tv-tea-card__placeholder">🍃</div>'}
                </div>
                <div class="tv-tea-card__content">
                    <div class="tv-tea-card__header">
                        <h3 class="tv-tea-card__title">${teaName}</h3>
                        ${tea.subtitle ? `<span class="tv-tea-card__subtitle">${tea.subtitle}</span>` : ''}
                    </div>
                    <p class="tv-tea-card__desc">${tea.description || ''}</p>
                    <div class="tv-tea-card__tags">
                        ${(tea.tags || []).map(tag => `<span class="tv-tag">${tag}</span>`).join('')}
                    </div>
                </div>
            </div>`;
        }).join('');
    },
};
