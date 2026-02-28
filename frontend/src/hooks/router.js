/**
 * TeaVision V12 | 前端路由系统
 *
 * 基于 hash 的 SPA 路由器：
 * - 监听 hashchange 事件
 * - 匹配路由并渲染对应页面
 * - 管理导航状态高亮
 *
 * 用法：
 *   Router.register('/', HomePage.render);
 *   Router.register('/detection', DetectionPage.render);
 *   Router.start();
 */

const Router = {
    /** @type {Map<string, Function>} 路由表 */
    _routes: new Map(),

    /** @type {HTMLElement} 页面容器 */
    _container: null,

    /** @type {string} 当前路由路径 */
    _currentPath: '',

    /**
     * 注册路由
     * @param {string} path - 路由路径（如 '/', '/detection'）
     * @param {Function} renderFn - 页面渲染函数，接收容器元素作为参数
     */
    register(path, renderFn) {
        this._routes.set(path, renderFn);
    },

    /**
     * 启动路由器
     *
     * 绑定 hashchange 事件监听，并触发初始路由。
     */
    start() {
        this._container = document.getElementById('app');

        // 监听 hash 变化
        window.addEventListener('hashchange', () => this._onRouteChange());

        // 初始加载
        this._onRouteChange();
    },

    /**
     * 编程式导航
     * @param {string} path - 目标路由路径
     */
    navigate(path) {
        window.location.hash = path;
    },

    /**
     * 路由变化处理（内部方法）
     */
    _onRouteChange() {
        // 解析 hash 得到路径（去掉 #）
        const hash = window.location.hash || '#/';
        const path = hash.slice(1) || '/';

        // 避免重复渲染
        if (path === this._currentPath) return;
        this._currentPath = path;

        // 查找匹配的路由
        const renderFn = this._routes.get(path);

        if (renderFn) {
            // 清空容器并渲染新页面
            this._container.innerHTML = '';
            this._container.scrollTop = 0;
            window.scrollTo(0, 0);

            renderFn(this._container);

            // 渲染完成后刷新 Lucide 图标
            if (window.lucide) {
                setTimeout(() => lucide.createIcons(), 50);
            }
        } else {
            // 404：未找到页面
            this._container.innerHTML = `
                <div class="tv-page-header" style="text-align:center;padding:6rem 2rem">
                    <h1 class="tv-page-title">404</h1>
                    <p class="tv-page-subtitle">页面未找到</p>
                    <a href="#/" class="tv-btn tv-btn--primary" style="margin-top:2rem">返回首页</a>
                </div>
            `;
        }

        // 更新导航栏高亮状态
        this._updateNavActiveState(path);
    },

    /**
     * 更新导航链接的激活状态
     * @param {string} currentPath - 当前路由路径
     */
    _updateNavActiveState(currentPath) {
        document.querySelectorAll('.tv-nav-link').forEach(link => {
            const href = link.getAttribute('href');
            const linkPath = href ? href.slice(1) : '/';
            link.classList.toggle('active', linkPath === currentPath);
        });

        // 同步更新移动端抽屉导航
        document.querySelectorAll('.tv-drawer__link').forEach(link => {
            const href = link.getAttribute('href');
            const linkPath = href ? href.slice(1) : '/';
            link.classList.toggle('active', linkPath === currentPath);
        });
    },
};
