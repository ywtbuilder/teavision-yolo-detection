/**
 * TeaVision V12 | 导航栏控制器
 *
 * 管理导航栏的交互行为：
 * - 移动端汉堡菜单开关
 * - 移动端抽屉导航
 * - 滚动时导航栏收缩效果
 */

const Navbar = {
    /**
     * 初始化导航栏
     */
    init() {
        this._initHamburger();
        this._initDrawer();
        this._initScrollEffect();
    },

    /**
     * 汉堡菜单按钮
     */
    _initHamburger() {
        const btn = document.getElementById('hamburgerBtn');
        if (!btn) return;

        btn.addEventListener('click', () => {
            const drawer = document.getElementById('mobileDrawer');
            const isOpen = drawer.classList.contains('open');

            if (isOpen) {
                this._closeDrawer();
            } else {
                this._openDrawer();
            }

            btn.classList.toggle('active', !isOpen);
        });
    },

    /**
     * 移动端抽屉导航
     */
    _initDrawer() {
        // 复制桌面导航链接到抽屉
        const navLinks = document.getElementById('navLinks');
        const drawerLinks = document.getElementById('drawerLinks');
        if (!navLinks || !drawerLinks) return;

        const links = navLinks.querySelectorAll('.tv-nav-link');
        links.forEach(link => {
            const clone = link.cloneNode(true);
            clone.className = 'tv-drawer__link' + (link.classList.contains('active') ? ' active' : '');
            clone.addEventListener('click', () => this._closeDrawer());
            drawerLinks.appendChild(clone);
        });

        // 点击遮罩关闭
        const overlay = document.getElementById('drawerOverlay');
        if (overlay) {
            overlay.addEventListener('click', () => this._closeDrawer());
        }
    },

    /**
     * 滚动时导航栏视觉效果
     */
    _initScrollEffect() {
        const navbar = document.getElementById('navbar');
        if (!navbar) return;

        let lastScroll = 0;
        window.addEventListener('scroll', () => {
            const current = window.scrollY;
            navbar.classList.toggle('scrolled', current > 20);
            lastScroll = current;
        }, { passive: true });
    },

    _openDrawer() {
        const drawer = document.getElementById('mobileDrawer');
        drawer.classList.add('open');
        document.body.style.overflow = 'hidden';
    },

    _closeDrawer() {
        const drawer = document.getElementById('mobileDrawer');
        const btn = document.getElementById('hamburgerBtn');
        drawer.classList.remove('open');
        btn.classList.remove('active');
        document.body.style.overflow = '';
    },
};
