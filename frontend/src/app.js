/**
 * TeaVision V12 | 应用入口
 *
 * 初始化所有模块并启动 SPA 路由。
 * 加载顺序：DOM → 导航栏 → 主题 → 路由注册 → 路由启动
 */

document.addEventListener('DOMContentLoaded', () => {
    // 1. 初始化导航栏
    Navbar.init();

    // 2. 初始化主题控制器
    ThemeController.init();

    // 3. 注册页面路由
    Router.register('/', (c) => HomePage.render(c));
    Router.register('/detection', (c) => DetectionPage.render(c));
    Router.register('/training', (c) => TrainingPage.render(c));
    Router.register('/comparison', (c) => ComparisonPage.render(c));
    Router.register('/augmentation', (c) => AugmentationPage.render(c));
    Router.register('/knowledge', (c) => KnowledgePage.render(c));
    Router.register('/statistics', (c) => StatisticsPage.render(c));

    // 4. 注册 /docs 路由（重定向到后端 API 文档）
    Router.register('/docs', () => {
        window.open(CONFIG.API_BASE_URL + '/docs', '_blank');
        // 回退到首页，避免空白页面
        window.location.hash = '#/';
    });

    // 5. 启动路由
    Router.start();

    // 6. 初始化 Lucide 图标
    if (window.lucide) {
        lucide.createIcons();
    }

    // 7. 获取系统硬件信息
    api.get('/')
        .then(data => {
            const el = document.getElementById('systemInfo');
            if (el && data.device) {
                const icon = data.device === 'cuda' ? '⚡' : '🐢';
                el.innerHTML = `${icon} ${data.device_name}`;
                el.title = `运行时环境: ${data.device_name} (${data.device})`;

                if (data.device === 'cpu') {
                    el.style.color = '#E6A23C'; // Warning color
                    // Add tooltip or click to show help?
                    el.style.cursor = 'help';
                    el.onclick = () => Toast.show('当前使用 CPU 运行，速度较慢。建议配置 NVIDIA 显卡加速。', 'warning');
                } else {
                    el.style.color = '#67C23A'; // Success color
                    el.style.fontWeight = 'bold';
                }
            }
        })
        .catch(console.warn);

    console.log(`%c🍃 TeaVision V12 %c${CONFIG.VERSION}`,
        'color:#B8956A;font-weight:800;font-size:14px',
        'color:#8A7E72;font-weight:400;font-size:12px'
    );
    console.log(`%cAPI → ${CONFIG.API_BASE_URL}`, 'color:#6F806D');
});
