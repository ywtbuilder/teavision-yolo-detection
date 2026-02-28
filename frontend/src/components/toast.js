/**
 * TeaVision V12 | Toast 通知组件
 *
 * 提供轻量级的通知提示功能。
 * 支持四种类型：success, error, warning, info
 */

const Toast = {
    /**
     * 显示 Toast 通知
     * @param {string} message - 通知文本
     * @param {'success'|'error'|'warning'|'info'} type - 通知类型
     * @param {number} duration - 显示时长（毫秒），默认 3000
     */
    show(message, type = 'info', duration = 3000) {
        const container = document.getElementById('toastContainer');
        if (!container) return;

        // 图标映射
        const icons = {
            success: 'check-circle',
            error: 'x-circle',
            warning: 'alert-triangle',
            info: 'info',
        };

        const toast = document.createElement('div');
        toast.className = `tv-toast tv-toast--${type}`;
        toast.innerHTML = `
            <i data-lucide="${icons[type]}" class="tv-toast__icon"></i>
            <span class="tv-toast__text">${message}</span>
        `;

        container.appendChild(toast);

        // 刷新图标
        if (window.lucide) lucide.createIcons({ nodes: [toast] });

        // 入场动画
        requestAnimationFrame(() => toast.classList.add('show'));

        // 自动消失
        setTimeout(() => {
            toast.classList.remove('show');
            toast.addEventListener('transitionend', () => toast.remove());
        }, duration);
    },
};
