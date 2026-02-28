/**
 * TeaVision V12 | 工具函数库
 *
 * 提供通用的格式化和 DOM 操作工具。
 */

const utils = {
    /**
     * 格式化数字：添加千位分隔符
     * @param {number} num - 待格式化的数字
     * @returns {string} 格式化后的字符串
     */
    formatNumber(num) {
        if (num === null || num === undefined) return '0';
        return num.toLocaleString('zh-CN');
    },

    /**
     * 格式化文件大小
     * @param {number} bytes - 字节数
     * @returns {string} 人类可读的大小字符串
     */
    formatFileSize(bytes) {
        if (bytes === 0) return '0 B';
        const units = ['B', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(1024));
        return (bytes / Math.pow(1024, i)).toFixed(1) + ' ' + units[i];
    },

    /**
     * 防抖函数
     * @param {Function} fn - 目标函数
     * @param {number} delay - 延迟毫秒数
     * @returns {Function} 防抖包装后的函数
     */
    debounce(fn, delay = 300) {
        let timer;
        return (...args) => {
            clearTimeout(timer);
            timer = setTimeout(() => fn(...args), delay);
        };
    },

    /**
     * 安全地设置元素内容（防 XSS）
     * @param {HTMLElement} el - 目标元素
     * @param {string} text - 纯文本内容
     */
    setText(el, text) {
        if (el) el.textContent = text;
    },

    /**
     * 创建 Lucide 图标元素
     * @param {string} name - 图标名称
     * @param {number} size - 图标大小（像素）
     * @returns {string} 图标 HTML
     */
    icon(name, size = 18) {
        return `<i data-lucide="${name}" style="width:${size}px;height:${size}px"></i>`;
    },

    /**
     * 等待指定毫秒
     * @param {number} ms - 毫秒数
     * @returns {Promise<void>}
     */
    sleep(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    },

    /**
     * 生成唯一 ID
     * @returns {string}
     */
    uid() {
        return Date.now().toString(36) + Math.random().toString(36).slice(2, 7);
    },
};
