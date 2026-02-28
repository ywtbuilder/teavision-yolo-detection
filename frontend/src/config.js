/**
 * TeaVision V13 | 全局配置
 *
 * API 地址自动检测：
 * - localhost / 127.0.0.1 → http://localhost:8000
 * - 公网 IP / 域名       → http://{hostname}:8000
 */

const CONFIG = {
    // 自动检测 API 基础地址
    API_BASE_URL: (() => {
        const hostname = window.location.hostname;

        // 本地开发环境
        if (hostname === 'localhost' || hostname === '127.0.0.1') {
            return 'http://localhost:8000';
        }

        // 公网环境：使用相同主机名 + 后端端口
        return `http://${hostname}:8000`;
    })(),

    // 应用版本号
    VERSION: '4.0.0',

    // 默认检测参数
    DEFAULT_CONF: 0.25,
    DEFAULT_IOU: 0.45,
};
