/**
 * TeaVision V12 | API 请求封装
 *
 * 统一的 HTTP 请求层：
 * - 自动拼接 API 基础地址
 * - 统一的错误处理与 Toast 提示
 * - 支持 JSON 和 FormData 请求
 */

const api = {
    /**
     * 发送 GET 请求
     * @param {string} endpoint - API 路径（如 '/stats'）
     * @param {Object} params - URL 查询参数
     * @returns {Promise<Object>} 响应 JSON 数据
     */
    async get(endpoint, params = {}) {
        const url = new URL(CONFIG.API_BASE_URL + endpoint);
        Object.entries(params).forEach(([k, v]) => {
            if (v !== undefined && v !== null) url.searchParams.set(k, v);
        });

        try {
            const res = await fetch(url.toString());
            if (!res.ok) throw new Error(`HTTP ${res.status}`);
            return await res.json();
        } catch (err) {
            console.error(`[API] GET ${endpoint} 失败:`, err);
            throw err;
        }
    },

    /**
     * 发送 POST 请求（JSON 格式）
     * @param {string} endpoint - API 路径
     * @param {Object} data - 请求体数据
     * @returns {Promise<Object>} 响应 JSON 数据
     */
    async post(endpoint, data = {}) {
        try {
            const res = await fetch(CONFIG.API_BASE_URL + endpoint, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(data),
            });
            if (!res.ok) throw new Error(`HTTP ${res.status}`);
            return await res.json();
        } catch (err) {
            console.error(`[API] POST ${endpoint} 失败:`, err);
            throw err;
        }
    },

    /**
     * 上传文件（FormData 格式）
     * @param {string} endpoint - API 路径
     * @param {FormData} formData - 包含文件的表单数据
     * @param {Object} params - URL 查询参数
     * @returns {Promise<Object>} 响应 JSON 数据
     */
    async upload(endpoint, formData, params = {}) {
        const url = new URL(CONFIG.API_BASE_URL + endpoint);
        Object.entries(params).forEach(([k, v]) => {
            if (v !== undefined && v !== null) url.searchParams.set(k, v);
        });

        try {
            const res = await fetch(url.toString(), {
                method: 'POST',
                body: formData,
            });
            if (!res.ok) throw new Error(`HTTP ${res.status}`);
            return await res.json();
        } catch (err) {
            console.error(`[API] UPLOAD ${endpoint} 失败:`, err);
            throw err;
        }
    },
};
