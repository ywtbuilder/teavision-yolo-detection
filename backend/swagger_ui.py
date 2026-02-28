# -*- coding: utf-8 -*-
"""
TeaVision V13 | 自定义 Swagger UI 文档界面

提供经过视觉定制的 API 文档页面，采用茶文化大地色调设计。
所有样式均为手工定制，遵循 Anti-AI 美学原则。

路由：
- GET /docs → 自定义 Swagger UI 页面
"""

from fastapi import FastAPI
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.responses import HTMLResponse


def register_swagger_ui(app: FastAPI):
    """
    注册自定义 Swagger UI 路由

    在 FastAPI 应用上注册 /docs 端点，替换默认的 Swagger UI 界面。
    注入自定义 CSS 实现视觉定制。

    Args:
        app: FastAPI 应用实例
    """

    @app.get("/docs", include_in_schema=False)
    async def custom_swagger_ui_html():
        """自定义 Swagger UI 文档页面"""
        html = get_swagger_ui_html(
            openapi_url=app.openapi_url,
            title=app.title + " - 接口文档",
            oauth2_redirect_url=app.swagger_ui_oauth2_redirect_url,
            swagger_js_url="https://unpkg.com/swagger-ui-dist@5/swagger-ui-bundle.js",
            swagger_css_url="https://unpkg.com/swagger-ui-dist@5/swagger-ui.css",
            swagger_favicon_url=(
                "https://yolo-tea-vision.oss-cn-hangzhou.aliyuncs.com/favicon.ico"
            ),
        )

        # 注入自定义样式
        body = html.body.decode("utf-8")
        return HTMLResponse(body.replace("</head>", f"{_CUSTOM_CSS}</head>"))


# ==================== 自定义样式表 ====================

_CUSTOM_CSS = """
<style>
/* ==========================================================================
   TeaVision | API 文档定制主题
   设计理念：手工质感 · 大地色调 · 专业排版

   色彩体系：
   - 背景色：暖白宣纸色 (#F9F7F5)
   - 主文字：墨黑色 (#2C2825)
   - 辅助文字：暖灰色 (#898075)
   - 强调色：大地金 (#B8956A)
   - HTTP 方法色：抹茶绿/沙色/木色/赭红

   排版：Inter + JetBrains Mono
   ========================================================================== */

:root {
    /* 色彩系统 */
    --c-bg: #F9F7F5;
    --c-surface: #FFFFFF;
    --c-text: #2C2825;
    --c-text-light: #898075;
    --c-accent: #B8956A;
    --c-accent-hover: #9E7D54;
    --c-border: #E8E5E0;

    /* HTTP 方法色（低饱和自然色调） */
    --c-get: #6F806D;
    --c-post: #B08968;
    --c-put: #D0A77A;
    --c-delete: #A8655F;

    /* 排版 */
    --font-stack: 'Inter', system-ui, -apple-system, sans-serif;

    /* 投影（柔和多层） */
    --shadow-sm: 0 2px 4px rgba(44, 40, 37, 0.02);
    --shadow-md: 0 8px 16px -4px rgba(44, 40, 37, 0.06);
    --shadow-lg: 0 16px 32px -8px rgba(44, 40, 37, 0.08);
}

/* 全局基础 */
body {
    background-color: var(--c-bg) !important;
    color: var(--c-text) !important;
    font-family: var(--font-stack) !important;
    -webkit-font-smoothing: antialiased;
    background-image: radial-gradient(var(--c-border) 1px, transparent 1px);
    background-size: 32px 32px;
}

.swagger-ui .wrapper {
    max-width: 960px !important;
    margin: 0 auto;
    padding: 0 2rem;
}

/* 顶部导航栏 */
.swagger-ui .topbar {
    background-color: rgba(255, 255, 255, 0.8) !important;
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border-bottom: 1px solid var(--c-border) !important;
    box-shadow: none;
    padding: 1rem 0;
    position: sticky;
    top: 0;
    z-index: 100;
}

.swagger-ui .topbar-wrapper {
    max-width: 960px !important;
    padding: 0 2rem;
}

.swagger-ui .topbar a {
    max-width: fit-content !important;
    flex: none;
}

.swagger-ui .topbar img { display: none; }

.swagger-ui .topbar a::after {
    content: 'TeaVision | API';
    font-weight: 700;
    font-size: 1.1rem;
    color: var(--c-text);
    letter-spacing: -0.01em;
}

/* 信息区域 */
.swagger-ui .info {
    margin: 5rem 0 4rem !important;
    text-align: left;
}

.swagger-ui .info .title {
    font-family: var(--font-stack) !important;
    font-weight: 800 !important;
    color: var(--c-text) !important;
    font-size: 3rem !important;
    letter-spacing: -0.04em;
    margin-bottom: 1.5rem;
    line-height: 1.1;
}

.swagger-ui .info .title small {
    background: var(--c-text) !important;
    color: #fff !important;
    border-radius: 999px;
    padding: 4px 12px;
    font-size: 12px;
    font-weight: 600;
    vertical-align: middle;
    top: -6px;
    position: relative;
}

/* 描述区排版 */
.swagger-ui .info .description {
    color: var(--c-text-light) !important;
    font-size: 1.125rem !important;
    line-height: 1.75 !important;
}

.swagger-ui .info .description h2 {
    font-size: 1.75rem !important;
    margin-top: 3rem !important;
    color: var(--c-text) !important;
    border: none;
    font-weight: 700;
}

.swagger-ui .info .description h3 {
    font-size: 1.25rem !important;
    margin-top: 2rem !important;
    font-weight: 600;
}

/* 协议选择区 */
.swagger-ui .scheme-container {
    background: transparent !important;
    box-shadow: none !important;
    border-top: 1px dashed var(--c-border);
    border-bottom: 1px dashed var(--c-border);
    margin: 4rem 0;
    padding: 2rem 0;
}

/* 接口卡片 */
.swagger-ui .opblock {
    border-radius: 12px !important;
    border: 1px solid var(--c-border) !important;
    box-shadow: var(--shadow-sm) !important;
    background: var(--c-surface) !important;
    margin: 0 0 1.5rem 0 !important;
    transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
    overflow: hidden;
}

.swagger-ui .opblock:hover {
    box-shadow: var(--shadow-md) !important;
    transform: translateY(-2px);
    border-color: var(--c-accent) !important;
}

/* HTTP 方法标签 */
.swagger-ui .opblock .opblock-summary-method {
    border-radius: 6px !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-weight: 600 !important;
    min-width: 80px !important;
    text-align: center;
    color: #fff !important;
    text-shadow: none !important;
    font-size: 0.8rem !important;
}

.swagger-ui .opblock.opblock-post .opblock-summary-method { background: var(--c-post) !important; }
.swagger-ui .opblock.opblock-get .opblock-summary-method { background: var(--c-get) !important; }
.swagger-ui .opblock.opblock-put .opblock-summary-method { background: var(--c-put) !important; }
.swagger-ui .opblock.opblock-delete .opblock-summary-method { background: var(--c-delete) !important; }

/* 摘要行 */
.swagger-ui .opblock-summary {
    border: none !important;
    padding: 1rem 1.5rem !important;
}

.swagger-ui .opblock-post .opblock-summary { border-left: 4px solid var(--c-post) !important; }
.swagger-ui .opblock-get .opblock-summary { border-left: 4px solid var(--c-get) !important; }

.swagger-ui .opblock-summary-path {
    font-family: 'JetBrains Mono', 'Fira Code', monospace !important;
    font-size: 0.95rem !important;
    font-weight: 500 !important;
    color: var(--c-text) !important;
    letter-spacing: -0.02em;
}

.swagger-ui .opblock-summary-description {
    color: var(--c-text-light) !important;
    font-size: 0.9rem !important;
}

/* 展开区域 */
.swagger-ui .opblock-body {
    border-top: 1px solid var(--c-border);
    padding-top: 2rem;
}

/* 执行按钮 */
.swagger-ui .btn.execute {
    background-color: var(--c-text) !important;
    color: #fff !important;
    width: 100%;
    padding: 12px 0 !important;
    border-radius: 8px !important;
    font-weight: 600;
    letter-spacing: 0.02em;
    text-transform: uppercase;
    font-size: 0.85rem;
}

.swagger-ui .btn.execute:hover {
    background-color: #000 !important;
    box-shadow: var(--shadow-md) !important;
}

/* 试用按钮 */
.swagger-ui .btn.try-out__btn {
    background: transparent !important;
    border: 1px solid var(--c-border) !important;
    color: var(--c-text) !important;
    border-radius: 6px !important;
    padding: 6px 16px;
}

.swagger-ui .btn.try-out__btn:hover {
    border-color: var(--c-text) !important;
    background: var(--c-bg) !important;
}

/* 数据模型区 */
.swagger-ui section.models {
    border: 1px solid var(--c-border) !important;
    border-radius: 12px;
    padding: 2rem;
    background: var(--c-surface);
    margin-top: 5rem;
    box-shadow: var(--shadow-sm);
}

.swagger-ui section.models h4 {
    font-family: var(--font-stack) !important;
    color: var(--c-text);
    font-size: 1.25rem;
    border-bottom: 1px solid var(--c-border);
    padding-bottom: 1rem;
    margin-bottom: 2rem;
    font-weight: 700;
}

.swagger-ui .model-box {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.85rem;
}

/* 隐藏不需要的元素 */
.swagger-ui .topbar .download-url-wrapper { display: none !important; }
.swagger-ui .info .main { margin: 0 !important; }

/* =============================================
   移动端响应式适配
   ============================================= */
@media screen and (max-width: 768px) {
    body { background-size: 24px 24px !important; }

    .swagger-ui .wrapper {
        max-width: 100% !important;
        padding: 0 1rem !important;
    }

    .swagger-ui .topbar { padding: 0.75rem 0 !important; }
    .swagger-ui .topbar-wrapper { padding: 0 1rem !important; }
    .swagger-ui .topbar a::after { font-size: 0.95rem !important; }

    .swagger-ui .info { margin: 2rem 0 2rem !important; }
    .swagger-ui .info .title {
        font-size: 1.75rem !important;
        line-height: 1.2 !important;
        word-break: break-word;
    }
    .swagger-ui .info .title small {
        display: block !important;
        margin-top: 0.5rem !important;
        width: fit-content;
    }

    .swagger-ui .info .description { font-size: 0.95rem !important; line-height: 1.65 !important; }
    .swagger-ui .info .description h2 { font-size: 1.25rem !important; margin-top: 2rem !important; }
    .swagger-ui .info .description h3 { font-size: 1rem !important; margin-top: 1.5rem !important; }
    .swagger-ui .info .description p,
    .swagger-ui .info .description li { font-size: 0.9rem !important; }

    .swagger-ui .scheme-container { margin: 2rem 0 !important; padding: 1rem 0 !important; }
    .swagger-ui .schemes-title,
    .swagger-ui .scheme-container .schemes {
        flex-direction: column !important;
        align-items: flex-start !important;
        gap: 0.5rem !important;
    }

    .swagger-ui .opblock-tag-section { margin-bottom: 1rem !important; }
    .swagger-ui .opblock-tag { font-size: 1.1rem !important; padding: 0.75rem 0 !important; }
    .swagger-ui h3.opblock-tag small {
        display: block !important;
        margin-left: 0 !important;
        margin-top: 0.25rem !important;
        font-size: 0.8rem !important;
    }

    .swagger-ui .opblock { border-radius: 8px !important; margin: 0 0 1rem 0 !important; }
    .swagger-ui .opblock:hover { transform: none !important; }
    .swagger-ui .opblock-summary { padding: 0.75rem 1rem !important; flex-wrap: wrap !important; gap: 0.5rem; }
    .swagger-ui .opblock .opblock-summary-method { min-width: 60px !important; font-size: 0.7rem !important; padding: 4px 8px !important; }
    .swagger-ui .opblock-summary-path { font-size: 0.8rem !important; word-break: break-all; flex: 1; min-width: 0; }
    .swagger-ui .opblock-summary-description { font-size: 0.75rem !important; width: 100% !important; margin-left: 0 !important; margin-top: 0.25rem !important; }
    .swagger-ui .opblock-summary-controls { margin-left: auto !important; }

    .swagger-ui .opblock-body { padding: 1rem !important; }
    .swagger-ui .opblock-section-header { padding: 0.5rem 0 !important; }
    .swagger-ui .opblock-section-header h4 { font-size: 0.9rem !important; }

    .swagger-ui .parameters-container,
    .swagger-ui .table-container { overflow-x: auto !important; }
    .swagger-ui table.parameters { font-size: 0.8rem !important; }
    .swagger-ui table.parameters tbody tr td:first-child { min-width: 100px !important; max-width: 120px !important; }
    .swagger-ui .parameter__name,
    .swagger-ui .parameter__type { font-size: 0.75rem !important; }

    .swagger-ui .responses-wrapper { padding: 0 !important; overflow-x: visible !important; }
    .swagger-ui .responses-inner { width: 100% !important; overflow-x: auto !important; }
    .swagger-ui .response-col_links { display: none !important; }

    .swagger-ui table.responses-table { width: 100% !important; display: table !important; table-layout: fixed !important; }
    .swagger-ui .responses-table thead { display: none !important; }
    .swagger-ui .responses-table tbody tr {
        display: block !important;
        margin-bottom: 1rem !important;
        border: 1px solid var(--c-border) !important;
        border-radius: 8px !important;
        padding: 0.75rem !important;
        background: var(--c-bg) !important;
    }
    .swagger-ui .responses-table tbody tr td { display: block !important; width: 100% !important; padding: 0.25rem 0 !important; border: none !important; }
    .swagger-ui .response-col_status { font-size: 1rem !important; font-weight: 700 !important; color: var(--c-get) !important; margin-bottom: 0.5rem !important; }
    .swagger-ui .response-col_status::before { content: 'Status: '; font-weight: 400; color: var(--c-text-light); }
    .swagger-ui .response-col_description { font-size: 0.85rem !important; width: 100% !important; }
    .swagger-ui .response-col_description h4,
    .swagger-ui .response-col_description h5 { font-size: 0.9rem !important; margin: 0.75rem 0 0.5rem !important; }

    .swagger-ui .response-controls { display: flex !important; flex-direction: column !important; gap: 0.5rem !important; }
    .swagger-ui select { font-size: 0.85rem !important; max-width: 100% !important; width: 100% !important; padding: 0.5rem !important; border-radius: 6px !important; }
    .swagger-ui .content-type { width: 100% !important; }
    .swagger-ui label[for*="accept"] { font-size: 0.75rem !important; }

    .swagger-ui .response-col_description .response-content-type { margin-top: 0.75rem !important; }
    .swagger-ui .tab { flex-wrap: wrap !important; padding: 0 !important; }
    .swagger-ui .tab li { font-size: 0.75rem !important; padding: 0.4rem 0.75rem !important; }

    .swagger-ui .example,
    .swagger-ui .model-example { font-size: 0.75rem !important; margin-top: 0.5rem !important; }
    .swagger-ui pre.microlight { font-size: 0.7rem !important; padding: 0.75rem !important; border-radius: 6px !important; overflow-x: auto !important; max-width: 100% !important; }
    .swagger-ui .highlight-code { max-width: 100% !important; overflow-x: auto !important; }

    .swagger-ui .btn.execute { padding: 10px 0 !important; font-size: 0.8rem !important; }
    .swagger-ui .btn.try-out__btn { padding: 4px 12px !important; font-size: 0.75rem !important; }

    .swagger-ui section.models { padding: 1rem !important; margin-top: 2rem !important; border-radius: 8px !important; }
    .swagger-ui section.models h4 { font-size: 1rem !important; margin-bottom: 1rem !important; }
    .swagger-ui .model-box { font-size: 0.75rem !important; }
    .swagger-ui section.models .model-container {
        margin: 0 0 0.75rem 0 !important;
        background: var(--c-bg) !important;
        border-radius: 8px !important;
        border: 1px solid var(--c-border) !important;
        overflow: hidden !important;
    }
    .swagger-ui section.models .model-container:last-child { margin-bottom: 0 !important; }
    .swagger-ui .models .model-box { padding: 0.75rem 1rem !important; }
    .swagger-ui .model-title { font-size: 0.6rem !important; word-break: break-all !important; overflow-wrap: break-word !important; white-space: normal !important; display: block !important; width: 100% !important; letter-spacing: -0.02em !important; line-height: 1.3 !important; }
    .swagger-ui .model-title__text { font-size: 0.6rem !important; word-break: break-all !important; white-space: normal !important; }
    .swagger-ui .model-toggle { margin-left: 0.5rem !important; flex-shrink: 0 !important; }
    .swagger-ui .models table { font-size: 0.7rem !important; }
    .swagger-ui .models .model { font-size: 0.7rem !important; }
    .swagger-ui .model .property { font-size: 0.65rem !important; }
    .swagger-ui .model .property.primitive { padding: 0.2rem 0.4rem !important; }
    .swagger-ui .models-control { font-size: 0.7rem !important; padding: 0.4rem !important; color: var(--c-text-light) !important; }
    .swagger-ui .model-container .model-box span:first-child { font-size: 0.65rem !important; color: var(--c-text-light) !important; }
    .swagger-ui .model-hint { font-size: 0.6rem !important; padding: 1px 5px !important; background: rgba(0,0,0,0.05) !important; border-radius: 3px !important; color: var(--c-accent) !important; }
    .swagger-ui .model-box .inner-object { padding-left: 0.75rem !important; }
    .swagger-ui .model-box table.model { width: 100% !important; }
    .swagger-ui .model-box table.model tbody tr td { padding: 0.25rem 0.5rem !important; font-size: 0.7rem !important; word-break: break-all !important; }
    .swagger-ui .dialog-ux .modal-ux { width: 95% !important; max-width: 95% !important; padding: 1rem !important; }
    .swagger-ui .tab li { font-size: 0.8rem !important; padding: 0.5rem 1rem !important; }
}

/* 超小屏幕 (≤480px) */
@media screen and (max-width: 480px) {
    .swagger-ui .wrapper { padding: 0 0.75rem !important; }
    .swagger-ui .info .title { font-size: 1.5rem !important; }
    .swagger-ui .opblock-summary-path { font-size: 0.7rem !important; }
    .swagger-ui .opblock .opblock-summary-method { min-width: 50px !important; font-size: 0.65rem !important; }
}
</style>
"""
