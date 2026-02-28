@echo off
chcp 65001 >nul
title TeaVision V13 - 一键启动所有服务

echo ============================================
echo    TeaVision V13 一键启动脚本
echo    茶芽智识 - 茶叶形态智能检测系统
echo ============================================
echo.
echo    架构: 前端 + 后端(接口层/业务层/数据层/工具层)
echo.

:: 获取脚本所在目录（项目根目录）
set "PROJECT_DIR=%~dp0"
cd /d "%PROJECT_DIR%"

echo [INFO] 项目目录: %PROJECT_DIR%
echo.

:: ==================== 检测 Python 环境 ====================
echo [STEP 1] 检测 Python 环境...

:: 优先检查 conda 环境
where conda >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo [INFO] 检测到 Conda 环境
    goto :START_BACKEND
)

:: 检查 venv 虚拟环境
if exist "%PROJECT_DIR%venv\Scripts\activate.bat" (
    echo [INFO] 检测到 venv 虚拟环境
    call "%PROJECT_DIR%venv\Scripts\activate.bat"
    goto :START_BACKEND
)

:: 检查系统 Python
where python >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo [INFO] 使用系统 Python
    goto :START_BACKEND
)

echo [ERROR] 未找到 Python 环境！请先安装 Python 或配置虚拟环境。
pause
exit /b 1


:: ==================== 启动后端服务 ====================
:START_BACKEND
echo.
echo [STEP 2] 启动后端 API 服务 (端口 8000)...

:: 检查依赖
pip show fastapi >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [INFO] 安装后端依赖...
    pip install -r "%PROJECT_DIR%requirements.txt" -q
)

:: 启动 FastAPI 后端（V13: backend.app:app）
start "TeaVision Backend" cmd /k "cd /d %PROJECT_DIR% && python -m uvicorn backend.app:app --host 0.0.0.0 --port 8000 --reload"

:: 等待后端启动
echo [INFO] 等待后端启动...
timeout /t 3 /nobreak >nul


:: ==================== 启动前端服务 ====================
echo.
echo [STEP 3] 启动前端服务 (端口 3000)...

:: 使用 Python 内置 HTTP 服务器提供前端静态文件
start "TeaVision Frontend" cmd /k "cd /d %PROJECT_DIR%frontend && python -m http.server 3000"


:: ==================== 完成 ====================
echo.
echo ============================================
echo    所有服务已启动！
echo.
echo    前端界面: http://localhost:3000
echo    后端 API: http://localhost:8000
echo    API 文档: http://localhost:8000/docs
echo ============================================
echo.
echo 按任意键打开浏览器...
pause >nul

:: 打开浏览器
start http://localhost:3000

exit
