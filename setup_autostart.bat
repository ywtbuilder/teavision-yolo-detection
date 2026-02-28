@echo off
chcp 65001 >nul
title TeaVision - 配置开机自启

echo ========================================================
echo    TeaVision 开机自启配置工具 (用户登录版)
echo ========================================================
echo.
echo    功能说明：
echo    将启动脚本添加到 Windows "启动" 文件夹。
echo    效果：当您远程登录服务器(或服务器自动登录)时，服务会自动运行。
echo.

:: 1. 定义路径
set "TARGET_SCRIPT=%~dp0start_all_services.bat"
set "STARTUP_FOLDER=%APPDATA%\Microsoft\Windows\Start Menu\Programs\Startup"
set "SHORTCUT_NAME=TeaVision_AutoStart.lnk"
set "LINK_PATH=%STARTUP_FOLDER%\%SHORTCUT_NAME%"

:: 2. 检查是否已存在
if exist "%LINK_PATH%" (
    echo [INFO] 检测到已配置自启动。
    echo.
    set /p "CHOICE=是否要 重新配置/覆盖 ? (Y/N): "
    if /i "%CHOICE%" neq "Y" goto :END
    echo.
)

:: 3. 使用 PowerShell 创建快捷方式
echo [PROCESS] 正在创建快捷方式...
echo    - 源文件: %TARGET_SCRIPT%
echo    - 目标位置: %LINK_PATH%

powershell -Command "$ws = New-Object -ComObject WScript.Shell; $s = $ws.CreateShortcut('%LINK_PATH%'); $s.TargetPath = '%TARGET_SCRIPT%'; $s.WorkingDirectory = '%~dp0'; $s.WindowStyle = 1; $s.Description = 'TeaVision 自动启动'; $s.Save()"

if %ERRORLEVEL% EQU 0 (
    echo.
    echo [SUCCESS] ✅ 配置成功！
    echo.
    echo 注意事项：
    echo 1. 此方式依赖用户登录。服务器重启后，您需要【远程连接】一次才会启动。
    echo 2. 若要取消自启，请删除以下文件：
    echo    %LINK_PATH%
) else (
    echo.
    echo [ERROR] ❌ 配置失败，请尝试以管理员身份运行。
)

:END
echo.
echo 按任意键退出...
pause >nul
