@echo off
chcp 65001 >nul
echo ========================================
echo   Building Techpixel MSI Installer
echo ========================================
echo.

echo [1/3] Generating assets.wxs via WiX Heat...
"%WIX%\bin\heat.exe" dir ".\assets" -dr Bin -cg AssetsComponentGroup -gg -scom -sreg -sfrag -out "wix\assets.wxs"
if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Failed to run heat.exe. Make sure WiX is installed and the %%WIX%% environment variable is set.
    pause
    exit /b %errorlevel%
)
echo.

echo [2/3] Fixing paths in assets.wxs (replacing SourceDir\ with assets\)...
powershell -Command "(Get-Content 'wix\assets.wxs' -Raw) -replace 'SourceDir\\', 'assets\' | Set-Content 'wix\assets.wxs' -NoNewline"
if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Failed to replace paths in the file.
    pause
    exit /b %errorlevel%
)
echo.

echo [3/3] Running build via cargo-dist...
dist build
if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Build failed with an error.
    pause
    exit /b %errorlevel%
)

echo.
echo ========================================
echo   Build completed successfully!
echo ========================================
pause