@echo off
chcp 65001 >nul
echo ========================================
echo   Сборка MSI установщика Techpixel
echo ========================================
echo.

echo [1/3] Генерация assets.wxs через WiX Heat...
"%WIX%\bin\heat.exe" dir ".\assets" -dr Bin -cg AssetsComponentGroup -gg -scom -sreg -sfrag -out "wix\assets.wxs"
if %errorlevel% neq 0 (
    echo.
    echo [ОШИБКА] Не удалось запустить heat.exe. Проверьте, установлен ли WiX и задана ли переменная %%WIX%%.
    pause
    exit /b %errorlevel%
)
echo.

echo [2/3] Исправление путей в assets.wxs (замена SourceDir\ на assets\)...
powershell -Command "(Get-Content 'wix\assets.wxs' -Raw) -replace 'SourceDir\\', 'assets\' | Set-Content 'wix\assets.wxs' -NoNewline"
if %errorlevel% neq 0 (
    echo.
    echo [ОШИБКА] Ошибка при замене путей в файле.
    pause
    exit /b %errorlevel%
)
echo.

echo [3/3] Запуск сборки через cargo-dist...
dist build
if %errorlevel% neq 0 (
    echo.
    echo [ОШИБКА] Сборка завершилась с ошибкой.
    pause
    exit /b %errorlevel%
)

echo.
echo ========================================
echo   Сборка успешно завершена!
echo ========================================
pause