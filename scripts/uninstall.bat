@echo off
echo Uninstalling LabeliT...

set "REPO_DIR=%~dp0.."
set "DESKTOP_DIR=%USERPROFILE%\OneDrive\Desktop"
if not exist "%DESKTOP_DIR%" set "DESKTOP_DIR=%USERPROFILE%\Desktop"

set "SHORTCUT_PATH=%DESKTOP_DIR%\LabeliT.lnk"

echo Removing Desktop shortcut...
if exist "%SHORTCUT_PATH%" del "%SHORTCUT_PATH%"

echo Removing virtual environment...
if exist "%REPO_DIR%\.venv" rmdir /s /q "%REPO_DIR%\.venv"

echo.
echo Uninstallation complete!
echo Note: Your user data and exported datasets in %DESKTOP_DIR%\auto_annotate\ have been preserved.
echo If you wish to delete them, you can manually remove the auto_annotate folder from your Desktop.
pause
