@echo off
echo Installing LabeliT...

:: Check for uv
where uv >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo Error: "uv" package manager not found.
    echo Please install uv ^(https://docs.astral.sh/uv/^) and make sure it is in your PATH.
    pause
    exit /b 1
)

:: Get absolute path of the directory containing the project root
set "REPO_DIR=%~dp0.."
cd /d "%REPO_DIR%"

echo Creating virtual environment and installing dependencies...
call uv venv --clear
call uv pip install -e .

echo Creating runtime directories...
set "DESKTOP_DIR=%USERPROFILE%\OneDrive\Desktop"
if not exist "%DESKTOP_DIR%" set "DESKTOP_DIR=%USERPROFILE%\Desktop"

set "WORK_DIR=%DESKTOP_DIR%\auto_annotate"
if not exist "%WORK_DIR%\input_images" mkdir "%WORK_DIR%\input_images"
if not exist "%WORK_DIR%\datasets" mkdir "%WORK_DIR%\datasets"
if not exist "%WORK_DIR%\model" mkdir "%WORK_DIR%\model"

echo Creating Desktop shortcut...
set "SHORTCUT_PATH=%DESKTOP_DIR%\LabeliT.lnk"
powershell -Command "$WshShell = New-Object -comObject WScript.Shell; $Shortcut = $WshShell.CreateShortcut('%SHORTCUT_PATH%'); $Shortcut.TargetPath = 'uv.exe'; $Shortcut.Arguments = 'run python -m auto_annotator'; $Shortcut.WorkingDirectory = '%REPO_DIR%'; $Shortcut.WindowStyle = 1; $Shortcut.Save()"

echo.
echo Installation complete! 
echo You can now launch LabeliT from the shortcut on your Desktop.
echo Put your ONNX models and labels in: %WORK_DIR%\model\
echo Put your images in: %WORK_DIR%\input_images\
pause
