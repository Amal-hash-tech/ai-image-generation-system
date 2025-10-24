@echo off
echo ========================================
echo   Image Generation System
echo   Starting Web Interface...
echo ========================================
echo.

REM Activate virtual environment
call E:\PROJECTS\LLM\Scripts\activate.bat

REM Start the Gradio UI
python gradio_ui.py

pause