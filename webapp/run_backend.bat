@echo off
rem Chay backend bang env kaggle-env (khong can activate). Them tham so uvicorn neu can, vi du: run_backend.bat --port 8001
set "PY=C:\Users\thinh\anaconda3\envs\kaggle-env\python.exe"
if not exist "%PY%" (echo Khong thay %PY% & exit /b 1)
rem Phai chay tu thu muc chua webapp/ vi code dung import tuong doi
cd /d "%~dp0.."
"%PY%" -m uvicorn webapp.backend.main:app --reload --port 8000 %*
