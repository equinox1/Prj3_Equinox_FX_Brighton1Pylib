@echo off
set PORT=8000
for /f "tokens=5" %%a in ('netstat -aon ^| findstr :%PORT%') do (
    taskkill /PID %%a /F
)
echo Port %PORT% has been killed.
echo Port %PORT% is now free to use.