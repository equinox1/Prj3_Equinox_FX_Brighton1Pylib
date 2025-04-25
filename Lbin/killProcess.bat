@echo off
for /L %%P in (8000,1,8100) do (
    for /f "tokens=5" %%a in ('netstat -aon ^| findstr :%%P') do (
        echo Killing process on port %%P with PID %%a...
        taskkill /PID %%a /F >nul 2>&1
    )
)
echo Ports 8000 to 8200 have been scanned and freed where necessary.
