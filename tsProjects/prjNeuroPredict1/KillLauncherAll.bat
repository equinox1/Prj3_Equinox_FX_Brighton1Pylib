@echo off
:: +------------------------------------------------------------------+
:: | KillLauncher.bat (One-click to kill chief + workers)             |
:: +------------------------------------------------------------------+

echo 🔍 Searching for launcher processes...

:: Kill any python.exe running chief or worker scripts

for /f "tokens=2 delims=," %%a in ('tasklist /v /fo csv ^| findstr /i "tsNeuroPredictWinMql_chief.py tsNeuroPredictWinMql_worker.py"') do (
    echo 🔪 Killing PID %%a ...
    taskkill /PID %%a /F
)

echo ✅ Done. All matching chief and worker processes have been killed.
pause
