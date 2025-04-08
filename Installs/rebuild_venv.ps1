# Path to your virtual environment
$venvPath = "venvwin1"

# Optional: Path to requirements.txt (if you have one)
$requirementsFile = "requirements.txt"

Write-Host "🔍 Checking if virtual environment exists..."
if (Test-Path $venvPath) {
    Write-Host "🗑️ Removing old virtual environment at $venvPath"
    Remove-Item -Recurse -Force $venvPath
}

Write-Host "⚙️ Creating new virtual environment..."
python -m venv $venvPath

Write-Host "✅ Virtual environment created. Activating..."

# Activate the virtual environment in this script session
& "$venvPath\Scripts\activate.ps1"

Write-Host "📦 Upgrading pip..."
python -m pip install --upgrade pip

# If you have a requirements.txt, install dependencies
if (Test-Path $requirementsFile) {
    Write-Host "📄 Installing dependencies from $requirementsFile..."
    python -m pip install -r $requirementsFile
} else {
    Write-Host "⚠️ No requirements.txt found. Skipping package installation."
}

Write-Host "🎉 Done!"
