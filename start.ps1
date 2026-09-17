param([int]$Port = 8000)
$ErrorActionPreference = 'Stop'
$backendPath = Join-Path $PSScriptRoot 'backend'
$pythonPath = Join-Path $backendPath '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $pythonPath)) {
    throw 'Create backend/.venv and install backend/requirements.txt first. See documentation/SETUP.md.'
}
if (-not (Test-Path -LiteralPath (Join-Path $backendPath '.env'))) {
    throw 'Configure backend/.env using .env.example first. See documentation/SETUP.md.'
}
if (-not (Test-Path -LiteralPath (Join-Path $PSScriptRoot 'frontend/dist/index.html'))) {
    throw 'Run npm install and npm run build in source_code/frontend first.'
}
Push-Location $backendPath
try {
    Write-Host "Research AI dashboard: http://127.0.0.1:$Port"
    & $pythonPath -m uvicorn app.main:app --host 127.0.0.1 --port $Port --workers 1
} finally {
    Pop-Location
}
