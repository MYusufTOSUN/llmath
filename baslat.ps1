# API'yi ve Streamlit arayüzünü iki ayrı pencerede başlatır.
# Kullanım (proje klasöründe): .\baslat.ps1
$root = $PSScriptRoot
$python = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    Write-Error "Sanal ortam bulunamadı. README'deki kurulum adımlarını uygulayın."
    exit 1
}
Start-Process powershell -WorkingDirectory $root -ArgumentList "-NoExit", "-Command", "& '$python' -m uvicorn api.main:app --port 8000"
Start-Sleep -Seconds 3
Start-Process powershell -WorkingDirectory $root -ArgumentList "-NoExit", "-Command", "& '$python' -m streamlit run ui/app.py"
Write-Host "API: http://127.0.0.1:8000/docs  ·  Arayüz: http://localhost:8501"
