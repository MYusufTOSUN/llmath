@echo off
rem LLMath: API'yi ve Streamlit arayuzunu baslatir, arayuzu tarayicida acar.
rem Proje klasorunde cift tiklayarak calistirin. Zaten aciksa yeniden baslatmaz.
setlocal
cd /d "%~dp0"
set "PY=.venv\Scripts\python.exe"
set PYTHONIOENCODING=utf-8
if not exist "%PY%" (
    echo Sanal ortam bulunamadi: %PY%
    echo README'deki kurulum adimlarini uygulayin.
    pause
    exit /b 1
)

rem Yerel modeller (Qwen) icin Ollama gerekir; aciksa dokunulmaz.
tasklist /fi "imagename eq ollama.exe" | find /i "ollama.exe" >nul
if errorlevel 1 if exist "%LOCALAPPDATA%\Programs\Ollama\ollama app.exe" (
    echo Ollama baslatiliyor...
    start "" "%LOCALAPPDATA%\Programs\Ollama\ollama app.exe"
)

curl -s -f -o nul http://127.0.0.1:8000/health && goto api_ready
echo API baslatiliyor...
start "LLMath API" cmd /k %PY% -m uvicorn api.main:app --port 8000
set /a tries=0
:wait_api
rem ping ~1 sn bekler; timeout komutu konsol girdisi olmadan calismaz
ping -n 2 127.0.0.1 >nul
curl -s -f -o nul http://127.0.0.1:8000/health && goto api_ready
set /a tries+=1
if %tries% lss 60 goto wait_api
echo API 60 saniyede acilmadi; "LLMath API" penceresindeki hataya bakin.
pause
exit /b 1
:api_ready

curl -s -f -o nul http://localhost:8501/_stcore/health && goto ui_ready
echo Arayuz baslatiliyor...
start "LLMath arayuz" cmd /k %PY% -m streamlit run ui/app.py --server.headless true
set /a tries=0
:wait_ui
ping -n 2 127.0.0.1 >nul
curl -s -f -o nul http://localhost:8501/_stcore/health && goto ui_ready
set /a tries+=1
if %tries% lss 60 goto wait_ui
echo Arayuz 60 saniyede acilmadi; "LLMath arayuz" penceresindeki hataya bakin.
pause
exit /b 1
:ui_ready

start "" http://localhost:8501
echo Arayuz: http://localhost:8501   API belgeleri: http://127.0.0.1:8000/docs
echo Kapatmak icin "LLMath API" ve "LLMath arayuz" pencerelerini kapatin.
ping -n 6 127.0.0.1 >nul
