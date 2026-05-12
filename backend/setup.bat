@echo off
echo Creando entorno virtual...
python -m venv .venv
call .venv\Scripts\activate
echo Instalando dependencias...
pip install -r requirements.txt
echo.
echo Listo. Para iniciar el servidor:
echo   .venv\Scripts\activate
echo   uvicorn main:app --reload
pause
