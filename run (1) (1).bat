@echo off
setlocal
cd /d "D:\aihackathon"

echo 🔍 Checking dependencies...
:: Use the same python instance that will run the app to install the package
python -m pip install python-dotenv uvicorn streamlit fastapi

echo 🚀 Launching API Server...
:: We use 'python -m uvicorn' to ensure it uses the specific python environment
start "FastAPI Server" cmd /k "python -m uvicorn api:app --reload"

timeout /t 5

echo 📊 Launching Streamlit App...
start "Streamlit Dashboard" cmd /k "python -m streamlit run app.py"

echo ✅ Both services are starting. Check the individual windows for logs.
pause