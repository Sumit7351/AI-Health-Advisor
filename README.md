# AI Health Advisor - Run Instructions

This guide explains how to run the AI Health Advisor application from your terminal.

## Prerequisites
- **Python** (3.8 or higher)
- **Node.js** (16 or higher)

## Quick Start (Windows)
Double-click the `run_app.bat` file in this directory. It will automatically start both the backend and frontend.

---

## Manual Start (Terminal)

If you prefer to run it manually, you will need **two separate terminal windows**.

### Terminal 1: Backend (API)
1. Open a terminal/command prompt.
2. Navigate to the `backend` folder:
   ```powershell
   cd backend
   ```
3. Run the Flask app:
   ```powershell
   python app.py
   ```
   *You should see "Running on http://127.0.0.1:5000"*

### Terminal 2: Frontend (UI)
1. Open a **new** terminal window.
2. Navigate to the `frontend` folder:
   ```powershell
   cd frontend
   ```
3. Start the development server:
   ```powershell
   npm run dev
   ```
   *You should see "Local: http://localhost:5173"*

## Accessing the App
Open your web browser and go to:
**http://localhost:5173**

## Troubleshooting
- **Backend fails?** Make sure you installed dependencies: `pip install pandas numpy scikit-learn shap flask flask-cors`
- **Frontend fails?** Make sure you installed dependencies: `cd frontend` then `npm install`
