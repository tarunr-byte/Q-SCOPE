# Q-SCOPE: Quantum Systems Operations Studio (Production Deployment)

A unified, production-ready hypersonic quantum telemetry analysis and flight operations studio. Built with a seven-agent autonomous pipeline, real ADXL345 sensor streaming, Azhva cognitive translation, and persistent PostgreSQL database storage.

---

## Architecture Overview

- **Frontend**: Premium Liquid Glass responsive UI with live 7-agent pipeline tracking journey, real-time audio machine translation, spectrogram generation, and an ultra-minimalist page loader.
- **Backend**: FastAPI with asynchronous SSE progress streaming and modular 7-agent execution engine.
- **Database Layer**: SQLAlchemy with automatic PostgreSQL connection pooling (for Railway.com) and automatic SQLite fallback (for local development).
- **Inference Engines**: High-performance classical & quantum estimators (Kaiser filtering, Kalman filtering, two-qubit quantum state simulator, and Groq/NVIDIA LLMs).

---

## 1-Click Railway.com Deployment Guide

### Step 1: Push to GitHub
Open your terminal in the `qscope-deploy` folder:

```bash
cd qscope-deploy
git init
git add .
git commit -m "Initial commit: Production Q-SCOPE with PostgreSQL database and clean UI"
git branch -M main
git remote add origin https://github.com/<YOUR_USERNAME>/<YOUR_REPO_NAME>.git
git push -u origin main
```

### Step 2: Deploy on Railway.com
1. Log in to [Railway.com](https://railway.com).
2. Click **+ New Project** $\rightarrow$ **Deploy from GitHub repo**.
3. Select your newly created repository.
4. Click **Add a Service** $\rightarrow$ **Database** $\rightarrow$ **Add PostgreSQL**.
5. Railway will automatically link the database and inject `DATABASE_URL` into your web application service!
6. Under your Web Service **Settings** $\rightarrow$ **Networking**, click **Generate Domain** (e.g. `qscope-production.up.railway.app`).

### Step 3: Add Environment Variables (Optional)
In your Railway web service, open the **Variables** tab to add optional keys:
- `GROQ_API_KEY`: *(Optional)* Enables live Groq LLaMA 3.3 70B cognitive translation and chat.
- `NVIDIA_API_KEY`: *(Optional)* Enables NVIDIA NIM model routing.
- `APP_ENV`: `production`

---

## Local Development (Quick Start)

### 1. Install Dependencies
```bash
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Start the Server
```bash
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```
Open [http://localhost:8000](http://localhost:8000) in your web browser. The application will initialize an SQLite database (`qscope.db`) automatically.

### 3. Run Automated Tests
```bash
python -m pytest tests
```

---

## API Reference

| Endpoint | Method | Description |
|---|---|---|
| `/` | `GET` | Serves the Liquid Glass frontend interface. |
| `/api/projects` | `GET` | Retrieves all saved flight projects from PostgreSQL / SQLite. |
| `/api/projects` | `POST` | Creates or updates a flight project with instrument configurations. |
| `/api/projects/{id}` | `DELETE` | Deletes a project and its runtimes from the database. |
| `/api/datasets` | `GET` | Lists available real ADXL345 flight CSV datasets on the server. |
| `/api/analyze/from-dataset` | `POST` | Executes the full 7-agent pipeline against real sensor data. |
| `/api/analyze/from-dataset/stream` | `POST` | Streams live step-by-step progress via Server-Sent Events (SSE). |
| `/api/azhva/translate` | `POST` | Generates cognitive translation, voice synthesis data, and spectrograms. |
| `/api/azhva/chat` | `POST` | Autonomous flight operations copilot conversation. |
| `/api/health` | `GET` | Health status and database connectivity indicator. |

---

## Directory Structure

```
qscope-deploy/
├── app/
│   ├── api/
│   │   ├── routes.py      # REST & SSE endpoints with DB persistence
│   │   └── schemas.py     # Pydantic validation contracts
│   ├── core/
│   │   ├── config.py      # Environment & Railway configuration
│   │   ├── database.py    # SQLAlchemy PostgreSQL / SQLite session engine
│   │   └── models.py      # Database models for Projects & AnalysisRuns
│   └── main.py            # FastAPI entry point & static frontend mount
├── agents/                # 7-agent pipeline + Azhva cognitive translator
│   └── nodes/             # Deterministic pipeline stages (Agents 1-7)
├── physics/               # Kaiser, Kalman, and 2-qubit quantum simulator
├── dataset/               # Real ADXL345 hypersonic accelerometer CSVs
├── frontend/              # Static single-page application & visual assets
├── tests/                 # Full automated test suite (API, physics, DB)
├── Dockerfile             # Production container definition
├── Procfile               # Railway / Heroku process runner
├── railway.json           # Railway configuration schema
├── requirements.txt       # Production dependencies
└── README.md
```
