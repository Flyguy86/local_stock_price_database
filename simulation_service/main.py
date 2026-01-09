from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import logging
from pathlib import Path
from .core import get_available_models, get_available_tickers, run_simulation

app = FastAPI(title="Simulation Service")
log = logging.getLogger("simulation.web")
logging.basicConfig(level=logging.INFO)

# Locate templates relative to this file
BASE_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = BASE_DIR / "templates"
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("dashboard.html", {"request": request})

@app.get("/api/config")
async def get_config():
    return {
        "models": get_available_models(),
        "tickers": get_available_tickers()
    }

class SimulationRequest(BaseModel):
    model_id: str
    ticker: str
    initial_cash: float = 10000.0

@app.post("/api/simulate")
async def simulate(req: SimulationRequest):
    try:
        log.info(f"Request: {req}")
        result = run_simulation(req.model_id, req.ticker, req.initial_cash)
        return result
    except Exception as e:
        log.error(f"Simulation failed: {e}", exc_info=True)
        return {"error": str(e)}
