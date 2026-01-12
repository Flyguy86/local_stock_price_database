from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import logging
from pathlib import Path
from .core import (
    get_available_models,
    get_available_tickers,
    run_simulation,
    train_trading_bot,
    get_simulation_history,
    get_top_strategies,
    delete_all_simulation_history
)

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

@app.get("/api/history")
async def get_history_endpoint():
    return get_simulation_history()

@app.get("/history/top")
def get_top_history_endpoint(limit: int = 15, offset: int = 0):
    """
    Returns paginated top strategies sorted by SQN.
    
    Query params:
        limit: Number of records per page (default 15)
        offset: Starting record index (default 0)
    
    Returns:
        {"items": [...], "total": N}
    """
    return get_top_strategies(limit, offset)


@app.delete("/history/all")
def delete_all_history_endpoint():
    """
    Deletes all simulation history records.
    
    Returns:
        {"status": "deleted"} on success
    """
    success = delete_all_simulation_history()
    if not success:
        raise HTTPException(status_code=500, detail="Failed to delete history")
    return {"status": "deleted", "message": "All simulation history has been deleted"}

class SimulationRequest(BaseModel):
    model_id: str
    ticker: str
    initial_cash: float = 10000.0
    use_bot: bool = False
    min_prediction_threshold: float = 0.0
    enable_z_score_check: bool = False
    volatility_normalization: bool = False
    regime_col: str | None = None
    allowed_regimes: list[int] | None = None

class BatchSimulationRequest(BaseModel):
    model_id: str
    tickers: list[str]
    initial_cash: float = 10000.0
    use_bot: bool = False
    min_prediction_threshold: float = 0.0
    enable_z_score_check: bool = False
    volatility_normalization: bool = False
    regime_col: str | None = None
    allowed_regimes: list[int] | None = None

class TrainBotRequest(BaseModel):
    model_id: str
    ticker: str
    min_prediction_threshold: float = 0.0
    enable_z_score_check: bool = False
    volatility_normalization: bool = False

@app.post("/api/simulate")
async def simulate(req: SimulationRequest):
    try:
        log.info(f"Request: {req}")
        result = run_simulation(
            req.model_id, req.ticker, req.initial_cash, req.use_bot,
            min_prediction_threshold=req.min_prediction_threshold,
            enable_z_score_check=req.enable_z_score_check,
            volatility_normalization=req.volatility_normalization,
            regime_col=req.regime_col,
            allowed_regimes=req.allowed_regimes
        )
        return result
    except Exception as e:
        log.error(f"Simulation failed: {e}", exc_info=True)
        return {"error": str(e)}

@app.post("/api/batch_simulate")
async def batch_simulate(req: BatchSimulationRequest):
    try:
        log.info(f"Batch Request: {req}")
        results = []
        for ticker in req.tickers:
            try:
                res = run_simulation(
                    req.model_id, ticker, req.initial_cash, req.use_bot,
                    min_prediction_threshold=req.min_prediction_threshold,
                    enable_z_score_check=req.enable_z_score_check,
                    volatility_normalization=req.volatility_normalization,
                    regime_col=req.regime_col,
                    allowed_regimes=req.allowed_regimes
                )
                results.append({

                    "ticker": ticker, 
                    "status": "success", 
                    "return_pct": res["stats"]["strategy_return_pct"],
                    "hit_rate_pct": res["stats"]["hit_rate_pct"],
                    "trades": res["stats"]["total_trades"]
                })
            except Exception as e:
                log.error(f"Batch sim failed for {ticker}: {e}")
                results.append({"ticker": ticker, "status": "error", "message": str(e)})
                
        return {"results": results}
    except Exception as e:
        log.error(f"Batch simulation failed: {e}", exc_info=True)
        return {"error": str(e)}

@app.post("/api/train_bot")
async def train_bot_endpoint(req: TrainBotRequest):
    try:
        result = train_trading_bot(
             req.model_id, req.ticker,
             min_prediction_threshold=req.min_prediction_threshold,
             enable_z_score_check=req.enable_z_score_check,
             volatility_normalization=req.volatility_normalization
        )
        return result
    except Exception as e:
        log.error(f"Bot training failed: {e}", exc_info=True)
        return {"error": str(e)}
