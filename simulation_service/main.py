from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import logging
import threading
from collections import deque
from pathlib import Path
from contextlib import asynccontextmanager
from .core import (
    get_available_models,
    get_available_tickers,
    run_simulation,
    train_trading_bot
)
from .pg_db import (
    get_pool,
    close_pool,
    ensure_tables,
    SimulationDB
)

# Log Buffer for Live Logs
log_buffer = deque(maxlen=5000)
log_lock = threading.Lock()

class BufferHandler(logging.Handler):
    """Custom handler to capture logs in memory buffer."""
    def emit(self, record):
        try:
            msg = self.format(record)
            with log_lock:
                log_buffer.append(msg)
        except Exception:
            self.handleError(record)

# Configure logging with buffer handler BEFORE app creation
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s [%(name)s] %(message)s')
_handler = BufferHandler()
_handler.setLevel(logging.INFO)
_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s [%(name)s] %(message)s'))
logging.getLogger().addHandler(_handler)

log = logging.getLogger("simulation.web")

# Database instance
db = SimulationDB()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage PostgreSQL connection pool lifecycle."""
    log.info("=" * 60)
    log.info("SIMULATION SERVICE STARTING")
    log.info("=" * 60)
    
    # Startup: Create PostgreSQL connection pool
    await ensure_tables()
    log.info("✓ PostgreSQL connection pool created")
    
    try:
        models = get_available_models()
        tickers = get_available_tickers()
        log.info(f"✓ Loaded {len(models)} models")
        log.info(f"✓ Loaded {len(tickers)} tickers: {tickers}")
        log.info("✓ Simulation service ready")
    except Exception as e:
        log.error(f"✗ Startup validation failed: {e}", exc_info=True)
    
    log.info("=" * 60)
    
    yield
    
    # Shutdown: Close PostgreSQL connection pool
    await close_pool()
    log.info("PostgreSQL connection pool closed")

app = FastAPI(title="Simulation Service", lifespan=lifespan)

# Locate templates relative to this file
BASE_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = BASE_DIR / "templates"
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring."""
    try:
        # Quick sanity checks
        models = get_available_models()
        tickers = get_available_tickers()
        return {
            "status": "healthy",
            "models_available": len(models),
            "tickers_available": len(tickers),
            "service": "simulation"
        }
    except Exception as e:
        log.error(f"Health check failed: {e}", exc_info=True)
        return {
            "status": "unhealthy",
            "error": str(e),
            "service": "simulation"
        }

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("dashboard.html", {"request": request})

@app.get("/api/config")
async def get_config():
    return {
        "models": get_available_models(),
        "tickers": get_available_tickers()
    }

@app.get("/logs")
def get_logs():
    """Get recent simulation logs from buffer."""
    with log_lock:
        logs_list = list(log_buffer)
        if not logs_list:
            return ["[No simulation logs yet]"]
        return logs_list

@app.get("/api/history")
async def get_history_endpoint(limit: int = 50):
    """Get recent simulation history from PostgreSQL."""
    return await db.get_history(limit)

@app.get("/history/top")
async def get_top_history_endpoint(limit: int = 15, offset: int = 0):
    """
    Returns paginated top strategies sorted by SQN.
    
    Query params:
        limit: Number of records per page (default 15)
        offset: Starting record index (default 0)
    
    Returns:
        {"items": [...], "total": N}
    """
    return await db.get_top_strategies(limit, offset)


@app.delete("/history/all")
async def delete_all_history_endpoint():
    """
    Deletes all simulation history records.
    
    Returns:
        {"status": "deleted"} on success
    """
    success = await db.delete_all_history()
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
        log.info("="*60)
        log.info(f"SIMULATION REQUEST: {req.model_id[:12]}... | {req.ticker}")
        log.info(f"  Threshold: {req.min_prediction_threshold}")
        log.info(f"  Z-Score: {req.enable_z_score_check}")
        log.info(f"  Vol Norm: {req.volatility_normalization}")
        log.info(f"  Regime: {req.regime_col}={req.allowed_regimes}")
        log.info("="*60)
        
        result = run_simulation(
            req.model_id, req.ticker, req.initial_cash, req.use_bot,
            min_prediction_threshold=req.min_prediction_threshold,
            enable_z_score_check=req.enable_z_score_check,
            volatility_normalization=req.volatility_normalization,
            regime_col=req.regime_col,
            allowed_regimes=req.allowed_regimes
        )
        
        log.info(f"SIMULATION COMPLETE: SQN={result.get('stats', {}).get('sqn', 'N/A')}")
        return result
    except Exception as e:
        log.error("="*60)
        log.error(f"SIMULATION FAILED: {e}", exc_info=True)
        log.error(f"  Model: {req.model_id}")
        log.error(f"  Ticker: {req.ticker}")
        log.error("="*60)
        raise HTTPException(status_code=500, detail=str(e))

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
