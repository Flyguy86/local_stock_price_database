"""
Holy Grail Success Criteria for model promotion.
Defines the conditions a simulation result must meet.
"""
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import statistics


@dataclass
class HolyGrailCriteria:
    """Configurable success thresholds."""
    sqn_min: float = 3.0
    sqn_max: float = 5.0
    profit_factor_min: float = 2.0
    profit_factor_max: float = 4.0
    trade_count_min: int = 200
    trade_count_max: int = 10000
    weekly_consistency_max: float = 0.5  # StdDev/Mean ratio


@dataclass
class CriteriaResult:
    """Result of criteria evaluation."""
    meets_all: bool
    sqn_ok: bool
    pf_ok: bool
    trades_ok: bool
    consistency_ok: bool
    sqn: float
    profit_factor: float
    trade_count: int
    weekly_consistency: Optional[float]
    details: Dict[str, Any]


def evaluate_holy_grail(
    result: Dict[str, Any],
    criteria: Optional[HolyGrailCriteria] = None
) -> CriteriaResult:
    """
    Evaluate if a simulation result meets the Holy Grail criteria.
    
    Args:
        result: Simulation result dict with keys:
            - sqn: System Quality Number
            - profit_factor: Gross Profit / Gross Loss
            - trades_count or trade_count: Number of round-trip trades
            - trades_per_week: Optional list of weekly trade counts
        criteria: Custom thresholds (uses defaults if None)
    
    Returns:
        CriteriaResult with pass/fail for each criterion
    """
    if criteria is None:
        criteria = HolyGrailCriteria()
    
    # Extract values (handle different key names)
    sqn = float(result.get("sqn", 0))
    pf = float(result.get("profit_factor", 0))
    trades = int(result.get("trades_count", result.get("trade_count", 0)))
    
    # SQN check
    sqn_ok = criteria.sqn_min <= sqn <= criteria.sqn_max
    
    # Profit Factor check
    pf_ok = criteria.profit_factor_min <= pf <= criteria.profit_factor_max
    
    # Trade count check
    trades_ok = criteria.trade_count_min <= trades <= criteria.trade_count_max
    
    # Weekly consistency check
    weekly_trades = result.get("trades_per_week", [])
    weekly_consistency = None
    consistency_ok = True  # Default to True if no weekly data
    
    if weekly_trades and len(weekly_trades) > 1:
        mean_trades = statistics.mean(weekly_trades)
        if mean_trades > 0:
            std_trades = statistics.stdev(weekly_trades)
            weekly_consistency = std_trades / mean_trades
            consistency_ok = weekly_consistency < criteria.weekly_consistency_max
    
    # Overall pass
    meets_all = sqn_ok and pf_ok and trades_ok and consistency_ok
    
    return CriteriaResult(
        meets_all=meets_all,
        sqn_ok=sqn_ok,
        pf_ok=pf_ok,
        trades_ok=trades_ok,
        consistency_ok=consistency_ok,
        sqn=sqn,
        profit_factor=pf,
        trade_count=trades,
        weekly_consistency=weekly_consistency,
        details={
            "thresholds": {
                "sqn": f"{criteria.sqn_min}-{criteria.sqn_max}",
                "profit_factor": f"{criteria.profit_factor_min}-{criteria.profit_factor_max}",
                "trades": f"{criteria.trade_count_min}-{criteria.trade_count_max}",
                "weekly_consistency": f"< {criteria.weekly_consistency_max}"
            },
            "actual": {
                "sqn": sqn,
                "profit_factor": pf,
                "trades": trades,
                "weekly_consistency": weekly_consistency
            }
        }
    )


def format_criteria_result(result: CriteriaResult) -> str:
    """Format criteria result for logging."""
    status = "✅ PROMOTED" if result.meets_all else "❌ NOT PROMOTED"
    checks = []
    
    checks.append(f"SQN: {result.sqn:.2f} {'✓' if result.sqn_ok else '✗'}")
    checks.append(f"PF: {result.profit_factor:.2f} {'✓' if result.pf_ok else '✗'}")
    checks.append(f"Trades: {result.trade_count} {'✓' if result.trades_ok else '✗'}")
    
    if result.weekly_consistency is not None:
        checks.append(f"Consistency: {result.weekly_consistency:.2f} {'✓' if result.consistency_ok else '✗'}")
    
    return f"{status} | " + " | ".join(checks)
