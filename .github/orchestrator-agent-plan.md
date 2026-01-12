## Code goals
lets make sure to keep javascript / hmtl  separate file from python.

# Orchestrator Agent Plan: Recursive Strategy Factory

## Mission
Automate the Train → Prune → Simulate loop without modifying existing service internals. All integration happens via **new endpoints** and a **new orchestrator service**.

## Critical Constraint: Do Not Break Existing Services
- **NO modifications** to existing endpoint signatures or return types
- **NO changes** to existing database schemas (only ADD new tables)
- **Expose new APIs** on existing services rather than modifying internal logic
- **New orchestrator service** coordinates via HTTP calls to existing services
- All existing tests must continue to pass

---

## Design Decisions (User Confirmed)

| Decision | Choice | Rationale |
|----------|--------|-----------|
| **Shared Results DB** | PostgreSQL | True distributed locking for multi-node workers |
| **Evolution Stopping Criterion** | Input parameter, default=4 | Configurable max generations per evolution run |
| **Priority Queuing** | YES | Child models from high-SQN parents get simulation priority |

---

## Architecture Overview
┌──────────────────────────────────────────────────────────────────┐
│ ORCHESTRATOR SERVICE (NEW) │
│ Port 8400 │
│ - Polls completed models from training_service │
│ - Computes fingerprints, checks deduplication │
│ - Triggers pruning + retraining │
│ - Queues simulation batches via optimization API │
│ - Flags success candidates │
└────────────────────────┬─────────────────────────────────────────┘
│ HTTP calls only
┌───────────────┼───────────────┐
▼ ▼ ▼
┌────────────────┐ ┌──────────────┐ ┌────────────────┐
│ training:8200 │ │ optim:8002 │ │ simulation:8300│
│ (unchanged) │ │ (unchanged) │ │ (unchanged) │
│ +new endpoints │ │ +new tables │ │ │
└────────────────┘ └──────────────┘ └────────────────┘
│
▼
┌────────────────┐
│ PostgreSQL │
│ (NEW) │
│ Port 5432 │
└────────────────┘


### Simulation requirments 
The simulation needs to be robust, expressing each step its taking to get the predictions.  

Simulation 
 INFO:simulation.core:Trading Bot: DISABLED (using raw model predictions)
optimization_c2       | INFO:simulation.core:Applying Regime Filter: regime_gmm MUST BE in [1]
optimization_c2       | INFO:simulation.core:Regime Filter blocked 1732 signals in unfavorable market conditions
optimization_c2       | INFO:simulation.core:SLIPPAGE MODEL: 4-bar execution delay with midpoint pricing
optimization_c2       | INFO:simulation.core:  -> Orders fill at mean(open, close) of bar T+4
optimization_c2       | INFO:simulation.core:  -> Simulates market impact, order routing delays, and partial fills
optimization_c2       | INFO:simulation.core:TRANSACTION COSTS: $0.02 per trade (entry + exit = $0.04 round-trip)
optimization_c2       | INFO:simulation.core:  -> Includes: SEC fees, exchange fees, clearing fees, and rounding
optimization_c2       | INFO:simulation.core:------------------------------------------------------------
optimization_c2       | INFO:simulation.core:Beginning Walk-Forward Backtest...
optimization_c2       | INFO:simulation.core:------------------------------------------------------------
optimization_c2       | INFO:simulation.core:Total Round-Trip Trades: 0
optimization_c2       | INFO:simulation.core:Win Rate: 0.0% (0 wins, 0 losses)
optimization_c2       | INFO:simulation.core:Benchmark Strategy: Buy & Hold 185.39 shares at $53.94
optimization_c2       | INFO:simulation.core:------------------------------------------------------------
optimization_c2       | INFO:simulation.core:Backtest Complete - Calculating Performance Metrics...
optimization_c2       | INFO:simulation.core:------------------------------------------------------------
optimization_c2       | INFO:simulation.core:Total Round-Trip Trades: 54
optimization_c2       | INFO:simulation.core:Win Rate: 53.7% (29 wins, 25 losses)
optimization_c2       | INFO:simulation.core:Average Win: $85.47 | Average Loss: $43.10
optimization_c2       | INFO:simulation.core:Expectancy (Avg P&L per trade): $25.95
optimization_c2       | INFO:simulation.core:Profit Factor (Gross Profit / Gross Loss): 2.30
optimization_c2       | INFO:simulation.core:System Quality Number (SQN): 1.82 (Fair)
optimization_c2       | INFO:simulation.core:Total Fees Paid: $2.16 (0.02% of capital)
optimization_c2       | INFO:simulation.core:Hit Rate (Directional Accuracy): 52.7%
optimization_c2       | INFO:simulation.core:------------------------------------------------------------
optimization_c2       | INFO:simulation.core:FINAL RESULTS:
optimization_c2       | INFO:simulation.core:  Strategy Return: +14.01% ($11,401.03)
optimization_c2       | INFO:simulation.core:  Benchmark Return: +353.67% ($45,367.07)
optimization_c2       | INFO:simulation.core:  Alpha (Outperformance): -339.66%
optimization_c2       | INFO:simulation.core:============================================================

