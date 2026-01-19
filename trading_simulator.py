#!/usr/bin/env python3
"""
Trading simulation backtest with realistic transaction costs and performance metrics.

Generates comprehensive performance statistics including:
- Total return, annualized return
- Sharpe ratio, Sortino ratio
- Maximum drawdown
- Win rate, profit factor
- Transaction costs impact
"""

import numpy as np
import pandas as pd
from typing import Dict, List
from dataclasses import dataclass


@dataclass
class TradeStats:
    """Statistics for a single trade."""
    entry_date: str
    exit_date: str
    entry_price: float
    exit_price: float
    shares: int
    pnl: float
    pnl_pct: float
    commission: float
    slippage: float


class TradingSimulator:
    """Simulate trading based on model predictions with realistic costs."""
    
    def __init__(
        self,
        initial_capital: float = 100000.0,
        position_size_pct: float = 0.95,  # Use 95% of capital per trade
        slippage_pct: float = 0.001,  # 0.1% slippage
        commission_per_share: float = 0.005,  # $0.005 per share
        risk_free_rate: float = 0.04  # 4% annual risk-free rate for Sharpe
    ):
        self.initial_capital = initial_capital
        self.position_size_pct = position_size_pct
        self.slippage_pct = slippage_pct
        self.commission_per_share = commission_per_share
        self.risk_free_rate = risk_free_rate
        
    def run_backtest(
        self,
        df: pd.DataFrame,
        predictions_col: str = 'predicted_return',
        actual_return_col: str = 'forward_return_1d',
        price_col: str = 'close',
        threshold: float = 0.001  # Only trade if prediction > 0.1%
    ) -> Dict:
        """
        Run trading simulation.
        
        Args:
            df: DataFrame with predictions and actual returns
            predictions_col: Column name for predicted returns
            actual_return_col: Column name for actual returns
            price_col: Column name for prices
            threshold: Minimum predicted return to enter trade
            
        Returns:
            Dictionary with performance metrics and trade log
        """
        df = df.copy()
        df = df.sort_values('ts').reset_index(drop=True)
        
        # Trading state
        cash = self.initial_capital
        shares = 0
        entry_price = None
        entry_date = None
        
        # Track performance
        equity_curve = []
        trades: List[TradeStats] = []
        daily_returns = []
        
        for idx, row in df.iterrows():
            date = row['ts']
            price = row[price_col]
            pred_return = row[predictions_col]
            actual_return = row.get(actual_return_col, 0)
            
            # Current position value
            position_value = shares * price
            total_equity = cash + position_value
            equity_curve.append({
                'date': date,
                'equity': total_equity,
                'cash': cash,
                'position_value': position_value
            })
            
            # Calculate daily return
            if len(equity_curve) > 1:
                prev_equity = equity_curve[-2]['equity']
                daily_return = (total_equity - prev_equity) / prev_equity
                daily_returns.append(daily_return)
            
            # Trading logic
            if shares == 0:  # Not in position
                # Enter long if prediction exceeds threshold
                if pred_return > threshold:
                    # Calculate position size
                    position_value_target = total_equity * self.position_size_pct
                    
                    # Apply slippage (buy at worse price)
                    entry_price_with_slippage = price * (1 + self.slippage_pct)
                    
                    # Calculate shares
                    shares = int(position_value_target / entry_price_with_slippage)
                    if shares > 0:
                        cost = shares * entry_price_with_slippage
                        commission = shares * self.commission_per_share
                        total_cost = cost + commission
                        
                        if total_cost <= cash:
                            cash -= total_cost
                            entry_price = price  # Track entry at market price
                            entry_date = date
                            
            else:  # In position
                # Exit if prediction turns negative or end of data
                if pred_return < 0 or idx == len(df) - 1:
                    # Apply slippage (sell at worse price)
                    exit_price_with_slippage = price * (1 - self.slippage_pct)
                    
                    proceeds = shares * exit_price_with_slippage
                    commission = shares * self.commission_per_share
                    cash += proceeds - commission
                    
                    # Record trade
                    pnl = (price - entry_price) * shares - (2 * commission) - (shares * entry_price * 2 * self.slippage_pct)
                    pnl_pct = (price - entry_price) / entry_price
                    
                    trades.append(TradeStats(
                        entry_date=str(entry_date),
                        exit_date=str(date),
                        entry_price=entry_price,
                        exit_price=price,
                        shares=shares,
                        pnl=pnl,
                        pnl_pct=pnl_pct,
                        commission=2 * commission,
                        slippage=shares * entry_price * 2 * self.slippage_pct
                    ))
                    
                    shares = 0
                    entry_price = None
                    entry_date = None
        
        # Calculate performance metrics
        equity_df = pd.DataFrame(equity_curve)
        final_equity = equity_df['equity'].iloc[-1]
        
        metrics = self._calculate_metrics(equity_df, daily_returns, trades)
        
        return {
            'metrics': metrics,
            'equity_curve': equity_df.to_dict('records'),
            'trades': [self._trade_to_dict(t) for t in trades],
            'summary': {
                'initial_capital': self.initial_capital,
                'final_equity': final_equity,
                'total_return_pct': ((final_equity - self.initial_capital) / self.initial_capital) * 100,
                'num_trades': len(trades),
                'slippage_pct': self.slippage_pct * 100,
                'commission_per_share': self.commission_per_share
            }
        }
    
    def _calculate_metrics(self, equity_df: pd.DataFrame, daily_returns: List[float], trades: List[TradeStats]) -> Dict:
        """Calculate comprehensive performance metrics."""
        if len(daily_returns) == 0:
            return {}
        
        returns_array = np.array(daily_returns)
        final_equity = equity_df['equity'].iloc[-1]
        initial_equity = equity_df['equity'].iloc[0]
        
        # Total return
        total_return = (final_equity - initial_equity) / initial_equity
        
        # Annualized return (assuming 252 trading days)
        num_days = len(equity_df)
        years = num_days / 252
        annualized_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
        
        # Volatility (annualized)
        volatility = np.std(returns_array) * np.sqrt(252)
        
        # Sharpe ratio
        excess_return = annualized_return - self.risk_free_rate
        sharpe_ratio = excess_return / volatility if volatility > 0 else 0
        
        # Sortino ratio (downside deviation)
        downside_returns = returns_array[returns_array < 0]
        downside_std = np.std(downside_returns) * np.sqrt(252) if len(downside_returns) > 0 else 0
        sortino_ratio = excess_return / downside_std if downside_std > 0 else 0
        
        # Maximum drawdown
        cumulative = (1 + equity_df['equity'].pct_change().fillna(0)).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Trade statistics
        if trades:
            winning_trades = [t for t in trades if t.pnl > 0]
            losing_trades = [t for t in trades if t.pnl < 0]
            
            win_rate = len(winning_trades) / len(trades) if trades else 0
            avg_win = np.mean([t.pnl for t in winning_trades]) if winning_trades else 0
            avg_loss = np.mean([t.pnl for t in losing_trades]) if losing_trades else 0
            
            total_wins = sum(t.pnl for t in winning_trades)
            total_losses = abs(sum(t.pnl for t in losing_trades))
            profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')
            
            total_commission = sum(t.commission for t in trades)
            total_slippage = sum(t.slippage for t in trades)
        else:
            win_rate = 0
            avg_win = 0
            avg_loss = 0
            profit_factor = 0
            total_commission = 0
            total_slippage = 0
        
        return {
            'total_return_pct': total_return * 100,
            'annualized_return_pct': annualized_return * 100,
            'volatility_pct': volatility * 100,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown_pct': max_drawdown * 100,
            'win_rate_pct': win_rate * 100,
            'profit_factor': profit_factor,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'total_commission': total_commission,
            'total_slippage': total_slippage,
            'num_trades': len(trades),
            'num_winning_trades': len(winning_trades) if trades else 0,
            'num_losing_trades': len(losing_trades) if trades else 0
        }
    
    def _trade_to_dict(self, trade: TradeStats) -> Dict:
        """Convert TradeStats to dictionary."""
        return {
            'entry_date': trade.entry_date,
            'exit_date': trade.exit_date,
            'entry_price': trade.entry_price,
            'exit_price': trade.exit_price,
            'shares': trade.shares,
            'pnl': trade.pnl,
            'pnl_pct': trade.pnl_pct * 100,
            'commission': trade.commission,
            'slippage': trade.slippage
        }
