from __future__ import annotations
import logging
import pandas as pd
import yfinance as yf
import asyncio

log = logging.getLogger("app.yahoo")

class YahooClient:
    def __init__(self):
        pass

    async def fetch_earnings(self, symbol: str) -> pd.DataFrame:
        log.info("fetching earnings from yahoo", extra={"symbol": symbol})
        try:
            ticker = yf.Ticker(symbol)
            # Run blocking call in thread
            earnings_dates = await asyncio.to_thread(lambda: ticker.earnings_dates)
            
            if earnings_dates is None or earnings_dates.empty:
                return pd.DataFrame()

            # yfinance returns index as "Earnings Date" (datetime)
            # columns: "EPS Estimate", "Reported EPS", "Surprise(%)"
            df = earnings_dates.reset_index()
            
            # Use specific columns if they exist
            rename_map = {
                "Earnings Date": "announce_date",
                "EPS Estimate": "estimated_eps",
                "Reported EPS": "actual_eps"
            }
            df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
            
            if "announce_date" not in df.columns:
                return pd.DataFrame()

            # Convert timestamp to date
            df["announce_date"] = pd.to_datetime(df["announce_date"]).dt.date
            
            # Fill missing required columns
            if "estimated_eps" not in df.columns: df["estimated_eps"] = None
            if "actual_eps" not in df.columns: df["actual_eps"] = None
            
            df["report_time"] = "unknown"
            df["fiscal_period"] = None
            df["fiscal_end_date"] = None
            
            # Simple validation
            df = df.dropna(subset=["announce_date"])
            
            return df
        except Exception as e:
            log.warning("yahoo fetch failed", extra={"symbol": symbol, "error": str(e)})
            return pd.DataFrame()

def get_yahoo_client() -> YahooClient:
    return YahooClient()
