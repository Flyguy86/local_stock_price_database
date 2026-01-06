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
            
            # Blocking call in thread
            def _get_dates():
                try:
                    return ticker.earnings_dates
                except Exception as exc:
                    log.warning("ticker.earnings_dates failed", extra={"symbol": symbol, "error": str(exc)})
                    return None

            earnings_dates = await asyncio.to_thread(_get_dates)
            
            if earnings_dates is None or earnings_dates.empty:
                log.info("earnings_dates empty/none, trying calendar", extra={"symbol": symbol})
                # Fallback to calendar (single upcoming date)
                try:
                    cal = await asyncio.to_thread(lambda: ticker.calendar)
                    log.info("calendar fetch result", extra={"symbol": symbol, "type": str(type(cal)), "content": str(cal)})
                    
                    dates = []
                    if cal and isinstance(cal, dict):
                        # Try various keys where dates might hide
                        if "Earnings Date" in cal:
                            dates = cal["Earnings Date"]
                        elif "Earnings High" in cal and "Earnings Low" in cal:
                             # Sometimes it returns a range? No, usually specific dates.
                             pass
                        # If values are lists of dates?
                        for v in cal.values():
                            if isinstance(v, list) and len(v) > 0 and hasattr(v[0], "year"):
                                dates = v
                                break

                    if dates:
                        # Ensure list
                        if not isinstance(dates, list):
                            dates = [dates]
                            
                        return pd.DataFrame({
                            "announce_date": [pd.to_datetime(d).date() for d in dates],
                            "report_time": ["unknown"] * len(dates),
                            "fiscal_period": [None] * len(dates),
                            "fiscal_end_date": [None] * len(dates),
                            "actual_eps": [None] * len(dates),
                            "estimated_eps": [None] * len(dates),
                            "symbol": [symbol] * len(dates)
                        })
                except Exception as exc:
                    log.warning("ticker.calendar failed", extra={"symbol": symbol, "error": str(exc)})
                
                return pd.DataFrame()

            # yfinance returns index as "Earnings Date" (datetime)
            # columns: "EPS Estimate", "Reported EPS", "Surprise(%)"
            df = earnings_dates.reset_index()
            
            # Normalize column names (handle potential variations)
            col_map = {
                "Earnings Date": "announce_date",
                "EPS Estimate": "estimated_eps", 
                "Reported EPS": "actual_eps",
                "Surprise(%)": "surprise_pct"
            }
            df = df.rename(columns=col_map)
            
            if "announce_date" not in df.columns:
                log.warning("missing announce_date column", extra={"cols": list(df.columns)})
                return pd.DataFrame()

            # Convert timestamp to date
            df["announce_date"] = pd.to_datetime(df["announce_date"]).dt.date
            
            # Fill missing required columns
            for col in ["estimated_eps", "actual_eps", "fiscal_period", "fiscal_end_date", "report_time"]:
                if col not in df.columns:
                    df[col] = None
            
            # Ensure report_time is string
            df["report_time"] = df["report_time"].fillna("unknown")
            
            # Drop invalid dates
            df = df.dropna(subset=["announce_date"])
            
            return df

        except Exception as e:
            log.warning("yahoo fetch exception", extra={"symbol": symbol, "error": str(e)})
            return pd.DataFrame()

def get_yahoo_client() -> YahooClient:
    return YahooClient()
