from __future__ import annotations
import logging
import pandas as pd
import yfinance as yf
import asyncio
import httpx
import io

from typing import AsyncIterator

log = logging.getLogger("app.yahoo")

class YahooClient:
    def __init__(self):
        pass

    async def fetch_bars(self, symbol: str, start: str | None = None, end: str | None = None) -> AsyncIterator[pd.DataFrame]:
        log.info("fetching bars from yahoo", extra={"symbol": symbol, "start": start, "end": end})
        try:
            # yfinance expects YYYY-MM-DD
            start_date = pd.to_datetime(start).strftime("%Y-%m-%d") if start else None
            end_date = pd.to_datetime(end).strftime("%Y-%m-%d") if end else None
            
            # Use '1d' for longer history as '1m' is limited to 30 days
            # If we need 1m data for VIX, yfinance might not have it for long periods.
            interval = "1d" 
            
            def _get_history():
                ticker = yf.Ticker(symbol)
                # If start/end not provided, fetch max? Or some default?
                # poller usually provides start/end.
                if start_date and end_date:
                    return ticker.history(start=start_date, end=end_date, interval=interval)
                elif start_date:
                    return ticker.history(start=start_date, interval=interval)
                else:
                    return ticker.history(period="1y", interval=interval)

            df = await asyncio.to_thread(_get_history)
            
            if df.empty:
                log.warning("yahoo history empty", extra={"symbol": symbol})
                return

            # Reset index to get Date/Datetime as column
            df = df.reset_index()
            
            # yfinance columns: Date (or Datetime), Open, High, Low, Close, Volume, ...
            # Normalize column names
            df.columns = [c.lower() for c in df.columns]
            
            rename_map = {
                "date": "ts",
                "datetime": "ts",
                "stock splits": "splits"
            }
            df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
            
            if "ts" not in df.columns:
                 # Check if index name was "Date" but didn't come out?
                 # If reset_index worked, it should be there.
                 log.warning("missing ts column after reset_index", extra={"cols": list(df.columns)})
                 return

            # Convert ts to UTC
            df["ts"] = pd.to_datetime(df["ts"], utc=True)
            
            # Ensure required columns
            required = ["open", "high", "low", "close", "volume"]
            for col in required:
                if col not in df.columns:
                    df[col] = 0.0 # or NaN
            
            # Add missing schema columns
            if "vwap" not in df.columns:
                df["vwap"] = (df["high"] + df["low"] + df["close"]) / 3
            if "trade_count" not in df.columns:
                df["trade_count"] = 0
                
            # Filter by requested time range if needed (yfinance includes start, excludes end usually)
            if start:
                df = df[df["ts"] >= pd.to_datetime(start, utc=True)]
            if end:
                df = df[df["ts"] < pd.to_datetime(end, utc=True)]
                
            if not df.empty:
                yield df

        except Exception as e:
            log.warning("yahoo history fetch exception", extra={"symbol": symbol, "error": str(e)})

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

                # Fallback: Direct Scrape of Calendar
                log.info("fallback to detailed calendar scrape", extra={"symbol": symbol})
                try:
                    url = f"https://finance.yahoo.com/calendar/earnings/?symbol={symbol}"
                    headers = {
                        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
                    }
                    async with httpx.AsyncClient() as client:
                        resp = await client.get(url, headers=headers, follow_redirects=True, timeout=10.0)
                        if resp.status_code == 200:
                            # Use io.StringIO for pd.read_html
                            dfs = pd.read_html(io.StringIO(resp.text))
                            if dfs:
                                df_cal = dfs[0]
                                log.info("scraped calendar table", extra={"symbol": symbol, "cols": list(df_cal.columns), "rows": len(df_cal)})
                                
                                # Expected cols: Symbol, Company, Earnings Date, EPS Estimate, Reported EPS, Surprise(%)
                                rename_map = {
                                    "Earnings Date": "announce_date",
                                    "EPS Estimate": "estimated_eps",
                                    "Reported EPS": "actual_eps",
                                    "Symbol": "symbol"
                                }
                                df_cal = df_cal.rename(columns={k: v for k, v in rename_map.items() if k in df_cal.columns})
                                
                                if "announce_date" in df_cal.columns:
                                    # Yahoo formatting often includes time "Oct 24, 2023, 4:00 PM EST"
                                    # We just want the date part
                                    df_cal["announce_date"] = pd.to_datetime(df_cal["announce_date"], errors="coerce").dt.date
                                    df_cal = df_cal.dropna(subset=["announce_date"])
                                    
                                    # Check symbol match if column exists
                                    if "symbol" in df_cal.columns:
                                        df_cal = df_cal[df_cal["symbol"] == symbol]
                                    else:
                                        df_cal["symbol"] = symbol

                                    # Fill missing
                                    for col in ["estimated_eps", "actual_eps"]:
                                        if col not in df_cal.columns: df_cal[col] = None
                                        else: df_cal[col] = pd.to_numeric(df_cal[col], errors="coerce")
                                        
                                    df_cal["report_time"] = "unknown"
                                    df_cal["fiscal_period"] = None
                                    df_cal["fiscal_end_date"] = None
                                    
                                    log.info("scraped calendar success", extra={"symbol": symbol, "rows": len(df_cal)})
                                    return df_cal
                except Exception as scrape_exc:
                     log.warning("scraped calendar failed", extra={"symbol": symbol, "error": str(scrape_exc)})
                
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
