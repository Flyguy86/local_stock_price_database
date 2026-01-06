import yfinance as yf
import pandas as pd
import sys

def test_earnings(symbol):
    print(f"Testing {symbol}...")
    try:
        t = yf.Ticker(symbol)
        dates = t.earnings_dates
        if dates is None:
            print("dates is None")
        elif dates.empty:
            print("dates is empty")
        else:
            print("Columns:", dates.columns)
            print("Index:", dates.index.name)
            print(dates.head())
            
        # Also try other attributes just in case
        print("\nCalendar:")
        print(t.calendar)
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_earnings("AAPL")
