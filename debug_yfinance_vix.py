import yfinance as yf

def check_vix():
    print("Checking VIX...")
    ticker = yf.Ticker("VIX")
    hist = ticker.history(period="1mo")
    print(f"VIX history empty: {hist.empty}")
    if not hist.empty:
        print(hist.head())

    print("Checking ^VIX...")
    ticker = yf.Ticker("^VIX")
    hist = ticker.history(period="1mo")
    print(f"^VIX history empty: {hist.empty}")
    if not hist.empty:
        print(hist.head())

if __name__ == "__main__":
    check_vix()
