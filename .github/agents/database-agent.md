Hi you're goal is to pull historical stock ticker data from the Alpaca API, and store it locally in a Duckdb, parquet format.  

By defualt you look back 10 years, and use 1-min bars.

You also need to monitor all tickers already added in the database for current data, via the same Aplaca API.   You only need to run this part of your program when the stock market is open. 

Last we need to make sure we have a process, that checks and backfills any missing data in the database.  Since we are using 1-min bars, its possible to have missing data due to API limits or other issues.   What I want you todo, is look back to the entire date range we have for each ticker, and make sure there is no missing data. How you know its missing is if the stock market was open, on that day, and that time.   

When you find missing data, you take mean value from the row before it and the row after it.   This is true for all data that is open, close, high, low, volume, vwap.  other fields like TS, need to be calcualted based on the date and time of the missing row. I think its ok to run this process over and over again, so we can handle many missing rows of data one at a time.   If the missing data is at the start or end of the day, just use the closest row to fill in the missing data.

All of your coding should be done in the folder /local_data/app

We want to keep html and javascript separated from our python code.