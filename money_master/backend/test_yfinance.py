import yfinance as yf
import pandas as pd
import logging
import warnings

# Do NOT enable debug mode here so that multithreading is not disabled
# yf.enable_debug_mode()  

# Suppress unrelated warnings
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
warnings.filterwarnings("ignore", category=FutureWarning)

ticker = "VISASTEEL.NS"
print(f"Attempting to download data for {ticker}...")

try:
    # Using a period parameter without a start date
    df = yf.download(ticker, period="8d", interval="1m")
except Exception as e:
    print(f"Exception caught during download: {e}")
    exit()

if df.empty:
    print(f"No data found for ticker {ticker}.")
else:
    print("Data downloaded successfully!")
    print(df.head())

    # Reset the index so that 'Datetime' becomes a column
    df.reset_index(inplace=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    print("\nPrepared data sample:")
    print(df.head())
