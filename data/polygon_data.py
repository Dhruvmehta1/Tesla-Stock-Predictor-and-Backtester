import requests
import pandas as pd

def get_polygon_data(ticker, start_date, end_date, api_key):
    """Fetch historical stock data from Polygon.io API"""
    url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{start_date}/{end_date}?apiKey={api_key}"
    response = requests.get(url)

    if response.status_code != 200:
        raise RuntimeError(f"Failed to download data: {response.status_code}")

    data = response.json()
    if data['resultsCount'] == 0:
        raise RuntimeError(f"No data returned for {ticker}")

    df = pd.DataFrame(data['results'])
    df.rename(columns={'o': 'Open', 'h': 'High', 'l': 'Low', 'c': 'Close', 'v': 'Volume', 't': 'Date'}, inplace=True)
    df['Date'] = pd.to_datetime(df['Date'], unit='ms')
    df.set_index('Date', inplace=True)
    df.sort_index(inplace=True)
    return df
