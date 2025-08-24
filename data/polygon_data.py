import requests
import pandas as pd
import time
from datetime import datetime

def get_polygon_data(ticker, start_date, end_date, api_key):
    """
    Fetch historical stock data from Polygon.io API with robust error handling

    Args:
        ticker: Stock ticker symbol (e.g., 'TSLA')
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format (defaults to today if None)
        api_key: Polygon.io API key

    Returns:
        DataFrame with OHLCV data
    """
    # Validate inputs
    if not ticker or not start_date or not api_key:
        raise ValueError("Ticker, start_date, and api_key are required")

    # Use current date if end_date is None
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")

    # Build URL with proper parameters
    url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{start_date}/{end_date}?apiKey={api_key}"

    # Make request with retry logic
    max_retries = 3
    for attempt in range(max_retries):
        try:
            print(f"Fetching data for {ticker} from {start_date} to {end_date}...")
            response = requests.get(url, timeout=10)

            # Handle specific error codes
            if response.status_code == 400:
                error_msg = f"Bad request (400): Check your date format or ticker symbol ({ticker})"
                print(error_msg)
                # Try to get more detailed error info from response
                try:
                    error_details = response.json()
                    print(f"API error details: {error_details}")
                except:
                    pass
                raise ValueError(error_msg)

            elif response.status_code == 401:
                raise ValueError(f"Authentication error (401): Invalid API key")

            elif response.status_code == 403:
                raise ValueError(f"Forbidden (403): Check API key permissions")

            elif response.status_code == 429:
                if attempt < max_retries - 1:
                    wait_time = (2 ** attempt) + 1  # Exponential backoff
                    print(f"Rate limit hit. Waiting {wait_time} seconds...")
                    time.sleep(wait_time)
                    continue
                else:
                    raise ValueError(f"Rate limit exceeded (429): Too many requests")

            elif response.status_code != 200:
                raise RuntimeError(f"Failed to download data: HTTP {response.status_code}")

            # Parse JSON response
            data = response.json()

            # Check if we have results
            if 'resultsCount' not in data or data['resultsCount'] == 0:
                print(f"Warning: No data returned for {ticker} in the specified date range")
                # Return empty DataFrame with proper columns
                return pd.DataFrame(columns=['Open', 'High', 'Low', 'Close', 'Volume'])

            # Process data
            df = pd.DataFrame(data['results'])
            df.rename(columns={'o': 'Open', 'h': 'High', 'l': 'Low', 'c': 'Close', 'v': 'Volume', 't': 'Date'}, inplace=True)
            df['Date'] = pd.to_datetime(df['Date'], unit='ms')
            df.set_index('Date', inplace=True)
            df.sort_index(inplace=True)

            print(f"Successfully fetched {len(df)} days of data for {ticker}")
            return df

        except requests.exceptions.RequestException as e:
            if attempt < max_retries - 1:
                wait_time = (2 ** attempt) + 1
                print(f"Network error: {e}. Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                print(f"Failed to fetch data after {max_retries} attempts: {e}")
                raise
