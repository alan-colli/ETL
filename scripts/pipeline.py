import pandas as pd
from pathlib import Path

import os
from dotenv import load_dotenv
from sqlalchemy import create_engine

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_PATH = BASE_DIR / 'data' / 'raw_data' / 'sp500_raw' / 'sp500_master_50years.csv'

#EXCTRACT

def extract() -> pd.DataFrame:
    df = pd.read_csv(RAW_PATH)
    print(f"[EXTRACT] {df.shape[0]:,} rows | {df.shape[1]} columns")
    return df

#CLEAN

def clean(df: pd.DataFrame) -> pd.DataFrame:

    # 1. RENAME COLUMNS USING snake_case
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(' ', '_')
    )

    # 2. CONVERT TIME TO DATE (REMOVE TIMEZONE)
    df['date'] = pd.to_datetime(df['date'], utc=True).dt.date

    # 3. REMOVE DUPLICATES
    before = len(df)
    df = df.drop_duplicates(subset=['date', 'ticker'])
    print(f"[CLEAN] DUPLICATES REMOVED: {before - len(df)}")

    # 4. REMOVE NULLS IN ESSENTIAL COLUMNSq
    essential = ['date', 'ticker', 'open', 'high', 'low', 'close', 'volume']
    before = len(df)
    df = df.dropna(subset=essential)
    print(f"[CLEAN] LINES REMOVED DUE TO NULLS IN ESSENTIAL COLUMNS: {before - len(df)}")

    # 5. CHECK FOR LOGICAL INCONSISTENCIES (HIGH < LOW)
    invalid_prices = df[df['high'] < df['low']]
    if not invalid_prices.empty:
        print(f"[CLEAN] ATTENTION: {len(invalid_prices)} LINES WITH HIGH < LOW:")
        df = df[df['high'] >= df['low']]

    # 6. IVESTIGATE OUTLIERS IN VOLUME (ABOVE 2B)
    volume_threshold = 2_000_000_000
    outliers = df[df['volume'] > volume_threshold]
    if not outliers.empty:
        print(f"[CLEAN] Volumns above 2BI:")
        print(outliers[['date', 'ticker', 'volume']].to_string())

    print(f"[CLEAN] FINAL SHAPE: {df.shape[0]:,} LINES | {df.shape[1]} COLUMNS")
    return df

#TRANSFORM

def transform(df: pd.DataFrame) -> pd.DataFrame:
 
    df = df.sort_values(['ticker', 'date']).reset_index(drop=True)
 
    # 1. CUMULATIVE RETURN
    df['cumulative_return'] = (
        df.groupby('ticker')['daily_return']
        .transform(lambda x: (1 + x.fillna(0)).cumprod())
    )
 
    # 2. DRAWDOWN
    df['rolling_max'] = df.groupby('ticker')['close'].transform('cummax')
    df['drawdown'] = (df['close'] - df['rolling_max']) / df['rolling_max']
    df = df.drop(columns=['rolling_max'])
 
    # 3. PRICE VS SMA200
    df['price_vs_sma200'] = pd.NA
    has_sma = df['sma200'].notna()
    df.loc[has_sma & (df['close'] > df['sma200']), 'price_vs_sma200'] = 'above'
    df.loc[has_sma & (df['close'] <= df['sma200']), 'price_vs_sma200'] = 'below'
    df.loc[~has_sma, 'price_vs_sma200'] = 'no_data'
 
    # 4. SHARPE PROXY
    df['sharpe_proxy'] = df['daily_return'] / df['volatility'].replace(0, pd.NA)
 
    # 5. CRISIS PERIODS
    date_ts = pd.to_datetime(df['date'])
 
    crisis_bins = pd.to_datetime([
        '1900-01-01',
        '1973-10-01', '1974-12-31',
        '1987-10-01', '1987-12-31',
        '2000-03-01', '2002-10-31',
        '2007-10-01', '2009-06-30',
        '2020-02-01', '2020-12-31',
        '2100-01-01'
    ])
 
    crisis_labels = [
        'normal',
        'oil_crisis_1973', 'normal',
        'black_monday_1987', 'normal',
        'dotcom_bubble_2000', 'normal',
        'financial_crisis_2008', 'normal',
        'covid_2020', 'normal'
    ]
 
    df['crisis_period'] = pd.cut(
        date_ts,
        bins=crisis_bins,
        labels=crisis_labels,
        ordered=False
    ).astype(str)
 
    print(f"[TRANSFORM] Metrics added: cumulative_return, drawdown, price_vs_sma200, sharpe_proxy, crisis_period")
    print(f"[TRANSFORM] {df.shape[0]:,} rows | {df.shape[1]} columns")
    return df

#LOAD

load_dotenv()

def load(df: pd.DataFrame) -> None:
    engine = create_engine(
        f"postgresql+psycopg2://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}"
        f"@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
    )

    df.to_sql(
        name='sp500_prices',
        con=engine,
        if_exists='replace',
        index=False,
        chunksize=5000,
        method='multi'
    )

    print(f"[LOAD] Data loaded into table sp500_prices")

#SAVE

def save(df: pd.DataFrame) -> None:
    out_path = BASE_DIR / 'data' / 'processed_data' / 'sp500_clean.csv'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"[SAVE] FILE SAVED IN: {out_path}")

def main():
    df = extract()
    df = clean(df)
    df = transform(df)
    save(df)
    load(df)

if __name__ == '__main__':
    main()