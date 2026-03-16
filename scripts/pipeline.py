import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
df_sp500 = pd.read_csv(BASE_DIR / 'data' / 'raw_data' / 'sp500_raw' / 'sp500_master_50years.csv')


