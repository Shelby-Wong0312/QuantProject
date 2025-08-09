import sqlite3
import pandas as pd

# Check database schema
conn = sqlite3.connect('data/quant_trading.db')
cursor = conn.cursor()

# Get table info
cursor.execute('PRAGMA table_info(daily_data)')
cols = cursor.fetchall()
print('Database columns:')
for col in cols:
    print(f'  - {col[1]}: {col[2]}')

# Get record count
cursor.execute('SELECT COUNT(*) FROM daily_data')
count = cursor.fetchone()[0]
print(f'\nTotal records in database: {count:,}')

# Get sample data
cursor.execute('SELECT * FROM daily_data LIMIT 5')
samples = cursor.fetchall()
print('\nSample records:')
for sample in samples:
    print(f'  {sample}')

conn.close()

# Check parquet file details
print('\n' + '='*50)
print('Parquet file details:')
df = pd.read_parquet('scripts/download/historical_data/daily/EBAY_daily.parquet')
print(f'Shape: {df.shape}')
print(f'Columns: {df.columns.tolist()}')
print(f'Date range: {df["timestamp"].min()} to {df["timestamp"].max()}')
print(f'Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB')
print('\nData statistics:')
print(df.describe())