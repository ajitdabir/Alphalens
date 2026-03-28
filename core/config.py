from pathlib import Path

class Config:
    BASE_DIR = Path(__file__).resolve().parents[1]
    SAMPLE_FILE = BASE_DIR / "data" / "sample_input.xlsx"
    DEFAULT_RF_ANNUAL = 0.065
    NSE_EQUITY_URL = "https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv"
