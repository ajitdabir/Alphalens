import io
import re
import difflib
import requests
import pandas as pd
from .config import Config

class MasterDataEngine:
    def __init__(self):
        self.name_map = {}
        self._load_master_data()
        self._add_manual_overrides()

    def _normalize(self, text) -> str:
        if pd.isna(text):
            return ""
        text = str(text).upper().strip()
        text = re.sub(r"\s+", " ", text)
        for patt in [r" LIMITED$", r" LTD\.?$", r" PLC$", r" COMPANY$", r" CORPORATION$", r" INDUSTRIES$", r" ENTERPRISES?$"]:
            text = re.sub(patt, "", text)
        text = text.replace("&", " AND ")
        text = re.sub(r"[^A-Z0-9 ]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _load_master_data(self):
        try:
            r = requests.get(Config.NSE_EQUITY_URL, headers={"User-Agent": "Mozilla/5.0"}, timeout=20)
            r.raise_for_status()
            df = pd.read_csv(io.StringIO(r.text))
            df.columns = df.columns.str.strip().str.upper()
            if "NAME OF COMPANY" in df.columns and "SYMBOL" in df.columns:
                for _, row in df.iterrows():
                    clean = self._normalize(row["NAME OF COMPANY"])
                    sym = str(row["SYMBOL"]).strip().upper()
                    if clean and sym:
                        self.name_map[clean] = sym
        except Exception:
            pass

    def _add_manual_overrides(self):
        self.name_map.update({
            "HDFC BANK": "HDFCBANK",
            "ICICI BANK": "ICICIBANK",
            "RELIANCE": "RELIANCE",
            "RELIANCE INDUSTRIES": "RELIANCE",
            "INFOSYS": "INFY",
            "BHARTI AIRTEL": "BHARTIARTL",
            "LARSEN AND TOUBRO": "LT",
            "STATE BANK OF INDIA": "SBIN",
            "AXIS BANK": "AXISBANK",
            "TATA CONSULTANCY SERVICES": "TCS",
            "ITC": "ITC",
            "M AND M": "M&M",
            "MAHINDRA AND MAHINDRA": "M&M",
            "KOTAK MAHINDRA BANK": "KOTAKBANK",
            "SUN PHARMA": "SUNPHARMA",
            "ULTRATECH CEMENT": "ULTRACEMCO",
            "MARUTI SUZUKI": "MARUTI"
        })

    def resolve(self, name: str) -> str:
        clean = self._normalize(name)
        if not clean:
            return "UNMAPPED"
        if clean in self.name_map:
            return self.name_map[clean]
        words = clean.split()
        for k in [clean, " ".join(words[:2]), words[0] if words else ""]:
            if k and k in self.name_map:
                return self.name_map[k]
        candidates = difflib.get_close_matches(clean, list(self.name_map.keys()), n=1, cutoff=0.84)
        if candidates:
            return self.name_map[candidates[0]]
        return "UNMAPPED"

    def mapping_exception_report(self, df: pd.DataFrame) -> pd.DataFrame:
        x = df.copy()
        x["Ticker"] = x["Name"].apply(self.resolve)
        return x[x["Ticker"] == "UNMAPPED"][["Name", "Sector", "Weight"]].reset_index(drop=True)
