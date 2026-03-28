
import requests
import pandas as pd
from .config import Config


class NavFetcher:
    def fetch_nav_series(self, amfi_code: str) -> pd.DataFrame:
        url = Config.MFAPI_URL.format(amfi_code=str(amfi_code).strip())
        headers = {"User-Agent": "Mozilla/5.0"}
        r = requests.get(url, headers=headers, timeout=20)
        r.raise_for_status()
        payload = r.json()
        if "data" not in payload:
            raise RuntimeError("NAV API response missing 'data'")

        df = pd.DataFrame(payload["data"])
        df["date"] = pd.to_datetime(df["date"], format="%d-%m-%Y", errors="coerce")
        df["nav"] = pd.to_numeric(df["nav"], errors="coerce")
        df = df.dropna(subset=["date", "nav"]).sort_values("date").reset_index(drop=True)
        return df

    def nav_return_between(self, nav_df: pd.DataFrame, start, end):
        s_df = nav_df[nav_df["date"] <= start]
        e_df = nav_df[nav_df["date"] <= end]
        if s_df.empty or e_df.empty:
            return None, None, None

        s_row = s_df.iloc[-1]
        e_row = e_df.iloc[-1]
        ret = (float(e_row["nav"]) / float(s_row["nav"])) - 1.0
        return ret, s_row["date"].to_pydatetime(), e_row["date"].to_pydatetime()
