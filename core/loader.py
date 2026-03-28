import pandas as pd

class UploadedDataLoader:
    def __init__(self, uploaded_file):
        self.uploaded_file = uploaded_file

    def _resolve_sheet_case_insensitive(self, xls: pd.ExcelFile, desired: str):
        desired = desired.strip().upper()
        for s in xls.sheet_names:
            if str(s).strip().upper() == desired:
                return s
        return None

    def load(self, sheet_name: str) -> pd.DataFrame:
        if self.uploaded_file is None:
            return pd.DataFrame()
        xls = pd.ExcelFile(self.uploaded_file)
        actual_sheet = self._resolve_sheet_case_insensitive(xls, sheet_name)
        if actual_sheet is None:
            return pd.DataFrame()
        df = pd.read_excel(self.uploaded_file, sheet_name=actual_sheet)
        rename = {}
        for c in df.columns:
            cu = str(c).strip().upper()
            if "COMPANY" in cu and "NAME" in cu:
                rename[c] = "Name"
            elif "HOLDING" in cu and "%" in cu:
                rename[c] = "Weight"
            elif "SECTOR" in cu:
                rename[c] = "Sector"
        df = df.rename(columns=rename)
        if "Name" not in df.columns or "Weight" not in df.columns:
            return pd.DataFrame()
        if "Sector" not in df.columns:
            df["Sector"] = "Unclassified"
        df["Weight"] = pd.to_numeric(df["Weight"], errors="coerce").fillna(0.0)
        df = df[df["Weight"] > 0].copy()
        total_weight = float(df["Weight"].sum())
        if total_weight <= 1.5:
            df["Weight"] = df["Weight"] * 100.0
            total_weight = float(df["Weight"].sum())
        if total_weight > 0 and abs(total_weight - 100.0) > 5.0:
            df["Weight"] = (df["Weight"] / total_weight) * 100.0
        df["Name"] = df["Name"].astype(str).str.strip()
        df["Sector"] = df["Sector"].fillna("Unclassified").astype(str)
        return df.reset_index(drop=True)
