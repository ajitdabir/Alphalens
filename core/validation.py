import pandas as pd

def validate_inputs(fund_df: pd.DataFrame, bench_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    def add(group, check, status, detail):
        rows.append({"Dataset": group, "Check": check, "Status": status, "Detail": detail})
    for label, df in [("Fund", fund_df), ("Benchmark", bench_df)]:
        total = float(df["Weight"].sum()) if not df.empty else 0.0
        add(label, "Weight total", "Pass" if abs(total - 100.0) <= 2.0 else "Watch", f"Total weight = {total:.2f}%")
        dup_count = int(df["Name"].duplicated().sum())
        add(label, "Duplicate securities", "Pass" if dup_count == 0 else "Watch", f"{dup_count} duplicate rows found")
        blank_sector = int((df["Sector"].astype(str).str.strip() == "").sum())
        add(label, "Blank sectors", "Pass" if blank_sector == 0 else "Watch", f"{blank_sector} blank sector rows")
        non_positive = int((df["Weight"] <= 0).sum())
        add(label, "Non-positive weights", "Pass" if non_positive == 0 else "Concern", f"{non_positive} non-positive weight rows")
    return pd.DataFrame(rows)
