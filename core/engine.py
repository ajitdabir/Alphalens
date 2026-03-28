from datetime import timedelta
import numpy as np
import pandas as pd
import yfinance as yf

from .mapping import MasterDataEngine
from .utils import annualize_return_from_daily, annualize_vol_from_daily, max_drawdown, tanh_score, clamp

class MarketDataEngine:
    def fetch_price_panel(self, symbols: list[str], start_dt, end_dt) -> pd.DataFrame:
        symbols = sorted({str(s).strip().upper() for s in symbols if s and s != "UNMAPPED"})
        if not symbols:
            return pd.DataFrame()
        yf_symbols = [f"{s}.NS" for s in symbols]
        start_dt = pd.Timestamp(start_dt) - timedelta(days=10)
        end_dt = pd.Timestamp(end_dt) + timedelta(days=2)
        try:
            data = yf.download(yf_symbols, start=start_dt, end=end_dt, auto_adjust=False, progress=False, threads=True)
            if data is None or data.empty:
                return pd.DataFrame()
            if isinstance(data.columns, pd.MultiIndex):
                if "Close" not in data.columns.get_level_values(0):
                    return pd.DataFrame()
                px = data["Close"].copy()
                px.columns = [str(c).replace(".NS", "") for c in px.columns]
                return px.sort_index().dropna(how="all")
            if "Close" in data.columns and len(yf_symbols) == 1:
                return pd.DataFrame({symbols[0]: data["Close"]}).sort_index().dropna(how="all")
            return pd.DataFrame()
        except Exception:
            return pd.DataFrame()

def build_price_return_table(prices: pd.DataFrame, requested_start, requested_end) -> pd.DataFrame:
    cols = ["Ticker","Requested Start Date","Actual Start Date","Requested End Date","Actual End Date","Start Price","End Price","% Change"]
    rows = []
    if prices.empty:
        return pd.DataFrame(columns=cols)
    requested_start = pd.Timestamp(requested_start)
    requested_end = pd.Timestamp(requested_end)
    for ticker in prices.columns:
        s = prices[ticker].dropna().sort_index()
        if s.empty:
            continue
        start_candidates = s.loc[s.index >= requested_start]
        if start_candidates.empty:
            start_candidates = s.loc[s.index <= requested_start].tail(1)
        end_candidates = s.loc[s.index <= requested_end]
        if end_candidates.empty:
            end_candidates = s.loc[s.index >= requested_end].head(1)
        if start_candidates.empty or end_candidates.empty:
            continue
        actual_start_dt = start_candidates.index[0]
        actual_end_dt = end_candidates.index[-1]
        if actual_start_dt > actual_end_dt:
            continue
        start_price = float(start_candidates.iloc[0]); end_price = float(end_candidates.iloc[-1])
        if pd.isna(start_price) or pd.isna(end_price) or start_price <= 0:
            continue
        rows.append({
            "Ticker": ticker,
            "Requested Start Date": requested_start.date(),
            "Actual Start Date": actual_start_dt.date(),
            "Requested End Date": requested_end.date(),
            "Actual End Date": actual_end_dt.date(),
            "Start Price": start_price,
            "End Price": end_price,
            "% Change": (end_price / start_price) - 1.0,
        })
    return pd.DataFrame(rows, columns=cols)

def compute_security_attribution(fund_df, bench_df, prices, requested_start, requested_end):
    f_agg = fund_df.groupby(["Ticker","Name","Sector"], dropna=False)["Weight"].sum().reset_index()
    b_agg = bench_df.groupby(["Ticker","Name","Sector"], dropna=False)["Weight"].sum().reset_index()
    merged = pd.merge(f_agg, b_agg, on=["Ticker","Sector"], how="outer", suffixes=("_F","_B"))
    merged["Weight_F"] = pd.to_numeric(merged.get("Weight_F", 0), errors="coerce").fillna(0.0)
    merged["Weight_B"] = pd.to_numeric(merged.get("Weight_B", 0), errors="coerce").fillna(0.0)
    merged["Name"] = merged["Name_F"].fillna(merged["Name_B"])
    return_data = build_price_return_table(prices, requested_start, requested_end)
    ticker_returns = dict(zip(return_data["Ticker"], return_data["% Change"]))
    merged["Return"] = merged["Ticker"].map(ticker_returns).fillna(0.0)
    if not return_data.empty:
        merged = merged.merge(return_data, on="Ticker", how="left")
    merged["Fund_Contrib"] = (merged["Weight_F"] / 100.0) * merged["Return"]
    merged["Bench_Contrib"] = (merged["Weight_B"] / 100.0) * merged["Return"]
    merged["Active_Return"] = merged["Fund_Contrib"] - merged["Bench_Contrib"]
    totals = {
        "fund_total": float(merged["Fund_Contrib"].sum()),
        "bench_total": float(merged["Bench_Contrib"].sum()),
        "alpha": float(merged["Fund_Contrib"].sum() - merged["Bench_Contrib"].sum()),
    }
    return merged, totals, return_data

def compute_brinson_sector(merged, bench_total_return):
    sect = merged.groupby("Sector", dropna=False).agg(
        Weight_F=("Weight_F","sum"),
        Weight_B=("Weight_B","sum"),
        Fund_Contrib=("Fund_Contrib","sum"),
        Bench_Contrib=("Bench_Contrib","sum"),
    ).reset_index()
    sect["R_F_sector"] = np.where(sect["Weight_F"] > 0, sect["Fund_Contrib"] / (sect["Weight_F"]/100.0), np.nan)
    sect["R_B_sector"] = np.where(sect["Weight_B"] > 0, sect["Bench_Contrib"] / (sect["Weight_B"]/100.0), np.nan)
    sect["R_F_for_selection"] = sect["R_F_sector"].fillna(sect["R_B_sector"])
    rb = float(bench_total_return)
    sect["Allocation_Effect"] = np.where(sect["Weight_B"] > 0, (sect["Weight_F"] - sect["Weight_B"]) / 100.0 * (sect["R_B_sector"] - rb), 0.0)
    sect["Selection_Effect"] = np.where(sect["Weight_B"] > 0, (sect["Weight_B"] / 100.0) * (sect["R_F_for_selection"] - sect["R_B_sector"]), 0.0)
    sect["Interaction_Effect"] = np.where(sect["Weight_B"] > 0, (sect["Weight_F"] - sect["Weight_B"]) / 100.0 * (sect["R_F_for_selection"] - sect["R_B_sector"]), 0.0)
    sect["OffBenchmark_Effect"] = np.where((sect["Weight_B"] == 0) & (sect["Weight_F"] > 0), (sect["Weight_F"] / 100.0) * (sect["R_F_sector"] - rb), 0.0)
    sect["Explained_Alpha"] = sect["Allocation_Effect"] + sect["Selection_Effect"] + sect["Interaction_Effect"] + sect["OffBenchmark_Effect"]
    sect["Active_Contrib"] = sect["Fund_Contrib"] - sect["Bench_Contrib"]
    sect["Active_Weight"] = sect["Weight_F"] - sect["Weight_B"]
    return sect.sort_values("Explained_Alpha", ascending=False).reset_index(drop=True)

def compute_daily_portfolio_returns(merged, prices):
    if prices.empty or len(prices.index) < 3:
        return pd.Series(dtype=float), pd.Series(dtype=float)
    daily = prices.sort_index().ffill().pct_change().fillna(0.0)
    wF = merged.set_index("Ticker")["Weight_F"].to_dict()
    wB = merged.set_index("Ticker")["Weight_B"].to_dict()
    fund_ret = pd.Series(0.0, index=daily.index)
    bench_ret = pd.Series(0.0, index=daily.index)
    for t, w in wF.items():
        if t in daily.columns:
            fund_ret += (float(w) / 100.0) * daily[t]
    for t, w in wB.items():
        if t in daily.columns:
            bench_ret += (float(w) / 100.0) * daily[t]
    return fund_ret, bench_ret

def compute_coverage_metrics(merged):
    mapped = float(merged.loc[merged["Ticker"] != "UNMAPPED", "Weight_F"].sum())
    priced = float(merged.loc[merged["Actual Start Date"].notna(), "Weight_F"].sum()) if "Actual Start Date" in merged.columns else 0.0
    unmapped = float(merged.loc[merged["Ticker"] == "UNMAPPED", "Weight_F"].sum())
    return {"mapped_weight_pctpts": mapped, "priced_weight_pctpts": priced, "unmapped_weight_pctpts": unmapped, "coverage_ratio": clamp(priced / 100.0, 0.0, 1.0)}

def compute_process_metrics(merged):
    active_share = 0.5 * float((merged["Weight_F"] - merged["Weight_B"]).abs().sum()) / 100.0
    top10_weight_pctpts = float(merged.sort_values("Weight_F", ascending=False)["Weight_F"].head(10).sum())
    return {"active_share": clamp(active_share, 0.0, 1.0), "top10_weight_pctpts": clamp(top10_weight_pctpts, 0.0, 100.0)}

def compute_scorecard(model_fund_return, model_bench_return, alpha, daily_fund, daily_bench, brinson, coverage, process):
    active_daily = (daily_fund - daily_bench).dropna()
    te = annualize_vol_from_daily(active_daily)
    ann_active = annualize_return_from_daily(active_daily)
    ir = (ann_active / te) if te > 0 else 0.0
    dd_fund = max_drawdown(daily_fund)
    dd_bench = max_drawdown(daily_bench)
    rel_dd = dd_fund - dd_bench
    alloc_sum = float(brinson["Allocation_Effect"].sum()) if not brinson.empty else 0.0
    sel_sum = float(brinson["Selection_Effect"].sum()) if not brinson.empty else 0.0
    inter_sum = float(brinson["Interaction_Effect"].sum()) if not brinson.empty else 0.0
    off_sum = float(brinson["OffBenchmark_Effect"].sum()) if not brinson.empty else 0.0
    explained = float(brinson["Explained_Alpha"].sum()) if not brinson.empty else 0.0
    s_alpha = 50 + 25 * tanh_score(alpha, 0.02)
    s_ir = 50 + 25 * tanh_score(ir, 0.5)
    s_dd = 50 + 25 * tanh_score(-(rel_dd), 0.05)
    return_quality = clamp((0.45 * s_alpha + 0.35 * s_ir + 0.20 * s_dd), 0, 100)
    selection_share = abs(sel_sum) / abs(explained) if abs(explained) > 1e-9 else 0.0
    s_sel = clamp(50 + 50 * (selection_share - 0.5), 0, 100)
    s_off = 100 - min(100, abs(off_sum) * 100 * 5)
    attribution_quality = clamp(0.65 * s_sel + 0.35 * s_off, 0, 100)
    s_as = clamp(100 * process["active_share"], 0, 100)
    s_conc = clamp(100 - max(0.0, process["top10_weight_pctpts"] - 40.0) * 2.0, 0, 100)
    process_quality = clamp(0.55 * s_as + 0.45 * s_conc, 0, 100)
    s_cov = clamp(100 * coverage["coverage_ratio"], 0, 100)
    overall = clamp(0.35 * return_quality + 0.25 * attribution_quality + 0.20 * process_quality + 0.20 * s_cov, 0, 100)
    return pd.DataFrame([
        ["Overall Fund Health Score", overall, "score"],
        ["Return Quality Score", return_quality, "score"],
        ["Attribution Quality Score", attribution_quality, "score"],
        ["Process Quality Score", process_quality, "score"],
        ["Data Confidence Score", s_cov, "score"],
        ["Model Fund Return", model_fund_return, "ratio"],
        ["Model Benchmark Return", model_bench_return, "ratio"],
        ["Model Alpha", alpha, "ratio"],
        ["Tracking Error (annualized)", te, "ratio"],
        ["Information Ratio", ir, "number"],
        ["Fund Max Drawdown (model)", dd_fund, "ratio"],
        ["Benchmark Max Drawdown (model)", dd_bench, "ratio"],
        ["Active Share", process["active_share"], "ratio"],
        ["Top-10 Concentration", process["top10_weight_pctpts"], "pctpts"],
        ["Coverage Ratio", coverage["coverage_ratio"], "ratio"],
        ["Mapped Weight", coverage["mapped_weight_pctpts"], "pctpts"],
        ["Priced Weight", coverage["priced_weight_pctpts"], "pctpts"],
        ["Unmapped Weight", coverage["unmapped_weight_pctpts"], "pctpts"],
        ["Allocation Effect (sum)", alloc_sum, "ratio"],
        ["Selection Effect (sum)", sel_sum, "ratio"],
        ["Interaction Effect (sum)", inter_sum, "ratio"],
        ["OffBenchmark Effect (sum)", off_sum, "ratio"],
        ["Explained Alpha (sum)", explained, "ratio"],
    ], columns=["Metric","Value","Format"]), te, ir

def build_diagnostics(alpha, coverage, process, te, ir, brinson, mapped_preview):
    selection = float(brinson["Selection_Effect"].sum()) if not brinson.empty else 0.0
    allocation = float(brinson["Allocation_Effect"].sum()) if not brinson.empty else 0.0
    rows = []
    def add(metric, value, fmt, status, why, interp):
        rows.append({"Metric": metric, "Value": value, "Format": fmt, "Status": status, "Why it matters": why, "Simple interpretation": interp})
    cov = coverage["coverage_ratio"]
    add("Coverage ratio", cov, "ratio", "Good" if cov >= 0.95 else "Watch" if cov >= 0.85 else "Concern", "Higher coverage means more holdings were both mapped and priced.", "Most holdings were mapped and priced." if cov >= 0.95 else "Coverage is usable but not perfect." if cov >= 0.85 else "Low coverage reduces confidence.")
    add("Mapped fund weight", coverage["mapped_weight_pctpts"], "pctpts", "Good" if coverage["mapped_weight_pctpts"] >= 90 else "Watch" if coverage["mapped_weight_pctpts"] >= 75 else "Concern", "This shows how much of the fund resolved to symbols.", f"Mapped examples: {', '.join(mapped_preview[:8])}" if mapped_preview else "No mapped securities were resolved.")
    add("Priced fund weight", coverage["priced_weight_pctpts"], "pctpts", "Good" if coverage["priced_weight_pctpts"] >= 90 else "Watch" if coverage["priced_weight_pctpts"] >= 75 else "Concern", "This shows how much of the fund had usable price history.", "Most mapped names were priced." if coverage["priced_weight_pctpts"] >= 90 else "Some mapped names lacked usable price history." if coverage["priced_weight_pctpts"] >= 75 else "Too much of the fund lacked usable price history.")
    add("Information ratio", ir, "number", "Good" if ir >= 0.3 else "Watch" if ir >= 0 else "Concern", "This shows whether active risk was rewarded.", "Active bets were rewarded." if ir >= 0.3 else "Payoff from active risk is mixed." if ir >= 0 else "Active risk was not rewarded.")
    add("Tracking error", te, "ratio", "Good" if te <= 0.04 else "Watch" if te <= 0.06 else "Concern", "This measures how far the fund moves away from the benchmark.", "Active risk is controlled." if te <= 0.04 else "Active risk is meaningful." if te <= 0.06 else "Active risk is high.")
    add("Selection contribution", selection, "ratio", "Good" if selection > 0 else "Concern", "This shows whether stock picking added value.", "Stock selection added value." if selection > 0 else "Stock selection hurt performance.")
    add("Allocation contribution", allocation, "ratio", "Good" if allocation > 0 else "Watch" if allocation == 0 else "Concern", "This shows whether sector overweights and underweights helped.", "Allocation helped." if allocation > 0 else "Allocation was neutral." if allocation == 0 else "Allocation hurt.")
    flags = []
    if coverage["coverage_ratio"] < 0.90:
        flags.append({"Area": "Coverage", "Message": "Coverage is below 90%. Review company naming and symbol resolution before taking a hard call."})
    if te > 0.06:
        flags.append({"Area": "Active risk", "Message": "Tracking error is high. The portfolio is taking large benchmark-relative bets."})
    if process["top10_weight_pctpts"] > 55:
        flags.append({"Area": "Concentration", "Message": "Top-10 weight is high. Outcomes may be driven by a small number of names."})
    if ir < 0:
        flags.append({"Area": "Payoff", "Message": "Active risk is not yet being rewarded."})
    if alpha < 0 and selection < 0:
        flags.append({"Area": "Stock selection", "Message": "Underperformance is being driven mainly by stock selection."})
    return pd.DataFrame(rows), pd.DataFrame(flags if flags else [{"Area": "None", "Message": "No major red flag stands out from this run."}])

def build_decision_summary(scorecard, brinson, diagnostics_df):
    alpha = float(scorecard.loc[scorecard["Metric"]=="Model Alpha","Value"].iloc[0])
    active_share = float(scorecard.loc[scorecard["Metric"]=="Active Share","Value"].iloc[0])
    coverage = float(scorecard.loc[scorecard["Metric"]=="Coverage Ratio","Value"].iloc[0])
    selection = float(scorecard.loc[scorecard["Metric"]=="Selection Effect (sum)","Value"].iloc[0])
    allocation = float(scorecard.loc[scorecard["Metric"]=="Allocation Effect (sum)","Value"].iloc[0])
    off_benchmark = float(scorecard.loc[scorecard["Metric"]=="OffBenchmark Effect (sum)","Value"].iloc[0])
    if alpha > 0 and selection > abs(allocation):
        verdict = "Selection-driven outperformance"
    elif alpha > 0:
        verdict = "Positive result, but not purely stock-selection led"
    elif alpha < 0 and selection < 0:
        verdict = "Underperformance driven mainly by weak stock selection"
    elif alpha < 0:
        verdict = "Underperformance with mixed drivers"
    else:
        verdict = "Broadly neutral outcome"
    confidence = "Good confidence" if coverage >= 0.95 else "Moderate confidence" if coverage >= 0.85 else "Low confidence"
    top_risk = diagnostics_df.loc[diagnostics_df["Status"]=="Concern","Metric"].iloc[0] if (diagnostics_df["Status"]=="Concern").any() else "No major red flag"
    main_driver = "Selection" if abs(selection) >= abs(allocation) and abs(selection) >= abs(off_benchmark) else "Allocation" if abs(allocation) >= abs(off_benchmark) else "Off-benchmark sectors"
    return pd.DataFrame([
        {"Decision Area":"Overall verdict","Message": verdict},
        {"Decision Area":"Main driver","Message": f"{main_driver} was the biggest driver of relative result."},
        {"Decision Area":"Confidence","Message": f"{confidence} in the output. Coverage is {coverage:.2%}."},
        {"Decision Area":"Portfolio differentiation","Message": f"Active share is {active_share:.2%}, indicating how different the fund is from the benchmark."},
        {"Decision Area":"Top risk","Message": str(top_risk)},
    ])

def build_sector_buckets(brinson):
    if brinson.empty:
        empty = pd.DataFrame(columns=brinson.columns)
        return {"helped": empty, "hurt": empty}
    return {
        "helped": brinson[brinson["Explained_Alpha"] > 0.0025].sort_values("Explained_Alpha", ascending=False),
        "hurt": brinson[brinson["Explained_Alpha"] < -0.0025].sort_values("Explained_Alpha", ascending=True),
    }

def build_stock_buckets(security_attr, top_n=5):
    if security_attr.empty:
        empty = pd.DataFrame(columns=security_attr.columns)
        return {"creators": empty, "detractors": empty}
    return {"creators": security_attr.sort_values("Active_Return", ascending=False).head(top_n), "detractors": security_attr.sort_values("Active_Return", ascending=True).head(top_n)}

def build_plain_english_summary(summary, brinson, security_attr, diagnostics_df, decision_summary):
    alpha = summary["alpha"]; fund_ret = summary["model_fund"]; bench_ret = summary["model_bench"]
    process = summary["process"]; coverage = summary["coverage"]
    sector_helped = brinson.sort_values("Explained_Alpha", ascending=False).head(2)
    sector_hurt = brinson.sort_values("Explained_Alpha", ascending=True).head(2)
    creators = security_attr.sort_values("Active_Return", ascending=False).head(2)
    detractors = security_attr.sort_values("Active_Return", ascending=True).head(2)
    helped_text = ", ".join([f"{r['Sector']} ({r['Explained_Alpha']:.2%})" for _, r in sector_helped.iterrows()]) if not sector_helped.empty else "No strong positive sector driver"
    hurt_text = ", ".join([f"{r['Sector']} ({r['Explained_Alpha']:.2%})" for _, r in sector_hurt.iterrows()]) if not sector_hurt.empty else "No strong negative sector driver"
    creator_text = ", ".join([f"{r['Name']} ({r['Active_Return']:.2%})" for _, r in creators.iterrows()]) if not creators.empty else "No standout stock contributor"
    detractor_text = ", ".join([f"{r['Name']} ({r['Active_Return']:.2%})" for _, r in detractors.iterrows()]) if not detractors.empty else "No standout stock detractor"
    concern_line = diagnostics_df.loc[diagnostics_df["Status"]=="Concern","Simple interpretation"].iloc[0] if (diagnostics_df["Status"]=="Concern").any() else "No major red flag stands out from this run."
    verdict = decision_summary.loc[decision_summary["Decision Area"]=="Overall verdict","Message"].iloc[0]
    return pd.DataFrame([
        {"Section":"Bottom line","Commentary": f"The fund delivered {fund_ret:.2%} versus {bench_ret:.2%} for the benchmark, so relative performance was {alpha:.2%}. Overall verdict: {verdict}."},
        {"Section":"What helped","Commentary": f"The main sector supports came from {helped_text}. The main stock contributors were {creator_text}."},
        {"Section":"What hurt","Commentary": f"The biggest sector drags came from {hurt_text}. The main stock detractors were {detractor_text}."},
        {"Section":"Portfolio shape","Commentary": f"Active share is {process['active_share']:.2%}. Top-10 concentration is {process['top10_weight_pctpts']:.2f}%. Coverage is {coverage['coverage_ratio']:.2%}."},
        {"Section":"What to watch","Commentary": concern_line},
    ])

def period_comparison(symbols, end_dt):
    engine = MarketDataEngine()
    periods = {"1M": 31, "3M": 92, "6M": 183, "1Y": 366}
    rows = []
    for label, days in periods.items():
        start_dt = pd.Timestamp(end_dt) - timedelta(days=days)
        prices = engine.fetch_price_panel(symbols, start_dt, end_dt)
        if prices.empty:
            rows.append({"Period": label, "Coverage": 0.0, "Average Stock Return": np.nan, "Median Stock Return": np.nan})
            continue
        table = build_price_return_table(prices, start_dt, end_dt)
        rows.append({
            "Period": label,
            "Coverage": float(len(table)) / max(len(symbols), 1),
            "Average Stock Return": float(table["% Change"].mean()) if not table.empty else np.nan,
            "Median Stock Return": float(table["% Change"].median()) if not table.empty else np.nan,
        })
    return pd.DataFrame(rows)

def build_ppt_summary(summary, decision_summary, brinson, top_creators, top_detractors):
    verdict = decision_summary.loc[decision_summary["Decision Area"]=="Overall verdict","Message"].iloc[0] if not decision_summary.empty else ""
    helped = ", ".join(brinson.sort_values("Explained_Alpha", ascending=False)["Sector"].head(3).astype(str).tolist()) if not brinson.empty else ""
    hurt = ", ".join(brinson.sort_values("Explained_Alpha", ascending=True)["Sector"].head(3).astype(str).tolist()) if not brinson.empty else ""
    creators = ", ".join(top_creators["Name"].head(3).astype(str).tolist()) if not top_creators.empty else ""
    detractors = ", ".join(top_detractors["Name"].head(3).astype(str).tolist()) if not top_detractors.empty else ""
    return pd.DataFrame([
        {"Section":"Headline","Text": f"Fund return {summary['model_fund']:.2%} vs benchmark {summary['model_bench']:.2%}; alpha {summary['alpha']:.2%}."},
        {"Section":"Verdict","Text": verdict},
        {"Section":"What helped","Text": helped},
        {"Section":"What hurt","Text": hurt},
        {"Section":"Top creators","Text": creators},
        {"Section":"Top detractors","Text": detractors},
        {"Section":"Portfolio shape","Text": f"Active share {summary['process']['active_share']:.2%}; top-10 weight {summary['process']['top10_weight_pctpts']:.2f}%."},
    ])

def run_full_analysis(fund_df, bench_df, start_dt, end_dt, rf_annual=0.065):
    master = MasterDataEngine()
    fund_df = fund_df.copy(); bench_df = bench_df.copy()
    fund_df["Ticker"] = fund_df["Name"].apply(master.resolve)
    bench_df["Ticker"] = bench_df["Name"].apply(master.resolve)
    symbols = sorted(set(fund_df.loc[fund_df["Ticker"] != "UNMAPPED","Ticker"]).union(set(bench_df.loc[bench_df["Ticker"] != "UNMAPPED","Ticker"])))
    prices = MarketDataEngine().fetch_price_panel(symbols, start_dt, end_dt)
    window_text = f"{pd.Timestamp(start_dt).date()} to {pd.Timestamp(end_dt).date()} requested"
    if not prices.empty:
        window_text += f" | {prices.index.min().date()} to {prices.index.max().date()} market data used"
    else:
        window_text += " | no market prices fetched"
    merged, totals, return_data = compute_security_attribution(fund_df, bench_df, prices, start_dt, end_dt)
    model_fund = float(totals["fund_total"]); model_bench = float(totals["bench_total"]); alpha = float(totals["alpha"])
    brinson = compute_brinson_sector(merged, model_bench)
    daily_fund, daily_bench = compute_daily_portfolio_returns(merged, prices)
    coverage = compute_coverage_metrics(merged)
    process = compute_process_metrics(merged)
    scorecard, te, ir = compute_scorecard(model_fund, model_bench, alpha, daily_fund, daily_bench, brinson, coverage, process)
    mapped_preview = merged.loc[merged["Ticker"] != "UNMAPPED","Ticker"].dropna().astype(str).unique().tolist()
    diagnostics_df, flags_df = build_diagnostics(alpha, coverage, process, te, ir, brinson, mapped_preview)
    decision_summary = build_decision_summary(scorecard, brinson, diagnostics_df)
    sector_buckets = build_sector_buckets(brinson)
    stock_buckets = build_stock_buckets(merged)
    summary = {"model_fund": model_fund, "model_bench": model_bench, "alpha": alpha, "coverage": coverage, "process": process}
    plain_english = build_plain_english_summary(summary, brinson, merged, diagnostics_df, decision_summary)
    return {
        "scorecard": scorecard,
        "flags_df": flags_df,
        "diagnostics_df": diagnostics_df,
        "plain_english": plain_english,
        "decision_summary": decision_summary,
        "security_attr": merged.sort_values("Active_Return", ascending=False).reset_index(drop=True),
        "brinson": brinson.reset_index(drop=True),
        "return_data": return_data.reset_index(drop=True),
        "sector_helped": sector_buckets["helped"].reset_index(drop=True),
        "sector_hurt": sector_buckets["hurt"].reset_index(drop=True),
        "top_creators": stock_buckets["creators"].reset_index(drop=True),
        "top_detractors": stock_buckets["detractors"].reset_index(drop=True),
        "window_text": window_text,
        "summary": summary,
        "period_compare": period_comparison(symbols, end_dt),
        "ppt_summary": build_ppt_summary(summary, decision_summary, brinson, stock_buckets["creators"], stock_buckets["detractors"]),
    }
