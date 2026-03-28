from datetime import date, timedelta
from pathlib import Path
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from core.config import Config
from core.loader import UploadedDataLoader
from core.mapping import MasterDataEngine
from core.validation import validate_inputs
from core.engine import run_full_analysis
from core.report_writer import build_excel_report

st.set_page_config(page_title="AlphaLens", page_icon="📈", layout="wide", initial_sidebar_state="collapsed")

CSS = '''
<style>
@import url("https://fonts.googleapis.com/css2?family=Lato:wght@300;400;700;900&display=swap");
html, body, [class*="css"] {font-family:"Lato", sans-serif;}
.stApp {background: radial-gradient(circle at top left, #fff4ea 0%, #f7f7f5 40%, #f7f7f5 100%);}
.block-container {max-width:1450px; padding-top:1rem; padding-bottom:2rem;}
header[data-testid="stHeader"] {background:transparent;}
.brand-shell {background:linear-gradient(135deg,#ffffff 0%,#fff6ef 55%,#fffdf9 100%); border:1px solid #F0E4D9; border-radius:28px; padding:32px 36px; box-shadow:0 18px 40px rgba(17,17,17,0.06);}
.brand-header {display:flex; align-items:center; gap:18px;}
.brand-mark {width:48px; height:48px; border-left:8px solid #F58024; border-top:8px solid #F58024; border-right:8px solid #F58024; border-bottom:0; border-radius:4px; box-sizing:border-box;}
.brand-title {font-size:3rem; line-height:0.95; font-weight:900; color:#111; letter-spacing:-0.04em;}
.brand-subtitle {font-size:1.08rem; color:#5F6368; margin-top:0.35rem;}
.hero-copy {font-size:1.02rem; color:#4b5563; margin-top:1rem; max-width:980px;}
.input-shell {background:#FFFFFF; border:1px solid #ECECEC; border-radius:20px; padding:18px 18px 8px 18px; box-shadow:0 8px 22px rgba(17,17,17,0.04); margin-top:16px;}
.input-note {font-size:0.90rem; color:#6b7280; margin-top:8px;}
.metric-card {background:#FFFFFF; border:1px solid #EFEFEF; border-radius:20px; padding:18px; box-shadow:0 8px 18px rgba(17,17,17,0.04); height:100%;}
.metric-card.orange {border-top:4px solid #F58024;}
.metric-label {color:#6b7280; font-size:0.88rem; margin-bottom:0.15rem;}
.metric-value {color:#111111; font-size:1.72rem; font-weight:900;}
.metric-sub {color:#8a8f98; font-size:0.82rem; margin-top:0.35rem;}
.panel {background:#FFFFFF; border:1px solid #EFEFEF; border-radius:20px; padding:18px; box-shadow:0 8px 18px rgba(17,17,17,0.04);}
.panel-title {font-size:1.02rem; font-weight:800; color:#111; margin-bottom:0.55rem;}
.info-banner {background:#FFF7ED; border:1px solid #FCD9BD; color:#9A3412; border-radius:14px; padding:12px 14px; font-size:0.92rem;}
.stTabs [data-baseweb="tab-list"] {gap:8px; margin-top:8px;}
.stTabs [data-baseweb="tab"] {height:42px; padding-left:14px; padding-right:14px; background:#fff; border:1px solid #efefef; border-radius:10px 10px 0 0;}
.stTabs [aria-selected="true"] {color:#F58024 !important; border-bottom:2px solid #F58024 !important;}
div[data-testid="stDownloadButton"] > button, .stButton > button {background:#F58024; color:white; border:1px solid #F58024; border-radius:12px; font-weight:700;}
div[data-testid="stDownloadButton"] > button:hover, .stButton > button:hover {background:#de6d14; border-color:#de6d14; color:white;}
</style>
'''
st.markdown(CSS, unsafe_allow_html=True)

if "results" not in st.session_state:
    st.session_state.results = None
if "validation_df" not in st.session_state:
    st.session_state.validation_df = None
if "mapping_exceptions" not in st.session_state:
    st.session_state.mapping_exceptions = None

def pct(x):
    return "" if pd.isna(x) else f"{x:.2%}"
def num(x):
    return "" if pd.isna(x) else f"{x:,.2f}"
def pctpts(x):
    return "" if pd.isna(x) else f"{x:,.2f}%"

def format_scorecard(df):
    out = df.copy()
    def fmt_row(r):
        if r["Format"] == "ratio": return pct(r["Value"])
        if r["Format"] == "pctpts": return pctpts(r["Value"])
        if r["Format"] == "score": return f"{r['Value']:.1f}"
        return num(r["Value"])
    out["Display"] = out.apply(fmt_row, axis=1)
    return out[["Metric","Display"]]

def format_diagnostics(df):
    out = df.copy()
    def fmt_row(r):
        if r["Format"] == "ratio": return pct(r["Value"])
        if r["Format"] == "pctpts": return pctpts(r["Value"])
        return num(r["Value"])
    out["Display"] = out.apply(fmt_row, axis=1)
    return out[["Metric","Display","Status","Why it matters","Simple interpretation"]]

def format_security(df):
    out = df.copy()
    for c in ["Weight_F","Weight_B"]:
        if c in out.columns: out[c] = out[c].map(pctpts)
    for c in ["Return","Fund_Contrib","Bench_Contrib","Active_Return"]:
        if c in out.columns: out[c] = out[c].map(pct)
    return out

def format_brinson(df):
    out = df.copy()
    for c in ["Weight_F","Weight_B","Active_Weight"]:
        if c in out.columns: out[c] = out[c].map(pctpts)
    for c in ["Fund_Contrib","Bench_Contrib","R_F_sector","R_B_sector","Allocation_Effect","Selection_Effect","Interaction_Effect","OffBenchmark_Effect","Explained_Alpha"]:
        if c in out.columns: out[c] = out[c].map(pct)
    return out

def format_returns(df):
    out = df.copy()
    if "% Change" in out.columns: out["% Change"] = out["% Change"].map(pct)
    for c in ["Start Price","End Price"]:
        if c in out.columns: out[c] = out[c].map(num)
    return out

sample_bytes = Path(Config.SAMPLE_FILE).read_bytes()

st.markdown("<div class='brand-shell'><div class='brand-header'><div class='brand-mark'></div><div><div class='brand-title'>AlphaLens</div><div class='brand-subtitle'>Fund Health & Attribution Engine</div></div></div><div class='hero-copy'>Upload a fund and benchmark workbook, run the engine, and get decision-ready attribution, multi-period comparison, validation diagnostics, mapping checks, and a PowerPoint-ready summary page.</div></div>", unsafe_allow_html=True)

topa, topb = st.columns([0.72, 0.28])
with topa:
    with st.expander("Input guide / data dictionary", expanded=False):
        st.markdown("""
**Workbook structure**
- Sheet 1: `Fund`
- Sheet 2: `Benchmark`

**Required columns**
- `Company Name`
- `Holding(%)`
- `Sector`

**Format rules**
- Weights should ideally sum to ~100%
- Sector names can be free text
- Company names should be close to listed company names

**Common mistakes**
- Wrong sheet names
- Empty sector values
- Repeated holdings for the same stock
- Weights entered as decimals or totals far from 100%
""")
with topb:
    st.download_button("Download sample input file", data=sample_bytes, file_name="alphalens_sample_input.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)

st.markdown("<div class='input-shell'>", unsafe_allow_html=True)
c1, c2, c3, c4, c5 = st.columns([1.4, 0.85, 0.85, 0.55, 0.45])
with c1:
    uploaded_file = st.file_uploader("Upload workbook", type=["xlsx","xls"], help="Workbook must contain two sheets: Fund and Benchmark.")
with c2:
    default_end = date.today()
    default_start = default_end - timedelta(days=365)
    start_date = st.date_input("Start date", value=default_start)
with c3:
    end_date = st.date_input("End date", value=default_end)
with c4:
    rf_rate = st.number_input("Risk-free rate", min_value=0.00, max_value=0.20, value=0.065, step=0.005, format="%.3f")
with c5:
    st.write(""); st.write("")
    run_btn = st.button("Run analysis", type="primary", use_container_width=True)
st.markdown("<div class='input-note'>Use the sample file if you are unsure about input format.</div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

if run_btn:
    if uploaded_file is None:
        st.error("Please upload an Excel workbook first.")
    elif start_date >= end_date:
        st.error("Start date must be before end date.")
    else:
        try:
            with st.spinner("Running AlphaLens…"):
                loader = UploadedDataLoader(uploaded_file)
                fund_df = loader.load("Fund")
                bench_df = loader.load("Benchmark")
                if fund_df.empty or bench_df.empty:
                    st.error("Could not read Fund or Benchmark sheet. Check sheet names and required columns.")
                else:
                    mapper = MasterDataEngine()
                    st.session_state.validation_df = validate_inputs(fund_df, bench_df)
                    st.session_state.mapping_exceptions = pd.concat([
                        mapper.mapping_exception_report(fund_df).assign(Dataset="Fund"),
                        mapper.mapping_exception_report(bench_df).assign(Dataset="Benchmark"),
                    ], ignore_index=True)
                    st.session_state.results = run_full_analysis(
                        fund_df=fund_df,
                        bench_df=bench_df,
                        start_dt=pd.to_datetime(start_date).to_pydatetime(),
                        end_dt=pd.to_datetime(end_date).to_pydatetime(),
                        rf_annual=float(rf_rate),
                    )
        except Exception as e:
            st.exception(e)

results = st.session_state.results
validation_df = st.session_state.validation_df
mapping_exceptions = st.session_state.mapping_exceptions

if results is None:
    l, r = st.columns(2)
    with l:
        st.markdown("<div class='panel'><div class='panel-title'>What this app does</div>This app explains how a fund generated performance. It compares the uploaded fund against its benchmark, breaks active return into stock-level and sector-level components, and creates a strong decision-oriented summary.</div>", unsafe_allow_html=True)
    with r:
        st.markdown("<div class='panel'><div class='panel-title'>What is new</div>Validation engine, mapping exception report, multiple period comparison, PowerPoint-ready summary page, and downloadable sample file are now built into the workflow.</div>", unsafe_allow_html=True)
else:
    summary = results["summary"]
    scorecard = results["scorecard"]
    security_attr = results["security_attr"]
    brinson = results["brinson"]
    flags_df = results["flags_df"]
    diagnostics_df = results["diagnostics_df"]
    plain_english = results["plain_english"]
    return_data = results["return_data"]
    window_text = results["window_text"]
    decision_summary = results["decision_summary"]
    period_compare = results["period_compare"]
    ppt_summary = results["ppt_summary"]

    if return_data.empty:
        st.markdown("<div class='info-banner'>No price history could be built for the selected period. Review Validation and Diagnostics for mapping and priced-weight coverage.</div>", unsafe_allow_html=True)

    cards = [
        ("Fund return", pct(summary["model_fund"]), "Modelled return using price history"),
        ("Benchmark return", pct(summary["model_bench"]), "Modelled benchmark return over the same window"),
        ("Alpha", pct(summary["alpha"]), "Fund minus benchmark"),
        ("Fund health score", f"{float(scorecard.loc[scorecard['Metric']=='Overall Fund Health Score','Value'].iloc[0]):.1f}", "Composite score out of 100"),
        ("Active share", pct(summary["process"]["active_share"]), "How different the fund is from the benchmark"),
        ("Top-10 weight", pctpts(summary["process"]["top10_weight_pctpts"]), "Portfolio concentration in top 10 names"),
    ]
    cols = st.columns(6)
    for col, (label, value, sub) in zip(cols, cards):
        col.markdown(f"<div class='metric-card orange'><div class='metric-label'>{label}</div><div class='metric-value'>{value}</div><div class='metric-sub'>{sub}</div></div>", unsafe_allow_html=True)

    st.caption(f"Window used: {window_text}")

    t1, t2, t3, t4, t5, t6, t7, t8, t9 = st.tabs(["Summary", "Validation", "Security attribution", "Sector attribution", "Contributors", "Period compare", "PPT summary", "Returns data", "Download"])

    with t1:
        a, b = st.columns([0.58, 0.42])
        with a:
            st.markdown("<div class='panel'><div class='panel-title'>Decision summary</div></div>", unsafe_allow_html=True)
            st.dataframe(decision_summary, use_container_width=True, hide_index=True)
            st.markdown("<div class='panel' style='margin-top:12px;'><div class='panel-title'>Plain-English interpretation</div></div>", unsafe_allow_html=True)
            st.dataframe(plain_english, use_container_width=True, hide_index=True)
        with b:
            st.markdown("<div class='panel'><div class='panel-title'>Scorecard</div></div>", unsafe_allow_html=True)
            st.dataframe(format_scorecard(scorecard), use_container_width=True, hide_index=True)

    with t2:
        left, right = st.columns([0.58, 0.42])
        with left:
            st.markdown("<div class='panel'><div class='panel-title'>Pre-run validation engine</div></div>", unsafe_allow_html=True)
            st.dataframe(validation_df, use_container_width=True, hide_index=True)
        with right:
            st.markdown("<div class='panel'><div class='panel-title'>Mapping exception report</div></div>", unsafe_allow_html=True)
            st.dataframe(mapping_exceptions, use_container_width=True, hide_index=True)
            if mapping_exceptions is not None and not mapping_exceptions.empty:
                st.download_button("Download mapping exceptions CSV", mapping_exceptions.to_csv(index=False).encode("utf-8"), "mapping_exceptions.csv", "text/csv", use_container_width=True)

    with t3:
        sec_cols = ["Ticker","Name","Sector","Weight_F","Weight_B","Return","Fund_Contrib","Bench_Contrib","Active_Return"]
        st.dataframe(format_security(security_attr[sec_cols].copy()), use_container_width=True, hide_index=True)

    with t4:
        bri_cols = ["Sector","Weight_F","Weight_B","Active_Weight","Fund_Contrib","Bench_Contrib","R_F_sector","R_B_sector","Allocation_Effect","Selection_Effect","Interaction_Effect","OffBenchmark_Effect","Explained_Alpha"]
        st.dataframe(format_brinson(brinson[bri_cols].copy()), use_container_width=True, hide_index=True)
        ow, uw = st.columns(2)
        with ow:
            st.markdown("<div class='panel'><div class='panel-title'>Top sector overweights</div></div>", unsafe_allow_html=True)
            over = brinson.sort_values("Active_Weight", ascending=False)[["Sector","Active_Weight","Explained_Alpha"]].head(10).copy()
            st.dataframe(format_brinson(over), use_container_width=True, hide_index=True)
        with uw:
            st.markdown("<div class='panel'><div class='panel-title'>Top sector underweights</div></div>", unsafe_allow_html=True)
            under = brinson.sort_values("Active_Weight", ascending=True)[["Sector","Active_Weight","Explained_Alpha"]].head(10).copy()
            st.dataframe(format_brinson(under), use_container_width=True, hide_index=True)

    with t5:
        left, right = st.columns(2)
        top_creators = results["top_creators"]; top_detractors = results["top_detractors"]
        with left:
            st.markdown("<div class='panel'><div class='panel-title'>Top contributors</div></div>", unsafe_allow_html=True)
            chart = top_creators[["Name","Active_Return"]].copy()
            fig, ax = plt.subplots(figsize=(7,4))
            ax.barh(chart["Name"][::-1], chart["Active_Return"][::-1])
            ax.set_title("Top contributors", loc="left", fontsize=13, fontweight="bold")
            ax.axvline(0, linewidth=1)
            plt.tight_layout()
            st.pyplot(fig, clear_figure=True)
            st.dataframe(format_security(top_creators[["Name","Sector","Active_Return","Weight_F"]].copy()), use_container_width=True, hide_index=True)
        with right:
            st.markdown("<div class='panel'><div class='panel-title'>Top detractors</div></div>", unsafe_allow_html=True)
            chart = top_detractors[["Name","Active_Return"]].copy()
            fig, ax = plt.subplots(figsize=(7,4))
            ax.barh(chart["Name"][::-1], chart["Active_Return"][::-1])
            ax.set_title("Top detractors", loc="left", fontsize=13, fontweight="bold")
            ax.axvline(0, linewidth=1)
            plt.tight_layout()
            st.pyplot(fig, clear_figure=True)
            st.dataframe(format_security(top_detractors[["Name","Sector","Active_Return","Weight_F"]].copy()), use_container_width=True, hide_index=True)

    with t6:
        pc = period_compare.copy()
        if not pc.empty:
            pc["Coverage"] = pc["Coverage"].map(pct)
            for c in ["Average Stock Return","Median Stock Return"]:
                pc[c] = pc[c].map(pct)
        st.markdown("<div class='panel'><div class='panel-title'>Multiple period comparison</div></div>", unsafe_allow_html=True)
        st.dataframe(pc, use_container_width=True, hide_index=True)
        fig, ax = plt.subplots(figsize=(8,4.5))
        tmp = results["period_compare"].copy()
        if not tmp.empty:
            ax.plot(tmp["Period"], tmp["Average Stock Return"], marker="o", label="Average")
            ax.plot(tmp["Period"], tmp["Median Stock Return"], marker="o", label="Median")
            ax.set_title("Average vs median stock return by period", loc="left", fontsize=13, fontweight="bold")
            ax.legend()
            plt.tight_layout()
            st.pyplot(fig, clear_figure=True)

    with t7:
        st.markdown("<div class='panel'><div class='panel-title'>PowerPoint-ready summary page</div></div>", unsafe_allow_html=True)
        st.dataframe(ppt_summary, use_container_width=True, hide_index=True)
        ppt_text = "\n".join([f"{r['Section']}: {r['Text']}" for _, r in ppt_summary.iterrows()])
        st.download_button("Download PPT-ready summary text", ppt_text.encode("utf-8"), "ppt_ready_summary.txt", "text/plain", use_container_width=True)

    with t8:
        st.dataframe(format_returns(return_data.copy()), use_container_width=True, hide_index=True)
        st.markdown("<div class='panel' style='margin-top:12px;'><div class='panel-title'>Diagnostics</div></div>", unsafe_allow_html=True)
        st.dataframe(format_diagnostics(diagnostics_df), use_container_width=True, hide_index=True)

    with t9:
        excel_bytes = build_excel_report(
            security_attr=security_attr,
            brinson=brinson,
            scorecard=scorecard,
            flags_df=flags_df,
            diagnostics_df=diagnostics_df,
            plain_english=plain_english,
            decision_summary=decision_summary,
            return_data=return_data,
            sector_helped=results["sector_helped"],
            sector_hurt=results["sector_hurt"],
            top_creators=results["top_creators"],
            top_detractors=results["top_detractors"],
            validation_df=validation_df,
            mapping_exceptions=mapping_exceptions,
            period_compare=results["period_compare"],
            ppt_summary=results["ppt_summary"],
        )
        st.download_button("Download Excel report", data=excel_bytes, file_name="alphalens_report.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)
