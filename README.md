# AlphaLens

AlphaLens is a benchmark-relative attribution analysis app for mutual funds and PMS portfolios. It helps research teams understand performance drivers through stock-level attribution, sector-level attribution, validation checks, mapping diagnostics, and multi-period comparison.

---

## Key Features

- Stock-level attribution (Fund vs Benchmark)
- Sector-level Brinson attribution
- Pre-run validation engine
- Mapping exception report
- Sample input file download
- Top contributors and detractors
- Sector overweight and underweight analysis
- Multi-period comparison (1M, 3M, 6M, 1Y)
- PowerPoint-ready summary output
- Clean, professional UI for RM and IC use

---

## Input Format

Upload an Excel file with two sheets:

### Sheet 1: Fund  
### Sheet 2: Benchmark  

### Required columns:
- Company Name  
- Holding(%)  
- Sector  

---

## How to Run Locally

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
python -m streamlit run app.py
