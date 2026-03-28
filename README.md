Features added:
- Sample input file download
- Input guide / data dictionary
- Pre-run validation engine
- Mapping exception report
- Top contributors / detractors charts
- Sector overweight / underweight summary
- PowerPoint-ready summary page
- Multiple period comparison (1M, 3M, 6M, 1Y)
- Enhanced UI / UX

Replace these files in your local project:
app.py
requirements.txt
core/config.py
core/utils.py
core/loader.py
core/mapping.py
core/validation.py
core/engine.py
core/report_writer.py
data/sample_input.xlsx

Run:
python -m pip install -r requirements.txt
python -m streamlit run app.py
