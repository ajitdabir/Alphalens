from io import BytesIO
import pandas as pd

def build_excel_report(
    security_attr: pd.DataFrame,
    brinson: pd.DataFrame,
    scorecard: pd.DataFrame,
    flags_df: pd.DataFrame,
    diagnostics_df: pd.DataFrame,
    plain_english: pd.DataFrame,
    decision_summary: pd.DataFrame,
    return_data: pd.DataFrame,
    sector_helped: pd.DataFrame,
    sector_hurt: pd.DataFrame,
    top_creators: pd.DataFrame,
    top_detractors: pd.DataFrame,
    validation_df: pd.DataFrame,
    mapping_exceptions: pd.DataFrame,
    period_compare: pd.DataFrame,
    ppt_summary: pd.DataFrame,
):
    output = BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        decision_summary.to_excel(writer, sheet_name="Decision_Summary", index=False)
        plain_english.to_excel(writer, sheet_name="Plain_English", index=False)
        diagnostics_df.to_excel(writer, sheet_name="Diagnostics", index=False)
        flags_df.to_excel(writer, sheet_name="Red_Flags", index=False)
        scorecard.to_excel(writer, sheet_name="Scorecard", index=False)
        return_data.to_excel(writer, sheet_name="Returns_Data", index=False)
        security_attr.to_excel(writer, sheet_name="Attribution_Security", index=False)
        brinson.to_excel(writer, sheet_name="Attribution_Brinson", index=False)
        sector_helped.to_excel(writer, sheet_name="Sector_Helped", index=False)
        sector_hurt.to_excel(writer, sheet_name="Sector_Hurt", index=False)
        top_creators.to_excel(writer, sheet_name="Top_Creators", index=False)
        top_detractors.to_excel(writer, sheet_name="Top_Detractors", index=False)
        validation_df.to_excel(writer, sheet_name="Validation", index=False)
        mapping_exceptions.to_excel(writer, sheet_name="Mapping_Exceptions", index=False)
        period_compare.to_excel(writer, sheet_name="Period_Compare", index=False)
        ppt_summary.to_excel(writer, sheet_name="PPT_Summary", index=False)
    output.seek(0)
    return output.getvalue()
