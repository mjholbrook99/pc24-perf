
import streamlit as st
from pathlib import Path
import pandas as pd
import numpy as np
import re

st.set_page_config(layout="centered", page_title="PC-24 Takeoff & Landing Calculator")

st.title("PC-24 Takeoff & Landing — Extracted AFM Tables")
st.write("This tool uses OCR-extracted tables from the provided PDF. Select a table, confirm axes, and interpolate values. After selection you can apply temp/wind/slope/surface corrections.")

base_dir = Path("/mnt/data/extracted_pdf/cleaned_csv_tables")
numeric_dir = base_dir / "numeric_only"
csvs = sorted([p for p in numeric_dir.glob("*.csv")])
if not csvs:
    st.error("No numeric tables found. Check /mnt/data/extracted_pdf/cleaned_csv_tables/numeric_only/")
else:
    # Auto-classify by looking at original cluster CSV text for keywords
    classified = {"Takeoff": [], "Landing": [], "Other": []}
    for p in csvs:
        txt = p.read_text(encoding='utf-8')
        if re.search(r'\bTAKE[\s_-]*OFF\b', txt, re.I) or re.search(r'\bGROUND[\s_-]*ROLL\b', txt, re.I):
            classified["Takeoff"].append(p.name)
        elif re.search(r'\bLANDING\b', txt, re.I) or re.search(r'\bAPPROACH\b', txt, re.I) or re.search(r'\bFLARE\b', txt, re.I):
            classified["Landing"].append(p.name)
        else:
            classified["Other"].append(p.name)
    st.sidebar.header("Quick pick (auto-classified)")
    pick_group = st.sidebar.radio("Category", options=["Takeoff", "Landing", "Other"])
    options = classified[pick_group] if pick_group in classified else [p.name for p in csvs]
    selected = st.sidebar.selectbox("Select a numeric table", options=options)
    df_raw = pd.read_csv(numeric_dir / selected, header=None, dtype=str)
    st.subheader("Raw numeric parse preview (top 30 rows)")
    st.dataframe(df_raw.head(30))

    st.markdown("## Interpret table headers")
    cols = df_raw.shape[1]
    rows = df_raw.shape[0]
    st.write(f"Detected rows: {rows}, cols: {cols}")

    x_is = st.radio("What do the ROW headers represent?", options=["Pressure Altitude (ft)", "Outside Air Temp (°C)", "Other"], index=0)
    y_is = st.radio("What do the COLUMN headers represent?", options=["Weight (lbs)", "Flaps/Config", "Other"], index=0)

    st.markdown("Select indices in the table that correspond to the header axes")
    x_header_col = st.number_input("Column index that contains X axis header labels (0-indexed)", min_value=0, max_value=cols-1, value=0)
    y_header_row = st.number_input("Row index that contains Y axis header labels (0-indexed)", min_value=0, max_value=rows-1, value=0)

    # Build numeric table block assumption: data block is rows below y_header_row and cols right of x_header_col
    numeric = df_raw.applymap(lambda v: float(v) if (isinstance(v,str) and v.strip().replace('/','').replace('.','',1).isdigit()) else np.nan)
    x_vals = numeric.iloc[y_header_row+1:, x_header_col].dropna().astype(float).values if y_header_row+1 < rows else np.array([])
    y_vals = numeric.iloc[y_header_row, x_header_col+1:].dropna().astype(float).values if x_header_col+1 < cols else np.array([])
    st.write("Detected X axis (first 10):", x_vals[:10])
    st.write("Detected Y axis (first 10):", y_vals[:10])

    st.markdown("### Interpolation inputs (enter values in same units as axes)")
    x_input = st.number_input("X input value", value=float(x_vals[0]) if len(x_vals)>0 else 0.0)
    y_input = st.number_input("Y input value", value=float(y_vals[0]) if len(y_vals)>0 else 0.0)

    def bilinear_interp(x_vals, y_vals, table, x, y):
        x_vals = np.array(x_vals, dtype=float)
        y_vals = np.array(y_vals, dtype=float)
        t = np.array(table, dtype=float)
        ix = np.searchsorted(x_vals, x) - 1
        iy = np.searchsorted(y_vals, y) - 1
        ix = np.clip(ix, 0, len(x_vals)-2)
        iy = np.clip(iy, 0, len(y_vals)-2)
        x1, x2 = x_vals[ix], x_vals[ix+1]
        y1, y2 = y_vals[iy], y_vals[iy+1]
        Q11 = t[ix, iy]
        Q12 = t[ix, iy+1]
        Q21 = t[ix+1, iy]
        Q22 = t[ix+1, iy+1]
        if x2==x1 or y2==y1:
            return float(np.nanmean([Q11,Q12,Q21,Q22]))
        return (Q11*(x2-x)*(y2-y) + Q21*(x-x1)*(y2-y) + Q12*(x2-x)*(y-y1) + Q22*(x-x1)*(y-y1)) / ((x2-x1)*(y2-y1))

    # prepare table block
    try:
        table_block = numeric.iloc[y_header_row+1:, x_header_col+1:].astype(float).values
        st.write("Numeric block shape:", table_block.shape)
        st.sidebar.header("Environmental Corrections")
        weight = st.sidebar.number_input("Takeoff Weight (lbs)", value=10000.0, step=100.0)
        pressure_alt = st.sidebar.number_input("Pressure Altitude (ft)", value=0, step=100)
        oat = st.sidebar.number_input("OAT (°C)", value=15.0, step=1.0)
        flap_setting = st.sidebar.selectbox("Flap setting (if applicable)", ["0","5","10","15","20"], index=0)
        runway_slope = st.sidebar.number_input("Runway slope (%) (positive uphill)", value=0.0, step=0.1)
        wind = st.sidebar.number_input("Headwind component (kt, positive=headwind)", value=0.0, step=1.0)
        runway_surface = st.sidebar.selectbox("Runway surface", ["Dry","Wet"], index=0)

        if st.button("Interpolate and apply corrections"):
            # choose x/y for interpolation based on user selection mapping
            x_vals_use = x_vals if len(x_vals)>0 else np.arange(table_block.shape[0])
            y_vals_use = y_vals if len(y_vals)>0 else np.arange(table_block.shape[1])
            interp_val = bilinear_interp(x_vals_use, y_vals_use, table_block, x_input, y_input)
            # corrections (approximate)
            std_temp = 15.0 - (pressure_alt/1000.0)*2.0
            temp_diff = oat - std_temp
            temp_factor = 1.0 + 0.03 * (temp_diff/10.0)
            wind_factor = 1.0 - 0.01 * wind
            slope_factor = 1.0 + 0.05 * runway_slope
            surface_factor = 1.15 if runway_surface == "Wet" else 1.0
            final = interp_val * temp_factor / wind_factor * slope_factor * surface_factor
            st.success(f"Base interpolated value: {interp_val:.0f}  — Final corrected distance: {final:.0f} ft")
            st.write({"base": interp_val, "temp_factor": temp_factor, "wind_factor": wind_factor, "slope_factor": slope_factor, "surface_factor": surface_factor, "final": final})
    except Exception as e:
        st.error("Could not build numeric block from selected indices: " + str(e))

st.markdown("---")
st.write("Verify tables and headers. For highest accuracy, replace these CSVs with clean AFM tables or tell me to clean them for you.")

