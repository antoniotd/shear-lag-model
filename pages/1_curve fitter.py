import streamlit as st
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg') # 1. FORCE HEADLESS BACKEND TO GUARANTEE RENDERING
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from sklearn.metrics import r2_score

# 1. Page Configuration
st.set_page_config(page_title="Interfacial Curve Fitter", layout="wide")

# Custom Navigation Sidebar Replacement
st.sidebar.page_link("app.py", label="Simulator", icon="🔬")
st.sidebar.page_link("pages/1_curve fitter.py", label="Curve Fitter", icon="🎯")
st.sidebar.write("---")

st.title("Interfacial Curve Fitter")
st.markdown("""
Upload pull-out data (**Force vs. Displacement**) from your laboratory tests. 
The algorithm will isolate the linear elastic phase and mathematically extract the true 
interfacial shear stiffness ($k$) of your composite interface.
""")

st.write("---")

# 2. Sidebar Parameters (Known constants from your specimens)
st.sidebar.header("📋 Specimen Constants")
r_f_micron = st.sidebar.number_input("Fiber Radius (r_f) [μm]", min_value=1.0, max_value=50.0, value=3.5, step=0.1)
L = st.sidebar.number_input("Embedment Length (L) [mm]", min_value=0.5, max_value=20.0, value=5.0, step=0.1)
E_f_gpa = st.sidebar.number_input("Fiber Elastic Modulus (E_f) [GPa]", min_value=50, max_value=500, value=230, step=5)

# Convert units immediately to mm and MPa
r_f = r_f_micron / 1000.0   # μm to mm
E_f = E_f_gpa * 1000.0       # GPa to MPa
A_f = np.pi * (r_f ** 2)     # Fiber Area (mm^2)

# 3. Data Input Section
st.subheader("📂 Upload Laboratory Pull-Out Data")

# Initialize dataframe persistent storage in session state
if "df" not in st.session_state:
    st.session_state.df = None

col1, col2 = st.columns([2, 1])

with col2:
    st.write("**No lab file?**")
    csv_template = (
        "Displacement_mm,Force_N\n0.000,0.000\n0.002,0.007\n0.004,0.015\n"
        "0.006,0.022\n0.008,0.030\n0.010,0.037\n0.012,0.045\n0.014,0.052\n"
        "0.016,0.059\n0.018,0.067\n0.020,0.074\n0.022,0.081\n0.024,0.088\n"
        "0.026,0.095\n0.028,0.101\n0.030,0.107\n0.032,0.112\n0.034,0.116\n"
        "0.036,0.119\n0.038,0.121\n0.040,0.122"
    )
    st.download_button(
        label="📥 Download Sample Lab CSV",
        data=csv_template,
        file_name="lab_pullout_test.csv",
        mime="text/csv",
        use_container_width=True
    )

with col1:
    uploaded_file = st.file_uploader("Upload UTM Data File (CSV format)", type=["csv"])

# Read and save file data to state cache permanently upon upload
if uploaded_file is not None:
    st.session_state.df = pd.read_csv(uploaded_file)

# Fetch current active dataset from persistent memory
df = st.session_state.df

if df is None:
    st.info("Waiting for data. Download the sample CSV file or upload your data above.")

# 4. Data Processing
else:
    st.write("### 📊 Data & Range Selection")
    
    columns = df.columns.tolist()
    col_x = st.selectbox("Select Displacement Column (mm)", columns, index=0)
    col_y = st.selectbox("Select Force Column (N)", columns, index=1)
    
    # 2. FORCE STRICT NUMERIC DATA CASTING TO PREVENT EMPTY PLOTS
    df[col_x] = pd.to_numeric(df[col_x], errors='coerce')
    df[col_y] = pd.to_numeric(df[col_y], errors='coerce')
    df = df.dropna(subset=[col_x, col_y]).reset_index(drop=True)
    
    max_idx = len(df)

    # Callback function to force handles to maintain a minimum distance of 2
    def enforce_min_span():
        current_val = st.session_state.elastic_slider
        if current_val[1] - current_val[0] < 2:
            if current_val[0] + 2 <= max_idx:
                st.session_state.elastic_slider = (current_val[0], current_val[0] + 2)
            else:
                st.session_state.elastic_slider = (max(0, max_idx - 2), max_idx)

    # Automatically set up tracking or recalibrate if data length changes
    if "elastic_slider" not in st.session_state or st.session_state.elastic_slider[1] > max_idx:
        st.session_state.elastic_slider = (0, int(max_idx * 0.75))

    # The interactive slider linked directly to the session state key
    fit_range = st.slider(
        "Select Data Index Range for Elastic Fitting Zone", 
        min_value=0, 
        max_value=max_idx, 
        key="elastic_slider",
        on_change=enforce_min_span
    )

    start_idx, end_idx = fit_range

    # Safe slice that will never be empty or single-pointed
    df_fit = df.iloc[start_idx:end_idx]
    x_exp = df_fit[col_x].values
    y_exp = df_fit[col_y].values
    
    # Step A: Perform linear regression on experimental data to find the slope (S)
    slope_exp, intercept = np.polyfit(x_exp, y_exp, 1)
    
    # Step B: Analytical Inverse Solver
    def shear_lag_inverse(k_guess):
        if k_guess <= 0:
            return 99999.0 
        omega = np.sqrt((2 * k_guess) / (r_f * E_f))
        calculated_slope = A_f * E_f * omega * np.tanh(omega * L)
        return calculated_slope - slope_exp

    # Use numerical root-finding starting with an initial guess of k = 50 MPa/mm
    k_extracted = fsolve(shear_lag_inverse, x0=50.0)[0]
    
    # Step C: Generate the analytical curve using the optimized k value
    omega_fit = np.sqrt((2 * k_extracted) / (r_f * E_f))
    y_fit = (A_f * E_f * omega_fit * np.tanh(omega_fit * L)) * df[col_x].values + intercept
    
    # Calculate Goodness of Fit (R^2) for the selected fitting window
    y_pred_window = (A_f * E_f * omega_fit * np.tanh(omega_fit * L)) * x_exp + intercept
    r2 = r2_score(y_exp, y_pred_window)
    
    st.write("---")
    st.write("### 🎯 Optimization Outputs")
    
    # Display calculated metrics
    metric_col1, metric_col2, metric_col3 = st.columns(3)
    with metric_col1:
        st.metric(label="Measured Experimental Slope", value=f"{slope_exp:.4f} N/mm")
    with metric_col2:
        st.metric(label="Extracted Interface Stiffness (k)", value=f"{k_extracted:.2f} MPa/mm")
    with metric_col3:
        st.metric(label="Fit Confidence (R² Score)", value=f"{r2*100:.2f} %")
        
    # 5. Plotting Experimental vs Analytical
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(df[col_x], df[col_y], color='lightgray', s=10, label='Full Experimental Dataset')
    ax.scatter(x_exp, y_exp, color='darkorange', s=15, label='Selected Window for Fitting')
    ax.plot(df[col_x], y_fit, color='dodgerblue', linewidth=2.5, linestyle='--', label=f'Analytical Fit (k = {k_extracted:.1f} MPa/mm)')
    
    ax.set_title("Experimental UTM Data vs. Analytical Shear-Lag Optimization", fontsize=12, fontweight='bold')
    ax.set_xlabel("Displacement [mm]", fontsize=10)
    ax.set_ylabel("Pull-out Force [N]", fontsize=10)
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend()
    
    # Explicitly pass fig token to render engine
    st.pyplot(fig)
    plt.close(fig) # Prevent server-side memory bloating