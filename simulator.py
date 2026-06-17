import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# 1. Page Configuration
st.set_page_config(page_title="Shear-Lag Interface Simulator", layout="wide")

st.title("🔬 Nano-Engineered Interface Simulator")
st.markdown("""
This simulator uses the analytical **Shear-Lag Model** to show how tensile and shear stresses 
distribute along a carbon fiber filament embedded in a cement matrix. 
Adjust the material and geometric parameters in the sidebar to observe stress concentrations in real time.
""")

st.write("---")

# 2. Sidebar Controls (Inputs)
st.sidebar.header("🛠️ Input Parameters")

# Fiber Geometry
st.sidebar.subheader("Fiber Geometry")
r_f_micron = st.sidebar.slider("Fiber Radius (r_f) [μm]", min_value=1.0, max_value=25.0, value=3.5, step=0.1)
L = st.sidebar.slider("Embedment Length (L) [mm]", min_value=1.0, max_value=15.0, value=5.0, step=0.5)

# Material Properties
st.sidebar.subheader("Material Properties")
E_f_gpa = st.sidebar.slider("Fiber Elastic Modulus (E_f) [GPa]", min_value=50, max_value=400, value=230, step=10)
k = st.sidebar.slider("Interface Shear Stiffness (k) [MPa/mm]", min_value=5, max_value=400, value=40, step=5)

# Loading Conditions
st.sidebar.subheader("Loading")
P = st.sidebar.slider("Applied Pull-out Force (P) [N]", min_value=0.01, max_value=0.50, value=0.05, step=0.01)

# 3. Unit Conversions & Math Engine
# Convert inputs to standard mm and MPa units for matching equations
r_f = r_f_micron / 1000.0   # μm to mm
E_f = E_f_gpa * 1000.0       # GPa to MPa

# Cross-sectional Area of the fiber (mm^2)
A_f = np.pi * (r_f ** 2)

# Input Tensile Stress at the crack face (x = L)
sigma_0 = P / A_f

# Calculate system constant Omega (ω)
omega = np.sqrt((2 * k) / (r_f * E_f))

# Generate x-coordinate array from 0 (deep tip) to L (crack face)
x = np.linspace(0, L, 500)

# Calculate Hyperbolic Stress Distributions safely
try:
    # Fiber Tensile Stress along x
    sigma_f = sigma_0 * (np.sinh(omega * x) / np.sinh(omega * L))
    
    # Interfacial Shear Stress along x
    tau = (sigma_0 * r_f * omega / 2.0) * (np.cosh(omega * x) / np.sinh(omega * L))
    
    # Extract boundary values for display
    peak_tau = tau[-1]
    peak_sigma = sigma_f[-1]
except ZeroDivisionError:
    st.error("Math Error: Check that your input values are greater than zero.")

# 4. KPI Metrics Dashboard Display
col1, col2, col3 = st.columns(3)
with col1:
    st.metric(label="Calculated System Modality (ω)", value=f"{omega:.4f}")
with col2:
    st.metric(label="Tensile Stress at Crack Face (σ_0)", value=f"{peak_sigma:.2f} MPa")
with col3:
    st.metric(label="Peak Interfacial Shear Stress (τ_max)", value=f"{peak_tau:.2f} MPa")

st.write("---")

# 5. Data Visualization (Plotting Engine)
# Set up side-by-side Matplotlib subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Fiber Tensile Stress
ax1.plot(x, sigma_f, color='crimson', linewidth=2.5, label='Tensile Stress')
ax1.fill_between(x, sigma_f, color='crimson', alpha=0.1)
ax1.set_title("Fiber Tensile Stress Profile (σ_f)", fontsize=12, fontweight='bold')
ax1.set_xlabel("Distance along embedded fiber, x [mm]", fontsize=10)
ax1.set_ylabel("Tensile Stress [MPa]", fontsize=10)
ax1.grid(True, linestyle='--', alpha=0.6)
ax1.axvline(x=L, color='black', linestyle=':', label='Crack Face (x=L)')
ax1.legend()

# Plot 2: Interfacial Shear Stress
ax2.plot(x, tau, color='dodgerblue', linewidth=2.5, label='Shear Stress')
ax2.fill_between(x, tau, color='dodgerblue', alpha=0.1)
ax2.set_title("Interfacial Shear Stress Profile (τ)", fontsize=12, fontweight='bold')
ax2.set_xlabel("Distance along embedded fiber, x [mm]", fontsize=10)
ax2.set_ylabel("Shear Stress [MPa]", fontsize=10)
ax2.grid(True, linestyle='--', alpha=0.6)
ax2.axvline(x=L, color='black', linestyle=':', label='Crack Face (x=L)')
ax2.legend()

st.pyplot(fig)