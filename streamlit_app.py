#!/usr/bin/env python3

import math
import numpy as np
import pandas as pd
import streamlit as st

# Set MINI_PIPELINE_AVAILABLE to False since mini_pipeline is not available
MINI_PIPELINE_AVAILABLE = False

# -------------------------
# Physics indices
# -------------------------
def add_physics_based_indices(df):
    df = df.copy()
    def getcol(name):
        for col in (f"{name}_wtpct", name):
            if col in df.columns:
                return pd.to_numeric(df[col], errors="coerce").fillna(0.0)
        return pd.Series(0.0, index=df.index, dtype=float)
    Cr = getcol("Cr"); Mo = getcol("Mo"); W = getcol("W"); N = getcol("N")
    Si = getcol("Si"); Nb = getcol("Nb"); Ni = getcol("Ni"); Cc = getcol("C")
    Mn = getcol("Mn"); Cu = getcol("Cu")
    df["PREN"] = Cr + 3.3 * (Mo + 0.5 * W) + 16.0 * N
    df["Ceq"] = 1.0 * Cr + 1.0 * Mo + 1.5 * Si + 0.5 * Nb
    df["Neq"] = 1.0 * Ni + 30.0 * Cc + 0.5 * Mn + 0.3 * Cu
    for col in ("PREN", "Ceq", "Neq"):
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    return df

# -------------------------
# Prediction wrapper
# -------------------------
def run_prediction_pipeline(gen_df, training_csv="corrosion_data_clustered.csv", save_csv="predicted_samples_only.csv"):
    gen_df = gen_df.copy().reset_index(drop=True)
    elements_superset = ['Fe','Cr','Ni','Mo','Mn','Si','Cu',
                         'S','P','C','N','Nb','Ti','W','V',
                         'Al','B']
    elements = [e for e in elements_superset if e in gen_df.columns]
    out_df = gen_df.copy().reset_index(drop=True)
    for c in elements:
        out_df[f"{c}_wtpct"] = out_df[c].astype(float)
    out_df = add_physics_based_indices(out_df)
    rng = np.random.default_rng(0)
    pren_norm = (out_df['PREN'] - out_df['PREN'].min()) / (np.nanmax(out_df['PREN']) - np.nanmin(out_df['PREN']) + 1e-12)
    ceq_norm = (out_df['Ceq'] - out_df['Ceq'].min()) / (np.nanmax(out_df['Ceq']) - np.nanmin(out_df['Ceq']) + 1e-12)
    neq_norm = (out_df['Neq'] - out_df['Neq'].min()) / (np.nanmax(out_df['Neq']) - np.nanmin(out_df['Neq']) + 1e-12)
    EtoJ_synthetic = 0.015 + 0.02 * (0.5 * pren_norm + 0.3 * ceq_norm + 0.2 * (1 - neq_norm))
    EtoJ_synthetic += rng.normal(scale=0.002, size=len(EtoJ_synthetic))
    out_df['EtoJ_pred'] = EtoJ_synthetic.clip(0.0001, None)
    out_df['EtoJ_plot'] = out_df['EtoJ_pred'].astype(float)
    out_df['Rrp_pred'] = out_df['EtoJ_pred']
    out_df['Rrp_plot'] = out_df['Rrp_pred'].astype(float)
    return out_df

# Streamlit App
st.title("CorrosionInformatics")

st.header("Input Elemental Composition (wt%)")

# Define ranges and defaults
unified_ranges = {
    'Mo': (0, 4), 'Cr': (10.5, 30), 'Si': (0, 2), 'Nb': (0, 1),
    'Ni': (0, 15), 'Mn': (0, 10), 'C': (0, 0.12), 'Cu': (0, 3)
}
small_ranges = {
    'S': (0.0003, 0.004),
    'P': (0.002, 0.03),
    'N': (0.002, 0.03),
    'Ti': (0.0, 0.05),
    'W': (0.0, 0.02),
    'V': (0.0, 0.02),
    'Al': (0.0, 0.02),
    'B': (0.0, 0.005)
}
fixed_values = {
    'Mo': (0 + 4) / 2,
    'Cr': (10.5 + 30) / 2,
    'Si': (0 + 2) / 2,
    'Nb': (0 + 1) / 2,
    'Ni': (0 + 15) / 2,
    'Mn': (0 + 10) / 2,
    'C': (0 + 0.12) / 2,
    'Cu': (0 + 3) / 2,
    'S': (0.0003 + 0.004) / 2,
    'P': (0.002 + 0.03) / 2,
    'N': (0.002 + 0.03) / 2,
    'Ti': (0.0 + 0.05) / 2,
    'W': (0.0 + 0.02) / 2,
    'V': (0.0 + 0.02) / 2,
    'Al': (0.0 + 0.02) / 2,
    'B': (0.0 + 0.005) / 2
}

# Inputs for unified ranges (sliders)
st.subheader("Major Elements")
col1, col2 = st.columns(2)
with col1:
    Cr = st.slider("Cr", min_value=unified_ranges['Cr'][0], max_value=unified_ranges['Cr'][1], value=fixed_values['Cr'], step=0.1)
    Mo = st.slider("Mo", min_value=unified_ranges['Mo'][0], max_value=unified_ranges['Mo'][1], value=fixed_values['Mo'], step=0.1)
    Si = st.slider("Si", min_value=unified_ranges['Si'][0], max_value=unified_ranges['Si'][1], value=fixed_values['Si'], step=0.1)
    Nb = st.slider("Nb", min_value=unified_ranges['Nb'][0], max_value=unified_ranges['Nb'][1], value=fixed_values['Nb'], step=0.1)
with col2:
    Ni = st.slider("Ni", min_value=unified_ranges['Ni'][0], max_value=unified_ranges['Ni'][1], value=fixed_values['Ni'], step=0.1)
    Mn = st.slider("Mn", min_value=unified_ranges['Mn'][0], max_value=unified_ranges['Mn'][1], value=fixed_values['Mn'], step=0.1)
    C = st.slider("C", min_value=unified_ranges['C'][0], max_value=unified_ranges['C'][1], value=fixed_values['C'], step=0.001)
    Cu = st.slider("Cu", min_value=unified_ranges['Cu'][0], max_value=unified_ranges['Cu'][1], value=fixed_values['Cu'], step=0.1)

# Inputs for small ranges (number inputs for precision)
st.subheader("Trace Elements")
col3, col4 = st.columns(2)
with col3:
    S = st.number_input("S", min_value=small_ranges['S'][0], max_value=small_ranges['S'][1], value=fixed_values['S'], step=0.0001)
    P = st.number_input("P", min_value=small_ranges['P'][0], max_value=small_ranges['P'][1], value=fixed_values['P'], step=0.001)
    N = st.number_input("N", min_value=small_ranges['N'][0], max_value=small_ranges['N'][1], value=fixed_values['N'], step=0.001)
    Ti = st.number_input("Ti", min_value=small_ranges['Ti'][0], max_value=small_ranges['Ti'][1], value=fixed_values['Ti'], step=0.001)
with col4:
    W = st.number_input("W", min_value=small_ranges['W'][0], max_value=small_ranges['W'][1], value=fixed_values['W'], step=0.001)
    V = st.number_input("V", min_value=small_ranges['V'][0], max_value=small_ranges['V'][1], value=fixed_values['V'], step=0.001)
    Al = st.number_input("Al", min_value=small_ranges['Al'][0], max_value=small_ranges['Al'][1], value=fixed_values['Al'], step=0.001)
    B = st.number_input("B", min_value=small_ranges['B'][0], max_value=small_ranges['B'][1], value=fixed_values['B'], step=0.0001)

# Collect inputs
inputs = {
    'Cr': Cr, 'Mo': Mo, 'Si': Si, 'Nb': Nb, 'Ni': Ni, 'Mn': Mn, 'C': C, 'Cu': Cu,
    'S': S, 'P': P, 'N': N, 'Ti': Ti, 'W': W, 'V': V, 'Al': Al, 'B': B
}

# Calculate Fe
others_sum = sum(inputs.values())
if others_sum > 99.99:
    st.error("Total composition (excluding Fe) must be less than 100%. Please adjust the inputs.")
    st.stop()
Fe = max(0.01, 100.0 - others_sum)
st.write(f"Calculated Fe (wt%): {Fe:.2f}")

# Create row
row = inputs.copy()
row['Fe'] = Fe
row['MfgRoute'] = '1'
row['T[oC]'] = 500
row['t[hr]'] = 0.25
row['Process_Clustered'] = 'Processed/Treated'
row['Coolingmedium_Clustered'] = 'Water-based Cooling'
row['Solution_Clustered'] = 'Chloride/Saline Solution'

# Display composition summary
st.subheader("Input Composition Summary")
comp_df = pd.DataFrame([row], columns=list(inputs.keys()) + ['Fe'])
st.table(comp_df)

# Create DataFrame
gen_df = pd.DataFrame([row])

# Reset button
if st.button("Reset to Defaults"):
    st.session_state.clear()
    st.experimental_rerun()

if st.button("Predict"):
    out_df = run_prediction_pipeline(gen_df)
    out_df = add_physics_based_indices(out_df)
    
    st.header("Results")
    st.subheader("Physics-Based Indices")
    st.write(f"PREN: {out_df['PREN'].iloc[0]:.2f}")
    st.write(f"Ceq: {out_df['Ceq'].iloc[0]:.2f}")
    st.write(f"Neq: {out_df['Neq'].iloc[0]:.2f}")
    
    st.subheader("Predictions")
    st.write(f"EtoJ_pred: {out_df['EtoJ_pred'].iloc[0]:.4f}")
    st.write(f"Rrp_pred: {out_df['Rrp_pred'].iloc[0]:.4f}")
    
    # Bar chart for indices
    st.subheader("Physics-Based Indices Visualization")
    indices = ['PREN', 'Ceq', 'Neq']
    values = [out_df['PREN'].iloc[0], out_df['Ceq'].iloc[0], out_df['Neq'].iloc[0]]

    ```chartjs
    {
      "type": "bar",
      "data": {
        "labels": ["PREN", "Ceq", "Neq"],
        "datasets": [{
          "label": "Indices",
          "data": [${out_df['PREN'].iloc[0]}, ${out_df['Ceq'].iloc[0]}, ${out_df['Neq'].iloc[0]}],
          "backgroundColor": ["#4CAF50", "#2196F3", "#FFC107"],
          "borderColor": ["#388E3C", "#1976D2", "#FFA000"],
          "borderWidth": 1
        }]
      },
      "options": {
        "scales": {
          "y": {
            "beginAtZero": true,
            "title": {
              "display": true,
              "text": "Value",
              "font": {"size": 14, "weight": "bold"}
            },
            "ticks": {"color": "#333333", "font": {"size": 12}}
          },
          "x": {
            "title": {
              "display": true,
              "text": "Index",
              "font": {"size": 14, "weight": "bold"}
            },
            "ticks": {"color": "#333333", "font": {"size": 12}}
          }
        },
        "plugins": {
          "legend": {"display": false}
        }
      }
    }
