
import math
import numpy as np
import pandas as pd
import streamlit as st
from mini_pipeline import featurize_and_refine_training_inmemory, featurize_new_and_align_inmemory, predict_on_dataframe_in_memory
import os

# Set MINI_PIPELINE_AVAILABLE to True
MINI_PIPELINE_AVAILABLE = True

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
# Prediction wrapper using mini_pipeline
# -------------------------
def run_prediction_pipeline(gen_df, training_csv="corrosion_data_clustered.csv", save_csv="predicted_samples_only.csv"):
    try:
        # Check if training CSV exists
        if not os.path.exists(training_csv):
            st.error(f"Training data file {training_csv} not found.")
            st.stop()

        # Process training data
        df_train_refined, encs, cols = featurize_and_refine_training_inmemory(training_csv)

        # Save input DataFrame to a temporary CSV for featurization
        temp_csv = "temp_new_data.csv"
        gen_df.to_csv(temp_csv, index=False)

        # Featurize and align new data
        df_ni_refined, _ = featurize_new_and_align_inmemory(temp_csv, encs, cols)

        # Predict
        out_df, preds = predict_on_dataframe_in_memory(df_ni_refined, label_encoders_provided=encs)

        # Clean up temporary CSV
        if os.path.exists(temp_csv):
            os.remove(temp_csv)

        # Add physics-based indices to output (for consistency with UI)
        out_df = add_physics_based_indices(out_df)

        # Ensure expected columns are present
        for col in ['EtoJ_pred', 'Rrp_pred']:
            if col not in out_df.columns:
                out_df[col] = 0.0  # Fallback if predictions are missing
        out_df['EtoJ_plot'] = out_df['EtoJ_pred'].astype(float)
        out_df['Rrp_plot'] = out_df['Rrp_pred'].astype(float)

        return out_df
    except Exception as e:
        st.error(f"Prediction pipeline failed: {e}")
        st.stop()

# Streamlit App
st.title("CorrosionInformatics")

# Check for training data at startup
training_csv = "corrosion_data_clustered.csv"  # Update path if in a subdirectory, e.g., "data/corrosion_data_clustered.csv"
if not os.path.exists(training_csv):
    st.warning(f"Training data file {training_csv} not found. Predictions may fail.")

st.header("Input Elemental Composition (wt%)")

# Define ranges and defaults
unified_ranges = {
    'Mo': (0.0, 4.0), 'Cr': (10.5, 30.0), 'Si': (0.0, 2.0), 'Nb': (0.0, 1.0),
    'Ni': (0.0, 15.0), 'Mn': (0.0, 10.0), 'C': (0.0, 0.12), 'Cu': (0.0, 3.0)
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
    'Mo': (0.0 + 4.0) / 2,
    'Cr': (10.5 + 30.0) / 2,
    'Si': (0.0 + 2.0) / 2,
    'Nb': (0.0 + 1.0) / 2,
    'Ni': (0.0 + 15.0) / 2,
    'Mn': (0.0 + 10.0) / 2,
    'C': (0.0 + 0.12) / 2,
    'Cu': (0.0 + 3.0) / 2,
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
    try:
        Cr = st.slider("Cr", min_value=float(unified_ranges['Cr'][0]), max_value=float(unified_ranges['Cr'][1]), 
                       value=float(fixed_values['Cr']), step=0.1, format="%.1f", key="Cr_slider")
        Mo = st.slider("Mo", min_value=float(unified_ranges['Mo'][0]), max_value=float(unified_ranges['Mo'][1]), 
                       value=float(fixed_values['Mo']), step=0.1, format="%.1f", key="Mo_slider")
        Si = st.slider("Si", min_value=float(unified_ranges['Si'][0]), max_value=float(unified_ranges['Si'][1]), 
                       value=float(fixed_values['Si']), step=0.1, format="%.1f", key="Si_slider")
        Nb = st.slider("Nb", min_value=float(unified_ranges['Nb'][0]), max_value=float(unified_ranges['Nb'][1]), 
                       value=float(fixed_values['Nb']), step=0.1, format="%.1f", key="Nb_slider")
    except Exception as e:
        st.error(f"Error in major elements sliders: {e}")
        st.stop()
with col2:
    try:
        Ni = st.slider("Ni", min_value=float(unified_ranges['Ni'][0]), max_value=float(unified_ranges['Ni'][1]), 
                       value=float(fixed_values['Ni']), step=0.1, format="%.1f", key="Ni_slider")
        Mn = st.slider("Mn", min_value=float(unified_ranges['Mn'][0]), max_value=float(unified_ranges['Mn'][1]), 
                       value=float(fixed_values['Mn']), step=0.1, format="%.1f", key="Mn_slider")
        C = st.slider("C", min_value=float(unified_ranges['C'][0]), max_value=float(unified_ranges['C'][1]), 
                      value=float(fixed_values['C']), step=0.001, format="%.3f", key="C_slider")
        Cu = st.slider("Cu", min_value=float(unified_ranges['Cu'][0]), max_value=float(unified_ranges['Cu'][1]), 
                       value=float(fixed_values['Cu']), step=0.1, format="%.1f", key="Cu_slider")
    except Exception as e:
        st.error(f"Error in major elements sliders: {e}")
        st.stop()

# Inputs for small ranges (number inputs for precision)
st.subheader("Trace Elements")
col3, col4 = st.columns(2)
with col3:
    S = st.number_input("S", min_value=float(small_ranges['S'][0]), max_value=float(small_ranges['S'][1]), 
                        value=float(fixed_values['S']), step=0.0001, format="%.4f", key="S_input")
    P = st.number_input("P", min_value=float(small_ranges['P'][0]), max_value=float(small_ranges['P'][1]), 
                        value=float(fixed_values['P']), step=0.001, format="%.3f", key="P_input")
    N = st.number_input("N", min_value=float(small_ranges['N'][0]), max_value=float(small_ranges['N'][1]), 
                        value=float(fixed_values['N']), step=0.001, format="%.3f", key="N_input")
    Ti = st.number_input("Ti", min_value=float(small_ranges['Ti'][0]), max_value=float(small_ranges['Ti'][1]), 
                         value=float(fixed_values['Ti']), step=0.001, format="%.3f", key="Ti_input")
with col4:
    W = st.number_input("W", min_value=float(small_ranges['W'][0]), max_value=float(small_ranges['W'][1]), 
                        value=float(fixed_values['W']), step=0.001, format="%.3f", key="W_input")
    V = st.number_input("V", min_value=float(small_ranges['V'][0]), max_value=float(small_ranges['V'][1]), 
                        value=float(fixed_values['V']), step=0.001, format="%.3f", key="V_input")
    Al = st.number_input("Al", min_value=float(small_ranges['Al'][0]), max_value=float(small_ranges['Al'][1]), 
                         value=float(fixed_values['Al']), step=0.001, format="%.3f", key="Al_input")
    B = st.number_input("B", min_value=float(small_ranges['B'][0]), max_value=float(small_ranges['B'][1]), 
                        value=float(fixed_values['B']), step=0.0001, format="%.4f", key="B_input")

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
    st.rerun()

if st.button("Predict"):
    try:
        out_df = run_prediction_pipeline(gen_df)
        
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
        chart_data = pd.DataFrame({'Indices': indices, 'Values': values})
        st.bar_chart(chart_data.set_index('Indices'))

        # Download results
        csv = out_df.to_csv(index=False)
        st.download_button("Download Results", csv, "predicted_composition.csv", "text/csv")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
