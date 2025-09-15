import os
from typing import Dict, List, Tuple, Optional, Union
import math
import json
import random
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import joblib
import torch
import torch.nn as nn
from pymatgen.core import Composition
from matminer.featurizers.composition import (
    Stoichiometry, ElementProperty, AtomicOrbitals, Miedema,
    YangSolidSolution, ValenceOrbital, ElectronegativityDiff
)

# ---------- Minimal config ----------
ELEMENT_COLS = ['Fe','Cr','Ni','Mo','Mn','Si','Cu','S','P','C','N','O',
                'Nb','Ti','W','Co','V','Al','Zr','Ta','Ce','Re','B']
CATEGORICAL_COLS = ['MfgRoute','Process_Clustered','Coolingmedium_Clustered',
                    'Solution_Clustered','phase_class','PREN_band']
INTERACTION_FEATURES = [
    'T_t_interaction','T_squared','t_squared','T_over_t','T_log_t_interaction',
    'PREN','Creq','Nieq_withN','est_austenite_frac','Ni_over_Cr','Ni_times_Cr',
    'PREN_T_interaction','PREN_T_over_t','est_aust_T','Ni_squared','Ni_log','Ni_Cr_interaction'
]
TARGET_COLUMN = 'Rrp'
MODEL_DIR = "Models"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------- Helpers: featurization ----------
def read_csv_or_raise(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return pd.read_csv(path)

def make_composition(df: pd.DataFrame, element_cols: List[str], comp_col: str='Composition') -> pd.DataFrame:
    df = df.copy()
    for e in element_cols:
        if e not in df.columns:
            df[e] = 0.0
        df[e] = pd.to_numeric(df[e], errors='coerce').fillna(0.0)
    def row_to_comp(row):
        comp = {el: row[el] for el in element_cols if row[el] > 0}
        if not comp:
            return np.nan
        try:
            return Composition(comp)
        except Exception:
            return np.nan
    df[comp_col] = df.apply(row_to_comp, axis=1)
    return df

def apply_matminer(df: pd.DataFrame, comp_col='Composition') -> pd.DataFrame:
    featurizers = [
        Stoichiometry(),
        ElementProperty.from_preset("magpie"),
        AtomicOrbitals(),
        Miedema(),
        YangSolidSolution(),
        ValenceOrbital(),
        ElectronegativityDiff()
    ]
    df = df.copy()
    for f in featurizers:
        try:
            df = f.featurize_dataframe(df, col_id=comp_col, ignore_errors=True)
        except Exception as e:
            print(f"matminer {f.__class__.__name__} error: {e}")
    df.drop(columns=[comp_col], inplace=True, errors='ignore')
    return df

def compute_comp_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for e in ['Cr','Mo','N','Ni','C','Mn','Si','Fe']:
        if e not in df.columns:
            df[e] = 0.0
    df['PREN'] = df['Cr'] + 3.3*df['Mo'] + 16.0*df['N']
    df['Creq'] = df['Cr'] + df['Mo'] + 1.5*df['Si']
    df['Nieq_withN'] = df['Ni'] + 30.0*df['C'] + 0.5*df['Mn']
    if 'N' in df.columns:
        df.loc[df['N'].notna(), 'Nieq_withN'] = df.loc[df['N'].notna(), 'Nieq_withN'] + 25.0*df.loc[df['N'].notna(), 'N']
    df['Ni_minus_Cr_eq'] = df['Nieq_withN'] - df['Creq']
    df['est_austenite_frac'] = 1.0 / (1.0 + np.exp(-df['Ni_minus_Cr_eq'].fillna(0.0)/2.0))
    df['phase_class'] = 'ferritic'
    df.loc[df['Ni_minus_Cr_eq'].between(-2.0, 2.0), 'phase_class'] = 'duplex'
    df.loc[df['Ni_minus_Cr_eq'] > 2.0, 'phase_class'] = 'austenitic'
    df['PREN_band'] = pd.cut(df['PREN'].fillna(0.0),
                             bins=[-np.inf,25,35,45,np.inf],
                             labels=['low','moderate','good','excellent']).astype(str)
    return df

def add_interactions(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in ['T[oC]','t[hr]','Ni','Cr','PREN','est_austenite_frac']:
        if c not in df.columns:
            df[c] = 0.0
    df['T_t_interaction'] = df['T[oC]'] * df['t[hr]']
    df['T_squared'] = df['T[oC]']**2
    df['t_squared'] = df['t[hr]']**2
    df['T_over_t'] = df['T[oC]'] / (df['t[hr]'] + 1e-6)
    df['log_t'] = np.log1p(df['t[hr]'] + 1e-6)
    df['T_log_t_interaction'] = df['T[oC]'] * df['log_t']
    df['PREN_T_interaction'] = df['PREN'] * df['T[oC]']
    df['PREN_T_over_t'] = df['PREN'] * df['T_over_t']
    df['est_aust_T'] = df['est_austenite_frac'] * df['T[oC]']
    df['Ni_squared'] = df['Ni']**2
    df['Ni_log'] = np.log1p(df['Ni'] + 1e-6)
    df['Ni_Cr_interaction'] = df['Ni'] * df['Cr']
    return df

def drop_all_zero_numeric(df: pd.DataFrame, exclude: Optional[List[str]] = None) -> pd.DataFrame:
    exclude = exclude or []
    numeric = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c not in exclude]
    drop = [c for c in numeric if (df[c].fillna(0) == 0).all()]
    return df.drop(columns=drop, errors='ignore')

# ---------- Main simplified flows ----------
def featurize_and_refine_training_inmemory(
    train_input: Union[str, pd.DataFrame],
    element_cols: List[str] = ELEMENT_COLS,
    categorical_cols: List[str] = CATEGORICAL_COLS,
    target_col: str = TARGET_COLUMN,
    save_artifacts: bool = False,
) -> Tuple[pd.DataFrame, Dict[str, LabelEncoder], List[str]]:
    # Handle input: either CSV path or DataFrame
    if isinstance(train_input, str):
        df = read_csv_or_raise(train_input)
    elif isinstance(train_input, pd.DataFrame):
        df = train_input.copy()
    else:
        raise TypeError("train_input must be a string (CSV path) or pandas DataFrame")

    df = make_composition(df, element_cols)
    df = df.dropna(subset=['Composition']).copy()
    df = apply_matminer(df)
    df = compute_comp_features(df)
    df = add_interactions(df)

    encoders: Dict[str, LabelEncoder] = {}
    for col in (categorical_cols or []):
        if col in df.columns:
            s = df[col].astype(str).fillna("Missing_Category")
            le = LabelEncoder().fit(s)
            df[col] = le.transform(s.values)
            encoders[col] = le
        else:
            le = LabelEncoder().fit(np.array(["Missing_Category"], dtype=object))
            df[col] = le.transform(np.array(["Missing_Category"]*len(df)))
            encoders[col] = le

    for c in df.columns:
        if c == target_col:
            continue
        df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0.0)

    exclude = element_cols + INTERACTION_FEATURES + (categorical_cols or []) + [target_col]
    df_refined = drop_all_zero_numeric(df, exclude=exclude)
    refined_cols = list(df_refined.columns)

    if save_artifacts:
        df_refined.to_csv("corrosion_refined.csv", index=False)
        os.makedirs(MODEL_DIR, exist_ok=True)
        joblib.dump(refined_cols, os.path.join(MODEL_DIR,"refined_columns.joblib"))
        for k,v in encoders.items():
            joblib.dump(v, os.path.join(MODEL_DIR, f"label_encoder_{k}.joblib"))

    return df_refined, encoders, refined_cols

def featurize_new_and_align_inmemory(
    new_input: Union[str, pd.DataFrame],
    encoders: Dict[str, LabelEncoder],
    reference_columns: List[str],
    element_cols: List[str] = ELEMENT_COLS,
    categorical_cols: List[str] = CATEGORICAL_COLS,
    target_col: str = TARGET_COLUMN,
    save_artifacts: bool = False
) -> Tuple[pd.DataFrame, Dict[str, LabelEncoder]]:
    # Handle input: either CSV path or DataFrame
    if isinstance(new_input, str):
        df_new = read_csv_or_raise(new_input)
    elif isinstance(new_input, pd.DataFrame):
        df_new = new_input.copy()
    else:
        raise TypeError("new_input must be a string (CSV path) or pandas DataFrame")

    df_new = make_composition(df_new, element_cols)
    df_new = apply_matminer(df_new)
    df_new = compute_comp_features(df_new)
    df_new = add_interactions(df_new)

    for col in (categorical_cols or []):
        le = encoders.get(col)
        if le is None:
            le = LabelEncoder().fit(np.array(["Missing_Category"], dtype=object))
        if col not in df_new.columns:
            df_new[col] = "Missing_Category"
        s = df_new[col].astype(str).fillna("Missing_Category").values.astype(str)
        known = set(le.classes_.astype(str))
        unseen_mask = ~np.isin(s, list(known))
        if unseen_mask.any():
            if "Missing_Category" in known:
                s[unseen_mask] = "Missing_Category"
            else:
                s[unseen_mask] = str(le.classes_[0])
        df_new[col] = le.transform(s)

    for c in df_new.columns:
        if c == target_col:
            continue
        df_new[c] = pd.to_numeric(df_new[c], errors='coerce').fillna(0.0)

    for c in reference_columns:
        if c not in df_new.columns:
            df_new[c] = np.nan if c == target_col else 0.0
    extras = [c for c in df_new.columns if c not in reference_columns]
    if extras:
        df_new = df_new.drop(columns=extras)
    df_new = df_new[reference_columns]

    if save_artifacts:
        df_new.to_csv("ni_agent_refined.csv", index=False)

    return df_new, encoders

# ---------- Model class (for prediction) ----------
class ResidualBlock(nn.Module):
    def __init__(self, dim, hidden=None, dropout=0.3):
        super().__init__()
        h = hidden or dim
        self.fc1 = nn.Linear(dim, h)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(h, dim)
        self.norm = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        out = self.fc1(x)
        out = self.act(out)
        out = self.drop(self.fc2(out))
        out = out + x
        out = self.norm(out)
        return out

class TabularResNet(nn.Module):
    def __init__(self, cont_dim, cat_cardinalities, emb_dims, hidden_sizes=(128,64), n_blocks=3, dropout=0.3, ni_idx=None):
        super().__init__()
        self.ni_idx = ni_idx
        self.embeddings = nn.ModuleList([nn.Embedding(nc, ed) for nc, ed in zip(cat_cardinalities, emb_dims)]) if len(cat_cardinalities) > 0 else None
        emb_total = sum(emb_dims) if emb_dims else 0
        input_dim = cont_dim + emb_total
        self.attention = nn.Sequential(nn.Linear(input_dim, input_dim), nn.Sigmoid())
        self.input_norm = nn.LayerNorm(input_dim)
        self.initial = nn.Sequential(nn.Linear(input_dim, hidden_sizes[0]), nn.GELU(), nn.Dropout(dropout))
        prev = hidden_sizes[0]
        self.blocks = nn.Sequential(*[ResidualBlock(prev, hidden=prev*2, dropout=dropout) for _ in range(n_blocks)])
        head_layers = []
        for h in hidden_sizes[1:]:
            head_layers += [nn.Linear(prev, h), nn.GELU(), nn.Dropout(dropout)]
            prev = h
        head_layers.append(nn.Linear(prev, 1))
        self.head = nn.Sequential(*head_layers)

    def forward(self, x_cont, x_cat=None):
        if self.embeddings and x_cat is not None and x_cat.shape[1] > 0:
            embs = [emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)]
            x = torch.cat([x_cont] + embs, dim=1)
        else:
            x = x_cont
        attn_weights = self.attention(x)
        if self.ni_idx is not None:
            attn_weights = attn_weights.clone()
            attn_weights[:, self.ni_idx] = attn_weights[:, self.ni_idx] * 1.5
        x_weighted = x * attn_weights
        x = self.input_norm(x_weighted)
        x = self.initial(x)
        x = self.blocks(x)
        out = self.head(x)
        return out.squeeze(1)

# ---------- Prediction helpers ----------
def load_label_encoders_fallback(provided_encoders=None, model_dir=MODEL_DIR, cat_cols=None):
    if provided_encoders:
        return provided_encoders
    enc_path = os.path.join(model_dir, "label_encoders.joblib")
    if os.path.exists(enc_path):
        return joblib.load(enc_path)
    encs = {}
    if cat_cols is None:
        cat_cols = CATEGORICAL_COLS
    for c in cat_cols:
        p = os.path.join(model_dir, f"label_encoder_{c}.joblib")
        if os.path.exists(p):
            encs[c] = joblib.load(p)
    return encs

def build_training_arrays_from_artifacts(df, cont_imputer, cont_scaler, label_encoders, cont_selected, cat_cols_present, cont_reducer=None):
    missing = [c for c in cont_selected if c not in df.columns]
    if missing:
        raise KeyError(f"Missing continuous columns required for NN: {missing}")
    X_cont = cont_imputer.transform(df[cont_selected])
    X_cont = cont_scaler.transform(X_cont).astype(np.float32)
    if cont_reducer is not None:
        X_cont = cont_reducer.transform(X_cont).astype(np.float32)
    X_cat = np.zeros((len(df), len(cat_cols_present)), dtype=np.int64) if len(cat_cols_present) > 0 else np.zeros((len(df), 0), dtype=np.int64)
    for i, c in enumerate(cat_cols_present):
        vals = df[c].astype(str).replace(['Missing_','missing_','MISSING_','nan','NaN',''], 'Missing_Category').fillna("Missing_Category").values
        le = label_encoders.get(c)
        if le is None:
            le = LabelEncoder().fit(np.array(["Missing_Category"], dtype=object))
        mapped = []
        classes = set([str(x) for x in le.classes_])
        for v in vals:
            if v in classes:
                mapped.append(int(le.transform([v])[0]))
            else:
                if "Missing_Category" in classes:
                    mapped.append(int(le.transform(["Missing_Category"])[0]))
                else:
                    mapped.append(int(le.transform([le.classes_[0]])[0]))
        X_cat[:, i] = np.array(mapped, dtype=np.int64)
    return X_cont, X_cat

def predict_on_dataframe_in_memory(
    df_refined,
    label_encoders_provided=None,
    model_dir=MODEL_DIR,
    device=DEVICE,
    save_predictions=False
) -> Tuple[pd.DataFrame, np.ndarray]:
    nn_preproc_path = os.path.join(model_dir, "nn_preprocessor.joblib")
    if not os.path.exists(nn_preproc_path):
        raise FileNotFoundError(f"Missing nn_preprocessor.joblib at {nn_preproc_path}")
    nn_preprocessor = joblib.load(nn_preproc_path)
    cont_selected = nn_preprocessor['cont_selected']
    cont_imputer = nn_preprocessor['cont_imputer']
    cont_scaler = nn_preprocessor['cont_scaler']
    cont_reducer = nn_preprocessor.get('cont_reducer', None)
    medians = dict(zip(cont_selected, getattr(cont_imputer, "statistics_", [])))

    emb_path = os.path.join(model_dir, "embedding_config.joblib")
    if not os.path.exists(emb_path):
        raise FileNotFoundError(f"Missing embedding_config.joblib at {emb_path}")
    emb_cfg = joblib.load(emb_path)
    cat_cols_present = emb_cfg['cat_cols']
    cat_cardinalities = emb_cfg['cardinalities']
    emb_dims = emb_cfg['emb_dims']

    ts_path = os.path.join(model_dir, "target_scaler.joblib")
    if not os.path.exists(ts_path):
        raise FileNotFoundError(f"Missing target_scaler.joblib at {ts_path}")
    target_scaler = joblib.load(ts_path)
    y_mean = target_scaler['mean']
    y_std = target_scaler['std']

    summary_path = os.path.join(model_dir, "nn_training_summary.json")
    if not os.path.exists(summary_path):
        raise FileNotFoundError(f"Missing nn_training_summary.json at {summary_path}")
    with open(summary_path, "r") as f:
        summary = json.load(f)
    fold_metrics = summary.get('fold_metrics', [])
    num_folds = len(fold_metrics)
    if num_folds == 0:
        raise RuntimeError("No folds described in nn_training_summary.json")
    fold_r2_scores = [max(m.get('r2', 0.0), 0.0) for m in fold_metrics]
    total_r2 = sum(fold_r2_scores)
    if total_r2 > 0:
        weights = [r2/total_r2 for r2 in fold_r2_scores]
    else:
        weights = [1.0/num_folds]*num_folds

    label_encoders = load_label_encoders_fallback(label_encoders_provided, model_dir=model_dir, cat_cols=cat_cols_present)

    df = df_refined.copy().reset_index(drop=True)
    for c in cat_cols_present:
        if c not in df.columns:
            df[c] = "Missing_Category"
        else:
            df[c] = df[c].astype(str).replace(['Missing_','missing_','MISSING_','nan','NaN',''], 'Missing_Category').fillna("Missing_Category")

    for c in cont_selected:
        if c not in df.columns:
            df[c] = medians.get(c, 0.0)

    X_cont, X_cat = build_training_arrays_from_artifacts(df, cont_imputer, cont_scaler, label_encoders, cont_selected, cat_cols_present, cont_reducer)
    ni_idx = cont_selected.index('Ni') if 'Ni' in cont_selected else None

    preds_ensemble = np.zeros(len(df), dtype=float)
    for fold in range(1, num_folds+1):
        swa_path = os.path.join(model_dir, f"nn_best_fold{fold}_swa.pth")
        if not os.path.exists(swa_path):
            print(f"Warning: model file for fold {fold} not found at {swa_path}; skipping fold")
            continue
        model = TabularResNet(cont_dim=X_cont.shape[1], cat_cardinalities=cat_cardinalities, emb_dims=emb_dims, hidden_sizes=(128,64), n_blocks=3, dropout=0.3, ni_idx=ni_idx).to(device)
        state = torch.load(swa_path, map_location=device)
        model.load_state_dict(state)
        model.eval()
        with torch.no_grad():
            X_cont_t = torch.from_numpy(X_cont).float().to(device)
            X_cat_t = torch.from_numpy(X_cat).long().to(device) if X_cat.shape[1] > 0 else None
            preds_z = model(X_cont_t, X_cat_t).cpu().numpy().ravel()
        preds_log = preds_z * y_std + y_mean
        preds_orig = np.expm1(preds_log)
        weight = weights[fold-1] if (len(weights) >= fold and len(weights) > 0) else 1.0/num_folds
        preds_ensemble += weight * preds_orig

    out_df = df.copy()
    out_df['Predicted_Rrp'] = preds_ensemble

    if save_predictions:
        out_path = os.path.join(model_dir, "in_memory_predictions.csv")
        out_df.to_csv(out_path, index=False)
        print("Saved predictions to", out_path)

    if TARGET_COLUMN in out_df.columns:
        trues = pd.to_numeric(out_df[TARGET_COLUMN], errors='coerce').values
        mask = ~np.isnan(trues)
        if mask.any():
            try:
                from sklearn.metrics import r2_score
                r2 = r2_score(trues[mask], preds_ensemble[mask]) if len(trues[mask])>1 else float('nan')
            except Exception:
                r2 = float('nan')
            mae = float(np.mean(np.abs(trues[mask] - preds_ensemble[mask])))
            mse = float(np.mean((trues[mask] - preds_ensemble[mask])**2))
            rmse = math.sqrt(mse)
            print("Metrics on provided Rrp: r2:", r2, "mae:", mae, "rmse:", rmse)

    return out_df, preds_ensemble

# ---------- CLI / example run ----------
if __name__ == "__main__":
    TRAIN_CSV = "corrosion_data_clustered.csv"
    NEW_CSV = "ni_agent.csv"
    print("Featurizing training (in memory)...")
    df_train_refined, encs, refined_cols = featurize_and_refine_training_inmemory(TRAIN_CSV, save_artifacts=False)
    print("Training refined shape:", df_train_refined.shape)
    print("Featurizing + aligning new data (in memory)...")
    df_ni_refined, used_encs = featurize_new_and_align_inmemory(NEW_CSV, encs, refined_cols, save_artifacts=False)
    print("NI refined shape:", df_ni_refined.shape)
    print("Running predictions (requires Models/ artifacts)...")
    try:
        out_df, preds = predict_on_dataframe_in_memory(df_ni_refined, label_encoders_provided=used_encs, model_dir=MODEL_DIR, device=DEVICE, save_predictions=False)
        print("Predictions done. Sample:")
        print(out_df[['Predicted_Rrp']].head())
    except Exception as e:
        print("Prediction failed:", repr(e))
        print("Check Models/ contains nn_preprocessor.joblib, embedding_config.joblib, target_scaler.joblib, nn_training_summary.json, and nn_best_fold{n}_swa.pth files.")
