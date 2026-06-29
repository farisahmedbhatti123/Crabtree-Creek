# -*- coding: utf-8 -*-
"""
=============================================================================
CRABTREE CREEK FLASH FLOOD FORECASTING FRAMEWORK
Multi-Horizon Deep Learning with Physics-Informed Feature Engineering
=============================================================================

-----------------------------------------------------------------------------
v4 CHANGELOG — review-driven sampling, multi-task, ensemble UQ, metrics
-----------------------------------------------------------------------------
  • DATA  — BalancedStormBatchSampler: event-weighted batches (40% flood /
            30% rising / 20% falling / 10% normal) replace random shuffle.
  • ARCH  — Multi-task HydroLSTMSeq2Seq: stage + flood exceedance + peak
            timing + heteroscedastic uncertainty (log-variance) heads.
  • LOSS  — Explicit d(stage)/dt derivative penalty alongside FloodAwareLoss.
  • THR   — USGS-informed tiered stages: P95 elevated (train) / 9 ft significant
            (eval+sampler) / 12 ft extreme (tail); focal BCE for rare peaks.
  • v5    — Perfect-prog future rain ON; sig-peak-weighted timing; rising-limb
            anticipation loss; stronger derivative/DTW/peak-magnitude terms.
  • v6    — Ordinal cumulative severity head P(>4,6,8,9); continuous stage L1
            weights; moderated tail alphas; reduced flood oversampling.
  • v6b   — EXCLUDE_TEST_YEARS for CV; RUN_FORWARD_EVAL for scoring a future
            calendar year with a model trained on all prior years.
  • UQ    — Deep ensemble (5 members) as primary UQ; MC-Dropout retained
            for calibration comparison (PICP_flood publishable finding).
  • FEATS — Storm age, rainfall centroid, dynamic hw70→US-1 xcorr travel time.
  • FEATS — LightGBM/SHAP feature pruning before training.
  • EVAL  — CRPS, ETS, threshold-crossing error, rising-limb-only RMSE.
  • CACHE — Trained LSTM / Transformer checkpoints saved to disk; reload on
            subsequent runs when features + architecture fingerprint matches.
-----------------------------------------------------------------------------

-----------------------------------------------------------------------------
v2 CHANGELOG — delayed-peak / phase-lag mitigation (see SECTION 2 config flags)
-----------------------------------------------------------------------------
The original framework exhibited a systematic late-peak bias (+40–58 min median,
+130–165 min on the largest events). The root causes were: (i) a "peak timing"
loss term built on a non-differentiable argmax that contributed zero gradient;
(ii) an MSE-dominated objective that rewards lag on uncertain sharp rises;
(iii) no future rainfall information; (iv) last-state model heads with no
machinery to place a peak at a lead time; (v) centred smoothing that leaked the
future and rounded peaks. This version addresses all of them:

  • LOSS  — FloodAwareLoss v2: differentiable, ASYMMETRIC soft-argmax timing
            penalty (late penalised harder) + optional soft-DTW shape/time term.
  • DATA  — Future forcing: perfect-prog rainfall over (t, t+h] as input
            features (swap for QPF/nowcast operationally). THE main lead-time
            lever. Strict past-only ablation via USE_FUTURE_FORCING=False.
  • ARCH  — Encoder–decoder Seq2Seq LSTM with per-horizon cross-attention, plus
            a dilated causal TCN front-end; Transformer now uses learned
            attention pooling over the full sequence instead of the last token.
  • FEATS — Multi-scale rate-of-rise (dStage/dt, dQ/dt), event-cumulative
            rainfall since storm onset, and multiple routing-lag channels.
  • EVAL  — Event-based skill (POD/FAR/CSI, peak bias, late-peak fraction)
            reported alongside aggregate NSE.
  • FIX   — Raw-data smoothing is now causal (trailing) instead of centred.
  • OPT   — Optional Stage-2 residual error-learning corrector (default OFF).

Every change is gated by a flag in SECTION 2 for clean ablation studies. All
model variants preserve the (B, LOOKBACK, F) -> (B, OUTPUT_STEPS) I/O contract,
so inference, MC-dropout, and plotting are unchanged. NOTE: the number of input
features increases (≈47 -> ≈85) because of the new feature groups; figure
titles that hard-code "47" are cosmetic and can be updated to len(feat_cols).
-----------------------------------------------------------------------------

STUDY OVERVIEW
--------------
This script implements a comparative deep learning framework for multi-horizon
water level forecasting at Crabtree Creek at US-1 Bridge, Raleigh, North
Carolina (USGS gauge 02087324). The study targets six operationally relevant
forecast horizons spanning 30 to 180 minutes, corresponding to the critical
warning window for flash flood response in urban watersheds.

Crabtree Creek is a representative urban flashy watershed exhibiting
sub-hour concentration times, impervious surface fractions exceeding 40%,
and documented flood response times as short as 45 minutes following intense
convective rainfall. These characteristics present a stringent test for
data-driven forecasting approaches and motivate the multi-horizon evaluation
design adopted here.

STUDY SITE
----------
Target gauge  : Crabtree Creek at US-1, Raleigh, NC (USGS 02087324)
Watershed area: approximately 113 km² (43.6 mi²)
Land cover    : predominantly urban (>40% impervious surface)
Climate       : humid subtropical; peak flooding April–September
Data period   : 2020-10-01 to 2025-10-01 (5-year record, 5-min resolution)

FORECAST DESIGN
---------------
Input window  : t−480 min to t  (8 hours of antecedent conditions, 96 steps)
Output targets: t+30, t+60, t+90, t+120, t+150, t+180 min (6 horizons)
Variable      : Gage height (ft) at USGS 02087324

This multi-horizon design enables evaluation of forecast skill degradation
with increasing lead time, which is essential for characterising the
operational utility of the forecasting system.

STATION NETWORK
---------------
Station              USGS ID      Precip  Gauge  Discharge  Role
-------------------------------------------------------------------
Crabtree @ Hwy 70    02087275     YES     YES    YES        Upstream precip+flow
Crabtree @ Ebenezer  0208726005   NO      YES    YES        Intermediate routing
Crabtree @ US-1      02087324     NO      YES    NO         TARGET
Crabtree @ Old WF    02087322     YES     YES    NO         Eastern sub-basin
Crabtree @ Anderson  0208731190   NO      YES    NO         Downstream ref.
Pigeon Creek         0208720525   NO      YES    NO         Tributary

FEATURE ENGINEERING SUMMARY (47 features total)
-------------------------------------------------
Group                          Features                              Count
Raw precipitation              hw70, old_wf instantaneous              2
Rolling accumulations          2 stations × 4 windows × 2 stats        16
Antecedent 225-min window      2 stations × (sum + variance)            4
Antecedent Precipitation Index 2 stations (exponential decay API)       2
Upstream gage heights          5 stations (hw70 to pigeon_ck)           5
Travel-time lagged gages       hw70 lag 60 min, ebenezer lag 30 min     2
Discharge + derivatives        2 stations × (raw + roll_max + dQ)       6
Eckhardt baseflow separation   2 stations × (baseflow + quickflow)       4
Travel-time lagged discharge   hw70 lag 60 min, ebenezer lag 30 min     2
Cyclic time encodings          sin/cos hour + sin/cos day-of-year        4
Total                                                                   47

MODEL ARCHITECTURES
-------------------
HydroLSTM      : 2-layer stacked LSTM → Dense(32) → ReLU → Dense(6)
HydroTransformer: InputProj → PosEmbed → TransformerEncoder(4L) → Dense(6)

LOSS FUNCTION
-------------
FloodAwareLoss (composite, 4 components):
  L = L_wtMSE + λ_q·L_pinball + λ_p·L_peak + λ_t·L_timing

  L_wtMSE  : tail-weighted MSE, flood events upweighted 6× (α=5, u=90th pct)
  L_pinball: asymmetric pinball at τ=0.95, underprediction penalty ×2
  L_peak   : batch-level peak magnitude |max(obs)−max(pred)|
  L_timing : peak timing error |argmax(obs)−argmax(pred)| in timesteps

UNCERTAINTY QUANTIFICATION
--------------------------
Monte Carlo Dropout: 100 stochastic forward passes with dropout active
  → mean prediction, std, 95% prediction interval [mean±2σ]
  → Evaluated via PICP, PICP_flood, PINAW metrics

EVALUATION METRICS
------------------
Overall    : RMSE (ft), MAE (ft), NSE, KGE, PBIAS (%)
Peak-focused: Peak-RMSE, Median peak timing error (min)
Uncertainty : PICP (%), PICP_flood (%), PINAW
Per-horizon : All metrics computed at each of 6 lead times

OUTPUTS (→ outputs_v6_causal/ by default; see OUTPUT_DIR)
--------------------------------
Per-model plots : training history, full record overlay, storm zoom panels,
                  feature importance (saliency/attention), monthly RMSE,
                  optuna_history, optuna_importance
Comparison plots: head-to-head overlay, scatter, metrics bar chart,
                  flow duration curve, timing error histogram,
                  uncertainty calibration, multi-horizon skill degradation,
                  storm comparison panels with uncertainty bands
CSV tables      : comparison_summary.csv, per_storm_metrics.csv,
                  uncertainty_metrics.csv, storm_catalog_test.csv,
                  per_horizon_metrics.csv,
                  lstm_optuna_trials.csv, transformer_optuna_trials.csv

HYPERPARAMETER OPTIMISATION
---------------------------
Optuna TPE (Tree-structured Parzen Estimator) Bayesian optimisation.
Each architecture is independently tuned using walk-forward TimeSeriesSplit
cross-validation on the TRAINING partition only. The TEST partition is
never accessed during hyperparameter search, preventing data leakage.

Search spaces:
  HydroLSTM:
    hidden_size_1   : categorical [64, 128, 256, 512]
    hidden_size_2   : categorical [32, 64, 128, 256]
    learning_rate   : log-uniform [1e-5, 1e-3]
    dropout         : uniform [0.1, 0.5]
    weight_decay    : log-uniform [1e-6, 1e-3]

  HydroTransformer:
    d_model         : categorical [64, 128, 256]
    nhead           : categorical [4, 8]   (must divide d_model)
    num_layers      : int [2, 6]
    learning_rate   : log-uniform [1e-5, 1e-3]
    dropout         : uniform [0.05, 0.4]
    weight_decay    : log-uniform [1e-6, 1e-3]

Configuration:
  OPTUNA_N_TRIALS      = 50   (trials per architecture)
  OPTUNA_CV_FOLDS      = 3    (TimeSeriesSplit folds)
  OPTUNA_EPOCHS        = 30   (max epochs per fold — short for speed)
  OPTUNA_PATIENCE      = 5    (early stopping per fold)
  OPTUNA_SKIP          = False (set True to use hardcoded fallback params)

INSTALLATION
------------
Mac (Apple Silicon M-series):
    pip install pandas numpy scikit-learn matplotlib seaborn dataretrieval
    pip install torch torchvision torchaudio   # MPS-enabled build
    pip install scipy optuna

Linux / Windows (CUDA):
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    pip install pandas numpy scikit-learn matplotlib seaborn dataretrieval scipy optuna

REFERENCES
----------
Hochreiter & Schmidhuber (1997) — Long Short-Term Memory. Neural Computation.
Vaswani et al. (2017) — Attention Is All You Need. NeurIPS.
Lim et al. (2021) — Temporal Fusion Transformers. Int. J. Forecasting.
Kratzert et al. (2019) — Towards learning universal hydrological behaviours.
Nash & Sutcliffe (1970) — River flow forecasting. J. Hydrology.
Gupta et al. (2009) — Decomposition of the KGE. J. Hydrology.
Eckhardt (2005) — How to construct recursive digital baseflow filters.
Abnar & Zuidema (2020) — Quantifying attention flow. ACL.
Gal & Ghahramani (2016) — Dropout as Bayesian approximation. ICML.
Rasiya Koya & Roy (2024) — TFT for streamflow prediction. J. Hydrology.
=============================================================================
"""

# =============================================================================
# SECTION 0 — IMPORTS AND PACKAGE AVAILABILITY CHECK
# =============================================================================

import os
import sys
import json
import pickle
import hashlib
import random
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from copy import deepcopy
from scipy.signal import find_peaks

import matplotlib
matplotlib.rcParams.update({
    "font.family"     : "DejaVu Sans",
    "font.size"       : 11,
    "axes.titlesize"  : 12,
    "axes.labelsize"  : 11,
    "xtick.labelsize" : 10,
    "ytick.labelsize" : 10,
    "legend.fontsize" : 9,
    "figure.dpi"      : 150,
    "savefig.dpi"     : 300,
    "figure.facecolor": "white",
    "axes.facecolor"  : "white",
    "axes.spines.top" : False,
    "axes.spines.right": False,
    "axes.grid"       : True,
    "grid.alpha"      : 0.3,
    "grid.linestyle"  : "--",
})
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import seaborn as sns
from itertools import product as itertools_product

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Sampler
import torch.nn.functional as F

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

# Optuna — Bayesian hyperparameter optimisation
try:
    import optuna
    from optuna.samplers import TPESampler
    from optuna.pruners import MedianPruner
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("[WARNING] optuna not installed. Run: pip install optuna")
    print("          Falling back to hardcoded hyperparameters.")

# Optional USGS data retrieval — falls back to synthetic data if unavailable
try:
    from dataretrieval import nwis
    NWIS_AVAILABLE = True
except ImportError:
    NWIS_AVAILABLE = False
    print("[WARNING] dataretrieval not installed. Run: pip install dataretrieval")
    print("          Falling back to physics-inspired synthetic data.")

warnings.filterwarnings("ignore")

# =============================================================================
# SECTION 1 — REPRODUCIBILITY
# All random number generators are seeded identically to ensure that every
# training run produces bit-identical results. This is a prerequisite for
# scientific reproducibility and peer review.
# =============================================================================

GLOBAL_SEED = 42


def set_all_seeds(seed: int = GLOBAL_SEED) -> None:
    """
    Seed all random number generators used by Python, NumPy, and PyTorch.

    Parameters
    ----------
    seed : int
        Master seed value. Default 42 is used throughout the study.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


set_all_seeds()


def _worker_init_fn(worker_id: int) -> None:
    """
    DataLoader worker initialiser ensuring each worker receives a unique
    but deterministic seed derived from the global seed.
    """
    np.random.seed(GLOBAL_SEED + worker_id)
    random.seed(GLOBAL_SEED + worker_id)


# =============================================================================
# SECTION 2 — CONFIGURATION
# All tuneable constants are defined here. Modifying values in this section
# is sufficient to reconfigure the entire pipeline without editing model or
# training code.
# =============================================================================

# ── Temporal resolution ───────────────────────────────────────────────────────
TIMESTEP_MIN = 5                                    # USGS IV resolution (minutes)

# ── Input window ──────────────────────────────────────────────────────────────
# v3: cut from 480 -> 120 min. Empirical lookback sweep (flood-conditioned NSE)
# saturates by 60-120 min and DEGRADES beyond it at long horizons; physically the
# upstream->US-1 travel time is <=25 min, rainfall response <=40 min and US-1
# momentum (rate-of-change autocorrelation) decays to ~0 by ~120 min. The old
# 96-step window was ~4x larger than the information content and diluted the
# recent signal with noise. 24 steps is ~4x cheaper and slightly better.
LOOKBACK_MIN   = 120                                # was 480 (v3 change)
LOOKBACK_STEPS = LOOKBACK_MIN // TIMESTEP_MIN       # = 24 time steps (was 96)

# ── Forecast horizons ─────────────────────────────────────────────────────────
# Six operationally relevant lead times spanning 0.5 to 3 hours.
# These correspond to standard National Weather Service warning thresholds.
FORECAST_HORIZONS_MIN   = [30, 60, 90, 120, 150, 180]
FORECAST_HORIZONS_STEPS = [h // TIMESTEP_MIN for h in FORECAST_HORIZONS_MIN]
# = [6, 12, 18, 24, 30, 36]
OUTPUT_STEPS = len(FORECAST_HORIZONS_MIN)           # = 6

# ── Study period ──────────────────────────────────────────────────────────────
START_DATE = "2020-01-01"
END_DATE   = "2025-10-01"   # primary study / training span (through 2025)

# ── USGS site identifiers ─────────────────────────────────────────────────────
# Verified against USGS NWIS monitoring pages.
# Parameter codes:
#   00045 = Precipitation, total, inches (tipping-bucket)
#   00065 = Gage height, feet
#   00060 = Discharge, cubic feet per second
USGS_SITES = {
    "hw70"       : "02087275",    # Crabtree Ck @ Hwy 70          — precip + gauge + Q
    "ebenezer"   : "0208726005",  # Crabtree Ck @ Ebenezer Ch Rd  — gauge + Q
    "us1"        : "02087324",    # Crabtree Ck @ US-1  [TARGET]  — gauge only
    "old_wf"     : "02087322",    # Crabtree Ck @ Old Wake Forest — precip + gauge
    "anderson_dr": "0208731190",  # Crabtree Ck @ Anderson Dr     — gauge only
    "pigeon_ck"  : "0208720525",  # Pigeon Creek tributary        — gauge only
}

# ── Data processing ───────────────────────────────────────────────────────────
SMOOTH_WINDOW = 3                       # 3-point centred rolling mean on raw data
MERGE_TOL     = pd.Timedelta("10min")  # asof merge tolerance

# ── Dataset split fractions ───────────────────────────────────────────────────
TRAIN_FRAC = 0.70
VAL_FRAC   = 0.15
TEST_FRAC  = 0.15

# ── Training hyperparameters ──────────────────────────────────────────────────
EPOCHS     = 100
BATCH_SIZE = 64
PATIENCE   = 15

# ── Antecedent Precipitation Index decay constant ────────────────────────────
# k = 0.92^(TIMESTEP_MIN/1440) gives approximately 8-day soil moisture memory.
# Reference: standard API formulation (Kohler & Linsley, 1951).
K_API_DECAY = 0.92 ** (TIMESTEP_MIN / 1440.0)

# ── Eckhardt baseflow filter parameters ──────────────────────────────────────
# alpha  = 0.925: recession constant for perennial streams (Eckhardt, 2005)
# BFImax = 0.80 : maximum baseflow index for perennial streams
ECKHARDT_ALPHA  = 0.925
ECKHARDT_BFIMAX = 0.80

# ── Tiered stage thresholds (USGS 02087324, 2020–2025 IV analysis) ─────────────
# Elevated (P95 ≈ 3.7 ft): training reweighting only — NOT operational flood.
# Significant (9 ft): ~0.37% of timesteps, ~26 peaks/5 yr — primary eval target.
# Extreme (12 ft): ~0.06% of timesteps, ~6 peaks/5 yr — tail generalisation.
ELEVATED_PERCENTILE   = 95
SIGNIFICANT_STAGE_FT  = 9.0
EXTREME_STAGE_FT      = 12.0
FLOOD_PERCENTILE      = ELEVATED_PERCENTILE   # legacy alias (train reweight only)

# ── FloodAwareLoss hyperparameters ───────────────────────────────────────────
# L1 weighting: smooth convex growth above elevated_ft (replaces cliff alphas).
USE_CONTINUOUS_STAGE_WEIGHT = True
STAGE_WEIGHT_K              = 0.85    # multiplier growth per ft above elevated
STAGE_WEIGHT_GAMMA          = 1.35    # convex — emphasises tail without cliffs
STAGE_WEIGHT_CAP            = 12.0    # max L1 multiplier

# Legacy cliff alphas — used only when USE_CONTINUOUS_STAGE_WEIGHT=False
FLOOD_ALPHA           = 4.0     # upweight elevated (P95) timesteps in L1
SIGNIFICANT_ALPHA     = 8.0     # extra L1 weight when obs >= 9 ft (was 20)
EXTREME_ALPHA         = 12.0    # extra L1 weight when obs >= 12 ft (was 40)
LOSS_LAMBDA_Q         = 0.3     # pinball loss coefficient
LOSS_LAMBDA_P         = 0.45    # peak magnitude (significant-event windows)
LOSS_LAMBDA_P_UNDER   = 0.35    # extra squared penalty for underpredicting crest >=9 ft
LOSS_LAMBDA_T         = 0.1     # peak timing loss coefficient
LOSS_TAU              = 0.95    # quantile level for pinball loss
LOSS_UNDERPRED_FACTOR = 3.0     # extra pinball penalty underpredicting >=9 ft
USE_FLOOD_FOCAL_LOSS  = True    # focal BCE on rare >=9 ft classification head
FLOOD_FOCAL_GAMMA     = 2.0
LOSS_LAMBDA_Q_SIGNIFICANT = 0.15  # extra pinball on >=9 ft timesteps only

# ── MC Dropout samples ────────────────────────────────────────────────────────
MC_SAMPLES = 100

# =============================================================================
# CONFIGURATION — v4 REVIEW IMPROVEMENTS (sampler, multi-task, ensemble, feats)
# =============================================================================

# (A) Event-weighted storm sampler — each mini-batch draws from flood /
#     rising-limb / falling-limb / normal windows at hydrologically motivated
#     ratios (attacks 98%-baseflow imbalance without over-sampling recession).
USE_BALANCED_STORM_SAMPLER = True
STORM_SAMPLER_WEIGHTS   = (0.32, 0.38, 0.15, 0.15)   # flood@9ft, rising, falling, normal
FLOOD_SAMPLER_ALWAYS_REPLACE = False  # avoid memorising ~10–15 training storms
RISE_RATE_THRESH_FT_PER_MIN = 0.004   # rising/falling limb classifier threshold

# (B) Multi-task auxiliary heads on HydroLSTMSeq2Seq.
USE_FLOOD_AUX_HEAD      = True    # ordinal cumulative exceedance head
USE_ORDINAL_SEVERITY_HEAD = True  # P(>4), P(>6), P(>8), P(>9) with monotonicity
ORDINAL_STAGE_LEVELS_FT = (4.0, 6.0, 8.0, 9.0)
ORDINAL_LEVEL_WEIGHTS   = (0.15, 0.25, 0.30, 0.30)   # rarer levels weighted higher
ORDINAL_MONOTONE_PENALTY = 0.5    # penalise P(>6) > P(>4) etc.
ORDINAL_FOCAL_FROM_IDX  = 2       # focal BCE from 8 ft level upward (rare classes)
USE_PEAK_AUX_HEAD       = True    # which horizon contains the peak (softmax)
LOSS_LAMBDA_FLOOD_CLS   = 0.5     # ordinal / exceedance auxiliary head
LOSS_LAMBDA_PEAK_CLS    = 0.25

# (B2) Explicit hydrograph derivative loss: penalise d(stage)/dt mismatch.
USE_DERIVATIVE_LOSS     = True
LOSS_LAMBDA_DERIV       = 0.30
DERIVATIVE_DT_MIN       = 30      # minutes between consecutive forecast horizons

# (B3) Rising-limb anticipation — penalise predicted rise rate lagging observed.
USE_RISING_LIMB_ANTICIPATION_LOSS = True
LOSS_LAMBDA_RISE_LAG    = 0.35

# (C) Deep ensemble + heteroscedastic output (primary UQ); MC-Dropout kept for comparison.
USE_DEEP_ENSEMBLE       = False
ENSEMBLE_N_MODELS       = 5
USE_HETEROSCEDASTIC     = True
KEEP_MC_DROPOUT         = True    # publishable miscalibration baseline (PICP_flood)

# (D) Storm-dynamic features not already covered by antecedent / flashiness groups.
USE_STORM_AGE_FEAT              = True   # minutes since storm onset
USE_RAINFALL_CENTROID_FEAT      = True   # rainfall center-of-mass within event
USE_DYNAMIC_TRAVEL_TIME_FEAT    = True   # per-storm cross-correlation lag (hw70→US-1)

# (E) Feature pruning via LightGBM / SHAP before deep-model training.
RUN_FEATURE_PRUNING     = True
FEATURE_PRUNING_TOP_K   = 55    # retain top-K of ~73–85 features (redundancy trim)
FEATURE_PRUNING_MIN_KEEP = 30   # safety floor — never prune below this count

# (F) Model checkpointing — skip re-training when saved weights are valid.
SAVE_TRAINED_MODELS = True     # persist weights + scaler after each training run
USE_MODEL_CACHE     = True     # load from disk when a matching checkpoint exists
FORCE_RETRAIN       = False     # True → ignore cache and retrain everything

# =============================================================================
# CONFIGURATION — v2 IMPROVEMENTS (delayed-peak / phase-lag mitigation)
# ---------------------------------------------------------------------------
# Master switches for the upgrades added to address the systematic late-peak
# bias. Each can be toggled independently for ablation studies. Defaults are
# the recommended "all-on" configuration except residual correction, which is
# an optional second stage kept OFF by default.
# =============================================================================

# (1) Architecture: use the encoder–decoder Seq2Seq LSTM and attention-pooled
#     Transformer instead of the last-state → MLP heads. Same (B,L,F)→(B,6)
#     I/O contract, so the rest of the pipeline is unchanged.
USE_IMPROVED_MODELS = True
USE_TCN_FRONTEND    = True      # dilated TCN front-end before the encoder
TCN_N_BLOCKS        = 2         # v3: trimmed 3->2 blocks. With a 120-min window a
                                # geometric dilation stack spanning 8 h attends to
                                # noise; 2 blocks give a ~2 h receptive field that
                                # matches the new lookback.

# =============================================================================
# CONFIGURATION — v3 EVALUATION REDESIGN (extreme-aware, leakage-safe)
# ---------------------------------------------------------------------------
# The single chronological 70/15/15 split under-tested the dangerous tail: the
# largest events (>12 ft) all fell in train/val, so generalisation to record
# floods was never measured. v3 replaces it with EXPANDING-WINDOW ROLLING-ORIGIN
# cross-validation split by calendar year. Extremes occur in EVERY year, so each
# held-out test year contains a >11 ft event while training stays strictly causal
# (past-only). Metrics are aggregated as mean ± std across folds.
# =============================================================================

# (A) Rolling-origin (expanding-window) cross-validation, split by year.
USE_ROLLING_ORIGIN_CV = True    # v3 primary evaluation. False -> legacy chrono split.
CV_MIN_TRAIN_YEARS    = 2       # first fold trains on >=2 years before testing
CV_VAL_FRACTION       = 0.15    # last 15% of each fold's train block -> validation
# Embargo: drop the last max-horizon steps of each train block so no training
# target (shifted forward by up to +180 min) leaks into the held-out test year.
CV_EMBARGO_STEPS      = max(FORECAST_HORIZONS_STEPS)   # = 36 steps (180 min)
# Fold used for the detailed publication figures (storm panels, saliency, etc.).
# -1 = last remaining fold after EXCLUDE_TEST_YEARS is applied.
SHOWCASE_FOLD_INDEX   = -1

# Drop these calendar years from rolling-origin TEST folds (train blocks unchanged
# for remaining folds). Example: [2025] → CV tests 2022–2024 only; 2025 data may
# still be in the dataset for forward-evaluation training.
EXCLUDE_TEST_YEARS    = []

# ── Forward evaluation: train on all years before FORWARD_EVAL_YEAR, score that year
# Set RUN_FORWARD_EVAL=True to score 2026 (or any future year) after extending the
# download through FORWARD_EVAL_END_DATE. Requires a one-time train → lstm_forward
# checkpoint (or set FORWARD_FORCE_RETRAIN=True). This is NOT the same as reusing
# lstm_fold4 (fold 4 trains only through 2024 and tests 2025).
RUN_FORWARD_EVAL        = True
FORWARD_EVAL_YEAR       = 2026
FORWARD_EVAL_END_DATE   = "2026-10-01"   # USGS download span for forward year
FORWARD_CHECKPOINT_TAG  = "lstm_forward"
FORWARD_FORCE_RETRAIN   = False          # load lstm_forward if fingerprint matches
SKIP_ROLLING_CV         = True          # True → only forward eval, skip CV loop

# (B) Demote the Transformer to an optional baseline. The data has short memory
#     (<=2 h), so self-attention's long-range strength is wasted and it
#     under-performs the LSTM here. To honour the demotion without losing the
#     comparison figures, the Transformer is trained ONLY on the showcase fold
#     (once) as a baseline line in the plots — the LSTM is the primary model and
#     is the one evaluated across ALL rolling-origin folds. Set False to drop it
#     entirely (Transformer-specific plots are then skipped gracefully).
RUN_TRANSFORMER       = True    # v3: baseline, trained on showcase fold only

# (C) Gradient-boosted-tree baseline the deep model MUST beat. A lagged-feature
#     GBM already reaches flood-NSE ~0.95/0.87 at +30/+60 min; if HydroLSTM does
#     not clear it, the deep net is not earning its complexity. Printed per fold.
RUN_GBM_BASELINE      = True    # v3: HistGradientBoosting flood baseline

# (2) Future forcing (perfect-prog by default): inject the rainfall that will
#     fall over (t, t+h] as input features. This is the single largest lever
#     on lead time — without it the model stays reactive and peaks lag by ~1×
#     the forecast horizon. v6 default run: OFF (strict past-only / causal).
#     For skill-ceiling experiments set True (perfect_prog rain over (t,t+h]).
USE_FUTURE_FORCING  = True
FUTURE_FORCING_MODE = "perfect_prog"   # {"perfect_prog", "qpf", "none"}

# (3) Loss: replace the dead argmax timing term with a differentiable
#     soft-argmax timing penalty (asymmetric: late peaks penalised harder)
#     plus an optional within-window soft-DTW shape/time term.
USE_SOFT_TIMING     = True
SOFT_TIMING_BETA    = 12.0      # sharper soft-argmax — tighter peak placement
LOSS_LAMBDA_T_LATE  = 0.50      # weight on LATE peak timing (dangerous)
LOSS_LAMBDA_T_EARLY = 0.05      # weight on EARLY peak timing error
USE_SOFT_DTW        = True
LOSS_LAMBDA_DTW     = 0.25      # weight on soft-DTW shape/time term
SOFT_DTW_GAMMA      = 0.1       # smoothing of soft-DTW (smaller = closer DTW)

# (4) Feature engineering additions.
USE_RISE_RATE_FEATS = True      # multi-scale dStage/dt and dQ/dt
USE_EVENT_CUM_RAIN  = True      # rainfall accumulated since storm onset
USE_MULTI_LAG_GAUGE = True      # several routing-lag channels (learned weighting)
USE_US1_AUTOREGRESSIVE = True   # v3: US-1's OWN observed past (level, lags, rise
                                # rate, rolling max). This was MISSING — gauge_us1
                                # was only the target, never an input — yet at
                                # issue time t it is observed and is the single
                                # strongest short-horizon predictor (it is exactly
                                # what the persistence baseline uses).
US1_LAG_STEPS_MIN   = [30, 60, 90]    # autoregressive lag channels for US-1
MULTI_LAG_STEPS_MIN = [30, 60, 90]   # gauge lag channels added per main station
RISE_RATE_SCALES_MIN = [5, 15, 30]   # finite-difference scales for rate features
DRY_SPELL_RESET_MIN = 180       # gap of near-zero rain that resets event accum

# (5) Optional residual error-learning second stage (Stage-2 corrector).
#     Trains a small model on the residuals of Stage-1 and adds the correction.
#     Kept OFF by default: enable once the Tier-1/2 upgrades are validated.
USE_RESIDUAL_CORRECTION = False
RESIDUAL_HIDDEN         = 96
RESIDUAL_EPOCHS         = 40

# (6) Physical post-processing: gage height cannot go below the channel
#     invert. Clamping removes unphysical sub-zero predictions that bias
#     PBIAS low and distort the flow-duration curve tail. Set to None to
#     disable. A small negative floor tolerates minor datum noise.
CLAMP_STAGE_FLOOR = -0.5   # ft; set None to disable

# (7) Future-forcing ablation: emulate an operational rainfall nowcast by
#     degrading the perfect-prog future rain with multiplicative + additive
#     error and temporal blurring. QPF_ERROR_LEVEL=0 -> perfect; larger ->
#     noisier nowcast. Used by FUTURE_FORCING_MODE="qpf" and the ablation
#     harness. Replace with a real MRMS/HRRR feed for an operational study.
QPF_ERROR_LEVEL   = 0.35   # 0..1, fractional nowcast error
QPF_BLUR_STEPS    = 6      # temporal smoothing window (steps) of the nowcast
# v3: run the 3-way QPF skill matrix — perfect-prog (ceiling) / blurred nowcast
# (operational) / gauges-only (floor) — because the entire +120/+180 flood skill
# comes from the future-rain assumption (flood-NSE goes -0.60 -> +0.67 at +180 when
# perfect rain is added). Report the bracket, never the ceiling alone.
RUN_FORCING_ABLATION = False   # if True, main() runs the 3-mode comparison

# =============================================================================
# OPTUNA HYPERPARAMETER SEARCH CONFIGURATION
# ---------------------------------------------------------------------------
# OPTUNA_SKIP = True   → Uses hardcoded params below (current setting)
# OPTUNA_SKIP = False  → Re-runs full Bayesian search (only if needed)
#
# LSTM params: Found by Optuna TPE, 50 trials × 3 folds on training partition
#              Run date: 2026-06-19 on Google Colab A100 GPU
#              Best trial #48, best mean CV val loss = 0.48674
#
# Transformer params: Manually tuned (Optuna search timed out on A100 after
#                     17 hours — LSTM search consumed full session).
#                     Architecture confirmed optimal by LSTM search pattern.
# =============================================================================
OPTUNA_SKIP     = True    # Search already done — params hardcoded below
OPTUNA_N_TRIALS = 50      # kept for reference (used during original search)
OPTUNA_CV_FOLDS = 3       # kept for reference
OPTUNA_EPOCHS   = 30      # kept for reference
OPTUNA_PATIENCE = 5       # kept for reference

# ── LSTM: Optuna TPE best parameters (Trial #48, val_loss=0.48674) ───────────
# hidden_sizes confirmed same as manual — architecture was already optimal.
# Key changes vs manual: lr 2.4× lower, dropout 0.4→0.149, weight_decay 11× higher.
LSTM_BEST_PARAMS = {
    "hidden_sizes" : (256, 128),          # confirmed by Optuna (same as manual)
    "learning_rate": 4.248e-05,           # Optuna found (was 1e-4)
    "dropout"      : 0.1491,              # Optuna found (was 0.4)
    "weight_decay" : 1.141e-04,           # Optuna found (was 1e-5)
}

# ── Transformer: manually tuned parameters ────────────────────────────────────
# Optuna search not completed due to compute time constraints.
# These values produced NSE=0.905, RMSE=0.387 ft at t+150 min in prior run.
TRANSFORMER_BEST_PARAMS = {
    "d_model"      : 128,
    "nhead"        : 8,
    "num_layers"   : 4,
    "learning_rate": 1e-4,
    "dropout"      : 0.25,
    "weight_decay" : 1e-5,
}

# ── Device selection ──────────────────────────────────────────────────────────
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")     # Apple Silicon M-series
else:
    DEVICE = torch.device("cpu")

# ── Output directory ──────────────────────────────────────────────────────────
OUTPUT_DIR = Path("outputs_v6_causal")
OUTPUT_DIR.mkdir(exist_ok=True)

# Preprocessed-data persistence: download raw -> engineer features -> SAVE to
# disk BEFORE training, so the dataset is cached, inspectable and reproducible.
PREPROCESSED_DIR       = OUTPUT_DIR / "preprocessed"
PREPROCESSED_DIR.mkdir(exist_ok=True)
CHECKPOINT_DIR         = OUTPUT_DIR / "checkpoints"
CHECKPOINT_DIR.mkdir(exist_ok=True)
SAVE_PREPROCESSED      = True    # write raw + engineered features before training


def _data_end_date() -> str:
    """Download/engineer features through END_DATE, extended for forward eval."""
    end = pd.Timestamp(END_DATE)
    if RUN_FORWARD_EVAL and FORWARD_EVAL_END_DATE:
        end = max(end, pd.Timestamp(FORWARD_EVAL_END_DATE))
    return end.strftime("%Y-%m-%d")

USE_PREPROCESSED_CACHE = True   # if True, load cached features when END_DATE matches

print("=" * 70)
print("  CRABTREE CREEK FLASH FLOOD FORECASTING FRAMEWORK")
print("=" * 70)
print(f"  Device       : {DEVICE}")
print(f"  Lookback     : {LOOKBACK_MIN} min  ({LOOKBACK_STEPS} steps)")
print(f"  Horizons     : {FORECAST_HORIZONS_MIN} min")
print(f"  Output steps : {OUTPUT_STEPS}")
print(f"  Study period : {START_DATE} → {_data_end_date()}"
      f"{' (END_DATE=' + END_DATE + ')' if _data_end_date() != END_DATE else ''}")
if EXCLUDE_TEST_YEARS:
    print(f"  CV exclude   : test years {EXCLUDE_TEST_YEARS}")
if RUN_FORWARD_EVAL:
    print(f"  Forward eval : year {FORWARD_EVAL_YEAR}  "
          f"(checkpoint={FORWARD_CHECKPOINT_TAG})")
print(f"  Output dir   : {OUTPUT_DIR.resolve()}/")
print(f"  Model cache  : {'ON' if USE_MODEL_CACHE else 'OFF'}"
      f"{' (FORCE_RETRAIN)' if FORCE_RETRAIN else ''}"
      f" → {CHECKPOINT_DIR.resolve()}/")
print(f"  Optuna       : {'ENABLED' if (OPTUNA_AVAILABLE and not OPTUNA_SKIP) else 'DISABLED (fallback params)'}")
if OPTUNA_AVAILABLE and not OPTUNA_SKIP:
    print(f"  Optuna trials: {OPTUNA_N_TRIALS} per architecture")
    print(f"  Optuna folds : {OPTUNA_CV_FOLDS} (walk-forward TimeSeriesSplit)")
print("=" * 70)


# =============================================================================
# SECTION 3 — DATA LOADING
# Instantaneous-value (IV) data are retrieved from the USGS National Water
# Information System (NWIS) API via the dataretrieval package. A 3-point
# rolling mean is applied to suppress tipping-bucket artefacts. Multiple
# station time series are merged using an asof (nearest-neighbour in time)
# join with a ±10-minute tolerance, then resampled to a uniform 5-minute
# grid. Gaps of ≤30 minutes are filled by linear time interpolation.
# =============================================================================

def download_usgs_5min(site_id: str,
                       param_codes: list,
                       start: str,
                       end: str) -> pd.DataFrame:
    """
    Retrieve USGS NWIS instantaneous-value data for a single site.

    Parameters
    ----------
    site_id     : 8-digit USGS site number (string)
    param_codes : list of USGS parameter codes, e.g. ['00065', '00045']
    start / end : ISO date strings 'YYYY-MM-DD'

    Returns
    -------
    pd.DataFrame with DatetimeIndex (UTC→local) and one column per parameter.
    Returns empty DataFrame if retrieval fails.
    """
    if not NWIS_AVAILABLE:
        return pd.DataFrame()

    try:
        df, _ = nwis.get_iv(sites=site_id,
                             parameterCd=param_codes,
                             start=start,
                             end=end)
        if df.empty:
            print(f"    [WARNING] No data returned for site {site_id}")
            return pd.DataFrame()

        df.index = pd.to_datetime(df.index, utc=True).tz_convert("US/Eastern")
        df.index = df.index.tz_localize(None)
        df.index.name = "datetime"

        # Retain only numeric columns (drop USGS qualifier codes)
        df = df.select_dtypes(include=[np.number])

        # Remove duplicate timestamps produced by DST fall-back transitions.
        # At the autumn clock change, two readings share the same local time;
        # keeping the last occurrence is the standard convention.
        if df.index.duplicated().any():
            n_dups = int(df.index.duplicated().sum())
            print(f"    [INFO] Dropping {n_dups} duplicate timestamps "
                  f"(DST artefacts) at site {site_id}")
            df = df[~df.index.duplicated(keep="last")]

        # Ensure strictly monotonic index after duplicate removal
        df = df.sort_index()

        # Suppress tipping-bucket noise with a CAUSAL (trailing) rolling mean.
        # v2 fix: a centred window uses t+1 to compute t, which (a) leaks the
        # future into the target gauge and (b) rounds off the rising-limb sharpness
        # and peak magnitude the model must learn. Trailing-only avoids both.
        df = df.rolling(SMOOTH_WINDOW, center=False, min_periods=1).mean()

        return df

    except Exception as exc:
        print(f"    [ERROR] Failed to retrieve site {site_id}: {exc}")
        return pd.DataFrame()


def save_preprocessed_data(df_raw, df_feat, feat_cols, target_cols,
                           df_tr=None, df_va=None, df_te=None):
    """
    Persist the downloaded + preprocessed dataset to PREPROCESSED_DIR BEFORE
    training. This caches the data (so it can be reloaded via
    USE_PREPROCESSED_CACHE without re-downloading), makes it inspectable, and
    documents exactly what the models were trained on for reproducibility.
    """
    PREPROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    if df_raw is not None:
        df_raw.to_csv(PREPROCESSED_DIR / "01_raw_merged.csv")
    df_feat.to_csv(PREPROCESSED_DIR / "02_features_full.csv")
    (PREPROCESSED_DIR / "06_feature_list.txt").write_text("\n".join(feat_cols))
    if df_tr is not None:
        lines = []
        for name, d in [("Train", df_tr), ("Val", df_va), ("Test", df_te)]:
            lines.append(f"{name}: {d.index.min()} -> {d.index.max()} "
                         f"({len(d)} rows)")
        lines.append(f"Features: {len(feat_cols)} | Targets: {len(target_cols)}")
        (PREPROCESSED_DIR / "07_split_summary.txt").write_text("\n".join(lines))
    (PREPROCESSED_DIR / "data_end_date.txt").write_text(_data_end_date())
    print(f"  Saved preprocessed data -> {PREPROCESSED_DIR}/ "
          f"(raw, {len(feat_cols)}-feature matrix, split summary)")


def build_raw_dataframe() -> pd.DataFrame:
    """
    Construct the master merged DataFrame from all USGS stations.

    Processing steps:
        1. Download each station's precipitation and/or gage height and
           discharge data from USGS NWIS.
        2. Resample each series to a uniform 5-minute DatetimeIndex.
        3. Sequentially merge all series into a single DataFrame using
           an asof join (nearest timestamp within ±MERGE_TOL).
        4. Resample merged result to enforce uniform 5-minute spacing.
        5. Fill gaps ≤30 minutes by linear time interpolation; fill
           remaining edge gaps ≤10 minutes by forward/backward fill.

    Returns
    -------
    pd.DataFrame with columns named {variable}_{station} and a uniform
    5-minute DatetimeIndex spanning START_DATE to END_DATE.
    """
    if not NWIS_AVAILABLE:
        print("  [INFO] NWIS unavailable — generating synthetic data ...")
        return _build_synthetic_dataframe()

    print("=== Downloading USGS NWIS data ===")

    # Uniform 5-minute reference index for resampling
    ref_idx = pd.date_range(START_DATE, _data_end_date(), freq="5min")
    series_dict = {}

    # ── Precipitation stations ─────────────────────────────────────────────
    precip_sites = {"hw70": USGS_SITES["hw70"],
                    "old_wf": USGS_SITES["old_wf"]}
    for alias, site_id in precip_sites.items():
        print(f"  Downloading precip @ {alias} ({site_id}) ...")
        df = download_usgs_5min(site_id, ["00045"], START_DATE, _data_end_date())
        if not df.empty:
            col = df.columns[0]
            s   = (df[col]
                   .clip(lower=0)
                   .reindex(ref_idx, method="nearest", tolerance=MERGE_TOL)
                   .fillna(0))
            series_dict[f"precip_{alias}"] = s
        else:
            series_dict[f"precip_{alias}"] = pd.Series(0.0, index=ref_idx)

    # ── Gage height stations ───────────────────────────────────────────────
    gauge_sites = {k: v for k, v in USGS_SITES.items()}
    for alias, site_id in gauge_sites.items():
        print(f"  Downloading gage height @ {alias} ({site_id}) ...")
        df = download_usgs_5min(site_id, ["00065"], START_DATE, _data_end_date())
        if not df.empty:
            col = df.columns[0]
            s   = (df[col]
                   .clip(lower=0)
                   .reindex(ref_idx, method="nearest", tolerance=MERGE_TOL))
            series_dict[f"gauge_{alias}"] = s

    # ── Discharge stations ─────────────────────────────────────────────────
    discharge_sites = {"hw70": USGS_SITES["hw70"],
                       "ebenezer": USGS_SITES["ebenezer"]}
    for alias, site_id in discharge_sites.items():
        print(f"  Downloading discharge @ {alias} ({site_id}) ...")
        df = download_usgs_5min(site_id, ["00060"], START_DATE, _data_end_date())
        if not df.empty:
            col = df.columns[0]
            s   = (df[col]
                   .clip(lower=0)
                   .reindex(ref_idx, method="nearest", tolerance=MERGE_TOL))
            series_dict[f"q_{alias}"] = s

    # ── Assemble and clean ─────────────────────────────────────────────────
    merged = pd.DataFrame(series_dict, index=ref_idx)
    merged.index.name = "datetime"

    # Linear interpolation for gaps ≤30 minutes (6 steps)
    merged = merged.interpolate(method="time", limit=6)
    # Edge fill for gaps ≤10 minutes (2 steps)
    merged = merged.ffill(limit=2).bfill(limit=2)

    print("\n  Data availability summary:")
    for col in merged.columns:
        pct  = merged[col].notna().mean() * 100
        flag = "✓" if pct >= 90 else "⚠"
        print(f"    {flag}  {col:<35} {pct:.1f}% available")

    print(f"\n  Merged DataFrame shape: {merged.shape}")
    return merged


def _build_synthetic_dataframe() -> pd.DataFrame:
    """
    Generate physics-inspired synthetic Crabtree Creek data for testing
    when the USGS API is unavailable.

    A stochastic precipitation pulse train drives a gamma-function
    hydrograph response at the target gauge. Upstream stations receive
    time-lagged and spatially-scaled versions of the same response,
    approximating travel-time dynamics observed in the real network.

    Returns
    -------
    pd.DataFrame matching the schema of build_raw_dataframe().
    """
    print("  Generating synthetic Crabtree Creek data ...")
    idx = pd.date_range(START_DATE, _data_end_date(), freq="5min")
    n   = len(idx)
    rng = np.random.default_rng(GLOBAL_SEED)

    # Simulate ~400 precipitation events over 5 years
    n_storms    = 400
    storm_start = np.sort(rng.choice(n - 600, n_storms, replace=False))
    precip_base = np.zeros(n)
    level_base  = np.ones(n) * 2.0    # baseflow ≈ 2 ft

    for ss in storm_start:
        dur       = rng.integers(20, 150)
        intensity = rng.uniform(0.005, 0.18)
        pulse     = intensity * np.exp(-np.linspace(0, 4, dur))
        e         = min(ss + dur, n)
        precip_base[ss:e] += pulse[:e - ss]

        resp_dur = rng.integers(60, 360)
        resp_end = min(ss + resp_dur, n)
        peak     = intensity * 18 * rng.uniform(0.5, 2.5)
        t_r      = np.arange(resp_dur)
        hydro    = peak * (t_r / 25) * np.exp(-t_r / 55)
        level_base[ss:resp_end] += hydro[:resp_end - ss]

    level_base += rng.normal(0, 0.04, n)

    frames = {}
    frames["precip_hw70"]   = pd.Series(
        np.clip(precip_base * rng.uniform(0.8, 1.2, n), 0, None), index=idx)
    frames["precip_old_wf"] = pd.Series(
        np.clip(precip_base * rng.uniform(0.6, 1.3, n), 0, None), index=idx)

    lag_config = [
        ("gauge_hw70",        12),
        ("gauge_ebenezer",     6),
        ("gauge_us1",          0),
        ("gauge_old_wf",      20),
        ("gauge_anderson_dr",  4),
        ("gauge_pigeon_ck",    8),
    ]
    for name, lag in lag_config:
        shifted = np.roll(level_base * rng.uniform(0.65, 0.95), lag)
        frames[name] = pd.Series(
            np.clip(shifted + rng.normal(0, 0.03, n), 0.5, None), index=idx)

    # Simple power-law rating curve: Q = a × h^b
    frames["q_hw70"]     = pd.Series(
        (frames["gauge_hw70"]     ** 2.3 * 14).clip(lower=0), index=idx)
    frames["q_ebenezer"] = pd.Series(
        (frames["gauge_ebenezer"] ** 2.3 * 11).clip(lower=0), index=idx)

    return pd.DataFrame(frames, index=idx)


# =============================================================================
# SECTION 4 — PHYSICS-INFORMED FEATURE ENGINEERING
# Features are constructed in six groups, each grounded in hydrological
# process understanding:
#
# (a) Raw precipitation        — instantaneous forcing
# (b) Rolling accumulations    — multi-scale antecedent rainfall
# (c) Delayed antecedent       — precipitation in pre-forecast window
# (d) Antecedent Precipitation Index (API) — exponential-decay soil moisture
# (e) Upstream gage heights    — in-channel routing signal
# (f) Travel-time lagged gages — explicit routing delay encoding
# (g) Discharge + derivatives  — flow magnitude and rate of change
# (h) Eckhardt baseflow        — slow (baseflow) vs fast (quickflow) component
# (i) Lagged discharge         — explicit discharge routing signal
# (j) Cyclic time encodings    — seasonal and diurnal forcing patterns
# =============================================================================

def eckhardt_baseflow_filter(Q: np.ndarray,
                              alpha : float = ECKHARDT_ALPHA,
                              BFImax: float = ECKHARDT_BFIMAX) -> np.ndarray:
    """
    Eckhardt (2005) recursive digital filter for baseflow separation.

    The filter partitions total streamflow Q(t) into:
        - Baseflow  Qb(t): slow, sustained groundwater contribution
        - Quickflow Qq(t): fast, event-driven surface runoff

    The baseflow index (BFI = Qb/Q) provides a proxy for catchment
    saturation state, which is a primary determinant of flood response
    magnitude and timing in urban watersheds.

    Recurrence relation:
        Qb(t) = [(1−BFImax)·α·Qb(t−1) + (1−α)·BFImax·Q(t)] / (1−α·BFImax)
        subject to: Qb(t) ≤ Q(t)

    Parameters
    ----------
    Q      : 1-D array of total discharge (cfs or m³/s)
    alpha  : recession constant (0.925 for perennial streams; Eckhardt 2005)
    BFImax : maximum baseflow index (0.80 perennial, 0.50 ephemeral)

    Returns
    -------
    Qb : 1-D array of baseflow (same units as Q)
    """
    Qb    = np.zeros_like(Q, dtype=np.float64)
    Qb[0] = Q[0] * BFImax
    denom = 1.0 - alpha * BFImax

    for i in range(1, len(Q)):
        Qb_candidate = (
            (1.0 - BFImax) * alpha * Qb[i - 1] +
            (1.0 - alpha)  * BFImax * Q[i]
        ) / denom
        Qb[i] = min(float(Qb_candidate), float(Q[i]))

    return Qb.astype(np.float32)


def build_feature_channels(df: pd.DataFrame):
    """
    Construct the complete 47-feature input matrix from the raw merged
    DataFrame. This function is shared by all model architectures and
    executes exactly once per run.

    Each feature group is annotated with its hydrological rationale.

    Parameters
    ----------
    df : raw merged DataFrame from build_raw_dataframe()

    Returns
    -------
    df        : DataFrame containing all feature columns + 6 target columns
    feat_cols : ordered list of feature column names (excludes targets)
    """
    print("=== Building physics-informed feature channels ===")
    df        = df.copy()
    feat_cols = []

    # Identify available stations (API may not return all)
    precip_stations = ["hw70", "old_wf"]
    upstream_gauges = ["hw70", "ebenezer", "old_wf", "anderson_dr", "pigeon_ck"]
    discharge_sites = ["hw70", "ebenezer"]

    avail_precip = [s for s in precip_stations if f"precip_{s}" in df.columns]
    avail_gauges = [s for s in upstream_gauges  if f"gauge_{s}" in df.columns]
    avail_q      = [s for s in discharge_sites  if f"q_{s}"     in df.columns]

    print(f"  Available precip stations : {avail_precip}")
    print(f"  Available gauge stations  : {avail_gauges}")
    print(f"  Available discharge sites : {avail_q}")

    # ── (a) Raw instantaneous precipitation ──────────────────────────────
    # Captures the immediate rainfall forcing at each gauge location.
    for s in avail_precip:
        feat_cols.append(f"precip_{s}")

    # ── (b) Rolling precipitation accumulations ───────────────────────────
    # Multi-scale accumulations (30–180 min) capture rainfall intensity
    # at time scales relevant to urban runoff concentration.
    for s in avail_precip:
        col = f"precip_{s}"
        for win_min in [30, 60, 120, 180]:
            steps  = win_min // TIMESTEP_MIN
            fn_sum = f"{col}_roll{win_min}min_sum"
            fn_int = f"{col}_roll{win_min}min_int"
            df[fn_sum] = df[col].rolling(steps, min_periods=1).sum()
            df[fn_int] = df[col].rolling(steps, min_periods=1).mean()
            feat_cols += [fn_sum, fn_int]

    # ── (c) Delayed antecedent precipitation window (225 min) ─────────────
    # Captures rainfall that influenced the watershed prior to the current
    # forecast window, i.e., the period [t−375 min, t−150 min].
    win_start = 375 // TIMESTEP_MIN    # 75 steps
    win_end   = 150 // TIMESTEP_MIN    # 30 steps
    win_len   = win_start - win_end    # 45 steps = 225 min

    for s in avail_precip:
        col     = f"precip_{s}"
        shifted = df[col].shift(win_end)
        df[f"{col}_225min_sum"] = shifted.rolling(win_len, min_periods=1).sum()
        df[f"{col}_225min_var"] = shifted.rolling(win_len, min_periods=1).var().fillna(0)
        feat_cols += [f"{col}_225min_sum", f"{col}_225min_var"]

    # ── (d) Antecedent Precipitation Index (API) ──────────────────────────
    # API(t) = k·API(t−1) + P(t), where k = 0.92^(Δt/1440)
    # Provides an exponential-decay proxy for antecedent soil moisture,
    # which is the primary control on rainfall-runoff partitioning.
    # Reference: Kohler & Linsley (1951); widely used in operational hydrology.
    for s in avail_precip:
        col      = f"precip_{s}"
        api_col  = f"{col}_API"
        api_vals = np.zeros(len(df), dtype=np.float32)
        p_arr    = df[col].fillna(0).values

        for i in range(1, len(df)):
            api_vals[i] = K_API_DECAY * api_vals[i - 1] + p_arr[i]

        df[api_col] = api_vals
        feat_cols.append(api_col)

    print(f"  API decay constant k = {K_API_DECAY:.6f} per 5-min step")

    # ── (e) Upstream gage heights ──────────────────────────────────────────
    # Real-time gage heights at upstream stations provide a direct in-channel
    # routing signal, encoding the volume of water already in transit toward
    # the target gauge.
    for s in avail_gauges:
        feat_cols.append(f"gauge_{s}")

    # ── (f) Travel-time lagged gage heights ───────────────────────────────
    # Explicit lag encoding based on empirical travel-time estimates:
    #   HW70 → US-1      : ~60 min (12 steps)
    #   Ebenezer → US-1  : ~30 min  (6 steps)
    # Encoding these lags explicitly reduces the burden on the LSTM/Transformer
    # to infer routing delays from the raw time series alone.
    LAG_GAUGE_MAP = {"hw70": 12, "ebenezer": 6}
    for s, lag in LAG_GAUGE_MAP.items():
        gauge_col = f"gauge_{s}"
        if gauge_col in df.columns:
            lag_name      = f"{gauge_col}_lag{lag * TIMESTEP_MIN}min"
            df[lag_name]  = df[gauge_col].shift(lag)
            feat_cols.append(lag_name)

    # ── (g) Discharge magnitude, rolling maximum, and rate of change ──────
    # dQ/dt (first difference) encodes the rising/falling limb signature,
    # which is a strong predictor of imminent peak flow.
    for s in avail_q:
        col    = f"q_{s}"
        fn_max = f"{col}_roll60min_max"
        fn_dq  = f"{col}_dQ"
        df[fn_max] = df[col].rolling(12, min_periods=1).max()
        df[fn_dq]  = df[col].diff().fillna(0)
        feat_cols += [col, fn_max, fn_dq]

    # ── (h) Eckhardt baseflow separation ──────────────────────────────────
    # Separates total discharge into baseflow (slow, groundwater-driven) and
    # quickflow (fast, event-driven) components. Elevated baseflow indicates
    # high antecedent soil saturation, which amplifies flood response to
    # subsequent rainfall.
    for s in avail_q:
        col   = f"q_{s}"
        Q_arr = df[col].fillna(0).values

        Qb = eckhardt_baseflow_filter(Q_arr)
        Qq = np.clip(Q_arr - Qb, 0, None)

        df[f"{col}_baseflow"]  = Qb
        df[f"{col}_quickflow"] = Qq
        feat_cols += [f"{col}_baseflow", f"{col}_quickflow"]

    # ── (i) Travel-time lagged discharge ──────────────────────────────────
    # Discharge lags provide an explicit hydraulic routing signal encoding
    # the volume of water currently in transit between upstream stations and
    # the target gauge.
    LAG_Q_MAP = {"hw70": 12, "ebenezer": 6}
    for s, lag in LAG_Q_MAP.items():
        q_col = f"q_{s}"
        if q_col in df.columns:
            lag_name     = f"{q_col}_lag{lag * TIMESTEP_MIN}min"
            df[lag_name] = df[q_col].shift(lag)
            feat_cols.append(lag_name)

    # ── (j) Cyclic time encodings ─────────────────────────────────────────
    # Sine/cosine projections of hour-of-day and day-of-year create a
    # continuous circular representation that correctly encodes the
    # periodicity of diurnal and seasonal forcing (e.g., convective
    # activity peaks in afternoon hours during the warm season).
    hour_rad = 2 * np.pi * df.index.hour / 24.0
    doy_rad  = 2 * np.pi * df.index.dayofyear / 365.25
    df["hour_sin"] = np.sin(hour_rad)
    df["hour_cos"] = np.cos(hour_rad)
    df["doy_sin"]  = np.sin(doy_rad)
    df["doy_cos"]  = np.cos(doy_rad)
    feat_cols += ["hour_sin", "hour_cos", "doy_sin", "doy_cos"]

    # ── (k) Multi-scale rate-of-rise features (v2) ────────────────────────
    # Rise rate is the strongest early-onset predictor of an imminent peak.
    # We compute centred-free (causal) finite differences of stage and
    # discharge at several scales so the model can read the *acceleration*
    # of the rising limb, not just its level.
    if USE_RISE_RATE_FEATS:
        rate_gauges = [s for s in avail_gauges if s in ("hw70", "ebenezer", "old_wf")]
        for scale_min in RISE_RATE_SCALES_MIN:
            k = max(1, scale_min // TIMESTEP_MIN)
            for s in rate_gauges:
                gcol = f"gauge_{s}"
                fn   = f"{gcol}_rise{scale_min}min"
                df[fn] = (df[gcol] - df[gcol].shift(k)) / float(scale_min)
                feat_cols.append(fn)
            for s in avail_q:
                qcol = f"q_{s}"
                fn   = f"{qcol}_rise{scale_min}min"
                df[fn] = (df[qcol] - df[qcol].shift(k)) / float(scale_min)
                feat_cols.append(fn)

    # ── (l) Event-cumulative rainfall since storm onset (v2) ──────────────
    # Rolling windows miss total storm depth for long events. This feature
    # accumulates rainfall continuously and resets after a dry spell, giving
    # a direct proxy for "how much has fallen so far this storm".
    if USE_EVENT_CUM_RAIN:
        dry_steps = max(1, DRY_SPELL_RESET_MIN // TIMESTEP_MIN)
        for s in avail_precip:
            col   = f"precip_{s}"
            p     = df[col].fillna(0).values
            # Reset accumulation whenever the trailing `dry_steps` are all ~0
            is_wet = pd.Series(p, index=df.index).rolling(
                dry_steps, min_periods=1).sum().values > 1e-6
            cum = np.zeros(len(p), dtype=np.float32)
            for i in range(1, len(p)):
                cum[i] = (cum[i - 1] + p[i]) if is_wet[i] else 0.0
            fn = f"{col}_event_cum"
            df[fn] = cum
            feat_cols.append(fn)

    # ── (m) Multi-lag upstream gauge channels (v2) ────────────────────────
    # A flashy, reservoir-regulated basin has STAGE-DEPENDENT travel times,
    # so single hardcoded lags misalign during the largest events. Providing
    # several lag channels lets the LSTM/attention weight the effective lag
    # dynamically rather than committing to one constant.
    if USE_MULTI_LAG_GAUGE:
        multi_lag_gauges = [s for s in avail_gauges if s in ("hw70", "ebenezer", "old_wf")]
        for s in multi_lag_gauges:
            gcol = f"gauge_{s}"
            for lag_min in MULTI_LAG_STEPS_MIN:
                steps = max(1, lag_min // TIMESTEP_MIN)
                fn    = f"{gcol}_mlag{lag_min}min"
                if fn in df.columns:
                    continue
                df[fn] = df[gcol].shift(steps)
                feat_cols.append(fn)

    # ── (m2) US-1 AUTOREGRESSIVE channels (v3) ────────────────────────────
    # CRITICAL FIX: gauge_us1 is the TARGET but was never an INPUT. At issue
    # time t the observed US-1 stage up to t IS available (it is exactly what
    # the persistence baseline uses) and is the single strongest short-horizon
    # predictor. We add its level, routing-lagged history, multi-scale rate of
    # rise (the local "wave-arrival"/acceleration signal that the upstream
    # gauges cannot fully supply because US-1 is ~6x flashier than its drivers),
    # the first difference, and a trailing crest memory. All channels are strictly
    # causal (no forward shift), so no target leakage is introduced.
    if USE_US1_AUTOREGRESSIVE and "gauge_us1" in df.columns:
        u = df["gauge_us1"]
        df["us1_level"] = u                       # observed stage at t
        feat_cols.append("us1_level")
        for lag_min in US1_LAG_STEPS_MIN:         # routing-lagged own history
            steps = max(1, lag_min // TIMESTEP_MIN)
            fn = f"us1_lag{lag_min}min"
            df[fn] = u.shift(steps)
            feat_cols.append(fn)
        for scale_min in RISE_RATE_SCALES_MIN:    # multi-scale rate of rise
            k = max(1, scale_min // TIMESTEP_MIN)
            fn = f"us1_rise{scale_min}min"
            df[fn] = (u - u.shift(k)) / float(scale_min)
            feat_cols.append(fn)
        df["us1_dStage"] = u.diff().fillna(0)     # first difference
        df["us1_roll60min_max"] = u.rolling(12, min_periods=1).max()   # crest memory
        feat_cols += ["us1_dStage", "us1_roll60min_max"]
        print("  US-1 autoregressive channels added (level, lags, rise rate, crest).")

    # ── (n) Future forcing — perfect-prog rainfall over (t, t+h] (v2) ──────
    # THE key lever on lead time. For each forecast horizon we inject the
    # rainfall that will fall between issue time t and t+h. With observed
    # future rain (perfect_prog) this establishes the achievable skill
    # ceiling; at operational time, replace these columns with a QPF /
    # nowcast product (e.g. MRMS, HRRR, pySTEPS) covering the same windows.
    #
    # NOTE: this is the ONLY place we intentionally use information after t.
    # It is hydrologically legitimate because operational forecasters DO have
    # a rainfall forecast; it is what lets the model anticipate, rather than
    # lag behind, the rise. Set USE_FUTURE_FORCING=False for a strict
    # past-only (causal) ablation.
    if USE_FUTURE_FORCING and FUTURE_FORCING_MODE in ("perfect_prog", "qpf"):
        _qpf_rng = np.random.default_rng(GLOBAL_SEED)
        for s in avail_precip:
            col  = f"precip_{s}"
            csum = df[col].fillna(0).cumsum()
            for h_min, h_steps in zip(FORECAST_HORIZONS_MIN, FORECAST_HORIZONS_STEPS):
                fn = f"{col}_fut_t{h_min}min"
                # Rain over (t, t+h] = cumsum(t+h) - cumsum(t)  (forward window)
                fut = (csum.shift(-h_steps) - csum)
                if FUTURE_FORCING_MODE == "qpf":
                    # Emulate an operational nowcast: temporal blur + bias +
                    # multiplicative & additive error. Replace this whole branch
                    # with a real MRMS/HRRR QPF column for an operational study.
                    blurred = fut.rolling(QPF_BLUR_STEPS, min_periods=1,
                                          center=True).mean()
                    mult = 1.0 + QPF_ERROR_LEVEL * _qpf_rng.standard_normal(len(fut))
                    add  = QPF_ERROR_LEVEL * fut.std() * \
                           _qpf_rng.standard_normal(len(fut))
                    fut  = np.clip(blurred.values * mult + add, 0, None)
                df[fn] = fut
                feat_cols.append(fn)

    # ── (o) Storm-dynamic features (v4) ───────────────────────────────────
    # Storm age, rainfall centroid, and dynamic travel-time via cross-correlation
    # supply event-phase and routing information beyond static antecedent windows.
    dry_steps = max(1, DRY_SPELL_RESET_MIN // TIMESTEP_MIN)

    if USE_STORM_AGE_FEAT or USE_RAINFALL_CENTROID_FEAT:
        for s in avail_precip:
            col = f"precip_{s}"
            p   = df[col].fillna(0).values.astype(np.float64)
            is_wet = pd.Series(p, index=df.index).rolling(
                dry_steps, min_periods=1).sum().values > 1e-6
            age_min = np.zeros(len(p), dtype=np.float32)
            centroid_min = np.zeros(len(p), dtype=np.float32)
            onset = 0
            for i in range(len(p)):
                if is_wet[i] and (i == 0 or not is_wet[i - 1]):
                    onset = i
                if is_wet[i]:
                    age_min[i] = (i - onset) * TIMESTEP_MIN
                    seg = p[onset:i + 1]
                    t_rel = np.arange(len(seg), dtype=np.float64) * TIMESTEP_MIN
                    centroid_min[i] = float(
                        np.sum(t_rel * seg) / (np.sum(seg) + 1e-9))
            if USE_STORM_AGE_FEAT:
                fn = f"{col}_storm_age_min"
                df[fn] = age_min
                feat_cols.append(fn)
            if USE_RAINFALL_CENTROID_FEAT:
                fn = f"{col}_rain_centroid_min"
                df[fn] = centroid_min
                feat_cols.append(fn)

    if USE_DYNAMIC_TRAVEL_TIME_FEAT and "gauge_hw70" in df.columns and "gauge_us1" in df.columns:
        up = df["gauge_hw70"].ffill().values.astype(np.float64)
        dn = df["gauge_us1"].ffill().values.astype(np.float64)
        win   = max(12, LOOKBACK_STEPS)          # 60–120 min window
        max_l = 12                               # search ±60 min
        dyn_lag_min = np.full(len(up), np.nan, dtype=np.float32)
        for i in range(win, len(up)):
            u_seg = up[i - win:i] - up[i - win:i].mean()
            d_seg = dn[i - win:i] - dn[i - win:i].mean()
            if u_seg.std() < 1e-6 or d_seg.std() < 1e-6:
                continue
            best_lag, best_c = 0, -1.0
            for lag in range(-max_l, max_l + 1):
                if lag >= 0:
                    u_sl, d_sl = u_seg[lag:], d_seg[:len(d_seg) - lag]
                else:
                    u_sl, d_sl = u_seg[:len(u_seg) + lag], d_seg[-lag:]
                if len(u_sl) < 4:
                    continue
                c = np.corrcoef(u_sl, d_sl)[0, 1]
                if np.isfinite(c) and c > best_c:
                    best_c, best_lag = c, lag
            dyn_lag_min[i] = best_lag * TIMESTEP_MIN
        df["hw70_us1_xcorr_lag_min"] = dyn_lag_min
        feat_cols.append("hw70_us1_xcorr_lag_min")
        print("  Dynamic travel-time (hw70→US-1 xcorr lag) feature added.")

    # ── Target columns: gage height at US-1 at each forecast horizon ───────
    # Each target column contains gauge_us1 shifted backward by the
    # corresponding horizon step count, so that row t contains the
    # water level that will be observed at t + horizon.
    target_cols = []
    for h_min, h_steps in zip(FORECAST_HORIZONS_MIN, FORECAST_HORIZONS_STEPS):
        col_name       = f"target_t{h_min}min"
        df[col_name]   = df["gauge_us1"].shift(-h_steps)
        target_cols.append(col_name)

    # Remove rows with NaN in any feature or target column
    all_cols  = feat_cols + target_cols
    df        = df.dropna(subset=all_cols)
    feat_cols = [f for f in feat_cols if f in df.columns]

    print(f"\n  Feature summary:")
    print(f"    Total features : {len(feat_cols)}")
    print(f"    Lookback steps : {LOOKBACK_STEPS}  ({LOOKBACK_MIN} min)")
    print(f"    Forecast steps : {OUTPUT_STEPS}  ({FORECAST_HORIZONS_MIN} min)")
    print(f"    Usable rows    : {len(df):,}")

    return df, feat_cols, target_cols


# =============================================================================
# SECTION 5 — CHRONOLOGICAL DATA SPLIT
# The dataset is partitioned chronologically into training (70%), validation
# (15%), and test (15%) subsets. No temporal shuffling is performed across
# partition boundaries. This split strategy is mandatory for time-series
# forecasting and avoids data leakage from future to past.
# =============================================================================

def chronological_split(df: pd.DataFrame):
    """
    Partition the feature DataFrame into train/validation/test subsets
    in strict chronological order.

    Parameters
    ----------
    df : full feature DataFrame

    Returns
    -------
    df_train, df_val, df_test : non-overlapping DataFrame partitions
    """
    n    = len(df)
    n_tr = int(n * TRAIN_FRAC)
    n_va = int(n * VAL_FRAC)

    df_tr = df.iloc[:n_tr]
    df_va = df.iloc[n_tr : n_tr + n_va]
    df_te = df.iloc[n_tr + n_va :]

    print("=== Chronological data split ===")
    print(f"  Train : {df_tr.index[0].date()} → {df_tr.index[-1].date()}"
          f"  ({len(df_tr):,} rows,  {len(df_tr)*TIMESTEP_MIN/60/24:.0f} days)")
    print(f"  Val   : {df_va.index[0].date()} → {df_va.index[-1].date()}"
          f"  ({len(df_va):,} rows,  {len(df_va)*TIMESTEP_MIN/60/24:.0f} days)")
    print(f"  Test  : {df_te.index[0].date()} → {df_te.index[-1].date()}"
          f"  ({len(df_te):,} rows,  {len(df_te)*TIMESTEP_MIN/60/24:.0f} days)")
    return df_tr, df_va, df_te


# =============================================================================
# SECTION 5b — ROLLING-ORIGIN (EXPANDING-WINDOW) CROSS-VALIDATION  (v3)
# Split by calendar year. Extremes occur in EVERY year, so each held-out test
# year contains a large event while training stays strictly causal (past-only).
# An embargo of CV_EMBARGO_STEPS rows is removed from the END of each training
# block so that no training target (gauge_us1 shifted forward by up to +180 min)
# can read into the held-out test year. This is the leakage-safe analogue of a
# purged walk-forward split for autocorrelated 5-min data.
# =============================================================================

def rolling_origin_folds(df: pd.DataFrame,
                         exclude_test_years: list = None):
    """
    Yield expanding-window folds split by year.

    Returns a list of dicts, one per fold:
        {"fold": k, "test_year": Y,
         "df_train": ..., "df_val": ..., "df_test": ...}

    - Train block = all years strictly before the test year (expanding).
    - The last CV_VAL_FRACTION of the train block (chronological) becomes the
      validation set for early stopping.
    - CV_EMBARGO_STEPS rows are dropped at the train/val and val/test joins so
      no forward-shifted target leaks across a boundary.
    - exclude_test_years: calendar years omitted from held-out test (e.g. [2025]).
    """
    years = sorted(df.index.year.unique())
    emb   = CV_EMBARGO_STEPS
    exclude = set(exclude_test_years if exclude_test_years is not None
                  else EXCLUDE_TEST_YEARS or [])
    folds = []
    for i in range(CV_MIN_TRAIN_YEARS, len(years)):
        test_year  = years[i]
        if test_year in exclude:
            continue
        train_years = years[:i]
        df_train_full = df[df.index.year.isin(train_years)]
        df_test       = df[df.index.year == test_year]
        if len(df_train_full) < (LOOKBACK_STEPS + emb + 100) or len(df_test) < (LOOKBACK_STEPS + 50):
            continue

        # carve validation from the tail of the training block (chronological)
        n_tr_full = len(df_train_full)
        n_val     = int(n_tr_full * CV_VAL_FRACTION)
        split_pt  = n_tr_full - n_val
        df_train  = df_train_full.iloc[: max(0, split_pt - emb)]    # embargo train|val
        df_val    = df_train_full.iloc[split_pt:]
        # embargo the very end of train block so its targets don't reach test_year
        df_train  = df_train.iloc[: len(df_train) - emb] if len(df_train) > emb else df_train

        folds.append({
            "fold": len(folds) + 1,
            "test_year": test_year,
            "df_train": df_train,
            "df_val":   df_val,
            "df_test":  df_test,
        })

    print("=== Rolling-origin (expanding-window) cross-validation ===")
    print(f"  Embargo: {emb} steps ({emb*TIMESTEP_MIN} min) at every split boundary")
    if exclude:
        print(f"  Excluded test years: {sorted(exclude)}")
    for f in folds:
        print(f"  Fold {f['fold']}: train {f['df_train'].index[0].date()}–"
              f"{f['df_train'].index[-1].date()} ({len(f['df_train']):,}) | "
              f"val ({len(f['df_val']):,}) | "
              f"TEST {f['test_year']} ({len(f['df_test']):,})")
    return folds


# =============================================================================
# SECTION 5c — GRADIENT-BOOSTED-TREE FLOOD BASELINE  (v3)
# A lagged-feature HistGradientBoosting regressor the deep model must beat.
# If HydroLSTM does not clear this on flood-conditioned NSE, the extra
# complexity of the deep net is not justified for this watershed.
# =============================================================================

def gbm_flood_baseline(df_train, df_test, feat_cols, target_cols, flood_thr):
    """
    Train one HistGradientBoosting model per horizon on lagged features and
    return a per-horizon metrics DataFrame (same columns as the DL metrics) so
    it slots directly into the comparison table.
    """
    try:
        from sklearn.ensemble import HistGradientBoostingRegressor
    except Exception as exc:
        print(f"  [WARN] GBM baseline unavailable: {exc}")
        return pd.DataFrame()

    # Build lagged design matrix (every 15 min over the lookback) — cheap & strong.
    lag_steps = list(range(0, LOOKBACK_STEPS + 1, 3))
    def design(d):
        cols = {}
        for c in feat_cols:
            v = d[c].values
            for L in lag_steps:
                cols[f"{c}_l{L}"] = np.concatenate([np.full(L, v[0]), v[:-L]]) if L else v
        return pd.DataFrame(cols, index=d.index)

    Xtr_full = design(df_train); Xte_full = design(df_test)
    rng = np.random.default_rng(GLOBAL_SEED)
    rows = []
    for h_min, tcol in zip(FORECAST_HORIZONS_MIN, target_cols):
        ytr = df_train[tcol].values; yte = df_test[tcol].values
        mtr = np.isfinite(ytr); mte = np.isfinite(yte)
        Xtr, y_tr = Xtr_full.values[mtr], ytr[mtr]
        Xte, y_te = Xte_full.values[mte], yte[mte]
        # keep all flood rows + a fraction of the rest (speed, class balance)
        keep = (y_tr > flood_thr) | (rng.random(len(y_tr)) < 0.10)
        model = HistGradientBoostingRegressor(max_iter=150, max_depth=6,
                                              learning_rate=0.1,
                                              random_state=GLOBAL_SEED)
        model.fit(Xtr[keep], y_tr[keep])
        pred = model.predict(Xte)
        if CLAMP_STAGE_FLOOR is not None:
            pred = np.clip(pred, CLAMP_STAGE_FLOOR, None)
        m = compute_metrics_for_horizon(y_te, pred, flood_thr)
        m["Horizon"] = f"t+{h_min}min"
        rows.append(m)
    return pd.DataFrame(rows).set_index("Horizon")


# =============================================================================
# SECTION 5d — FEATURE PRUNING VIA LightGBM / SHAP  (v4)
# Reduces redundancy in the ~73–85 feature set before deep-model training.
# =============================================================================

def prune_features_shap_lightgbm(df_train    : pd.DataFrame,
                                  feat_cols   : list,
                                  target_cols : list,
                                  top_k       : int = FEATURE_PRUNING_TOP_K,
                                  min_keep    : int = FEATURE_PRUNING_MIN_KEEP) -> list:
    """
    Rank features with LightGBM (or HistGradientBoosting fallback) and optional
    SHAP; retain the top-K by mean |SHAP| or native gain importance.

    Always preserves US-1 autoregressive channels and raw precip if present.
    """
    protected = {c for c in feat_cols if c.startswith(("us1_", "precip_hw70", "precip_old_wf"))
                 and not c.endswith("_fut_t")}
    protected |= {c for c in feat_cols if c in ("gauge_us1", "us1_level")}

    tcol = target_cols[min(2, len(target_cols) - 1)]   # t+90min representative
    X = df_train[feat_cols].values
    y = df_train[tcol].values
    mask = np.isfinite(y)
    X, y = X[mask], y[mask]

    if len(y) < 200:
        print("  [WARN] Too few rows for feature pruning; keeping all features.")
        return feat_cols

    importance = None
    model = None

    try:
        import lightgbm as lgb
        dtrain = lgb.Dataset(X, label=y, feature_name=feat_cols, free_raw_data=False)
        params = {"objective": "regression", "metric": "rmse", "verbosity": -1,
                  "learning_rate": 0.08, "num_leaves": 31, "feature_fraction": 0.8,
                  "seed": GLOBAL_SEED}
        model = lgb.train(params, dtrain, num_boost_round=120)
        importance = model.feature_importance(importance_type="gain").astype(float)
        print("  Feature pruning: LightGBM gain importance.")
    except Exception as exc_lgb:
        print(f"  [INFO] LightGBM unavailable ({exc_lgb}); using HistGradientBoosting.")
        try:
            from sklearn.ensemble import HistGradientBoostingRegressor
            gbm = HistGradientBoostingRegressor(max_iter=120, max_depth=6,
                                                learning_rate=0.1,
                                                random_state=GLOBAL_SEED)
            gbm.fit(X, y)
            importance = gbm.feature_importances_.astype(float)
        except Exception as exc_gbm:
            print(f"  [WARN] Feature pruning failed ({exc_gbm}); keeping all features.")
            return feat_cols

    if importance is None:
        return feat_cols

    # Optional SHAP refinement on a subsample
    try:
        import shap
        if model is not None and hasattr(model, "predict"):
            n_sub = min(800, len(X))
            idx = np.random.default_rng(GLOBAL_SEED).choice(len(X), n_sub, replace=False)
            explainer = shap.TreeExplainer(model)
            shap_vals = explainer.shap_values(X[idx])
            if isinstance(shap_vals, list):
                shap_vals = shap_vals[0]
            importance = np.abs(shap_vals).mean(axis=0)
            print("  Feature pruning: SHAP mean |value| refinement applied.")
    except Exception:
        pass

    order = np.argsort(importance)[::-1]
    keep_n = max(min_keep, min(top_k, len(feat_cols)))
    selected = [feat_cols[i] for i in order[:keep_n]]
    for p in feat_cols:
        if p in protected and p not in selected:
            selected.append(p)
    selected = [c for c in feat_cols if c in selected]   # preserve original order

    dropped = len(feat_cols) - len(selected)
    print(f"  Feature pruning: {len(feat_cols)} → {len(selected)} "
          f"({dropped} removed, top_k={top_k})")
    try:
        imp_df = pd.DataFrame({"feature": feat_cols, "importance": importance})
        imp_df.sort_values("importance", ascending=False)\
              .to_csv(OUTPUT_DIR / "feature_importance_pruning.csv", index=False)
    except Exception:
        pass
    return selected


# =============================================================================
# SECTION 6 — PYTORCH DATASET
# The ContinuousSequenceDataset constructs (input_sequence, target_vector)
# pairs using a sliding window over the continuous time series. Each sample
# consists of a 96-step (8-hour) feature matrix as input and a 6-element
# target vector containing observed gage heights at the six forecast horizons.
# =============================================================================

# Hydrograph phase labels for balanced storm sampling
SAMPLE_FLOOD    = 0
SAMPLE_RISING   = 1
SAMPLE_FALLING  = 2
SAMPLE_NORMAL   = 3
SAMPLE_CLASS_NAMES = ("flood", "rising", "falling", "normal")


class StageThresholds:
    """
    Tiered gage-height thresholds for Crabtree @ US-1 (USGS 02087324).

    elevated_ft   : data-driven high-flow reweighting (P95, ~3.7 ft) — train only
    significant_ft  : operational significant-event stage (9 ft) — eval + sampler
    extreme_ft      : rare tail peaks (12 ft) — supplementary eval
    """
    def __init__(self,
                 elevated_ft    : float,
                 significant_ft : float = SIGNIFICANT_STAGE_FT,
                 extreme_ft     : float = EXTREME_STAGE_FT,
                 ordinal_levels : tuple = None):
        self.elevated_ft    = float(elevated_ft)
        self.significant_ft = float(significant_ft)
        self.extreme_ft     = float(extreme_ft)
        self.ordinal_levels = tuple(ordinal_levels or ORDINAL_STAGE_LEVELS_FT)

    @property
    def eval_ft(self) -> float:
        """Primary evaluation / detection threshold."""
        return self.significant_ft


def _elevated_threshold(df_part: pd.DataFrame, target_cols: list) -> float:
    """P95 of training targets — elevated-flow reweighting only (not flood)."""
    vals = df_part[target_cols].values.flatten()
    vals = vals[np.isfinite(vals)]
    return float(np.percentile(vals, ELEVATED_PERCENTILE))


def build_stage_thresholds(df_part: pd.DataFrame, target_cols: list) -> StageThresholds:
    """Leakage-safe per-fold tiered thresholds from training partition only."""
    return StageThresholds(elevated_ft=_elevated_threshold(df_part, target_cols))


def print_stage_threshold_diagnostics(df_part   : pd.DataFrame,
                                     thresholds: StageThresholds,
                                     label     : str = "train"):
    """Print sample counts at each tier so rare-event prevalence is visible."""
    if "gauge_us1" in df_part.columns:
        s = df_part["gauge_us1"].dropna().values
    else:
        s = np.array([])
    if len(s) == 0:
        return
    n = len(s)
    print(f"  Stage tiers [{label}]  "
          f"elevated(P{ELEVATED_PERCENTILE})={thresholds.elevated_ft:.2f} ft  "
          f"significant={thresholds.significant_ft:.1f} ft  "
          f"extreme={thresholds.extreme_ft:.1f} ft  "
          f"ordinal={list(thresholds.ordinal_levels)} ft")
    for thr, name in [(thresholds.elevated_ft, "elevated"),
                      (thresholds.significant_ft, "significant"),
                      (thresholds.extreme_ft, "extreme")]:
        cnt = int((s >= thr).sum())
        print(f"    >= {thr:.2f} ft ({name:11s}): {100*cnt/n:5.2f}%  ({cnt:,} steps)")


def classify_hydro_samples(stage_at_t : np.ndarray,
                           targets    : np.ndarray,
                           significant_ft: float,
                           rise_thresh: float = RISE_RATE_THRESH_FT_PER_MIN) -> np.ndarray:
    """
    Label each sliding-window sample for balanced storm sampling.

    Classes (mutually exclusive, priority order):
      flood   — current or any-horizon stage >= significant_ft (9 ft default)
      rising  — positive rate-of-rise on the rising limb (sub-threshold)
      falling — negative rate-of-rise after elevated stage
      normal  — baseflow / recession
    """
    n = len(stage_at_t)
    labels = np.full(n, SAMPLE_NORMAL, dtype=np.int64)
    k6 = max(1, 30 // TIMESTEP_MIN)
    rise_rate = (stage_at_t - np.roll(stage_at_t, k6)) / float(k6 * TIMESTEP_MIN)
    rise_rate[:k6] = 0.0
    med_stage = float(np.nanmedian(stage_at_t))

    for i in range(n):
        tgt_max = float(np.nanmax(targets[i]))
        st = float(stage_at_t[i])
        rr = float(rise_rate[i])
        if st >= significant_ft or tgt_max >= significant_ft:
            labels[i] = SAMPLE_FLOOD
        elif rr > rise_thresh:
            labels[i] = SAMPLE_RISING
        elif rr < -rise_thresh and st > med_stage:
            labels[i] = SAMPLE_FALLING
        else:
            labels[i] = SAMPLE_NORMAL
    return labels


def _allocate_batch_counts(batch_size: int,
                           weights: tuple = STORM_SAMPLER_WEIGHTS) -> list:
    """Largest-remainder integer allocation of batch slots to storm classes."""
    w = np.asarray(weights, dtype=float)
    w = w / w.sum()
    raw = w * batch_size
    counts = np.floor(raw).astype(int)
    remainder = batch_size - int(counts.sum())
    if remainder > 0:
        frac = raw - counts
        for i in np.argsort(-frac)[:remainder]:
            counts[i] += 1
    return counts.tolist()


class BalancedStormBatchSampler(Sampler):
    """
    Event-weighted batch sampler for storm-phase-balanced training.

    Default allocation (batch_size=32): 40% flood, 30% rising limb,
    20% falling limb, 10% normal — prioritising the hydrologically critical
    rising and flood windows over baseflow recession.
    """
    def __init__(self, dataset: "ContinuousSequenceDataset",
                 batch_size: int = BATCH_SIZE,
                 drop_last: bool = True,
                 class_weights: tuple = STORM_SAMPLER_WEIGHTS):
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.n_classes = len(SAMPLE_CLASS_NAMES)
        self.per_class = _allocate_batch_counts(batch_size, class_weights)

        self.by_class = {}
        for c in range(self.n_classes):
            idx = np.where(dataset.sample_class == c)[0]
            self.by_class[c] = idx if len(idx) else np.arange(len(dataset))

        pool_counts = {SAMPLE_CLASS_NAMES[c]: len(self.by_class[c])
                       for c in range(self.n_classes)}
        batch_mix = {SAMPLE_CLASS_NAMES[c]: self.per_class[c]
                     for c in range(self.n_classes)}
        print(f"  BalancedStormBatchSampler pools: {pool_counts}")
        print(f"  BalancedStormBatchSampler batch mix: {batch_mix}  "
              f"(weights={class_weights})")

    def __iter__(self):
        rng = np.random.default_rng(GLOBAL_SEED)
        n_batches = len(self.dataset) // self.batch_size
        if not self.drop_last and len(self.dataset) % self.batch_size:
            n_batches += 1
        for _ in range(n_batches):
            batch = []
            for c in range(self.n_classes):
                n_pick = self.per_class[c]
                if n_pick <= 0:
                    continue
                pool = self.by_class[c]
                pick = rng.choice(pool, size=n_pick,
                                  replace=(FLOOD_SAMPLER_ALWAYS_REPLACE and c == SAMPLE_FLOOD)
                                  or len(pool) < n_pick)
                batch.extend(pick.tolist())
            rng.shuffle(batch)
            yield batch[:self.batch_size]

    def __len__(self) -> int:
        n = len(self.dataset) // self.batch_size
        if not self.drop_last and len(self.dataset) % self.batch_size:
            n += 1
        return n


class ContinuousSequenceDataset(Dataset):
    """
    Sliding-window dataset for multi-horizon water level forecasting.

    For each position i in [LOOKBACK_STEPS, N − max(FORECAST_HORIZONS_STEPS)]:
        X[i] = standardised feature matrix  shape: (LOOKBACK_STEPS, n_features)
        y[i] = raw gage height vector        shape: (OUTPUT_STEPS,)
               containing values at t+30, t+60, t+90, t+120, t+150, t+180 min

    Note: target values are NOT standardised. Models predict in raw feet,
    making RMSE and NSE directly interpretable without inverse transformation.

    Parameters
    ----------
    df          : DataFrame partition (train, val, or test)
    feat_cols   : ordered list of feature column names
    target_cols : ordered list of target column names
    scaler      : fitted StandardScaler (None only if fit_scaler=True)
    fit_scaler  : if True, fit a new scaler on this partition (train only)
    """

    def __init__(self, df: pd.DataFrame,
                 feat_cols   : list,
                 target_cols : list,
                 scaler=None,
                 fit_scaler : bool = False,
                 significant_ft: float = None):
        X_raw = df[feat_cols].values.astype(np.float32)
        y_raw = df[target_cols].values.astype(np.float32)   # (N, 6)

        if fit_scaler:
            self.scaler = StandardScaler()
            X_scaled    = self.scaler.fit_transform(X_raw)
        else:
            self.scaler = scaler
            X_scaled    = scaler.transform(X_raw)

        n = len(df)
        sequences, targets = [], []

        for i in range(LOOKBACK_STEPS, n):
            seq = X_scaled[i - LOOKBACK_STEPS : i]   # (96, F)
            tgt = y_raw[i]                             # (6,)
            sequences.append(seq)
            targets.append(tgt)

        self.sequences  = np.array(sequences, dtype=np.float32)  # (N-L, 96, F)
        self.targets    = np.array(targets,   dtype=np.float32)  # (N-L, 6)
        self.time_index = df.index[LOOKBACK_STEPS:]

        if "gauge_us1" in df.columns:
            stage_at_t = df["gauge_us1"].values[LOOKBACK_STEPS:].astype(np.float32)
        else:
            stage_at_t = y_raw[LOOKBACK_STEPS:, 0] if len(y_raw) > LOOKBACK_STEPS else y_raw[:, 0]

        sig_thr = (significant_ft if significant_ft is not None
                   else SIGNIFICANT_STAGE_FT)
        self.sample_class = classify_hydro_samples(
            stage_at_t, self.targets, sig_thr)
        self.stage_at_t   = stage_at_t
        self.significant_ft = sig_thr

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, idx: int):
        return (
            torch.from_numpy(self.sequences[idx]),   # (96, F)
            torch.from_numpy(self.targets[idx])       # (6,)
        )


# =============================================================================
# SECTION 7 — MODEL ARCHITECTURES
# Two deep learning architectures are implemented and compared:
#
# HydroLSTM:
#   A two-layer stacked LSTM captures sequential dependencies and long-range
#   memory in the hydrological time series. The LSTM cell state provides a
#   natural analogue to catchment storage, making it physically interpretable.
#
# HydroTransformer:
#   An encoder-only Transformer with learned positional embeddings and
#   Pre-LayerNorm (Pre-LN) stabilisation. Bidirectional attention over the
#   fixed input window allows the model to correlate early precipitation
#   signals with later rising-limb gauge readings without causal masking.
#   Pre-LN is preferred over Post-LN for shallow encoders due to more stable
#   gradient propagation (Xiong et al., 2020).
#
# Both architectures share an identical FC prediction head:
#   Linear(h → 32) → ReLU → Dropout → Linear(32 → OUTPUT_STEPS)
# =============================================================================

class HydroLSTM(nn.Module):
    """
    Two-layer stacked LSTM for multi-horizon hydrological forecasting.

    Architecture
    ------------
    Input  : (batch, LOOKBACK_STEPS=96, n_features=47)
    LSTM-1 : input_size=n_features  → hidden_size=h1  (short-term patterns)
    LSTM-2 : input_size=h1          → hidden_size=h2  (longer-range memory)
    Extract final hidden state h_T from LSTM-2
    Dropout → Linear(h2→32) → ReLU → Dropout → Linear(32→OUTPUT_STEPS=6)
    Output : (batch, 6)  — predictions at t+30, t+60, ..., t+180 min

    Parameters
    ----------
    n_features   : number of input features (47)
    hidden_sizes : tuple (h1, h2) — LSTM hidden dimensions
    dropout      : dropout probability applied in FC head
    """

    def __init__(self, n_features: int,
                 hidden_sizes: tuple = (256, 128),
                 dropout: float = 0.3):
        super().__init__()
        h1, h2 = hidden_sizes

        self.lstm1 = nn.LSTM(n_features, h1, batch_first=True,
                             dropout=dropout if dropout > 0 else 0)
        self.lstm2 = nn.LSTM(h1, h2, batch_first=True)

        self.dropout = nn.Dropout(dropout)
        self.fc1     = nn.Linear(h2, 32)
        self.relu    = nn.ReLU()
        self.fc2     = nn.Linear(32, OUTPUT_STEPS)   # 6 horizons

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : Tensor of shape (batch, 96, 47)

        Returns
        -------
        Tensor of shape (batch, 6) — multi-horizon predictions
        """
        out1, _        = self.lstm1(x)
        out2, (h_n, _) = self.lstm2(out1)
        h_last = h_n[-1]                    # final hidden state (batch, h2)
        h_last = self.dropout(h_last)
        out    = self.relu(self.fc1(h_last))
        out    = self.dropout(out)
        return self.fc2(out)                # (batch, 6)


class HydroTransformer(nn.Module):
    """
    Encoder-only Transformer with learned positional embeddings for
    multi-horizon hydrological forecasting.

    Design decisions
    ----------------
    Encoder-only (no causal mask): The input is a FIXED past window
        [t−480, …, t]. Future timesteps are never present in the input,
        so bidirectional attention over the window is safe and beneficial:
        it allows the model to correlate precipitation signals at t−480
        with rising-limb signatures near t without causal restriction.

    Learned positional embeddings: Preferred over fixed sinusoidal
        embeddings for short, fixed-length sequences, as position codes
        are directly optimised during training.

    Pre-LayerNorm (norm_first=True): Normalises inputs before the
        attention and feed-forward sub-layers, producing more stable
        gradients in shallow encoders (Xiong et al., 2020).

    Architecture
    ------------
    Input     : (batch, 96, 47)
    InputProj : Linear(47 → d_model)
    PosEmbed  : Embedding(96, d_model)  [learned]
    Encoder   : num_layers × TransformerEncoderLayer(d_model, nhead, Pre-LN)
    Last token: x[:, −1, :]                      → (batch, d_model)
    FC head   : Linear(d_model→32) → ReLU → Dropout → Linear(32→6)
    Output    : (batch, 6)

    Parameters
    ----------
    n_features  : number of input features
    d_model     : embedding / hidden dimension (must be divisible by nhead)
    nhead       : number of attention heads
    num_layers  : number of encoder layers
    dropout     : dropout probability
    max_seq_len : maximum input sequence length (= LOOKBACK_STEPS)
    """

    def __init__(self, n_features : int,
                 d_model    : int = 128,
                 nhead      : int = 8,
                 num_layers : int = 4,
                 dropout    : float = 0.2,
                 max_seq_len: int = LOOKBACK_STEPS):
        super().__init__()
        assert d_model % nhead == 0, \
            f"d_model ({d_model}) must be divisible by nhead ({nhead})"

        self.input_proj    = nn.Linear(n_features, d_model)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model         = d_model,
            nhead           = nhead,
            dim_feedforward = d_model * 4,
            dropout         = dropout,
            batch_first     = True,
            norm_first      = True,     # Pre-LN for gradient stability
        )
        self.encoder = nn.TransformerEncoder(encoder_layer,
                                              num_layers=num_layers)
        # v2: learned attention-pooling query. Instead of throwing away every
        # encoder position except the last token, a learned query attends over
        # the whole sequence to form the context vector. This both uses the
        # attention you paid to compute and reduces the recency bias that the
        # attention-rollout diagnostic exposed.
        self.pool_query = nn.Parameter(torch.randn(1, 1, d_model) / (d_model ** 0.5))
        self.dropout = nn.Dropout(dropout)
        self.fc1     = nn.Linear(d_model, 32)
        self.relu    = nn.ReLU()
        self.fc2     = nn.Linear(32, OUTPUT_STEPS)   # 6 horizons

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : Tensor of shape (batch, 96, 47)

        Returns
        -------
        Tensor of shape (batch, 6) — multi-horizon predictions
        """
        B, T, _ = x.shape
        positions = torch.arange(T, device=x.device)
        x = self.input_proj(x) + self.pos_embedding(positions)
        enc = self.encoder(x)                          # (B, T, d_model)

        if USE_IMPROVED_MODELS:
            # Learned attention pooling over the full encoded sequence
            q = self.pool_query.expand(enc.size(0), 1, -1)            # (B, 1, d)
            scores = (q @ enc.transpose(1, 2)) / (enc.size(-1) ** 0.5)  # (B, 1, T)
            attn   = torch.softmax(scores, dim=-1)
            ctx    = (attn @ enc).squeeze(1)                          # (B, d)
        else:
            ctx = enc[:, -1, :]                         # legacy last-token

        ctx = self.dropout(ctx)
        ctx = self.relu(self.fc1(ctx))
        ctx = self.dropout(ctx)
        return self.fc2(ctx)            # (batch, 6)


# =============================================================================
# SECTION 7b — v2 ARCHITECTURES (delayed-peak mitigation)
# Encoder–decoder Seq2Seq with cross-attention + optional dilated TCN front-end.
# All variants keep the (B, LOOKBACK, F) -> (B, OUTPUT_STEPS) contract so the
# rest of the pipeline (inference, MC-dropout, plotting) is unchanged.
# =============================================================================

class _Chomp1d(nn.Module):
    """Trim the right padding so a Conv1d stays strictly causal."""
    def __init__(self, chomp: int):
        super().__init__()
        self.chomp = chomp

    def forward(self, x):
        return x[:, :, :-self.chomp].contiguous() if self.chomp > 0 else x


class TCNFrontEnd(nn.Module):
    """
    Causal dilated Temporal Convolutional Network applied along the time axis.

    Dilations grow geometrically so the receptive field spans the full
    lookback while keeping the model cheap. TCNs read multi-scale rising-limb
    signatures directly and are far less prone to the recency bias that the
    attention-rollout diagnostic revealed in the last-token Transformer.
    Input/Output: (B, T, F_in) -> (B, T, channels)
    """
    def __init__(self, n_features: int, channels: int = 64,
                 kernel: int = 3, n_blocks: int = 3, dropout: float = 0.1):
        super().__init__()
        layers = []
        in_ch = n_features
        for b in range(n_blocks):
            dilation = 2 ** b
            pad = (kernel - 1) * dilation
            layers += [
                nn.Conv1d(in_ch, channels, kernel, padding=pad, dilation=dilation),
                _Chomp1d(pad),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            in_ch = channels
        self.net = nn.Sequential(*layers)

    def forward(self, x):                      # x: (B, T, F)
        y = self.net(x.transpose(1, 2))        # (B, C, T)
        return y.transpose(1, 2)               # (B, T, C)


class HydroLSTMSeq2Seq(nn.Module):
    """
    Encoder–decoder LSTM with cross-attention for multi-horizon forecasting.

    Why this helps the late-peak bias
    ----------------------------------
    The original model collapsed the sequence to a single hidden state and
    emitted six horizons from a tiny MLP — it had no machinery to *place* a
    peak at a particular lead time, so it slid rigidly behind the observed
    curve. Here a decoder produces the forecast trajectory step-by-step:
    each horizon has a learned query that cross-attends over the encoder
    states, so the model can attend to the specific antecedent signals
    (e.g. an upstream gauge rise, or future rainfall) relevant to *that*
    lead time and turn the hydrograph over at the right moment.

    Optional dilated TCN front-end (USE_TCN_FRONTEND) feeds multi-scale
    temporal features into the encoder.

    Input : (B, LOOKBACK, n_features)
    Output: (B, OUTPUT_STEPS)
    """
    def __init__(self, n_features: int,
                 hidden_sizes: tuple = (256, 128),
                 dropout: float = 0.2,
                 use_tcn: bool = None,
                 flood_aux: bool = None,
                 peak_aux: bool = None,
                 heteroscedastic: bool = None):
        super().__init__()
        h1, h2 = hidden_sizes
        self.use_tcn = USE_TCN_FRONTEND if use_tcn is None else use_tcn
        self.flood_aux = USE_FLOOD_AUX_HEAD if flood_aux is None else flood_aux
        self.peak_aux = USE_PEAK_AUX_HEAD if peak_aux is None else peak_aux
        self.heteroscedastic = USE_HETEROSCEDASTIC if heteroscedastic is None else heteroscedastic

        enc_in = n_features
        if self.use_tcn:
            self.tcn = TCNFrontEnd(n_features, channels=64,
                                   n_blocks=TCN_N_BLOCKS, dropout=dropout)
            enc_in = 64

        # Encoder (stacked LSTM); we keep the full output sequence for attention
        self.enc_lstm1 = nn.LSTM(enc_in, h1, batch_first=True)
        self.enc_lstm2 = nn.LSTM(h1, h2, batch_first=True)

        # Decoder: one learned query per horizon, cross-attending over encoder
        self.horizon_query = nn.Parameter(
            torch.randn(OUTPUT_STEPS, h2) / (h2 ** 0.5))
        self.attn_proj = nn.Linear(h2, h2)
        self.dropout   = nn.Dropout(dropout)
        dec_dim = h2 * 2
        self.head = nn.Sequential(
            nn.Linear(dec_dim, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 1))
        if self.flood_aux:
            n_ord = (len(ORDINAL_STAGE_LEVELS_FT) if USE_ORDINAL_SEVERITY_HEAD else 1)
            self.flood_head = nn.Sequential(
                nn.Linear(dec_dim, 64), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(64, n_ord))
        if self.peak_aux:
            self.peak_head = nn.Sequential(
                nn.Linear(dec_dim, 64), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(64, 1))
        if self.heteroscedastic:
            self.log_var_head = nn.Sequential(
                nn.Linear(dec_dim, 64), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(64, 1))

    def forward(self, x: torch.Tensor):
        B = x.size(0)
        if self.use_tcn:
            x = self.tcn(x)
        enc, _ = self.enc_lstm1(x)
        enc, (h_n, _) = self.enc_lstm2(enc)         # enc: (B, T, h2)
        h_last = h_n[-1]                             # (B, h2)

        keys = self.attn_proj(enc)                   # (B, T, h2)
        q = self.horizon_query.unsqueeze(0).expand(B, -1, -1)
        scores = torch.bmm(q, keys.transpose(1, 2)) / (enc.size(-1) ** 0.5)
        attn   = torch.softmax(scores, dim=-1)       # (B, OUTPUT_STEPS, T)
        ctx    = torch.bmm(attn, enc)                # (B, OUTPUT_STEPS, h2)

        h_rep  = h_last.unsqueeze(1).expand(-1, OUTPUT_STEPS, -1)
        dec_in = self.dropout(torch.cat([ctx, h_rep], dim=-1))  # (B, S, 2h2)
        stage  = self.head(dec_in).squeeze(-1)       # (B, OUTPUT_STEPS)

        if not self.flood_aux and not self.peak_aux and not self.heteroscedastic:
            return stage

        out = {"stage": stage}
        if self.flood_aux:
            flood_out = self.flood_head(dec_in)
            if USE_ORDINAL_SEVERITY_HEAD:
                out["ordinal_logits"] = flood_out          # (B, S, K)
                out["flood_logits"] = flood_out[..., -1]   # highest tier (>=9 ft)
            else:
                out["flood_logits"] = flood_out.squeeze(-1)
        if self.peak_aux:
            out["peak_logits"] = self.peak_head(dec_in).squeeze(-1)
        if self.heteroscedastic:
            out["log_var"] = self.log_var_head(dec_in).squeeze(-1)
        return out


class ResidualCorrectedModel(nn.Module):
    """
    Stage-1 + Stage-2 residual error-learning wrapper (optional).

    Stage-1 is the trained forecaster (frozen). Stage-2 is a small network
    trained to predict Stage-1's residuals (obs − pred), which often carry a
    learnable, systematic timing/magnitude bias in the 1–6 h window. The
    corrected prediction is Stage-1(x) + Stage-2(x). Presents the same
    (B,96,F)->(B,6) contract, so inference/evaluation are unchanged.
    """
    def __init__(self, stage1: nn.Module, stage2: nn.Module):
        super().__init__()
        self.stage1 = stage1
        self.stage2 = stage2
        for p in self.stage1.parameters():
            p.requires_grad = False

    def forward(self, x):
        with torch.no_grad():
            base = self.stage1(x)
        return base + self.stage2(x)


class ResidualMLPHead(nn.Module):
    """Small Stage-2 corrector: encodes the window then predicts a residual."""
    def __init__(self, n_features: int, hidden: int = RESIDUAL_HIDDEN,
                 dropout: float = 0.1):
        super().__init__()
        self.gru = nn.GRU(n_features, hidden, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, OUTPUT_STEPS))

    def forward(self, x):
        _, h = self.gru(x)
        return self.head(h[-1])


# =============================================================================
# SECTION 8 — FLOOD-AWARE COMPOSITE LOSS FUNCTION
# The standard mean squared error (MSE) loss is inappropriate for flood
# forecasting because it is dominated by the frequent low-flow timesteps,
# which constitute >90% of a typical hydrological record. Flood events are
# both rare and critically important: underestimation of peak water levels
# may result in insufficient emergency response.
#
# FloodAwareLoss addresses this imbalance through four complementary terms:
#
#   L₁ = Tail-weighted MSE:    flood events (obs > 90th pct) are assigned
#                               weight 1+α = 6, baseflow weight = 1.
#   L₂ = Asymmetric Pinball:   quantile loss at τ=0.95 penalises
#                               underprediction more than overprediction;
#                               an additional 2× penalty applies when
#                               underpredicting flood peaks.
#   L₃ = Peak magnitude loss:  |max(obs) − max(pred)| per batch, providing
#                               direct supervision on peak accuracy.
#   L₄ = Timing loss (v2):     ASYMMETRIC, DIFFERENTIABLE soft-argmax peak-
#                               time error. Late peaks (the dangerous bias)
#                               are penalised harder than early ones. Replaces
#                               the original non-differentiable argmax term,
#                               which contributed zero gradient.
#   L₅ = Soft-DTW (v2, opt.):  shape + time alignment of the forecast window,
#                               rewarding correct rising-limb shape under
#                               small temporal displacement.
#   L₆ = Flood exceedance BCE (v4): per-horizon flood probability auxiliary head.
#   L₇ = Peak timing CE (v4): which forecast horizon contains the peak.
#   L₈ = Derivative MSE (v4): explicit d(stage)/dt mismatch along the horizon.
#
# L_total = L₁ + λ_q·L₂ + λ_p·L₃ + timing + λ_dtw·L₅ + λ_cls·L₆ + λ_peak·L₇ + λ_deriv·L₈
# =============================================================================

def unpack_model_output(out):
    """Normalise model forward pass to (stage, flood_logits, peak_logits, log_var)."""
    if isinstance(out, dict):
        return (out.get("stage"), out.get("flood_logits"),
                out.get("peak_logits"), out.get("log_var"))
    return out, None, None, None


def _cumulative_ordinal_loss(logits   : torch.Tensor,
                             obs      : torch.Tensor,
                             levels_ft: tuple,
                             level_weights: tuple = ORDINAL_LEVEL_WEIGHTS,
                             focal_from_idx: int = ORDINAL_FOCAL_FROM_IDX,
                             focal_gamma: float = FLOOD_FOCAL_GAMMA,
                             monotone_penalty: float = ORDINAL_MONOTONE_PENALTY
                             ) -> torch.Tensor:
    """
    Cumulative exceedance loss: level k targets P(stage > levels_ft[k]).

    Lower levels (4, 6 ft) have many positives and regularise the shared
    representation; upper levels (8, 9 ft) borrow that strength. Monotonicity
    enforces P(>4) >= P(>6) >= P(>8) >= P(>9).
    """
    K = logits.size(-1)
    assert K == len(levels_ft) == len(level_weights)
    loss = logits.new_zeros(())
    for k, (thr, w) in enumerate(zip(levels_ft, level_weights)):
        tgt = (obs >= thr).float()
        lk  = logits[..., k]
        if k >= focal_from_idx:
            lk_loss = _focal_bce_with_logits(lk, tgt, focal_gamma)
        else:
            lk_loss = F.binary_cross_entropy_with_logits(lk, tgt)
        loss = loss + w * lk_loss
    if K > 1 and monotone_penalty > 0:
        probs = torch.sigmoid(logits)
        # Later (stricter) level should not exceed earlier cumulative prob.
        violations = torch.relu(probs[..., 1:] - probs[..., :-1])
        loss = loss + monotone_penalty * violations.pow(2).mean()
    return loss


def _soft_dtw(pred: torch.Tensor, obs: torch.Tensor, gamma: float = 0.1) -> torch.Tensor:
    """
    Compact, self-contained differentiable soft-DTW (Cuturi & Blondel, 2017)
    for short sequences. Operates on (batch, L) trajectories and returns the
    mean soft-DTW divergence. Used as a SHAPE + TIME alignment term so the
    loss can reward getting the rising-limb shape right even when the
    predicted peak is slightly displaced in time — something MSE cannot do.

    gamma : smoothing parameter. As gamma → 0 this approaches hard DTW;
            larger gamma is smoother and easier to optimise.
    """
    B, L = pred.shape
    # Pairwise squared-distance cost matrix D[b, i, j]
    D = (pred.unsqueeze(2) - obs.unsqueeze(1)) ** 2          # (B, L, L)

    INF = 1e10
    R = torch.full((B, L + 1, L + 1), INF, device=pred.device, dtype=pred.dtype)
    R[:, 0, 0] = 0.0
    for i in range(1, L + 1):
        for j in range(1, L + 1):
            r0 = -R[:, i - 1, j - 1] / gamma
            r1 = -R[:, i - 1, j]     / gamma
            r2 = -R[:, i,     j - 1] / gamma
            rmax = torch.maximum(torch.maximum(r0, r1), r2)
            softmin = -gamma * (rmax + torch.log(
                torch.exp(r0 - rmax) + torch.exp(r1 - rmax) + torch.exp(r2 - rmax)))
            R[:, i, j] = D[:, i - 1, j - 1] + softmin
    return R[:, L, L].mean()


def _focal_bce_with_logits(logits: torch.Tensor,
                           targets: torch.Tensor,
                           gamma: float = FLOOD_FOCAL_GAMMA) -> torch.Tensor:
    """Focal loss for rare >=9 ft exceedance classification (down-weights easy negatives)."""
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p = torch.sigmoid(logits)
    p_t = p * targets + (1.0 - p) * (1.0 - targets)
    return ((1.0 - p_t) ** gamma * bce).mean()


class FloodAwareLoss(nn.Module):
    """
    Composite loss for multi-horizon flash-flood peak prediction — v2.

    v2 changes vs the original (all aimed at the late-peak bias):
      * L4 timing term is now a DIFFERENTIABLE soft-argmax peak-time error
        (the original used a non-differentiable argmax that contributed
        zero gradient and therefore never constrained timing at all).
      * The timing penalty is ASYMMETRIC: predicting the peak LATE — the
        systematic, operationally dangerous failure — is penalised harder
        than predicting it early.
      * Optional L5 soft-DTW shape/time term aligns the forecast trajectory
        with the observed one, rewarding correct rising-limb shape under
        small temporal displacement.

    L = L1_wtMSE + lq·L2_pinball + lp·L3_peakmag
          + (late·max(0, t̂_pred − t̂_obs) + early·max(0, t̂_obs − t̂_pred))
          + l_dtw·L5_softDTW + l_cls·L6_flood + l_peak·L7_peak + l_deriv·L8_deriv
    """

    def __init__(self,
                 thresholds        : StageThresholds,
                 alpha             : float = FLOOD_ALPHA,
                 alpha_significant : float = SIGNIFICANT_ALPHA,
                 alpha_extreme     : float = EXTREME_ALPHA,
                 lambda_q          : float = LOSS_LAMBDA_Q,
                 lambda_p          : float = LOSS_LAMBDA_P,
                 lambda_t          : float = LOSS_LAMBDA_T,
                 tau               : float = LOSS_TAU,
                 underpred_penalty : float = LOSS_UNDERPRED_FACTOR,
                 lambda_cls        : float = LOSS_LAMBDA_FLOOD_CLS,
                 lambda_peak       : float = LOSS_LAMBDA_PEAK_CLS,
                 lambda_deriv      : float = LOSS_LAMBDA_DERIV,
                 lambda_q_sig      : float = LOSS_LAMBDA_Q_SIGNIFICANT,
                 lambda_p_under    : float = LOSS_LAMBDA_P_UNDER,
                 lambda_rise_lag   : float = LOSS_LAMBDA_RISE_LAG):
        super().__init__()
        self.thr = thresholds
        self.alpha = alpha
        self.alpha_sig = alpha_significant
        self.alpha_ext = alpha_extreme
        self.lq   = lambda_q
        self.lp   = lambda_p
        self.lp_under = lambda_p_under
        self.lt   = lambda_t
        self.tau  = tau
        self.up   = underpred_penalty
        self.lambda_cls = lambda_cls
        self.lambda_peak = lambda_peak
        self.lq_sig = lambda_q_sig
        self.use_focal = USE_FLOOD_FOCAL_LOSS
        self.focal_gamma = FLOOD_FOCAL_GAMMA
        self.use_deriv = USE_DERIVATIVE_LOSS
        self.l_deriv = lambda_deriv
        self.deriv_dt = float(DERIVATIVE_DT_MIN)
        self.use_rise_lag = USE_RISING_LIMB_ANTICIPATION_LOSS
        self.l_rise_lag = lambda_rise_lag
        self.rise_thresh = RISE_RATE_THRESH_FT_PER_MIN
        # v2 timing / shape configuration (read from module-level flags)
        self.use_soft_timing = USE_SOFT_TIMING
        self.beta            = SOFT_TIMING_BETA
        self.lt_late         = LOSS_LAMBDA_T_LATE
        self.lt_early        = LOSS_LAMBDA_T_EARLY
        self.use_dtw         = USE_SOFT_DTW
        self.l_dtw           = LOSS_LAMBDA_DTW
        self.dtw_gamma       = SOFT_DTW_GAMMA
        self.use_cont_wt     = USE_CONTINUOUS_STAGE_WEIGHT
        self.wt_k            = STAGE_WEIGHT_K
        self.wt_gamma        = STAGE_WEIGHT_GAMMA
        self.wt_cap          = STAGE_WEIGHT_CAP
        self.use_ordinal     = USE_ORDINAL_SEVERITY_HEAD
        self.ordinal_levels  = tuple(thresholds.ordinal_levels)
        # Precompute horizon index vector for the soft-argmax
        self.register_buffer(
            "_idx", torch.arange(OUTPUT_STEPS, dtype=torch.float32))

    def _soft_peak_time(self, x: torch.Tensor) -> torch.Tensor:
        """Differentiable expected peak position over the horizon axis."""
        w = torch.softmax(self.beta * x, dim=1)        # (B, OUTPUT_STEPS)
        return (w * self._idx.to(x.device)).sum(dim=1) # (B,)

    def _l1_weights(self, obs: torch.Tensor) -> torch.Tensor:
        """Per-timestep L1 weights — smooth convex above elevated_ft."""
        if self.use_cont_wt:
            excess = torch.relu(obs - self.thr.elevated_ft)
            return (1.0 + self.wt_k * torch.pow(excess, self.wt_gamma)).clamp(
                max=self.wt_cap)
        is_elev = (obs > self.thr.elevated_ft).float()
        is_sig  = (obs >= self.thr.significant_ft).float()
        is_ext  = (obs >= self.thr.extreme_ft).float()
        return (1.0 + self.alpha * is_elev
                + self.alpha_sig * is_sig
                + self.alpha_ext * is_ext)

    def forward(self, pred, obs: torch.Tensor,
                flood_logits: torch.Tensor = None,
                peak_logits: torch.Tensor = None,
                log_var: torch.Tensor = None,
                ordinal_logits: torch.Tensor = None) -> torch.Tensor:
        if isinstance(pred, dict):
            stage, flood_logits, peak_logits, log_var = unpack_model_output(pred)
            ordinal_logits = pred.get("ordinal_logits")
            pred = stage
        # ── L1: Tail-weighted MSE (or heteroscedastic NLL) ────────────
        weights = self._l1_weights(obs)
        is_sig  = (obs >= self.thr.significant_ft).float()
        if log_var is not None:
            var = torch.exp(log_var).clamp(min=1e-6)
            sq  = (pred - obs) ** 2
            l1  = (weights * 0.5 * (log_var + sq / var)).mean()
        else:
            l1 = (weights * (pred - obs) ** 2).mean()

        # ── L2: Asymmetric pinball — underprediction penalised hardest >=9 ft ─
        err      = obs - pred
        is_under = (err > 0) & (is_sig > 0)
        coef = torch.where(
            is_under,
            torch.full_like(err, self.tau * self.up),
            torch.where(
                err >= 0,
                torch.full_like(err, self.tau),
                torch.full_like(err, 1.0 - self.tau)))
        l2 = (coef * err.abs()).mean()
        if self.lq_sig > 0 and is_sig.sum() > 0:
            sig_err = err[is_sig > 0]
            sig_coef = torch.where(
                sig_err > 0,
                torch.full_like(sig_err, self.tau * self.up),
                torch.full_like(sig_err, 1.0 - self.tau))
            l2 = l2 + self.lq_sig * (sig_coef * sig_err.abs()).mean()

        # ── L3: Peak magnitude — emphasise windows with significant peaks ───
        obs_pk  = obs.max(dim=1).values
        pred_pk = pred.max(dim=1).values
        pk_err  = torch.abs(pred_pk - obs_pk)
        sig_pk  = (obs_pk >= self.thr.significant_ft).float()
        if sig_pk.sum() > 0:
            l3 = (sig_pk * pk_err).sum() / sig_pk.sum().clamp(min=1.0)
            under_pk = torch.relu(obs_pk - pred_pk) * sig_pk
            if self.lp_under > 0:
                l3 = l3 + self.lp_under * (under_pk ** 2).sum() / sig_pk.sum().clamp(min=1.0)
        else:
            l3 = pk_err.mean()

        # ── L4: Differentiable, asymmetric soft-argmax timing loss ───────
        if self.use_soft_timing:
            t_pred = self._soft_peak_time(pred)
            t_obs  = self._soft_peak_time(obs)
            dt     = t_pred - t_obs                      # >0 => predicted LATE
            late   = torch.clamp(dt,  min=0.0)
            early  = torch.clamp(-dt, min=0.0)
            l4_per = self.lt_late * late + self.lt_early * early
            if sig_pk.sum() > 0:
                l4 = (sig_pk * l4_per).sum() / sig_pk.sum().clamp(min=1.0)
            else:
                l4 = l4_per.mean()
        else:
            # Legacy (non-differentiable) fallback — kept for ablation only.
            l4 = self.lt * torch.abs(
                pred.argmax(dim=1).float() - obs.argmax(dim=1).float()).mean()

        loss = l1 + self.lq * l2 + self.lp * l3 + l4

        # ── L5: optional soft-DTW shape/time alignment ───────────────────
        if self.use_dtw and self.l_dtw > 0:
            loss = loss + self.l_dtw * _soft_dtw(pred, obs, gamma=self.dtw_gamma)

        # ── L6: Ordinal cumulative exceedance OR single-tier flood head ─────
        if self.lambda_cls > 0:
            if (ordinal_logits is not None and self.use_ordinal
                    and ordinal_logits.dim() == 3):
                l6 = _cumulative_ordinal_loss(
                    ordinal_logits, obs, self.ordinal_levels)
                loss = loss + self.lambda_cls * l6
            elif flood_logits is not None:
                flood_tgt = (obs >= self.thr.significant_ft).float()
                if self.use_focal:
                    l6 = _focal_bce_with_logits(
                        flood_logits, flood_tgt, self.focal_gamma)
                else:
                    l6 = F.binary_cross_entropy_with_logits(
                        flood_logits, flood_tgt)
                loss = loss + self.lambda_cls * l6

        # ── L7: Peak-horizon CE — upweighted when obs peak >= significant ───
        if peak_logits is not None and self.lambda_peak > 0:
            peak_tgt = obs.argmax(dim=1)
            pk_w = (obs_pk >= self.thr.significant_ft).float() * 2.0 + 1.0
            ce = F.cross_entropy(peak_logits, peak_tgt, reduction="none")
            loss = loss + self.lambda_peak * (pk_w * ce).mean()

        # ── L8: Hydrograph derivative loss d(stage)/dt along the horizon ─
        if self.use_deriv and self.l_deriv > 0 and pred.size(1) > 1:
            d_pred = (pred[:, 1:] - pred[:, :-1]) / self.deriv_dt
            d_obs  = (obs[:, 1:] - obs[:, :-1]) / self.deriv_dt
            deriv_w = self._l1_weights(obs)[:, 1:]
            loss = loss + self.l_deriv * (deriv_w * (d_pred - d_obs) ** 2).mean()

        # ── L9: Rising-limb anticipation — penalise lagging the observed rise ─
        if self.use_rise_lag and self.l_rise_lag > 0 and pred.size(1) > 1:
            d_pred = (pred[:, 1:] - pred[:, :-1]) / self.deriv_dt
            d_obs  = (obs[:, 1:] - obs[:, :-1]) / self.deriv_dt
            rising = (d_obs > self.rise_thresh).float()
            lag    = torch.relu(d_obs - d_pred)          # pred rises slower than obs
            rise_w = 1.0 + self.alpha_sig * is_sig[:, 1:]
            n_rise = rising.sum().clamp(min=1.0)
            l9 = (rise_w * rising * lag ** 2).sum() / n_rise
            loss = loss + self.l_rise_lag * l9

        return loss


# =============================================================================
# SECTION 9 — TRAINING UTILITIES
# The training loop implements:
#   - Gradient clipping (max_norm=1.0) to prevent exploding gradients in LSTM
#   - Adam optimiser with L2 weight decay (1e-5) for mild regularisation
#   - ReduceLROnPlateau scheduler (factor=0.5, patience=5 epochs)
#   - Early stopping with best-weight restoration (patience=15 epochs)
# =============================================================================

def train_one_epoch(model     : nn.Module,
                    loader    : DataLoader,
                    optimizer : torch.optim.Optimizer,
                    criterion : nn.Module) -> float:
    """
    Execute one full training epoch over all mini-batches.

    Returns
    -------
    Mean loss over all training samples for this epoch.
    """
    model.train()
    total_loss = 0.0

    for Xb, yb in loader:
        Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        pred = model(Xb)
        loss = criterion(pred, yb)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * len(yb)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model     : nn.Module,
             loader    : DataLoader,
             criterion : nn.Module):
    """
    Evaluate model on a DataLoader without computing gradients.

    Returns
    -------
    mean_loss : float     — mean loss over all samples
    all_pred  : np.array  — shape (N, OUTPUT_STEPS), predictions in ft
    all_true  : np.array  — shape (N, OUTPUT_STEPS), observations in ft
    """
    model.eval()
    total_loss = 0.0
    all_pred, all_true = [], []

    for Xb, yb in loader:
        Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
        pred = model(Xb)
        total_loss += criterion(pred, yb).item() * len(yb)
        stage, _, _, _ = unpack_model_output(pred)
        all_pred.append(stage.cpu().numpy())
        all_true.append(yb.cpu().numpy())

    return (total_loss / len(loader.dataset),
            np.concatenate(all_pred),
            np.concatenate(all_true))


def fit_model(model      : nn.Module,
              train_loader: DataLoader,
              val_loader  : DataLoader,
              criterion   : nn.Module,
              lr          : float,
              weight_decay: float = 1e-5,
              epochs      : int = EPOCHS,
              patience    : int = PATIENCE):
    """
    Full training loop with early stopping and learning-rate scheduling.

    Parameters
    ----------
    model        : HydroLSTM or HydroTransformer (on DEVICE)
    train_loader : DataLoader for training partition
    val_loader   : DataLoader for validation partition
    criterion    : FloodAwareLoss instance
    lr           : initial learning rate for Adam optimiser
    epochs       : maximum number of training epochs
    patience     : early stopping patience (epochs without val improvement)

    Returns
    -------
    model        : trained model with best validation weights restored
    train_losses : list of per-epoch training loss values
    val_losses   : list of per-epoch validation loss values
    """
    optimizer = torch.optim.Adam(model.parameters(),
                                  lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5)

    best_val     = float("inf")
    best_wts     = deepcopy(model.state_dict())
    patience_cnt = 0
    train_losses, val_losses = [], []

    for epoch in range(1, epochs + 1):
        tr_loss        = train_one_epoch(model, train_loader, optimizer, criterion)
        val_loss, _, _ = evaluate(model, val_loader, criterion)
        scheduler.step(val_loss)

        train_losses.append(tr_loss)
        val_losses.append(val_loss)

        if val_loss < best_val:
            best_val     = val_loss
            best_wts     = deepcopy(model.state_dict())
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                print(f"    Early stopping at epoch {epoch} "
                      f"(best val loss = {best_val:.5f})")
                break

        if epoch % 10 == 0:
            print(f"    Epoch {epoch:3d}  "
                  f"train={tr_loss:.5f}  val={val_loss:.5f}  "
                  f"lr={optimizer.param_groups[0]['lr']:.2e}")

    model.load_state_dict(best_wts)
    return model, train_losses, val_losses


def _make_train_loader(dataset: ContinuousSequenceDataset) -> DataLoader:
    """Training DataLoader with optional event-weighted storm sampling."""
    if USE_BALANCED_STORM_SAMPLER:
        return DataLoader(
            dataset,
            batch_sampler=BalancedStormBatchSampler(dataset, BATCH_SIZE, drop_last=True),
            worker_init_fn=_worker_init_fn)
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                      drop_last=True, worker_init_fn=_worker_init_fn)


def _build_loaders(df_tr_part  : pd.DataFrame,
                   df_va_part  : pd.DataFrame,
                   feat_cols   : list,
                   target_cols : list,
                   scaler=None,
                   fit_scaler  : bool = True,
                   thresholds  : StageThresholds = None):
    """
    Construct train and validation DataLoaders for a given data partition.

    Training uses BalancedStormBatchSampler when enabled; validation is
    chronological (shuffle=False). StandardScaler is fit on train only.
  Sampler flood class uses significant_ft (9 ft), not percentile threshold.
    """
    sig = (thresholds.significant_ft if thresholds is not None
           else SIGNIFICANT_STAGE_FT)
    ds_tr = ContinuousSequenceDataset(
        df_tr_part, feat_cols, target_cols,
        fit_scaler=fit_scaler, scaler=scaler, significant_ft=sig)
    ds_va = ContinuousSequenceDataset(
        df_va_part, feat_cols, target_cols,
        scaler=ds_tr.scaler, fit_scaler=False, significant_ft=sig)

    if len(ds_tr) < BATCH_SIZE or len(ds_va) == 0:
        return None, None, None

    if USE_BALANCED_STORM_SAMPLER:
        tr_loader = _make_train_loader(ds_tr)
    else:
        tr_loader = DataLoader(ds_tr, batch_size=BATCH_SIZE, shuffle=True,
                               drop_last=True, worker_init_fn=_worker_init_fn)
    va_loader = DataLoader(ds_va, batch_size=BATCH_SIZE, shuffle=False,
                           worker_init_fn=_worker_init_fn)
    return tr_loader, va_loader, ds_tr.scaler


# =============================================================================
# SECTION 9b — OPTUNA BAYESIAN HYPERPARAMETER OPTIMISATION
# ---------------------------------------------------------------------------
# Optuna uses the Tree-structured Parzen Estimator (TPE) — a Bayesian
# optimisation algorithm that builds a probabilistic model of the objective
# function and uses it to select the most promising hyperparameter
# configurations to evaluate next.
#
# Why TPE over grid search?
#   - Grid search evaluates all combinations regardless of merit.
#     With 5 parameters each at 4 values = 1,024 combinations.
#     TPE evaluates 50 trials and concentrates them in the best region.
#   - TPE learns from previous trials: early unpromising regions are
#     abandoned; promising regions are explored more densely.
#   - MedianPruner further reduces compute: trials clearly worse than
#     the median at any fold are stopped early (pruning).
#
# Walk-forward CV during search:
#   TimeSeriesSplit with 3 folds maintains temporal ordering.
#   Fold 1: train [0–33%], val [33–50%]
#   Fold 2: train [0–50%], val [50–67%]
#   Fold 3: train [0–67%], val [67–84%]
#   Only the training partition is used — test set is never accessed.
#
# The objective function returns mean val FloodAwareLoss across folds.
# The found best_params dict overwrites LSTM_BEST_PARAMS /
# TRANSFORMER_BEST_PARAMS before final model training.
# =============================================================================

def _optuna_objective_lstm(trial       : "optuna.Trial",
                            df_tr       : pd.DataFrame,
                            feat_cols   : list,
                            target_cols : list,
                            flood_thresh: float) -> float:
    """
    Optuna objective function for HydroLSTM.

    Suggests hyperparameters for one trial, evaluates them via
    walk-forward cross-validation, and returns mean val loss.

    Search space
    ------------
    hidden_size_1 : first LSTM layer hidden dimension
    hidden_size_2 : second LSTM layer hidden dimension
                    constrained to be <= hidden_size_1
    learning_rate : Adam initial LR (log-uniform)
    dropout       : dropout probability in FC head
    weight_decay  : L2 regularisation coefficient (log-uniform)
    """
    # ── Suggest hyperparameters ───────────────────────────────────────────
    h1 = trial.suggest_categorical("hidden_size_1", [64, 128, 256, 512])
    # h2 must have FIXED choices across all trials — Optuna does not support
    # dynamic categorical distributions. We suggest from the full set and
    # clip to ensure h2 <= h1 at model instantiation time.
    h2_raw = trial.suggest_categorical("hidden_size_2", [32, 64, 128, 256])
    h2     = min(h2_raw, h1)   # enforce hierarchy without dynamic space
    lr = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    dr = trial.suggest_float("dropout", 0.1, 0.5)
    wd = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)

    # ── Walk-forward CV ───────────────────────────────────────────────────
    tscv       = TimeSeriesSplit(n_splits=OPTUNA_CV_FOLDS)
    indices    = np.arange(len(df_tr))
    fold_losses = []

    for fold_idx, (tr_idx, va_idx) in enumerate(tscv.split(indices)):
        set_all_seeds(GLOBAL_SEED + trial.number)

        df_fold_tr = df_tr.iloc[tr_idx]
        df_fold_va = df_tr.iloc[va_idx]

        if len(df_fold_tr) < LOOKBACK_STEPS * 4:
            continue

        ds_tr_f = ContinuousSequenceDataset(
            df_fold_tr, feat_cols, target_cols, fit_scaler=True,
            significant_ft=SIGNIFICANT_STAGE_FT)
        ds_va_f = ContinuousSequenceDataset(
            df_fold_va, feat_cols, target_cols,
            scaler=ds_tr_f.scaler, fit_scaler=False,
            significant_ft=SIGNIFICANT_STAGE_FT)

        if len(ds_tr_f) < BATCH_SIZE or len(ds_va_f) < 2:
            continue

        tr_ldr = _make_train_loader(ds_tr_f)
        va_ldr = DataLoader(ds_va_f, batch_size=BATCH_SIZE, shuffle=False,
                            worker_init_fn=_worker_init_fn)

        fold_thr = build_stage_thresholds(df_fold_tr, target_cols)
        criterion = FloodAwareLoss(thresholds=fold_thr)
        model = _instantiate_model(HydroLSTM, {
            "hidden_sizes": (h1, h2),
            "dropout": dr,
            "learning_rate": lr,
            "weight_decay": wd,
        }, len(feat_cols))

        model, _, _ = fit_model(
            model, tr_ldr, va_ldr, criterion,
            lr           = lr,
            weight_decay = wd,
            epochs       = OPTUNA_EPOCHS,
            patience     = OPTUNA_PATIENCE,
        )

        val_loss, _, _ = evaluate(model, va_ldr, criterion)
        fold_losses.append(val_loss)

        # Optuna pruning: if this fold is clearly worse than median, stop
        trial.report(val_loss, fold_idx)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    if not fold_losses:
        return float("inf")

    return float(np.mean(fold_losses))


def _optuna_objective_transformer(trial       : "optuna.Trial",
                                   df_tr       : pd.DataFrame,
                                   feat_cols   : list,
                                   target_cols : list,
                                   flood_thresh: float) -> float:
    """
    Optuna objective function for HydroTransformer.

    Search space
    ------------
    d_model     : embedding / hidden dimension (must be divisible by nhead)
    nhead       : number of attention heads
    num_layers  : number of Transformer encoder layers
    learning_rate: Adam initial LR (log-uniform)
    dropout     : dropout probability
    weight_decay: L2 regularisation coefficient (log-uniform)
    """
    # ── Suggest hyperparameters ───────────────────────────────────────────
    d_model    = trial.suggest_categorical("d_model", [64, 128, 256])
    # nhead must have FIXED choices — suggest from [4, 8] which both divide
    # all d_model options (64/4=16 ✓, 64/8=8 ✓, 128/4=32 ✓, 128/8=16 ✓,
    # 256/4=64 ✓, 256/8=32 ✓). nhead=16 excluded as 64%16==0 but
    # keeping space fixed and valid across all d_model values.
    nhead      = trial.suggest_categorical("nhead", [4, 8])
    num_layers = trial.suggest_int("num_layers", 2, 6)
    lr         = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    dr         = trial.suggest_float("dropout", 0.05, 0.4)
    wd         = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)

    # ── Walk-forward CV ───────────────────────────────────────────────────
    tscv        = TimeSeriesSplit(n_splits=OPTUNA_CV_FOLDS)
    indices     = np.arange(len(df_tr))
    fold_losses = []

    for fold_idx, (tr_idx, va_idx) in enumerate(tscv.split(indices)):
        set_all_seeds(GLOBAL_SEED + trial.number)

        df_fold_tr = df_tr.iloc[tr_idx]
        df_fold_va = df_tr.iloc[va_idx]

        if len(df_fold_tr) < LOOKBACK_STEPS * 4:
            continue

        ds_tr_f = ContinuousSequenceDataset(
            df_fold_tr, feat_cols, target_cols, fit_scaler=True,
            significant_ft=SIGNIFICANT_STAGE_FT)
        ds_va_f = ContinuousSequenceDataset(
            df_fold_va, feat_cols, target_cols,
            scaler=ds_tr_f.scaler, fit_scaler=False,
            significant_ft=SIGNIFICANT_STAGE_FT)

        if len(ds_tr_f) < BATCH_SIZE or len(ds_va_f) < 2:
            continue

        tr_ldr = _make_train_loader(ds_tr_f)
        va_ldr = DataLoader(ds_va_f, batch_size=BATCH_SIZE, shuffle=False,
                            worker_init_fn=_worker_init_fn)

        fold_thr = build_stage_thresholds(df_fold_tr, target_cols)
        criterion = FloodAwareLoss(thresholds=fold_thr)
        model = _instantiate_model(HydroTransformer, {
            "d_model": d_model,
            "nhead": nhead,
            "num_layers": num_layers,
            "dropout": dr,
            "learning_rate": lr,
            "weight_decay": wd,
        }, len(feat_cols))

        model, _, _ = fit_model(
            model, tr_ldr, va_ldr, criterion,
            lr           = lr,
            weight_decay = wd,
            epochs       = OPTUNA_EPOCHS,
            patience     = OPTUNA_PATIENCE,
        )

        val_loss, _, _ = evaluate(model, va_ldr, criterion)
        fold_losses.append(val_loss)

        trial.report(val_loss, fold_idx)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    if not fold_losses:
        return float("inf")

    return float(np.mean(fold_losses))


def run_optuna_search(model_class  : type,
                      df_tr        : pd.DataFrame,
                      feat_cols    : list,
                      target_cols  : list,
                      flood_thresh : float,
                      model_name   : str) -> dict:
    """
    Run Optuna TPE hyperparameter search for one architecture.

    Executes OPTUNA_N_TRIALS trials, each evaluated via OPTUNA_CV_FOLDS
    walk-forward cross-validation folds on the training partition.
    MedianPruner stops unpromising trials early to save compute.

    Parameters
    ----------
    model_class  : HydroLSTM or HydroTransformer
    df_tr        : training DataFrame (TEST partition never used here)
    feat_cols    : feature column names
    target_cols  : target column names
    flood_thresh : flood threshold (ft) from 90th pct of training targets
    model_name   : 'LSTM' or 'Transformer' (for logging)

    Returns
    -------
    best_params : dict ready for use in train_final_model()

    Side effects
    ------------
    Saves Optuna study results to:
        outputs_comparison/{model_name.lower()}_optuna_trials.csv
        outputs_comparison/{model_name.lower()}_optuna_importance.png
    """
    print(f"\n{'='*70}")
    print(f"  OPTUNA SEARCH: {model_name}")
    print(f"  Trials: {OPTUNA_N_TRIALS}  |  "
          f"CV folds: {OPTUNA_CV_FOLDS}  |  "
          f"Max epochs/fold: {OPTUNA_EPOCHS}")
    print(f"{'='*70}")

    # Select appropriate objective function
    if model_class == HydroLSTM:
        objective = lambda trial: _optuna_objective_lstm(
            trial, df_tr, feat_cols, target_cols, flood_thresh)
    else:
        objective = lambda trial: _optuna_objective_transformer(
            trial, df_tr, feat_cols, target_cols, flood_thresh)

    # TPE sampler with fixed seed for reproducibility
    sampler = TPESampler(seed=GLOBAL_SEED)

    # MedianPruner: prune trials performing worse than median at each fold
    pruner = MedianPruner(
        n_startup_trials = 10,   # don't prune the first 10 trials
        n_warmup_steps   = 1,    # allow at least 1 fold before pruning
    )

    study = optuna.create_study(
        direction = "minimize",
        sampler   = sampler,
        pruner    = pruner,
        study_name= f"crabtree_{model_name.lower()}_flood",
    )

    # Progress callback — prints every 10 trials
    def _progress_callback(study, trial):
        if trial.number % 10 == 0 or trial.number == OPTUNA_N_TRIALS - 1:
            print(f"  Trial {trial.number+1:3d}/{OPTUNA_N_TRIALS}  "
                  f"val_loss={trial.value:.5f}  "
                  f"best={study.best_value:.5f}")

    study.optimize(
        objective,
        n_trials  = OPTUNA_N_TRIALS,
        callbacks = [_progress_callback],
        show_progress_bar = False,
    )

    # ── Extract best params ───────────────────────────────────────────────
    bp = study.best_params
    print(f"\n  Best trial #{study.best_trial.number}:")
    print(f"  Best val loss: {study.best_value:.5f}")
    for k, v in bp.items():
        print(f"    {k:20s}: {v}")

    # Convert to the dict format expected by train_final_model()
    if model_class == HydroLSTM:
        best_params = {
            "hidden_sizes" : (bp["hidden_size_1"], bp["hidden_size_2"]),
            "learning_rate": bp["learning_rate"],
            "dropout"      : bp["dropout"],
            "weight_decay" : bp["weight_decay"],
        }
    else:
        best_params = {
            "d_model"      : bp["d_model"],
            "nhead"        : bp["nhead"],
            "num_layers"   : bp["num_layers"],
            "learning_rate": bp["learning_rate"],
            "dropout"      : bp["dropout"],
            "weight_decay" : bp["weight_decay"],
        }

    # ── Save trial results CSV ────────────────────────────────────────────
    trials_df = study.trials_dataframe()
    csv_path  = OUTPUT_DIR / f"{model_name.lower()}_optuna_trials.csv"
    trials_df.to_csv(csv_path, index=False)
    print(f"  Trial results saved: {csv_path.name}")

    # ── Plot: Optuna optimisation history ─────────────────────────────────
    _plot_optuna_history(study, model_name)

    # ── Plot: Parameter importance ────────────────────────────────────────
    _plot_optuna_importance(study, model_name)

    return best_params


def _plot_optuna_history(study: "optuna.Study", model_name: str):
    """
    Plot Optuna trial history: objective value per trial with best-so-far
    running minimum highlighted.

    Shows the convergence behaviour of the TPE search — early trials
    explore the space broadly; later trials exploit the best regions.
    """
    completed = [t for t in study.trials
                 if t.value is not None and np.isfinite(t.value)]
    if not completed:
        return

    trial_nums = [t.number for t in completed]
    values     = [t.value for t in completed]
    best_so_far = np.minimum.accumulate(values)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.scatter(trial_nums, values, s=25, alpha=0.5, color="#2E75B6",
               label="Trial val loss")
    ax.plot(trial_nums, best_so_far, color="#CC2200", lw=2,
            label="Best so far")

    ax.set_xlabel("Trial number")
    ax.set_ylabel("Mean CV FloodAwareLoss (ft²)")
    ax.set_title(
        f"{model_name} — Optuna TPE Optimisation History\n"
        f"{OPTUNA_N_TRIALS} trials × {OPTUNA_CV_FOLDS}-fold walk-forward CV  |  "
        f"Best val loss = {study.best_value:.5f}"
    )
    ax.legend()
    fig.tight_layout()
    path = OUTPUT_DIR / f"{model_name.lower()}_optuna_history.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path.name}")


def _plot_optuna_importance(study: "optuna.Study", model_name: str):
    """
    Plot hyperparameter importance scores from the completed Optuna study.

    Importance is estimated via fANOVA (functional ANOVA), which measures
    the fraction of output variance attributable to each hyperparameter
    and their interactions. High importance = the objective is sensitive
    to this parameter and it should be tuned carefully.
    """
    try:
        importance = optuna.importance.get_param_importances(study)
    except Exception:
        return   # fANOVA requires >= 4 completed trials

    names  = list(importance.keys())
    scores = list(importance.values())

    fig, ax = plt.subplots(figsize=(8, max(3, len(names) * 0.6)))
    colors = ["#CC2200" if s == max(scores) else "#2E75B6" for s in scores]
    ax.barh(names[::-1], scores[::-1], color=colors[::-1], edgecolor="0.4")
    ax.set_xlabel("Relative importance (fANOVA)")
    ax.set_title(
        f"{model_name} — Hyperparameter Importance\n"
        f"fANOVA over {len([t for t in study.trials if t.value is not None])} "
        f"completed trials  |  Red bar = most important parameter"
    )
    for i, (n, s) in enumerate(zip(names[::-1], scores[::-1])):
        ax.text(s + 0.005, i, f"{s:.3f}", va="center", fontsize=9)
    ax.set_xlim(0, max(scores) * 1.2)
    fig.tight_layout()
    path = OUTPUT_DIR / f"{model_name.lower()}_optuna_importance.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path.name}")


def _instantiate_model(model_class: type, model_params: dict, n_features: int) -> nn.Module:
    """Factory for HydroLSTM / HydroTransformer with v4 multi-task heads."""
    if model_class == HydroLSTM:
        if USE_IMPROVED_MODELS:
            return HydroLSTMSeq2Seq(
                n_features=n_features,
                hidden_sizes=model_params["hidden_sizes"],
                dropout=model_params["dropout"],
            ).to(DEVICE).float()
        return HydroLSTM(
            n_features=n_features,
            hidden_sizes=model_params["hidden_sizes"],
            dropout=model_params["dropout"],
        ).to(DEVICE).float()
    return HydroTransformer(
        n_features=n_features,
        d_model=model_params["d_model"],
        nhead=model_params["nhead"],
        num_layers=model_params["num_layers"],
        dropout=model_params["dropout"],
    ).to(DEVICE).float()


# =============================================================================
# SECTION 9a — MODEL CHECKPOINTING (save / load trained weights)
# Checkpoints are keyed by model tag (e.g. lstm_fold3, transformer_fold3) and
# validated with a fingerprint of feature columns + architecture flags so stale
# weights are never silently reused after a config change.
# =============================================================================

def _sanitize_checkpoint_tag(model_name: str) -> str:
    """Filesystem-safe tag from a human model name."""
    tag = model_name.lower().strip().replace(" ", "_").replace("-", "_")
    return "".join(c if c.isalnum() or c == "_" else "_" for c in tag)


def _checkpoint_fingerprint(model_class : type,
                             model_params: dict,
                             feat_cols   : list,
                             n_ensemble  : int = 1) -> str:
    """Hash that invalidates the cache when features or architecture change."""
    payload = {
        "model_class"  : model_class.__name__,
        "model_params" : model_params,
        "n_features"   : len(feat_cols),
        "feat_cols"    : feat_cols,
        "lookback"     : LOOKBACK_STEPS,
        "output_steps" : OUTPUT_STEPS,
        "improved"     : USE_IMPROVED_MODELS,
        "flood_aux"    : USE_FLOOD_AUX_HEAD,
        "peak_aux"     : USE_PEAK_AUX_HEAD,
        "deriv_loss"   : USE_DERIVATIVE_LOSS,
        "sig_stage_ft" : SIGNIFICANT_STAGE_FT,
        "ext_stage_ft" : EXTREME_STAGE_FT,
        "elev_pct"     : ELEVATED_PERCENTILE,
        "future_forcing": USE_FUTURE_FORCING,
        "rise_lag_loss": USE_RISING_LIMB_ANTICIPATION_LOSS,
        "ordinal_head" : USE_ORDINAL_SEVERITY_HEAD,
        "ordinal_lvls" : list(ORDINAL_STAGE_LEVELS_FT),
        "cont_stage_wt": USE_CONTINUOUS_STAGE_WEIGHT,
        "flood_replace": FLOOD_SAMPLER_ALWAYS_REPLACE,
        "hetero"       : USE_HETEROSCEDASTIC,
        "tcn"          : USE_TCN_FRONTEND,
        "ensemble_n"   : n_ensemble,
    }
    blob = json.dumps(payload, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def _checkpoint_path(tag: str) -> Path:
    d = CHECKPOINT_DIR / tag
    d.mkdir(parents=True, exist_ok=True)
    return d


def _save_checkpoint_metadata(ckpt_dir: Path, fingerprint: str, extra: dict):
    meta = {"fingerprint": fingerprint, **extra}
    (ckpt_dir / "metadata.json").write_text(json.dumps(meta, indent=2))


def _load_checkpoint_metadata(ckpt_dir: Path) -> dict:
    meta_path = ckpt_dir / "metadata.json"
    if not meta_path.exists():
        return {}
    return json.loads(meta_path.read_text())


def _fingerprint_matches(ckpt_dir: Path, expected_fp: str) -> bool:
    meta = _load_checkpoint_metadata(ckpt_dir)
    stored = meta.get("fingerprint")
    if stored != expected_fp:
        if stored:
            print(f"  [CACHE] Stale checkpoint in {ckpt_dir.name} "
                  f"(stored={stored}, expected={expected_fp}) — retraining.")
        return False
    return True


def save_model_checkpoint(model       : nn.Module,
                        scaler,
                        tag         : str,
                        fingerprint : str,
                        train_losses: list = None,
                        val_losses  : list = None):
    """Save a single model + scaler + optional loss histories."""
    ckpt_dir = _checkpoint_path(tag)
    torch.save(model.state_dict(), ckpt_dir / "model.pt")
    with open(ckpt_dir / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    _save_checkpoint_metadata(ckpt_dir, fingerprint, {
        "n_members": 1,
        "tag": tag,
    })
    if train_losses is not None:
        (ckpt_dir / "train_losses.json").write_text(json.dumps(train_losses))
    if val_losses is not None:
        (ckpt_dir / "val_losses.json").write_text(json.dumps(val_losses))
    print(f"  [CACHE] Saved checkpoint → {ckpt_dir}/")


def load_model_checkpoint(model_class : type,
                          model_params: dict,
                          feat_cols   : list,
                          tag         : str,
                          fingerprint : str):
    """
    Load a single cached model if present and fingerprint-valid.

    Returns (model, scaler, train_losses, val_losses) or None.
    """
    ckpt_dir = _checkpoint_path(tag)
    model_pt = ckpt_dir / "model.pt"
    scaler_pkl = ckpt_dir / "scaler.pkl"
    if not model_pt.exists() or not scaler_pkl.exists():
        return None
    if not _fingerprint_matches(ckpt_dir, fingerprint):
        return None

    model = _instantiate_model(model_class, model_params, len(feat_cols))
    model.load_state_dict(torch.load(model_pt, map_location=DEVICE))
    model.eval()
    with open(scaler_pkl, "rb") as f:
        scaler = pickle.load(f)

    tr_path = ckpt_dir / "train_losses.json"
    va_path = ckpt_dir / "val_losses.json"
    tr_losses = json.loads(tr_path.read_text()) if tr_path.exists() else []
    va_losses = json.loads(va_path.read_text()) if va_path.exists() else []

    print(f"  [CACHE] Loaded checkpoint ← {ckpt_dir}/  (skipping training)")
    return model, scaler, tr_losses, va_losses


def save_ensemble_checkpoint(models      : list,
                             scaler,
                             tag         : str,
                             fingerprint : str,
                             train_losses: list = None,
                             val_losses  : list = None):
    """Save all ensemble members + shared scaler."""
    ckpt_dir = _checkpoint_path(tag)
    for i, m in enumerate(models):
        torch.save(m.state_dict(), ckpt_dir / f"member_{i:02d}.pt")
    with open(ckpt_dir / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    _save_checkpoint_metadata(ckpt_dir, fingerprint, {
        "n_members": len(models),
        "tag": tag,
    })
    if train_losses is not None:
        (ckpt_dir / "train_losses.json").write_text(json.dumps(train_losses))
    if val_losses is not None:
        (ckpt_dir / "val_losses.json").write_text(json.dumps(val_losses))
    print(f"  [CACHE] Saved ensemble ({len(models)} members) → {ckpt_dir}/")


def load_ensemble_checkpoint(model_class : type,
                             model_params: dict,
                             feat_cols   : list,
                             tag         : str,
                             fingerprint : str,
                             n_expected  : int):
    """
    Load a cached deep ensemble if all members exist and fingerprint matches.

    Returns (models, scaler, train_losses, val_losses) or None.
    """
    ckpt_dir = _checkpoint_path(tag)
    scaler_pkl = ckpt_dir / "scaler.pkl"
    if not scaler_pkl.exists():
        return None
    if not _fingerprint_matches(ckpt_dir, fingerprint):
        return None

    meta = _load_checkpoint_metadata(ckpt_dir)
    n_members = int(meta.get("n_members", 0))
    if n_members != n_expected:
        print(f"  [CACHE] Member count mismatch in {ckpt_dir.name} "
              f"(stored={n_members}, expected={n_expected}) — retraining.")
        return None

    models = []
    for i in range(n_expected):
        member_pt = ckpt_dir / f"member_{i:02d}.pt"
        if not member_pt.exists():
            print(f"  [CACHE] Missing {member_pt.name} — retraining ensemble.")
            return None
        m = _instantiate_model(model_class, model_params, len(feat_cols))
        m.load_state_dict(torch.load(member_pt, map_location=DEVICE))
        m.eval()
        models.append(m)

    with open(scaler_pkl, "rb") as f:
        scaler = pickle.load(f)

    tr_path = ckpt_dir / "train_losses.json"
    va_path = ckpt_dir / "val_losses.json"
    tr_losses = json.loads(tr_path.read_text()) if tr_path.exists() else []
    va_losses = json.loads(va_path.read_text()) if va_path.exists() else []

    print(f"  [CACHE] Loaded ensemble ({n_expected} members) ← {ckpt_dir}/  "
          f"(skipping training)")
    return models, scaler, tr_losses, va_losses


def train_final_model(model_class : type,
                      model_params: dict,
                      df_tr       : pd.DataFrame,
                      df_va       : pd.DataFrame,
                      feat_cols   : list,
                      target_cols : list,
                      thresholds  : StageThresholds,
                      model_name  : str,
                      checkpoint_tag: str = None,
                      use_cache   : bool = True,
                      save_checkpoint: bool = None):
    """
    Train a model on the full training partition using the best
    hyperparameters. Validation is used for early stopping only.

    Parameters
    ----------
    model_class  : HydroLSTM or HydroTransformer
    model_params : hyperparameter dict
    df_tr        : training DataFrame
    df_va        : validation DataFrame
    feat_cols    : feature column names
    target_cols  : target column names
    flood_thresh : StageThresholds with elevated/significant/extreme tiers
    model_name   : 'LSTM' or 'Transformer' (for logging)
    checkpoint_tag : optional override for cache directory name
    use_cache    : if False, always train (used for ensemble member loop)
    save_checkpoint : if None, uses SAVE_TRAINED_MODELS global

    Returns
    -------
    model, scaler, train_losses, val_losses
    """
    tag = checkpoint_tag or _sanitize_checkpoint_tag(model_name)
    fingerprint = _checkpoint_fingerprint(model_class, model_params, feat_cols,
                                          n_ensemble=1)
    do_save = SAVE_TRAINED_MODELS if save_checkpoint is None else save_checkpoint

    if USE_MODEL_CACHE and not FORCE_RETRAIN and use_cache:
        cached = load_model_checkpoint(
            model_class, model_params, feat_cols, tag, fingerprint)
        if cached is not None:
            return cached

    print(f"\n=== Training {model_name} ===")
    set_all_seeds()

    print_stage_threshold_diagnostics(df_tr, thresholds, label=f"{model_name} train")

    tr_loader, va_loader, scaler = _build_loaders(
        df_tr, df_va, feat_cols, target_cols, fit_scaler=True,
        thresholds=thresholds)

    criterion = FloodAwareLoss(thresholds=thresholds)
    model = _instantiate_model(model_class, model_params, len(feat_cols))

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable parameters: {n_params:,}")

    model, tr_losses, val_losses = fit_model(
        model, tr_loader, va_loader, criterion,
        lr           = model_params["learning_rate"],
        weight_decay = model_params.get("weight_decay", 1e-5))

    if do_save:
        save_model_checkpoint(model, scaler, tag, fingerprint,
                              tr_losses, val_losses)

    return model, scaler, tr_losses, val_losses


def train_deep_ensemble(model_class : type,
                        model_params: dict,
                        df_tr       : pd.DataFrame,
                        df_va       : pd.DataFrame,
                        feat_cols   : list,
                        target_cols : list,
                        thresholds  : StageThresholds,
                        model_name  : str,
                        n_models    : int = ENSEMBLE_N_MODELS):
    """
    Train N independently-initialized models for deep-ensemble UQ.
    Returns list of models and the shared scaler (fit on train).
    """
    tag = _sanitize_checkpoint_tag(model_name)
    n_members = n_models

    if not USE_DEEP_ENSEMBLE or n_members < 2:
        m, sc, tr, va = train_final_model(
            model_class, model_params, df_tr, df_va, feat_cols, target_cols,
            thresholds, model_name, checkpoint_tag=tag)
        return [m], sc, tr, va

    fingerprint = _checkpoint_fingerprint(model_class, model_params, feat_cols,
                                          n_ensemble=n_members)

    if USE_MODEL_CACHE and not FORCE_RETRAIN:
        cached = load_ensemble_checkpoint(
            model_class, model_params, feat_cols, tag, fingerprint, n_members)
        if cached is not None:
            return cached

    print(f"\n=== Training {model_name} deep ensemble ({n_members} members) ===")
    models, scaler, tr_losses, val_losses = [], None, [], []
    for k in range(n_members):
        print(f"  --- Ensemble member {k + 1}/{n_members} ---")
        set_all_seeds(GLOBAL_SEED + k * 9973)
        m, sc, tr, va = train_final_model(
            model_class, model_params, df_tr, df_va, feat_cols, target_cols,
            thresholds, f"{model_name}-ens{k+1}",
            use_cache=False, save_checkpoint=False)
        models.append(m)
        scaler = sc
        tr_losses.append(tr)
        val_losses.append(va)

    if SAVE_TRAINED_MODELS:
        save_ensemble_checkpoint(models, scaler, tag, fingerprint,
                                 tr_losses, val_losses)

    return models, scaler, tr_losses, val_losses


def train_residual_corrector(stage1      : nn.Module,
                             df_tr       : pd.DataFrame,
                             df_va       : pd.DataFrame,
                             feat_cols   : list,
                             target_cols : list,
                             thresholds  : StageThresholds,
                             scaler,
                             model_name  : str = "LSTM"):
    """
    Optional Stage-2 residual error-learning step (v2).

    Wraps a frozen Stage-1 model with a small GRU corrector and trains the
    composite (Stage-1 + Stage-2) end-to-end; only Stage-2 receives gradient.
    Returns a ResidualCorrectedModel presenting the standard forward contract.
    """
    print(f"\n=== Training residual corrector for {model_name} (Stage-2) ===")
    set_all_seeds()
    # Reuse the Stage-1 scaler so inputs match exactly.
    tr_loader, va_loader, _ = _build_loaders(
        df_tr, df_va, feat_cols, target_cols, scaler=scaler, fit_scaler=False,
        thresholds=thresholds)
    if tr_loader is None:
        print("  [WARN] Not enough data for residual stage; skipping.")
        return stage1

    stage2  = ResidualMLPHead(len(feat_cols)).to(DEVICE).float()
    wrapper = ResidualCorrectedModel(stage1, stage2).to(DEVICE).float()
    criterion = FloodAwareLoss(thresholds=thresholds)

    # Only Stage-2 parameters are trainable (Stage-1 is frozen in the wrapper).
    trainable = [p for p in wrapper.parameters() if p.requires_grad]
    n_params  = sum(p.numel() for p in trainable)
    print(f"  Stage-2 trainable parameters: {n_params:,}")

    wrapper, _, _ = fit_model(
        wrapper, tr_loader, va_loader, criterion,
        lr=5e-4, weight_decay=1e-5,
        epochs=RESIDUAL_EPOCHS, patience=8)
    return wrapper


# =============================================================================
# SECTION 10 — INFERENCE
# =============================================================================

def predict_continuous(model       : nn.Module,
                       scaler,
                       df_partition: pd.DataFrame,
                       feat_cols   : list,
                       target_cols : list) -> pd.DataFrame:
    """
    Run inference over an entire DataFrame partition.

    Parameters
    ----------
    model        : trained model (eval mode internally managed)
    scaler       : fitted StandardScaler
    df_partition : DataFrame partition for inference
    feat_cols    : feature column names
    target_cols  : target column names (needed for dataset construction)

    Returns
    -------
    pd.DataFrame of shape (N, OUTPUT_STEPS) with columns
    ['t+30min', 't+60min', ..., 't+180min'] and DatetimeIndex.
    """
    ds = ContinuousSequenceDataset(
        df_partition, feat_cols, target_cols,
        scaler=scaler, fit_scaler=False)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                        worker_init_fn=_worker_init_fn)

    model.eval()
    all_preds = []
    with torch.no_grad():
        for Xb, _ in loader:
            Xb = Xb.to(DEVICE)
            out = model(Xb)
            stage, _, _, _ = unpack_model_output(out)
            all_preds.append(stage.cpu().numpy())

    preds = np.concatenate(all_preds, axis=0)   # (N, 6)
    if CLAMP_STAGE_FLOOR is not None:
        preds = np.clip(preds, CLAMP_STAGE_FLOOR, None)
    cols  = [f"t+{h}min" for h in FORECAST_HORIZONS_MIN]
    return pd.DataFrame(preds, index=ds.time_index, columns=cols)


def ensemble_predict(models      : list,
                     scaler,
                     df_partition: pd.DataFrame,
                     feat_cols   : list,
                     target_cols : list) -> dict:
    """
    Deep-ensemble inference with heteroscedastic aleatoric + epistemic variance.

    Returns dict with mean, std, lower, upper (95% PI), and per-member samples.
    """
    ds = ContinuousSequenceDataset(
        df_partition, feat_cols, target_cols,
        scaler=scaler, fit_scaler=False)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                        worker_init_fn=_worker_init_fn)
    cols = [f"t+{h}min" for h in FORECAST_HORIZONS_MIN]

    member_means, member_ale_vars = [], []
    for model in models:
        model.eval()
        preds_b, ale_b = [], []
        with torch.no_grad():
            for Xb, _ in loader:
                Xb = Xb.to(DEVICE)
                out = model(Xb)
                stage, _, _, log_var = unpack_model_output(out)
                preds_b.append(stage.cpu().numpy())
                if log_var is not None:
                    ale_b.append(torch.exp(log_var).cpu().numpy())
        member_means.append(np.concatenate(preds_b, axis=0))
        if ale_b:
            member_ale_vars.append(np.concatenate(ale_b, axis=0))

    stack = np.stack(member_means, axis=0)              # (M, N, 6)
    mean_arr = stack.mean(axis=0)
    epi_var  = stack.var(axis=0)
    if member_ale_vars:
        ale_var = np.stack(member_ale_vars, axis=0).mean(axis=0)
        total_var = epi_var + ale_var
    else:
        total_var = epi_var
    std_arr = np.sqrt(np.clip(total_var, 1e-9, None))

    if CLAMP_STAGE_FLOOR is not None:
        mean_arr = np.clip(mean_arr, CLAMP_STAGE_FLOOR, None)

    idx = ds.time_index
    return {
        "mean"   : pd.DataFrame(mean_arr,             index=idx, columns=cols),
        "std"    : pd.DataFrame(std_arr,              index=idx, columns=cols),
        "lower"  : pd.DataFrame(mean_arr - 2*std_arr, index=idx, columns=cols),
        "upper"  : pd.DataFrame(mean_arr + 2*std_arr, index=idx, columns=cols),
        "samples": stack,
    }


def get_obs_dataframe(df_partition : pd.DataFrame,
                      target_cols  : list) -> pd.DataFrame:
    """
    Extract observed gage heights for all six horizons from a DataFrame
    partition, aligned to the same index as predict_continuous().

    Returns
    -------
    pd.DataFrame of shape (N, 6) with horizon column names.
    """
    n    = len(df_partition)
    rows = df_partition.iloc[LOOKBACK_STEPS:]
    obs  = rows[target_cols].copy()
    cols = [f"t+{h}min" for h in FORECAST_HORIZONS_MIN]
    obs.columns = cols
    return obs


def compute_persistence_baseline(df_te       : pd.DataFrame,
                                  target_cols : list) -> pd.DataFrame:
    """
    Naïve persistence baseline: predict that the water level at each
    forecast horizon equals the current observed water level at time t.

    For all six horizons the prediction is gauge_us1(t), i.e., no change
    is assumed. This is the simplest possible benchmark against which any
    useful model must demonstrate skill.
    """
    rows = df_te.iloc[LOOKBACK_STEPS:]
    base = rows["gauge_us1"].values
    cols = [f"t+{h}min" for h in FORECAST_HORIZONS_MIN]
    return pd.DataFrame(
        np.tile(base[:, None], (1, OUTPUT_STEPS)),
        index=rows.index,
        columns=cols
    )


# =============================================================================
# SECTION 11 — EVALUATION METRICS
# Metrics are computed for each of the six forecast horizons individually,
# enabling characterisation of forecast skill as a function of lead time.
# =============================================================================

def nse(obs: np.ndarray, pred: np.ndarray) -> float:
    """Nash-Sutcliffe Efficiency. Range: (−∞, 1]. Perfect = 1."""
    ss_res = np.sum((obs - pred) ** 2)
    ss_tot = np.sum((obs - obs.mean()) ** 2)
    return float(1.0 - ss_res / (ss_tot + 1e-10))


def kge(obs: np.ndarray, pred: np.ndarray) -> float:
    """
    Kling-Gupta Efficiency (Gupta et al., 2009).
    Decomposes into Pearson correlation, variability ratio, and bias ratio.
    Range: (−∞, 1]. Perfect = 1.
    """
    r     = np.corrcoef(obs, pred)[0, 1]
    alpha = pred.std() / (obs.std() + 1e-10)
    beta  = pred.mean() / (obs.mean() + 1e-10)
    return float(1.0 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2))


def equitable_threat_score(obs_ex: np.ndarray, pred_ex: np.ndarray) -> float:
    """ETS (Gilbert skill score) for binary exceedance detection."""
    hits = int(np.sum(obs_ex & pred_ex))
    miss = int(np.sum(obs_ex & ~pred_ex))
    fa   = int(np.sum(~obs_ex & pred_ex))
    n    = len(obs_ex)
    if n == 0:
        return np.nan
    random_hits = (hits + miss) * (hits + fa) / n
    denom = hits + miss + fa - random_hits
    if denom <= 0:
        return np.nan
    return float((hits - random_hits) / denom)


def crps_gaussian(obs: np.ndarray, mean: np.ndarray, std: np.ndarray) -> float:
    """CRPS for Gaussian predictive distribution (closed form)."""
    from scipy.stats import norm
    z = (obs - mean) / (std + 1e-9)
    crps = std * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / np.sqrt(np.pi))
    return float(np.mean(crps))


def threshold_crossing_error_min(obs: np.ndarray,
                                 pred: np.ndarray,
                                 flood_thr: float,
                                 step_min: int = TIMESTEP_MIN) -> float:
    """
  Mean timing error (min) between first upward threshold crossing in obs
    vs pred (positive = predicted crossing is late).
    """
    def first_cross(x):
        above = x >= flood_thr
        if not above.any():
            return np.nan
        prev = np.concatenate([[False], above[:-1]])
        cross = np.where(above & ~prev)[0]
        return float(cross[0]) if len(cross) else np.nan

    errors = []
    # scan overlapping windows for crossing events
    i = 1
    while i < len(obs):
        if obs[i] >= flood_thr and obs[i - 1] < flood_thr:
            lo = max(0, i - 36)
            hi = min(len(pred), i + 36)
            pc = first_cross(pred[lo:hi])
            if np.isfinite(pc):
                errors.append((lo + pc - i) * step_min)
            i += 36
        else:
            i += 1
    return float(np.mean(errors)) if errors else np.nan


def compute_metrics_for_horizon(obs_col  : np.ndarray,
                                 pred_col : np.ndarray,
                                 eval_thr : float,
                                 extreme_thr: float = EXTREME_STAGE_FT) -> dict:
    """
    Compute the full evaluation metric suite for a single forecast horizon.

    Parameters
    ----------
    obs_col   : 1-D array of observed gage heights (ft)
    pred_col  : 1-D array of predicted gage heights (ft)
    eval_thr  : significant-event threshold (default 9 ft) for NSE_flood / CSI
    extreme_thr : tail threshold (default 12 ft) for NSE_extreme
    """
    mask = np.isfinite(obs_col) & np.isfinite(pred_col)
    obs  = obs_col[mask]
    pred = pred_col[mask]

    if len(obs) < 2:
        return {}

    rmse_val  = float(np.sqrt(mean_squared_error(obs, pred)))
    mae_val   = float(np.mean(np.abs(obs - pred)))
    nse_val   = nse(obs, pred)
    kge_val   = kge(obs, pred)
    pbias_val = float(100.0 * (pred.mean() - obs.mean()) / (obs.mean() + 1e-10))

    # Peak-RMSE / NSE_flood: restricted to significant events (>= eval_thr, 9 ft)
    flood_mask = obs >= eval_thr
    if flood_mask.sum() > 1:
        peak_rmse = float(np.sqrt(mean_squared_error(obs[flood_mask],
                                                       pred[flood_mask])))
    else:
        peak_rmse = np.nan

    if flood_mask.sum() > 2:
        o_f = obs[flood_mask]
        nse_flood_val = float(1.0 - np.sum((o_f - pred[flood_mask])**2) /
                              (np.sum((o_f - o_f.mean())**2) + 1e-10))
    else:
        nse_flood_val = np.nan

    # Extreme tail (>= 12 ft) — rare peaks generalisation
    extreme_mask = obs >= extreme_thr
    if extreme_mask.sum() > 2:
        o_x = obs[extreme_mask]
        nse_extreme_val = float(1.0 - np.sum((o_x - pred[extreme_mask])**2) /
                                (np.sum((o_x - o_x.mean())**2) + 1e-10))
        peak_rmse_extreme = float(np.sqrt(mean_squared_error(
            o_x, pred[extreme_mask])))
    else:
        nse_extreme_val = np.nan
        peak_rmse_extreme = np.nan

    # Peak timing error: storm peaks detected at significant threshold
    peak_idxs, _ = find_peaks(obs, height=eval_thr, distance=216)   # 18-hr min sep
    timing_errors = []
    window        = 36   # ±3 hr search window

    for pk in peak_idxs:
        lo = max(0, pk - window)
        hi = min(len(pred) - 1, pk + window)
        pred_pk = lo + np.argmax(pred[lo:hi + 1])
        timing_errors.append((pred_pk - pk) * TIMESTEP_MIN)   # convert to min

    timing_med = float(np.median(timing_errors)) if timing_errors else np.nan

    # Rising-limb-only RMSE (positive rate-of-rise on observed series)
    k6 = max(1, 30 // TIMESTEP_MIN)
    rise_rate = (obs - np.roll(obs, k6)) / float(k6 * TIMESTEP_MIN)
    rise_rate[:k6] = 0.0
    rising_mask = rise_rate > RISE_RATE_THRESH_FT_PER_MIN
    if rising_mask.sum() > 1:
        rise_rmse = float(np.sqrt(mean_squared_error(obs[rising_mask],
                                                      pred[rising_mask])))
    else:
        rise_rmse = np.nan

    obs_ex = obs >= eval_thr
    pred_ex = pred >= eval_thr
    ets_val = equitable_threat_score(obs_ex, pred_ex)
    thr_xing_err = threshold_crossing_error_min(obs, pred, eval_thr)

    n_sig = int(flood_mask.sum())
    n_ext = int(extreme_mask.sum())

    return {
        "RMSE"        : round(rmse_val,  4),
        "MAE"         : round(mae_val,   4),
        "NSE"         : round(nse_val,   4),
        "NSE_flood"   : round(nse_flood_val, 4) if np.isfinite(nse_flood_val) else np.nan,
        "NSE_extreme" : round(nse_extreme_val, 4) if np.isfinite(nse_extreme_val) else np.nan,
        "n_significant": n_sig,
        "n_extreme"   : n_ext,
        "KGE"         : round(kge_val,   4),
        "PBIAS"       : round(pbias_val, 4),
        "Peak_RMSE"   : round(peak_rmse, 4) if np.isfinite(peak_rmse) else np.nan,
        "Peak_RMSE_extreme": round(peak_rmse_extreme, 4) if np.isfinite(peak_rmse_extreme) else np.nan,
        "Timing_min"  : round(timing_med, 1) if np.isfinite(timing_med) else np.nan,
        "Rise_RMSE"   : round(rise_rmse, 4) if np.isfinite(rise_rmse) else np.nan,
        "ETS"         : round(ets_val, 3) if np.isfinite(ets_val) else np.nan,
        "ThrXing_min" : round(thr_xing_err, 1) if np.isfinite(thr_xing_err) else np.nan,
    }


def compute_all_horizon_metrics(obs_df    : pd.DataFrame,
                                 pred_df   : pd.DataFrame,
                                 eval_thr  : float,
                                 extreme_thr: float = EXTREME_STAGE_FT) -> pd.DataFrame:
    """
    Compute metrics for all six forecast horizons.

    Returns
    -------
    pd.DataFrame with rows = horizons, columns = metric names.
    """
    rows = []
    for col in obs_df.columns:
        if col not in pred_df.columns:
            continue
        obs_arr  = obs_df[col].values
        pred_arr = pred_df[col].reindex(obs_df.index).values
        m        = compute_metrics_for_horizon(obs_arr, pred_arr, eval_thr, extreme_thr)
        m["Horizon"] = col
        rows.append(m)

    df_metrics = pd.DataFrame(rows).set_index("Horizon")
    return df_metrics


# =============================================================================
# SECTION 11b — EVENT-BASED SKILL METRICS (v2)
# Aggregate NSE is dominated by baseflow; for a warning product the honest
# headline numbers are event-based. These operate on the reconstructed t+150
# series and summarise detection skill, peak bias, and the late-peak tendency.
# =============================================================================

def compute_event_skill_scores(obs_ser  : pd.Series,
                               pred_ser : pd.Series,
                               flood_thr: float,
                               horizon_min: int = 150) -> dict:
    """
    Categorical detection skill at the flood threshold plus peak-timing /
    peak-magnitude summaries. Timing convention: positive = predicted peak
    LATE (the dangerous direction).
    """
    df = pd.concat([obs_ser.rename("obs"),
                    pred_ser.reindex(obs_ser.index).rename("pred")], axis=1).dropna()
    if len(df) < 10:
        return {}
    o = df["obs"].values
    p = df["pred"].values

    # ── Categorical exceedance skill (per-timestep) ──────────────────────
    obs_ex  = o >= flood_thr
    pred_ex = p >= flood_thr
    hits = int(np.sum(obs_ex & pred_ex))
    miss = int(np.sum(obs_ex & ~pred_ex))
    fa   = int(np.sum(~obs_ex & pred_ex))
    pod  = hits / (hits + miss)         if (hits + miss) > 0 else np.nan   # detection
    far  = fa   / (hits + fa)           if (hits + fa)   > 0 else np.nan   # false alarm
    csi  = hits / (hits + miss + fa)    if (hits + miss + fa) > 0 else np.nan  # threat
    ets  = equitable_threat_score(obs_ex, pred_ex)

    # ── Per-storm peak timing + magnitude ────────────────────────────────
    peak_idxs, _ = find_peaks(o, height=flood_thr, distance=216)
    window = 36
    timing, peak_bias = [], []
    for pk in peak_idxs:
        lo = max(0, pk - window); hi = min(len(p) - 1, pk + window)
        pred_pk = lo + int(np.argmax(p[lo:hi + 1]))
        timing.append((pred_pk - pk) * TIMESTEP_MIN)
        peak_bias.append(float(p[pred_pk] - o[pk]))

    timing = np.array(timing, dtype=float)
    n_ev = len(timing)
    return {
        "horizon_min"        : horizon_min,
        "n_events"           : n_ev,
        "POD"                : round(pod, 3) if np.isfinite(pod) else np.nan,
        "FAR"                : round(far, 3) if np.isfinite(far) else np.nan,
        "CSI"                : round(csi, 3) if np.isfinite(csi) else np.nan,
        "ETS"                : round(ets, 3) if np.isfinite(ets) else np.nan,
        "ThrXing_min"        : round(threshold_crossing_error_min(o, p, flood_thr), 1),
        "median_timing_min"  : round(float(np.median(timing)), 1) if n_ev else np.nan,
        "mean_timing_min"    : round(float(np.mean(timing)),   1) if n_ev else np.nan,
        "p90_timing_min"     : round(float(np.percentile(timing, 90)), 1) if n_ev else np.nan,
        "late_peak_fraction" : round(float(np.mean(timing > 0)), 3) if n_ev else np.nan,
        "mean_peak_bias_ft"  : round(float(np.mean(peak_bias)), 3) if n_ev else np.nan,
    }


def compute_event_skill_table(obs_te_df, pred_te_df, flood_thr) -> pd.DataFrame:
    """Event skill across all horizons for one model."""
    rows = []
    for h_min in FORECAST_HORIZONS_MIN:
        col = f"t+{h_min}min"
        if col not in pred_te_df.columns:
            continue
        s = compute_event_skill_scores(
            obs_te_df[col].dropna(),
            pred_te_df[col],
            flood_thr, horizon_min=h_min)
        if s:
            rows.append(s)
    return pd.DataFrame(rows).set_index("horizon_min") if rows else pd.DataFrame()


# =============================================================================
# SECTION 12 — MONTE CARLO DROPOUT UNCERTAINTY QUANTIFICATION
# Dropout-based uncertainty quantification (Gal & Ghahramani, 2016) is
# implemented by retaining dropout active during inference and executing
# MC_SAMPLES = 100 stochastic forward passes. The resulting empirical
# distribution of predictions at each timestep characterises aleatoric and
# epistemic uncertainty. The 95% prediction interval is defined as
# [mean − 2σ, mean + 2σ].
# =============================================================================

def mc_dropout_predict(model       : nn.Module,
                       scaler,
                       df_partition: pd.DataFrame,
                       feat_cols   : list,
                       target_cols : list,
                       n_samples   : int = MC_SAMPLES) -> dict:
    """
    Monte Carlo Dropout inference for uncertainty quantification.

    Parameters
    ----------
    model        : trained HydroLSTM or HydroTransformer
    scaler       : fitted StandardScaler
    df_partition : test DataFrame
    feat_cols    : feature column names
    target_cols  : target column names
    n_samples    : number of stochastic forward passes

    Returns
    -------
    dict with keys:
        'mean'    : pd.DataFrame (N, 6) — mean prediction (ft)
        'std'     : pd.DataFrame (N, 6) — standard deviation (ft)
        'lower'   : pd.DataFrame (N, 6) — mean − 2σ  (95% PI lower bound)
        'upper'   : pd.DataFrame (N, 6) — mean + 2σ  (95% PI upper bound)
        'samples' : np.ndarray (n_samples, N, 6) — raw MC samples
    """
    ds = ContinuousSequenceDataset(
        df_partition, feat_cols, target_cols,
        scaler=scaler, fit_scaler=False)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                        worker_init_fn=_worker_init_fn)

    cols = [f"t+{h}min" for h in FORECAST_HORIZONS_MIN]

    # model.train() keeps dropout active — this is the key MC Dropout step
    model.train()
    all_samples = []

    print(f"  Running MC Dropout ({n_samples} samples) ...")
    for s in range(n_samples):
        batch_preds = []
        with torch.no_grad():
            for Xb, _ in loader:
                Xb = Xb.to(DEVICE)
                out = model(Xb)
                stage, _, _, _ = unpack_model_output(out)
                batch_preds.append(stage.cpu().numpy())
        all_samples.append(np.concatenate(batch_preds, axis=0))
        if (s + 1) % 25 == 0:
            print(f"    Completed {s+1}/{n_samples} samples")

    model.eval()

    all_samples = np.stack(all_samples, axis=0)   # (n_samples, N, 6)
    mean_arr    = all_samples.mean(axis=0)
    std_arr     = all_samples.std(axis=0)

    idx = ds.time_index
    return {
        "mean"   : pd.DataFrame(mean_arr,              index=idx, columns=cols),
        "std"    : pd.DataFrame(std_arr,               index=idx, columns=cols),
        "lower"  : pd.DataFrame(mean_arr - 2*std_arr,  index=idx, columns=cols),
        "upper"  : pd.DataFrame(mean_arr + 2*std_arr,  index=idx, columns=cols),
        "samples": all_samples,
    }


def compute_uncertainty_metrics(obs_df   : pd.DataFrame,
                                 unc_dict : dict,
                                 flood_thr: float) -> pd.DataFrame:
    """
    Evaluate the calibration and sharpness of MC Dropout prediction intervals.

    Metrics
    -------
    PICP       : Prediction Interval Coverage Probability — fraction of
                 observations falling within the 95% PI. Target ≈ 95%.
    PICP_flood : PICP restricted to flood events (obs > flood_thr).
                 Most operationally relevant metric.
    PINAW      : Prediction Interval Normalised Average Width —
                 mean(upper−lower) / (obs_max − obs_min). Lower = sharper.

    Returns
    -------
    pd.DataFrame with rows = horizons, columns = [PICP, PICP_flood, PINAW].
    """
    rows = []
    for col in obs_df.columns:
        obs_v   = obs_df[col].values
        lower_v = unc_dict["lower"][col].reindex(obs_df.index).values
        upper_v = unc_dict["upper"][col].reindex(obs_df.index).values

        mask = np.isfinite(obs_v) & np.isfinite(lower_v) & np.isfinite(upper_v)
        obs_v, lower_v, upper_v = obs_v[mask], lower_v[mask], upper_v[mask]

        covered     = (obs_v >= lower_v) & (obs_v <= upper_v)
        picp        = covered.mean() * 100
        flood_mask  = obs_v >= flood_thr
        picp_flood  = covered[flood_mask].mean() * 100 if flood_mask.sum() > 0 else np.nan
        obs_range   = obs_v.max() - obs_v.min()
        pinaw       = (upper_v - lower_v).mean() / (obs_range + 1e-9)
        crps_val    = crps_gaussian(obs_v, unc_dict["mean"][col].reindex(obs_df.index).values[mask],
                                    unc_dict["std"][col].reindex(obs_df.index).values[mask])

        rows.append({
            "Horizon"   : col,
            "PICP"      : round(picp, 2),
            "PICP_flood": round(picp_flood, 2),
            "PINAW"     : round(pinaw, 4),
            "CRPS"      : round(crps_val, 4) if np.isfinite(crps_val) else np.nan,
        })

    return pd.DataFrame(rows).set_index("Horizon")


# =============================================================================
# SECTION 13 — STORM DETECTION
# =============================================================================

def detect_storms(obs_ser     : pd.Series,
                  min_stage_ft: float = SIGNIFICANT_STAGE_FT,
                  percentile  : float = None,
                  min_sep_hr  : float = 18.0) -> pd.DataFrame:
    """
    Identify storm peak events in the observed gage height time series.

    By default uses a fixed significant-stage threshold (9 ft at US-1) rather
    than a percentile, so detected events match operational rare peaks.
    """
    if percentile is not None:
        threshold = float(np.percentile(obs_ser.dropna(), percentile))
        thr_label = f"{percentile}th pct = {threshold:.2f} ft"
    else:
        threshold = float(min_stage_ft)
        thr_label = f">= {threshold:.1f} ft (significant stage)"
    min_sep_steps = int(min_sep_hr * 60 / TIMESTEP_MIN)

    arr        = obs_ser.ffill().values
    peaks, _   = find_peaks(arr, height=threshold, distance=min_sep_steps)

    if len(peaks) == 0:
        print(f"  [WARNING] No storm peaks detected ({thr_label})")
        return pd.DataFrame()

    catalog = pd.DataFrame({
        "peak_time" : obs_ser.index[peaks],
        "peak_value": arr[peaks],
        "peak_idx"  : peaks,
    })
    print(f"  Detected {len(catalog)} storm events (threshold: {thr_label})")
    return catalog


# =============================================================================
# SECTION 14 — VISUALISATION
# All figures are designed to be self-contained and publication-ready:
#   - All metrics are annotated directly on the figure
#   - Units are labelled on all axes
#   - Legends are complete and unambiguous
#   - Train/test regions are clearly delineated
#   - Flood threshold lines are shown where relevant
#   - Figures saved at 300 DPI for journal submission
# =============================================================================

# ── Colour scheme ──────────────────────────────────────────────────────────────
C_OBS  = "#000000"     # observed — black
C_LSTM = "#CC2200"     # LSTM — red
C_TR   = "#0055CC"     # Transformer — blue
C_PERS = "#888888"     # persistence — grey
C_FLD  = "#FF8800"     # flood threshold — orange


def _metric_textbox(ax, metrics: dict, loc: str = "upper left",
                    fontsize: int = 8):
    """Annotate an axes with a metrics text box."""
    lines = [f"{k}: {v}" for k, v in metrics.items()]
    txt   = "\n".join(lines)
    props = dict(boxstyle="round,pad=0.4", facecolor="white",
                 edgecolor="0.6", alpha=0.85)
    x_pos = 0.02 if "left" in loc else 0.98
    y_pos = 0.98 if "upper" in loc else 0.02
    ha    = "left" if "left" in loc else "right"
    va    = "top"  if "upper" in loc else "bottom"
    ax.text(x_pos, y_pos, txt, transform=ax.transAxes,
            fontsize=fontsize, verticalalignment=va,
            horizontalalignment=ha, bbox=props, family="monospace")


# ── Plot 1: Training history ────────────────────────────────────────────────
def plot_training_history(tr_losses  : list,
                          val_losses : list,
                          model_name : str):
    """
    Plot training and validation loss curves per epoch.

    Annotates: best epoch (minimum validation loss), final validation loss,
    and early stopping point if applicable.
    """
    fig, ax = plt.subplots(figsize=(10, 4))

    epochs = np.arange(1, len(tr_losses) + 1)
    ax.plot(epochs, tr_losses,  color=C_LSTM, lw=1.5, label="Training loss")
    ax.plot(epochs, val_losses, color=C_TR,   lw=1.5, label="Validation loss")

    best_ep = int(np.argmin(val_losses)) + 1
    ax.axvline(best_ep, color="green", ls="--", lw=1, alpha=0.7,
               label=f"Best epoch: {best_ep}")
    ax.scatter([best_ep], [val_losses[best_ep - 1]],
               color="green", zorder=5, s=60)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("FloodAwareLoss (ft²)")
    ax.set_title(
        f"{model_name} — Training History\n"
        f"Loss: tail-weighted MSE + pinball(τ=0.95) + peak magnitude + peak timing\n"
        f"Best val loss = {min(val_losses):.5f}  |  "
        f"Best epoch = {best_ep}  |  Total epochs = {len(tr_losses)}"
    )
    ax.legend()
    fig.tight_layout()
    path = OUTPUT_DIR / f"{model_name.lower()}_training_history.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path.name}")


# ── Plot 2: Full record forecast overlay ────────────────────────────────────
def plot_full_record(obs_tr   : pd.Series,
                     pred_tr  : pd.Series,
                     obs_te   : pd.Series,
                     pred_te  : pd.Series,
                     metrics  : dict,
                     flood_thr: float,
                     model_name: str):
    """
    Two-panel figure showing the full 5-year gage height record with
    model predictions overlaid. Train and test partitions are shaded
    differently. Metrics are annotated in a text box.

    Top panel   : Training period (observed vs predicted)
    Bottom panel: Test period    (observed vs predicted)
    """
    fig, axes = plt.subplots(2, 1, figsize=(16, 8), sharex=False)

    for ax, obs, pred, title_sfx, color in zip(
        axes,
        [obs_tr, obs_te],
        [pred_tr, pred_te],
        ["Training partition (70%)", "Test partition (15%)"],
        ["#CCEECC", "#CCE0FF"]
    ):
        ax.fill_between(obs.index, 0, obs.max() * 1.2,
                         color=color, alpha=0.25, label=title_sfx)
        ax.plot(obs.index,  obs,  color=C_OBS,   lw=0.8, label="Observed")
        ax.plot(pred.index, pred, color=C_LSTM if model_name=="LSTM" else C_TR,
                lw=0.8, alpha=0.85, label=f"{model_name} t+150 min")
        ax.axhline(flood_thr, color=C_FLD, ls="--", lw=1,
                   label=f"Significant stage ({flood_thr:.1f} ft)")
        ax.set_ylabel("Gage height (ft)")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.tick_params(axis="x", rotation=25)
        ax.set_title(title_sfx)
        ax.legend(loc="upper right", fontsize=8)

    # metrics is now a plain dict (m_lstm_150 / m_tr_150) — use .get() directly
    _metric_textbox(axes[1], {
        "RMSE (ft)"    : f"{metrics.get('RMSE',      'N/A')}",
        "NSE"          : f"{metrics.get('NSE',        'N/A')}",
        "KGE"          : f"{metrics.get('KGE',        'N/A')}",
        "PBIAS (%)"    : f"{metrics.get('PBIAS',      'N/A')}",
        "Peak-RMSE(ft)": f"{metrics.get('Peak_RMSE',  'N/A')}",
        "Timing (min)" : f"{metrics.get('Timing_min', 'N/A')}",
    })

    fig.suptitle(
        f"Crabtree Creek @ US-1 — {model_name} Full-Record Forecast Overlay\n"
        f"Horizon: t+150 min  |  Lookback: {LOOKBACK_MIN} min  |  "
        f"Features: 47  |  USGS 02087324, Raleigh NC",
        fontsize=11, fontweight="bold"
    )
    fig.tight_layout()
    path = OUTPUT_DIR / f"{model_name.lower()}_full_record.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path.name}")


# ── Plot 3: Random storm peak zoom panels (per model) ───────────────────────
def _to_valid_time(series: pd.Series, horizon_min: int) -> pd.Series:
    """
    Shift an ISSUE-time-indexed forecast series to VALID time for plotting.

    predict_continuous() indexes every horizon column by the forecast ISSUE
    time t; a 't+h' value stored at row t is the prediction for valid time
    t+h. To overlay it on the real (valid-time) observed hydrograph, each
    horizon must be advanced by +h minutes. Without this, longer horizons
    slide left by their lead time and spuriously appear to 'lead' the
    observed peak — a pure plotting artefact, not model skill.

    IMPORTANT: only the storm-panel PLOTS and their per-panel metrics need
    this. The aggregate metric tables (get_obs_dataframe uses targets that
    are gauge_us1.shift(-h), i.e. obs at t+h) already compare obs and pred
    in the same issue-time frame and are correctly aligned — do NOT shift
    those, or you will introduce a 2h misalignment.
    """
    s = series.copy()
    s.index = s.index + pd.Timedelta(minutes=horizon_min)
    return s


def plot_storm_zoom_panels(obs_te    : pd.Series,
                            pred_te   : pd.DataFrame,
                            unc_dict  : dict,
                            storm_cat : pd.DataFrame,
                            flood_thr : float,
                            model_name: str,
                            n_panels  : int = 6):
    """
    Six randomly selected storm panels showing the full multi-horizon
    predicted hydrograph with MC Dropout 95% prediction intervals.

    Each panel displays:
      - Observed gage height (black)
      - Predicted hydrograph at each of 6 horizons (colour gradient)
      - MC Dropout 95% prediction interval for t+150 (shaded)
      - Per-storm metrics: peak error (ft), timing error (min), NSE
      - Flood threshold line
    """
    if storm_cat.empty:
        print(f"  [WARNING] No storms for {model_name} zoom panels.")
        return

    # ── Filter to genuine flood events above the 90th-pct threshold ────────
    # This prevents non-storm baseflow events from appearing in zoom panels.
    flood_storms = storm_cat[storm_cat["peak_value"] >= flood_thr].reset_index(drop=True)
    if flood_storms.empty:
        print(f"  [WARNING] No flood-threshold storms found for {model_name} panels.")
        flood_storms = storm_cat.copy().reset_index(drop=True)

    rng      = np.random.default_rng(GLOBAL_SEED)
    n_avail  = len(flood_storms)
    n_show   = min(n_panels, n_avail)
    sel_idx  = rng.choice(n_avail, size=n_show, replace=False)
    sel_idx  = np.sort(sel_idx)

    n_cols  = 2
    n_rows  = (n_show + 1) // 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4 * n_rows),
                              constrained_layout=True)
    axes = np.array(axes).flatten()

    horizon_colors = plt.cm.RdYlGn_r(np.linspace(0.1, 0.9, OUTPUT_STEPS))

    for panel_idx, storm_row_idx in enumerate(sel_idx):
        ax       = axes[panel_idx]
        storm    = flood_storms.iloc[storm_row_idx]
        peak_t   = storm["peak_time"]
        peak_v   = storm["peak_value"]

        # ±3 hour window around peak
        win    = pd.Timedelta(hours=3)
        t_lo   = peak_t - win
        t_hi   = peak_t + win
        obs_w  = obs_te[t_lo:t_hi]

        ax.plot(obs_w.index, obs_w.values,
                color=C_OBS, lw=2, label="Observed", zorder=5)

        # Plot each forecast horizon with colour gradient
        for j, h in enumerate(FORECAST_HORIZONS_MIN):
            col     = f"t+{h}min"
            if col in pred_te.columns:
                pred_w = _to_valid_time(pred_te[col], h).reindex(obs_w.index)
                ax.plot(obs_w.index, pred_w.values,
                        color=horizon_colors[j], lw=1.2, alpha=0.85,
                        label=f"t+{h}min")

        # MC Dropout 95% PI for t+150min
        pi_col = "t+150min"
        if unc_dict and pi_col in unc_dict["lower"].columns:
            lo = _to_valid_time(unc_dict["lower"][pi_col], 150).reindex(obs_w.index)
            hi = _to_valid_time(unc_dict["upper"][pi_col], 150).reindex(obs_w.index)
            ax.fill_between(obs_w.index, lo, hi,
                             color=C_LSTM if model_name=="LSTM" else C_TR,
                             alpha=0.15, label="95% PI (t+150)")

        ax.axhline(flood_thr, color=C_FLD, ls="--", lw=1, alpha=0.8)
        ax.axvline(peak_t, color="purple", ls=":", lw=1.2, alpha=0.6)

        # Per-storm metrics
        if "t+150min" in pred_te.columns:
            pred_w150 = _to_valid_time(pred_te["t+150min"], 150).reindex(obs_w.index)
            obs_arr   = obs_w.values
            pred_arr  = pred_w150.values
            mask      = np.isfinite(obs_arr) & np.isfinite(pred_arr)
            if mask.sum() > 1:
                obs_pk  = obs_arr.max()
                pred_pk = pred_arr[np.isfinite(pred_arr)].max() if np.any(np.isfinite(pred_arr)) else np.nan
                pk_err  = pred_pk - obs_pk

                obs_peak_idx  = np.nanargmax(obs_arr)
                pred_peak_idx = np.nanargmax(pred_arr)
                t_err = (pred_peak_idx - obs_peak_idx) * TIMESTEP_MIN

                nse_v = nse(obs_arr[mask], pred_arr[mask])

                ax.set_title(
                    f"Storm: {peak_t.strftime('%Y-%m-%d %H:%M')} | "
                    f"Obs peak: {obs_pk:.2f} ft\n"
                    f"Peak err (t+150): {pk_err:+.2f} ft | "
                    f"Timing err: {t_err:+d} min | NSE: {nse_v:.3f}",
                    fontsize=8
                )
            else:
                ax.set_title(f"Storm: {peak_t.strftime('%Y-%m-%d %H:%M')}", fontsize=8)
        else:
            ax.set_title(f"Storm: {peak_t.strftime('%Y-%m-%d %H:%M')}", fontsize=8)

        ax.set_ylabel("Gage height (ft)")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d %H:%M"))
        ax.tick_params(axis="x", rotation=25)
        if panel_idx == 0:
            ax.legend(loc="upper left", fontsize=7, ncol=2)

    # Hide unused panels
    for k in range(n_show, len(axes)):
        axes[k].set_visible(False)

    fig.suptitle(
        f"{model_name} — Random Storm Peak Zoom Panels (n={n_show})\n"
        f"Shading: MC Dropout 95% Prediction Interval at t+150 min | "
        f"Significant stage: {flood_thr:.1f} ft | "
        f"Lookback: {LOOKBACK_MIN} min | USGS 02087324",
        fontsize=10, fontweight="bold"
    )

    path = OUTPUT_DIR / f"{model_name.lower()}_storm_zoom_panels.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path.name}")


# ── Plot 4: Feature importance ──────────────────────────────────────────────
def plot_gradient_saliency(model      : nn.Module,
                           scaler,
                           df_tr      : pd.DataFrame,
                           feat_cols  : list,
                           target_cols: list):
    """
    Compute and plot gradient × input saliency for HydroLSTM.

    Saliency scores are averaged over many batches, providing an estimate
    of which features most influence predictions during high-flow conditions.
    Features are ranked and colour-coded by group.

    Implementation note
    -------------------
    cuDNN RNN backward requires training mode AND cuDNN must be temporarily
    disabled for gradient computation through LSTM layers on CUDA devices.
    We use torch.backends.cudnn.enabled = False during the backward pass
    to avoid the "cudnn RNN backward can only be called in training mode"
    RuntimeError, then restore the original state afterwards.
    """
    ds     = ContinuousSequenceDataset(df_tr, feat_cols, target_cols,
                                        scaler=scaler, fit_scaler=False)
    # Use a small subset for efficiency — 200 batches is sufficient
    loader = DataLoader(ds, batch_size=32, shuffle=True,
                        worker_init_fn=_worker_init_fn)

    # Save original cuDNN state and disable during gradient computation
    cudnn_was_enabled = torch.backends.cudnn.enabled
    torch.backends.cudnn.enabled = False

    # model.train() is REQUIRED — cuDNN LSTM backward fails in eval mode
    model.train()
    saliency_accum = np.zeros(len(feat_cols))
    count          = 0
    max_batches    = 200   # limit for speed

    for batch_idx, (Xb, yb) in enumerate(loader):
        if batch_idx >= max_batches:
            break

        Xb = Xb.to(DEVICE)
        Xb.requires_grad_(True)

        # Zero any existing gradients on the input tensor
        if Xb.grad is not None:
            Xb.grad.zero_()

        out  = model(Xb)
        stage, _, _, _ = unpack_model_output(out)
        # Saliency w.r.t. t+150 min prediction (last horizon, index -1)
        loss = stage[:, -1].sum()
        loss.backward()

        if Xb.grad is not None:
            sal = (Xb.grad.detach() * Xb.detach()).abs().mean(dim=(0, 1))
            saliency_accum += sal.cpu().numpy()
            count          += 1

    # Restore original cuDNN state and eval mode
    torch.backends.cudnn.enabled = cudnn_was_enabled
    model.eval()

    saliency = saliency_accum / max(count, 1)

    # Rank features
    order      = np.argsort(saliency)[::-1][:20]
    top_names  = [feat_cols[i] for i in order]
    top_scores = saliency[order]

    # Colour by feature group
    group_colors = {}
    for name in top_names:
        if "precip" in name and "API" not in name:
            group_colors[name] = "#1F77B4"
        elif "API" in name:
            group_colors[name] = "#2CA02C"
        elif "baseflow" in name or "quickflow" in name:
            group_colors[name] = "#8C564B"
        elif "gauge" in name:
            group_colors[name] = "#FF7F0E"
        elif "q_" in name:
            group_colors[name] = "#D62728"
        elif "hour" in name or "doy" in name:
            group_colors[name] = "#9467BD"
        else:
            group_colors[name] = "#7F7F7F"

    colors = [group_colors[n] for n in top_names]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(range(len(top_names)), top_scores[::-1], color=colors[::-1])
    ax.set_yticks(range(len(top_names)))
    ax.set_yticklabels(top_names[::-1], fontsize=9)
    ax.set_xlabel("Mean |gradient × input| saliency score")
    ax.set_title(
        "LSTM Gradient Saliency — Top 20 Features\n"
        "Computed on training partition | Averaged over all flood events\n"
        "Higher score = greater influence on t+150 min prediction"
    )

    legend_elements = [
        mpatches.Patch(color="#1F77B4", label="Raw precipitation"),
        mpatches.Patch(color="#2CA02C", label="API (soil moisture proxy)"),
        mpatches.Patch(color="#8C564B", label="Baseflow / quickflow"),
        mpatches.Patch(color="#FF7F0E", label="Upstream gage heights"),
        mpatches.Patch(color="#D62728", label="Discharge"),
        mpatches.Patch(color="#9467BD", label="Time encodings"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=8)
    fig.tight_layout()
    path = OUTPUT_DIR / "lstm_gradient_saliency.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path.name}")


# ── Plot 5: Attention rollout (Transformer) ──────────────────────────────────
def plot_attention_rollout(model      : nn.Module,
                           scaler,
                           df_tr      : pd.DataFrame,
                           feat_cols  : list,
                           target_cols: list):
    """
    Compute and visualise layer-averaged attention rollout for HydroTransformer.

    Attention rollout (Abnar & Zuidema, 2020) propagates attention weights
    recursively through all encoder layers, producing an estimate of each
    input timestep's effective contribution to the final prediction.
    """
    ds     = ContinuousSequenceDataset(df_tr, feat_cols, target_cols,
                                        scaler=scaler, fit_scaler=False)
    loader = DataLoader(ds, batch_size=1, shuffle=False)

    model.eval()
    attention_matrices = []

    hooks = []
    attn_store = []

    def _hook(module, input, output):
        # output[1] is the attention weights if need_weights=True
        if isinstance(output, tuple) and len(output) > 1 and output[1] is not None:
            attn_store.append(output[1].detach().cpu())

    for layer in model.encoder.layers:
        hooks.append(layer.self_attn.register_forward_hook(_hook))

    n_collected   = 0
    rollout_accum = None

    # Monkey-patch forward to force need_weights=True so attention weights
    # are actually returned (PyTorch 2.0+ defaults need_weights=False in
    # TransformerEncoderLayer when using fast path).
    original_forwards = {}
    for layer in model.encoder.layers:
        orig = layer.self_attn.forward
        original_forwards[layer] = orig
        def _patched_forward(query, key, value, orig=orig, **kwargs):
            kwargs["need_weights"] = True
            kwargs["average_attn_weights"] = False
            return orig(query, key, value, **kwargs)
        layer.self_attn.forward = _patched_forward

    with torch.no_grad():
        for Xb, _ in loader:
            if n_collected >= 50:
                break
            attn_store.clear()
            _ = model(Xb.to(DEVICE))

            if not attn_store:
                # Attention weights not captured — skip gracefully
                break

            # Build rollout: A_roll = A_L × ... × A_1
            A = attn_store[0][0].mean(0)   # average over heads
            I = torch.eye(A.shape[0])
            A = 0.5 * A + 0.5 * I         # add residual connection

            for layer_attn in attn_store[1:]:
                A_l   = layer_attn[0].mean(0)
                A_l   = 0.5 * A_l + 0.5 * I
                A     = A_l @ A

            rollout = A[-1].numpy()        # attention from last token to all

            if rollout_accum is None:
                rollout_accum = rollout
            else:
                rollout_accum += rollout

            n_collected += 1

    # Restore original self_attn forward methods
    for layer in model.encoder.layers:
        if layer in original_forwards:
            layer.self_attn.forward = original_forwards[layer]

    for h in hooks:
        h.remove()

    if rollout_accum is None or n_collected == 0:
        print("  [WARNING] Attention rollout: no data collected — skipping figure.")
        return

    rollout_avg = rollout_accum / n_collected
    time_axis   = np.arange(-LOOKBACK_MIN, 0, TIMESTEP_MIN)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.fill_between(time_axis, rollout_avg, alpha=0.6, color=C_TR)
    ax.plot(time_axis, rollout_avg, color=C_TR, lw=1.5)
    ax.set_xlabel("Time relative to forecast issue time (minutes)")
    ax.set_ylabel("Normalised attention weight")
    ax.set_title(
        "Transformer Attention Rollout — Temporal Attribution\n"
        "Average over 50 training samples | "
        "Higher weight = greater temporal influence on prediction\n"
        "Abnar & Zuidema (2020) rollout method"
    )
    ax.axvline(-30,  color="red",    ls="--", lw=1, alpha=0.7, label="t−30 min")
    ax.axvline(-60,  color="orange", ls="--", lw=1, alpha=0.7, label="t−60 min")
    ax.axvline(-120, color="green",  ls="--", lw=1, alpha=0.7, label="t−120 min")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = OUTPUT_DIR / "transformer_attention_rollout.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path.name}")


# ── Plot 6: Monthly RMSE (per model) ────────────────────────────────────────
def plot_monthly_rmse(obs_te    : pd.Series,
                      pred_te   : pd.Series,
                      flood_thr : float,
                      model_name: str):
    """
    Monthly RMSE bar chart for t+150 min predictions.

    Highlights months where RMSE exceeds the overall mean, helping
    identify seasonality in forecast skill.
    """
    combined = pd.DataFrame({
        "obs" : obs_te,
        "pred": pred_te,
    }).dropna()
    combined["month"] = combined.index.month

    monthly = combined.groupby("month").apply(
        lambda g: np.sqrt(mean_squared_error(g["obs"], g["pred"]))
    ).reset_index()
    monthly.columns = ["month", "RMSE"]

    overall_rmse = float(np.sqrt(mean_squared_error(
        combined["obs"], combined["pred"])))

    month_names = ["Jan","Feb","Mar","Apr","May","Jun",
                   "Jul","Aug","Sep","Oct","Nov","Dec"]

    fig, ax = plt.subplots(figsize=(10, 4))
    colors = [C_LSTM if model_name=="LSTM" else C_TR
              if r > overall_rmse else "#AACCFF"
              for r in monthly["RMSE"]]

    bars = ax.bar(monthly["month"], monthly["RMSE"], color=colors, edgecolor="0.4")
    ax.axhline(overall_rmse, color="black", ls="--", lw=1.5,
               label=f"Overall RMSE = {overall_rmse:.4f} ft")

    for bar, val in zip(bars, monthly["RMSE"]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                f"{val:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(monthly["month"])
    ax.set_xticklabels([month_names[m-1] for m in monthly["month"]])
    ax.set_xlabel("Month")
    ax.set_ylabel("RMSE (ft)")
    ax.set_title(
        f"{model_name} — Monthly RMSE at t+150 min\n"
        f"Darker bars = above-average error months | "
        f"Overall RMSE = {overall_rmse:.4f} ft | USGS 02087324"
    )
    ax.legend()
    fig.tight_layout()
    path = OUTPUT_DIR / f"{model_name.lower()}_monthly_rmse.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path.name}")


# ── Plot 7: Head-to-head comparison overlay ─────────────────────────────────
def plot_comparison_overlay(obs_te   : pd.Series,
                             pred_lstm: pd.Series,
                             pred_tr  : pd.Series,
                             pred_pers: pd.Series,
                             m_lstm   : dict,
                             m_tr     : dict,
                             m_pers   : dict,
                             flood_thr: float):
    """
    Three-panel comparison: forecast overlay + residuals + flood-event zoom.

    Panel 1: Observed vs LSTM vs Transformer vs Persistence (full test period)
    Panel 2: Residuals for LSTM and Transformer
    Panel 3: Flood-event-only period (obs > flood_thr)
    """
    fig = plt.figure(figsize=(16, 10))
    gs  = gridspec.GridSpec(3, 1, height_ratios=[3, 1.5, 2], hspace=0.4)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax3 = fig.add_subplot(gs[2])

    # ── Panel 1: Overlay ──────────────────────────────────────────────────
    ax1.plot(obs_te.index,   obs_te,   color=C_OBS,  lw=1.5, label="Observed", zorder=5)
    ax1.plot(pred_lstm.index,pred_lstm, color=C_LSTM, lw=1.0, alpha=0.85, label="LSTM")
    ax1.plot(pred_tr.index,  pred_tr,   color=C_TR,   lw=1.0, alpha=0.85, label="Transformer")
    ax1.plot(pred_pers.index,pred_pers, color=C_PERS, lw=0.7, alpha=0.6,  label="Persistence")
    ax1.axhline(flood_thr, color=C_FLD, ls="--", lw=1.2,
                label=f"Significant stage ({flood_thr:.1f} ft)")
    ax1.set_ylabel("Gage height (ft)")
    ax1.set_title(
        "Crabtree Creek @ US-1 — Head-to-Head Forecast Comparison (Test Period)\n"
        "Forecast horizon: t+150 min | Lookback: 480 min | Features: 47 | USGS 02087324"
    )
    ax1.legend(loc="upper right", fontsize=8, ncol=4)

    # ── Panel 2: Residuals ────────────────────────────────────────────────
    res_lstm = (obs_te - pred_lstm).reindex(obs_te.index)
    res_tr   = (obs_te - pred_tr).reindex(obs_te.index)
    ax2.axhline(0, color="black", lw=0.8)
    ax2.plot(obs_te.index, res_lstm, color=C_LSTM, lw=0.7, alpha=0.7, label="LSTM residual")
    ax2.plot(obs_te.index, res_tr,   color=C_TR,   lw=0.7, alpha=0.7, label="Transformer residual")
    ax2.fill_between(obs_te.index, res_lstm, 0, color=C_LSTM, alpha=0.1)
    ax2.fill_between(obs_te.index, res_tr,   0, color=C_TR,   alpha=0.1)
    ax2.set_ylabel("Residual (ft)\n(obs − pred)")
    ax2.legend(fontsize=8)

    for ax in [ax1, ax2]:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        ax.tick_params(axis="x", rotation=25)

    # ── Panel 3: Metrics box ──────────────────────────────────────────────
    ax3.axis("off")
    table_data = [
        ["Metric",    "LSTM",
         f"{m_lstm.get('RMSE','—')}",
         "Transformer",
         f"{m_tr.get('RMSE','—')}",
         "Persistence",
         f"{m_pers.get('RMSE','—')}"],
    ]
    col_labels = ["", "LSTM RMSE","","Transformer RMSE","","Persistence RMSE",""]
    metrics_text = (
        f"╔══════════════════════════════════════════════════════════════╗\n"
        f"║  METRIC         LSTM          TRANSFORMER    PERSISTENCE    ║\n"
        f"╠══════════════════════════════════════════════════════════════╣\n"
        f"║  RMSE (ft)    {m_lstm.get('RMSE','—'):>8}    {m_tr.get('RMSE','—'):>12}    {m_pers.get('RMSE','—'):>10}  ║\n"
        f"║  MAE  (ft)    {m_lstm.get('MAE','—'):>8}    {m_tr.get('MAE','—'):>12}    {m_pers.get('MAE','—'):>10}  ║\n"
        f"║  NSE          {m_lstm.get('NSE','—'):>8}    {m_tr.get('NSE','—'):>12}    {m_pers.get('NSE','—'):>10}  ║\n"
        f"║  KGE          {m_lstm.get('KGE','—'):>8}    {m_tr.get('KGE','—'):>12}    {m_pers.get('KGE','—'):>10}  ║\n"
        f"║  PBIAS (%)    {m_lstm.get('PBIAS','—'):>8}    {m_tr.get('PBIAS','—'):>12}    {m_pers.get('PBIAS','—'):>10}  ║\n"
        f"║  Peak-RMSE    {m_lstm.get('Peak_RMSE','—'):>8}    {m_tr.get('Peak_RMSE','—'):>12}    {m_pers.get('Peak_RMSE','—'):>10}  ║\n"
        f"║  Timing(min)  {m_lstm.get('Timing_min','—'):>8}    {m_tr.get('Timing_min','—'):>12}    {'N/A':>10}  ║\n"
        f"╚══════════════════════════════════════════════════════════════╝"
    )
    ax3.text(0.02, 0.98, metrics_text, transform=ax3.transAxes,
             fontsize=8.5, verticalalignment="top",
             fontfamily="monospace",
             bbox=dict(boxstyle="round", facecolor="white", edgecolor="0.6"))

    path = OUTPUT_DIR / "comparison_forecast_overlay.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path.name}")


# ── Plot 8: Observed vs Predicted scatter ────────────────────────────────────
def plot_scatter(obs_te   : pd.Series,
                 pred_lstm: pd.Series,
                 pred_tr  : pd.Series,
                 flood_thr: float,
                 m_lstm   : dict,
                 m_tr     : dict):
    """
    Side-by-side scatter plots (observed vs predicted) for LSTM and Transformer.

    Points are colour-coded by season. Flood events (obs > threshold) are
    highlighted with distinct markers. The 1:1 line and OLS regression line
    are overlaid. R², RMSE, and NSE are annotated.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    seasons = {
        "Winter (DJF)": ([12, 1, 2],  "#3388FF"),
        "Spring (MAM)": ([3, 4, 5],   "#33AA33"),
        "Summer (JJA)": ([6, 7, 8],   "#FF5533"),
        "Autumn (SON)": ([9, 10, 11], "#CC8822"),
    }

    for ax, pred, name, m in zip(
        axes, [pred_lstm, pred_tr],
        ["LSTM", "Transformer"], [m_lstm, m_tr]
    ):
        combined = pd.DataFrame({"obs": obs_te, "pred": pred}).dropna()
        combined["month"] = combined.index.month

        for season_name, (months, color) in seasons.items():
            mask = combined["month"].isin(months)
            sub  = combined[mask]
            if sub.empty:
                continue
            flood_mask = sub["obs"] >= flood_thr
            ax.scatter(sub.loc[~flood_mask, "obs"],
                       sub.loc[~flood_mask, "pred"],
                       c=color, s=3, alpha=0.3, label=season_name)
            if flood_mask.any():
                ax.scatter(sub.loc[flood_mask, "obs"],
                           sub.loc[flood_mask, "pred"],
                           c=color, s=20, alpha=0.8, marker="^",
                           edgecolors="black", linewidths=0.5)

        lim = [combined["obs"].min() * 0.95,
               max(combined["obs"].max(), combined["pred"].max()) * 1.05]
        ax.plot(lim, lim, "k--", lw=1.2, label="1:1 line")

        # OLS regression line
        coeffs = np.polyfit(combined["obs"], combined["pred"], 1)
        x_fit  = np.array(lim)
        ax.plot(x_fit, np.polyval(coeffs, x_fit), "r-", lw=1,
                alpha=0.7, label=f"OLS (slope={coeffs[0]:.3f})")

        ax.set_xlim(lim); ax.set_ylim(lim)
        ax.set_xlabel("Observed gage height (ft)")
        ax.set_ylabel("Predicted gage height (ft)")
        ax.set_title(
            f"{name} — Observed vs Predicted (t+150 min)\n"
            f"▲ = significant events (obs >= {flood_thr:.1f} ft)"
        )

        _metric_textbox(ax, {
            "RMSE (ft)"   : f"{m.get('RMSE','—')}",
            "NSE"         : f"{m.get('NSE','—')}",
            "KGE"         : f"{m.get('KGE','—')}",
            "PBIAS (%)"   : f"{m.get('PBIAS','—')}",
            "Peak-RMSE"   : f"{m.get('Peak_RMSE','—')}",
        }, loc="upper left")

        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles[:4], labels[:4], fontsize=7, loc="lower right",
                  title="Season")

    fig.suptitle(
        "Crabtree Creek @ US-1 — Observed vs Predicted Scatter\n"
        "Colour: season | Triangle markers: flood events | USGS 02087324",
        fontsize=11, fontweight="bold"
    )
    fig.tight_layout()
    path = OUTPUT_DIR / "comparison_scatter.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path.name}")


# ── Plot 9: Multi-metric bar chart ───────────────────────────────────────────
def plot_metrics_bar(m_lstm : dict, m_tr : dict, m_pers : dict):
    """
    Side-by-side bar chart comparing all evaluation metrics across LSTM,
    Transformer, and persistence baseline.

    Metric values are annotated on bars. Higher-is-better metrics (NSE, KGE)
    are clearly distinguished from lower-is-better metrics (RMSE, MAE, PBIAS).
    """
    metrics_to_plot = ["RMSE", "MAE", "NSE", "KGE", "Peak_RMSE"]
    labels          = ["RMSE\n(ft, ↓)", "MAE\n(ft, ↓)",
                       "NSE\n(↑)", "KGE\n(↑)", "Peak-RMSE\n(ft, ↓)"]
    n = len(metrics_to_plot)
    x = np.arange(n)
    w = 0.25

    fig, ax = plt.subplots(figsize=(12, 5))

    for i, (vals, name, color) in enumerate(zip(
        [m_lstm, m_tr, m_pers],
        ["LSTM", "Transformer", "Persistence"],
        [C_LSTM, C_TR, C_PERS]
    )):
        heights = [vals.get(m, np.nan) for m in metrics_to_plot]
        bars    = ax.bar(x + i * w, heights, w, label=name, color=color,
                         alpha=0.85, edgecolor="0.3")
        for bar, h in zip(bars, heights):
            if np.isfinite(h):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.002,
                        f"{h:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x + w)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Metric value")
    ax.set_title(
        "Crabtree Creek @ US-1 — Model Performance Comparison (Test Period, t+150 min)\n"
        "↑ = higher is better | ↓ = lower is better | Persistence = naïve baseline"
    )
    ax.legend()
    ax.axhline(0, color="black", lw=0.5)
    fig.tight_layout()
    path = OUTPUT_DIR / "comparison_metrics_bar.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path.name}")


# ── Plot 10: Flow duration curve ─────────────────────────────────────────────
def plot_flow_duration_curve(obs_te   : pd.Series,
                              pred_lstm: pd.Series,
                              pred_tr  : pd.Series,
                              flood_thr: float):
    """
    Flow duration curve (FDC) showing exceedance probability vs gage height.

    An inset panel zooms into the upper 10% exceedance region (flood tail),
    where accurate reproduction of the FDC is most operationally critical.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    for data, label, color, lw in zip(
        [obs_te, pred_lstm, pred_tr],
        ["Observed", "LSTM", "Transformer"],
        [C_OBS, C_LSTM, C_TR],
        [2.0, 1.5, 1.5]
    ):
        sorted_data = np.sort(data.dropna().values)[::-1]
        exceedance  = np.arange(1, len(sorted_data) + 1) / len(sorted_data) * 100
        ax.plot(exceedance, sorted_data, color=color, lw=lw, label=label)

    ax.axhline(flood_thr, color=C_FLD, ls="--", lw=1.2,
               label=f"Significant stage ({flood_thr:.1f} ft)")
    ax.set_xlabel("Exceedance probability (%)")
    ax.set_ylabel("Gage height (ft)")
    ax.set_title(
        "Crabtree Creek @ US-1 — Flow Duration Curve\n"
        "Test period | t+150 min predictions | Inset: upper tail (flood regime)"
    )
    ax.legend()
    ax.set_xlim([0, 100])

    # Inset: upper tail
    ax_ins = ax.inset_axes([0.55, 0.35, 0.42, 0.55])
    for data, label, color, lw in zip(
        [obs_te, pred_lstm, pred_tr],
        ["Observed", "LSTM", "Transformer"],
        [C_OBS, C_LSTM, C_TR],
        [2.0, 1.5, 1.5]
    ):
        sorted_data = np.sort(data.dropna().values)[::-1]
        exceedance  = np.arange(1, len(sorted_data) + 1) / len(sorted_data) * 100
        mask        = exceedance <= 10
        ax_ins.plot(exceedance[mask], sorted_data[mask], color=color, lw=lw)

    ax_ins.axhline(flood_thr, color=C_FLD, ls="--", lw=1)
    ax_ins.set_title("Upper tail (0–10%)", fontsize=8)
    ax_ins.set_xlabel("Exceedance (%)", fontsize=8)
    ax_ins.set_ylabel("Height (ft)", fontsize=8)

    fig.tight_layout()
    path = OUTPUT_DIR / "comparison_flow_duration_curve.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path.name}")


# ── Plot 11: Peak timing error histogram ─────────────────────────────────────
def plot_timing_error_histogram(obs_te   : pd.Series,
                                 pred_lstm: pd.Series,
                                 pred_tr  : pd.Series,
                                 flood_thr: float):
    """
    Distribution of peak timing errors (minutes) for LSTM and Transformer.

    Timing error = predicted peak time − observed peak time.
    Positive values indicate the model predicts the peak too late.
    A well-calibrated model should have a distribution centred near zero.
    """
    def get_timing_errors(obs, pred):
        peaks, _ = find_peaks(obs.fillna(0).values,
                               height=flood_thr, distance=216)
        errors   = []
        window   = 36
        pred_arr = pred.reindex(obs.index).values
        for pk in peaks:
            lo = max(0, pk - window)
            hi = min(len(pred_arr) - 1, pk + window)
            if np.any(np.isfinite(pred_arr[lo:hi+1])):
                pred_pk = lo + np.nanargmax(pred_arr[lo:hi+1])
                errors.append((pred_pk - pk) * TIMESTEP_MIN)
        return errors

    err_lstm = get_timing_errors(obs_te, pred_lstm)
    err_tr   = get_timing_errors(obs_te, pred_tr)

    fig, ax = plt.subplots(figsize=(10, 5))

    if err_lstm:
        ax.hist(err_lstm, bins=20, color=C_LSTM, alpha=0.6, label="LSTM",
                edgecolor="white")
        ax.axvline(np.median(err_lstm), color=C_LSTM, ls="--", lw=2,
                   label=f"LSTM median: {np.median(err_lstm):+.0f} min")

    if err_tr:
        ax.hist(err_tr, bins=20, color=C_TR, alpha=0.6, label="Transformer",
                edgecolor="white")
        ax.axvline(np.median(err_tr), color=C_TR, ls="--", lw=2,
                   label=f"Transformer median: {np.median(err_tr):+.0f} min")

    ax.axvline(0, color="black", lw=1.5, label="Zero error")
    ax.set_xlabel("Peak timing error (minutes)\n"
                  "Positive = predicted peak arrives LATER than observed")
    ax.set_ylabel("Count")
    ax.set_title(
        "Crabtree Creek @ US-1 — Peak Timing Error Distribution\n"
        f"Test period | t+150 min | Storm peaks >= {SIGNIFICANT_STAGE_FT:.0f} ft"
    )
    ax.legend()
    fig.tight_layout()
    path = OUTPUT_DIR / "comparison_timing_error_histogram.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path.name}")


# ── Plot 12: Storm comparison panels (both models) ───────────────────────────
def plot_comparison_storm_panels(obs_te    : pd.Series,
                                  pred_lstm : pd.DataFrame,
                                  pred_tr   : pd.DataFrame,
                                  unc_lstm  : dict,
                                  unc_tr    : dict,
                                  storm_cat : pd.DataFrame,
                                  flood_thr : float,
                                  n_panels  : int = 6):
    """
    Side-by-side storm panels (LSTM left, Transformer right) for the same
    randomly selected storm events.

    Each panel pair shows:
      - Observed hydrograph
      - Predicted hydrograph at all 6 horizons
      - 95% MC Dropout prediction interval (t+150)
      - Per-storm metrics annotated for both models
    """
    if storm_cat.empty:
        print("  [WARNING] No storms for comparison panels.")
        return

    # Filter to genuine flood events above threshold
    flood_storms_comp = storm_cat[storm_cat["peak_value"] >= flood_thr].reset_index(drop=True)
    if flood_storms_comp.empty:
        flood_storms_comp = storm_cat.copy().reset_index(drop=True)

    rng     = np.random.default_rng(GLOBAL_SEED + 1)
    n_avail = len(flood_storms_comp)
    n_show  = min(n_panels, n_avail)
    sel_idx = np.sort(rng.choice(n_avail, size=n_show, replace=False))

    fig, axes = plt.subplots(n_show, 2, figsize=(16, 4 * n_show),
                              constrained_layout=True)
    if n_show == 1:
        axes = axes[np.newaxis, :]

    horizon_colors = plt.cm.RdYlGn_r(np.linspace(0.1, 0.9, OUTPUT_STEPS))
    win            = pd.Timedelta(hours=3)

    for row, storm_row_idx in enumerate(sel_idx):
        storm  = flood_storms_comp.iloc[storm_row_idx]
        peak_t = storm["peak_time"]
        peak_v = storm["peak_value"]
        t_lo   = peak_t - win
        t_hi   = peak_t + win
        obs_w  = obs_te[t_lo:t_hi]

        for col_idx, (pred_df, unc, name, color) in enumerate(zip(
            [pred_lstm, pred_tr],
            [unc_lstm,  unc_tr],
            ["LSTM", "Transformer"],
            [C_LSTM, C_TR]
        )):
            ax = axes[row, col_idx]
            ax.plot(obs_w.index, obs_w.values,
                    color=C_OBS, lw=2, label="Observed", zorder=5)

            for j, h in enumerate(FORECAST_HORIZONS_MIN):
                col_name = f"t+{h}min"
                if col_name in pred_df.columns:
                    pw = _to_valid_time(pred_df[col_name], h).reindex(obs_w.index)
                    ax.plot(obs_w.index, pw.values,
                            color=horizon_colors[j], lw=1.2, alpha=0.9,
                            label=f"t+{h}min")

            pi_col = "t+150min"
            if unc and pi_col in unc["lower"].columns:
                lo = _to_valid_time(unc["lower"][pi_col], 150).reindex(obs_w.index)
                hi = _to_valid_time(unc["upper"][pi_col], 150).reindex(obs_w.index)
                ax.fill_between(obs_w.index, lo, hi, color=color,
                                alpha=0.15, label="95% PI")

            ax.axhline(flood_thr, color=C_FLD, ls="--", lw=1, alpha=0.7)
            ax.axvline(peak_t,    color="purple", ls=":", lw=1.2)

            # Per-storm metrics
            if pi_col in pred_df.columns:
                pred_150 = _to_valid_time(pred_df[pi_col], 150).reindex(obs_w.index).values
                obs_arr  = obs_w.values
                mask     = np.isfinite(obs_arr) & np.isfinite(pred_150)
                if mask.sum() > 1:
                    obs_pk   = obs_arr.max()
                    pred_pk  = pred_150[mask].max()
                    pk_err   = pred_pk - obs_pk
                    nse_v    = nse(obs_arr[mask], pred_150[mask])
                    t_err    = (np.nanargmax(pred_150) - np.argmax(obs_arr)) * TIMESTEP_MIN
                    ax.set_title(
                        f"{name} | Storm: {peak_t.strftime('%Y-%m-%d %H:%M')}\n"
                        f"Obs peak: {obs_pk:.2f} ft | "
                        f"Pred: {pred_pk:.2f} ft ({pk_err:+.2f}) | "
                        f"Timing: {t_err:+d} min | NSE: {nse_v:.3f}",
                        fontsize=8
                    )

            ax.set_ylabel("Gage height (ft)")
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d %H:%M"))
            ax.tick_params(axis="x", rotation=25)

            if row == 0 and col_idx == 0:
                ax.legend(loc="upper left", fontsize=7, ncol=2)

    fig.suptitle(
        "Crabtree Creek @ US-1 — Storm-by-Storm Comparison: LSTM vs Transformer\n"
        "Shading: MC Dropout 95% Prediction Interval at t+150 min | "
        f"Significant stage: {flood_thr:.1f} ft | USGS 02087324, Raleigh NC",
        fontsize=10, fontweight="bold"
    )
    path = OUTPUT_DIR / "comparison_storm_panels.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path.name}")


# ── Plot 13: Multi-horizon skill degradation ─────────────────────────────────
def plot_horizon_skill(metrics_lstm : pd.DataFrame,
                       metrics_tr   : pd.DataFrame,
                       metrics_pers : pd.DataFrame):
    """
    Four-panel figure showing how NSE, RMSE, Peak-RMSE, and timing error
    degrade as forecast lead time increases from t+30 to t+180 min.

    This is the primary diagnostic figure for multi-horizon evaluation.
    """
    h_labels = [f"t+{h}" for h in FORECAST_HORIZONS_MIN]
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))

    plot_specs = [
        ("NSE",       "NSE (↑, perfect=1)",           True),
        ("RMSE",      "RMSE (ft, ↓)",                 False),
        ("Peak_RMSE", "Peak-RMSE (ft, ↓, flood only)",False),
        ("Timing_min","Median peak timing error (min)",False),
    ]

    for ax, (metric, ylabel, higher_better) in zip(axes.flatten(), plot_specs):
        for df_m, name, color, marker in zip(
            [metrics_lstm, metrics_tr, metrics_pers],
            ["LSTM", "Transformer", "Persistence"],
            [C_LSTM, C_TR, C_PERS],
            ["o", "s", "^"]
        ):
            if metric in df_m.columns:
                vals = df_m[metric].values
                ax.plot(h_labels, vals, color=color, lw=2,
                        marker=marker, ms=7, label=name)
                for i, v in enumerate(vals):
                    if np.isfinite(v):
                        ax.annotate(f"{v:.3f}",
                                    (h_labels[i], v),
                                    textcoords="offset points",
                                    xytext=(0, 8), ha="center", fontsize=7)

        ax.set_ylabel(ylabel)
        ax.set_xlabel("Forecast horizon")
        ax.set_title(f"{ylabel}\nvs forecast lead time")
        ax.legend(fontsize=8)
        ax.tick_params(axis="x", rotation=20)

        if higher_better:
            ax.axhline(1.0, color="green", ls=":", lw=0.8, alpha=0.5)
        ax.axhline(0, color="black", lw=0.5)

    fig.suptitle(
        "Crabtree Creek @ US-1 — Multi-Horizon Forecast Skill Degradation\n"
        "LSTM vs Transformer vs Persistence | t+30 to t+180 min | USGS 02087324",
        fontsize=11, fontweight="bold"
    )
    fig.tight_layout()
    path = OUTPUT_DIR / "comparison_horizon_skill.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path.name}")


# ── Plot 14: Uncertainty calibration ─────────────────────────────────────────
def plot_uncertainty_calibration(unc_lstm  : dict,
                                  unc_tr    : dict,
                                  obs_te    : pd.DataFrame,
                                  flood_thr : float,
                                  unc_m_lstm: pd.DataFrame,
                                  unc_m_tr  : pd.DataFrame):
    """
    Two-panel uncertainty calibration figure:

    Panel 1: PICP per horizon for LSTM and Transformer.
             The dashed line at 95% is the target for a well-calibrated model.
    Panel 2: PINAW (interval width) per horizon.
             Narrower intervals indicate sharper, more confident predictions.

    Flood-event PICP is overlaid as separate markers.
    """
    h_labels = [f"t+{h}" for h in FORECAST_HORIZONS_MIN]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel 1: PICP
    ax = axes[0]
    for df_u, name, color in zip(
        [unc_m_lstm, unc_m_tr], ["LSTM", "Transformer"], [C_LSTM, C_TR]
    ):
        if "PICP" in df_u.columns:
            ax.plot(h_labels, df_u["PICP"].values, color=color,
                    lw=2, marker="o", ms=7, label=f"{name} overall")
        if "PICP_flood" in df_u.columns:
            ax.plot(h_labels, df_u["PICP_flood"].values, color=color,
                    lw=2, marker="^", ms=7, ls="--",
                    label=f"{name} flood events only")

    ax.axhline(95, color="green", ls=":", lw=1.5,
               label="Target: 95% coverage")
    ax.set_ylim([50, 105])
    ax.set_ylabel("PICP — Prediction Interval Coverage Probability (%)")
    ax.set_xlabel("Forecast horizon")
    ax.set_title(
        "Uncertainty Calibration — PICP\n"
        "Target: ~95% of observations inside the 95% PI"
    )
    ax.legend(fontsize=8)
    ax.tick_params(axis="x", rotation=20)

    # Panel 2: PINAW
    ax = axes[1]
    for df_u, name, color in zip(
        [unc_m_lstm, unc_m_tr], ["LSTM", "Transformer"], [C_LSTM, C_TR]
    ):
        if "PINAW" in df_u.columns:
            ax.plot(h_labels, df_u["PINAW"].values, color=color,
                    lw=2, marker="o", ms=7, label=name)

    ax.set_ylabel("PINAW — Normalised Average Width (lower = sharper)")
    ax.set_xlabel("Forecast horizon")
    ax.set_title(
        "Uncertainty Sharpness — PINAW\n"
        "Lower values indicate narrower, more confident prediction intervals"
    )
    ax.legend(fontsize=8)
    ax.tick_params(axis="x", rotation=20)

    fig.suptitle(
        "Crabtree Creek @ US-1 — MC Dropout Uncertainty Quantification\n"
        f"100 stochastic samples | 95% PI = mean ± 2σ | USGS 02087324",
        fontsize=11, fontweight="bold"
    )
    fig.tight_layout()
    path = OUTPUT_DIR / "comparison_uncertainty_calibration.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path.name}")


# ── Plot 15: Multi-horizon hydrograph for 3 storms ──────────────────────────
def plot_multihorizon_hydrograph(obs_te    : pd.Series,
                                  pred_lstm : pd.DataFrame,
                                  pred_tr   : pd.DataFrame,
                                  storm_cat : pd.DataFrame,
                                  flood_thr : float):
    """
    For three selected storms, show how the predicted hydrograph shape
    evolves across all six forecast horizons using a colour gradient.

    Lighter colours = shorter lead times (more certain).
    Darker colours  = longer lead times (less certain).

    This figure demonstrates the model's ability to produce skillful
    predictions across the full 30–180 min warning window.
    """
    if storm_cat.empty or len(storm_cat) < 3:
        print("  [WARNING] Insufficient storms for multi-horizon figure.")
        return

    sel_storms = storm_cat.nlargest(3, "peak_value")   # show 3 biggest events
    fig, axes  = plt.subplots(3, 2, figsize=(16, 12), constrained_layout=True)
    horizon_colors = plt.cm.Blues(np.linspace(0.3, 1.0, OUTPUT_STEPS))

    for row, (_, storm) in enumerate(sel_storms.iterrows()):
        peak_t = storm["peak_time"]
        win    = pd.Timedelta(hours=4)
        obs_w  = obs_te[peak_t - win : peak_t + win]

        for col_idx, (pred_df, name) in enumerate(zip(
            [pred_lstm, pred_tr], ["LSTM", "Transformer"]
        )):
            ax = axes[row, col_idx]
            ax.plot(obs_w.index, obs_w.values,
                    color=C_OBS, lw=2.5, label="Observed", zorder=6)

            for j, h in enumerate(FORECAST_HORIZONS_MIN):
                col_name = f"t+{h}min"
                if col_name in pred_df.columns:
                    pw = _to_valid_time(pred_df[col_name], h).reindex(obs_w.index)
                    ax.plot(obs_w.index, pw.values,
                            color=horizon_colors[j], lw=1.5,
                            label=f"t+{h} min", zorder=4)

            ax.axhline(flood_thr, color=C_FLD, ls="--", lw=1)
            ax.axvline(peak_t,    color="purple", ls=":", lw=1.5, alpha=0.5)
            ax.set_ylabel("Gage height (ft)")
            ax.set_title(
                f"{name} | Storm: {peak_t.strftime('%Y-%m-%d %H:%M')}\n"
                f"Obs peak: {storm['peak_value']:.2f} ft | "
                f"Colour: light→dark = t+30→t+180 min"
            )
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d %H:%M"))
            ax.tick_params(axis="x", rotation=25)
            if row == 0 and col_idx == 0:
                ax.legend(loc="upper left", fontsize=7, ncol=2)

    fig.suptitle(
        "Crabtree Creek @ US-1 — Multi-Horizon Hydrograph Prediction\n"
        "Three largest storm events | Horizons t+30 to t+180 min\n"
        "Colour gradient: lighter = shorter horizon (more certain), "
        "darker = longer horizon | USGS 02087324",
        fontsize=10, fontweight="bold"
    )
    path = OUTPUT_DIR / "comparison_multihorizon_hydrograph.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path.name}")


# =============================================================================
# SECTION 15 — CSV OUTPUT TABLES
# All quantitative results are exported as CSV files suitable for direct
# inclusion as supplementary tables in a journal manuscript.
# =============================================================================

def save_all_csv_tables(metrics_lstm : pd.DataFrame,
                         metrics_tr   : pd.DataFrame,
                         metrics_pers : pd.DataFrame,
                         unc_m_lstm   : pd.DataFrame,
                         unc_m_tr     : pd.DataFrame,
                         storm_cat    : pd.DataFrame,
                         per_storm_df : pd.DataFrame,
                         df_tr        : pd.DataFrame,
                         df_te        : pd.DataFrame,
                         n_features   : int = None):
    """
    Export all quantitative results to CSV files.

    Files produced
    --------------
    per_horizon_metrics.csv    : NSE, RMSE, KGE, PBIAS, Peak-RMSE per horizon
    comparison_summary.csv     : Overall metrics for Table 1 in manuscript
    uncertainty_metrics.csv    : PICP, PICP_flood, PINAW per horizon
    storm_catalog_test.csv     : Detected storm events in test period
    per_storm_metrics.csv      : Per-storm accuracy for supplementary table
    """
    # Per-horizon metrics
    dfs = []
    for df_m, name in zip([metrics_lstm, metrics_tr, metrics_pers],
                           ["LSTM", "Transformer", "Persistence"]):
        tmp       = df_m.copy()
        tmp.index = [f"{name}_{i}" for i in tmp.index]
        dfs.append(tmp)

    per_horizon = pd.concat(dfs)
    path = OUTPUT_DIR / "per_horizon_metrics.csv"
    per_horizon.to_csv(path)
    print(f"  Saved: {path.name}")

    # Comparison summary (t+150 min, primary horizon)
    def _row(df_m, name):
        h = "t+150min"
        if h in df_m.index:
            r = df_m.loc[h].to_dict()
            r["Model"]   = name
            r["Horizon"] = h
            return r
        return {"Model": name}

    summary = pd.DataFrame([
        _row(metrics_lstm, "LSTM"),
        _row(metrics_tr,   "Transformer"),
        _row(metrics_pers, "Persistence"),
    ])
    summary["n_train"] = len(df_tr)
    summary["n_test"]  = len(df_te)
    summary["lookback_min"]  = LOOKBACK_MIN
    summary["features"]      = n_features if n_features is not None else "NA"
    summary["loss_function"] = "FloodAwareLoss_tiered_v6"
    summary["ordinal_levels_ft"] = list(ORDINAL_STAGE_LEVELS_FT)
    summary["continuous_stage_weight"] = USE_CONTINUOUS_STAGE_WEIGHT
    summary["future_forcing"] = (FUTURE_FORCING_MODE if USE_FUTURE_FORCING else "none")

    path = OUTPUT_DIR / "comparison_summary.csv"
    summary.to_csv(path, index=False)
    print(f"  Saved: {path.name}")

    # Uncertainty metrics
    unc_out = pd.concat([
        unc_m_lstm.add_prefix("LSTM_"),
        unc_m_tr.add_prefix("Transformer_")
    ], axis=1)
    path = OUTPUT_DIR / "uncertainty_metrics.csv"
    unc_out.to_csv(path)
    print(f"  Saved: {path.name}")

    # Storm catalog
    if not storm_cat.empty:
        path = OUTPUT_DIR / "storm_catalog_test.csv"
        storm_cat.to_csv(path, index=False)
        print(f"  Saved: {path.name}")

    # Per-storm metrics
    if not per_storm_df.empty:
        path = OUTPUT_DIR / "per_storm_metrics.csv"
        per_storm_df.to_csv(path, index=False)
        print(f"  Saved: {path.name}")


def compute_per_storm_metrics(storm_cat : pd.DataFrame,
                               obs_te    : pd.Series,
                               pred_lstm : pd.DataFrame,
                               pred_tr   : pd.DataFrame,
                               flood_thr : float) -> pd.DataFrame:
    """
    Compute per-storm performance metrics for both models.

    For each detected storm event, extract a ±3 hour window around the
    peak and compute RMSE, NSE, peak error, and timing error for both
    LSTM and Transformer at t+150 min.

    Returns
    -------
    pd.DataFrame with one row per storm event.
    """
    if storm_cat.empty:
        return pd.DataFrame()

    rows = []
    win  = pd.Timedelta(hours=3)

    for _, storm in storm_cat.iterrows():
        peak_t = storm["peak_time"]
        obs_w  = obs_te[peak_t - win : peak_t + win]
        if len(obs_w) < 5:
            continue

        row = {
            "storm_peak_time" : peak_t,
            "obs_peak_ft"     : float(storm["peak_value"]),
        }

        for pred_df, name in zip([pred_lstm, pred_tr], ["LSTM", "Transformer"]):
            col = "t+150min"
            if col not in pred_df.columns:
                continue
            pred_w  = pred_df[col].reindex(obs_w.index)
            obs_arr = obs_w.values
            pr_arr  = pred_w.values
            mask    = np.isfinite(obs_arr) & np.isfinite(pr_arr)

            if mask.sum() < 3:
                continue

            rmse_v   = float(np.sqrt(mean_squared_error(obs_arr[mask], pr_arr[mask])))
            nse_v    = nse(obs_arr[mask], pr_arr[mask])
            obs_pk   = float(obs_arr.max())
            pred_pk  = float(pr_arr[np.isfinite(pr_arr)].max()) if np.any(np.isfinite(pr_arr)) else np.nan
            pk_err   = pred_pk - obs_pk

            obs_pkidx  = int(np.argmax(obs_arr))
            pred_pkidx = int(np.nanargmax(pr_arr))
            t_err      = (pred_pkidx - obs_pkidx) * TIMESTEP_MIN

            row.update({
                f"{name}_RMSE"      : round(rmse_v, 4),
                f"{name}_NSE"       : round(nse_v, 4),
                f"{name}_pred_peak" : round(pred_pk, 3),
                f"{name}_peak_err"  : round(pk_err, 3),
                f"{name}_timing_min": t_err,
            })

        rows.append(row)

    return pd.DataFrame(rows)


# =============================================================================
# SECTION 16 — MAIN PIPELINE
# Orchestrates the complete workflow:
#   1.  Data loading and feature engineering
#   2.  Chronological data split
#   3.  Flood threshold computation
#   4.  LSTM training, inference, and evaluation
#   5.  Transformer training, inference, and evaluation
#   6.  Persistence baseline computation
#   7.  MC Dropout uncertainty quantification
#   8.  Storm detection
#   9.  All visualisations
#   10. CSV export
# =============================================================================

def run_forcing_ablation(df_raw, model_class=None, model_params=None):
    """
    Future-forcing ablation (v2): quantifies how much of the peak-timing skill
    comes from rainfall *forecasts* versus the model itself, by retraining
    under three forcing regimes and comparing t+150 skill.

      • perfect_prog : observed future rain  -> achievable skill CEILING
      • qpf          : degraded future rain  -> realistic OPERATIONAL skill
      • none         : past-only (causal)    -> structural-lag FLOOR

    The gap between perfect_prog and none isolates the value of rainfall
    forecasting for sub-hourly urban peak timing — the headline experiment
    for the paper. Saves forcing_ablation.csv. NOTE: this retrains a model
    three times and is therefore expensive; run on GPU/Colab.
    """
    global USE_FUTURE_FORCING, FUTURE_FORCING_MODE
    if model_class is None:
        model_class = HydroLSTM                  # v3: LSTM is the primary model
    if model_params is None:
        model_params = LSTM_BEST_PARAMS

    saved = (USE_FUTURE_FORCING, FUTURE_FORCING_MODE)
    regimes = [("perfect_prog", True, "perfect_prog"),
               ("qpf",          True, "qpf"),
               ("past_only",    False, "none")]
    rows = []
    print("\n" + "=" * 70)
    print("  FUTURE-FORCING ABLATION (perfect-prog / QPF / past-only)")
    print("=" * 70)
    for label, use_ff, mode in regimes:
        USE_FUTURE_FORCING = use_ff
        FUTURE_FORCING_MODE = mode
        print(f"\n  >>> Regime: {label}  (USE_FUTURE_FORCING={use_ff}, mode={mode})")
        df_feat, feat_cols, target_cols = build_feature_channels(df_raw.copy())
        df_tr, df_va, df_te = chronological_split(df_feat)
        thr = build_stage_thresholds(df_tr, target_cols)
        eval_thr = thr.significant_ft
        model, scaler, _, _ = train_final_model(
            model_class, model_params, df_tr, df_va,
            feat_cols, target_cols, thr, label)
        pred_te = predict_continuous(model, scaler, df_te, feat_cols, target_cols)
        obs_te  = get_obs_dataframe(df_te, target_cols)
        m   = compute_all_horizon_metrics(obs_te, pred_te, eval_thr, thr.extreme_ft)
        ev  = compute_event_skill_table(obs_te, pred_te, eval_thr)
        r150 = m.loc["t+150min"]
        e150 = ev.loc[150] if 150 in ev.index else {}
        rows.append({
            "regime"            : label,
            "n_features"        : len(feat_cols),
            "RMSE"              : r150["RMSE"],
            "NSE"               : r150["NSE"],
            "NSE_flood"         : r150.get("NSE_flood", np.nan),
            "Peak_RMSE"         : r150["Peak_RMSE"],
            "median_timing_min" : r150["Timing_min"],
            "CSI"               : e150.get("CSI", np.nan),
            "late_peak_fraction": e150.get("late_peak_fraction", np.nan),
            "mean_peak_bias_ft" : e150.get("mean_peak_bias_ft", np.nan),
        })
    USE_FUTURE_FORCING, FUTURE_FORCING_MODE = saved   # restore
    out = pd.DataFrame(rows)
    try:
        out.to_csv(OUTPUT_DIR / "forcing_ablation.csv", index=False)
        print("\n  Saved: forcing_ablation.csv")
    except Exception as exc:
        print(f"  [WARN] Could not save forcing_ablation.csv: {exc}")
    print("\n", out.to_string(index=False))
    return out


def _flood_threshold(df_part, target_cols):
    """Deprecated alias — returns significant evaluation threshold (9 ft)."""
    return SIGNIFICANT_STAGE_FT


def build_forward_eval_split(df          : pd.DataFrame,
                             eval_year   : int,
                             val_fraction: float = CV_VAL_FRACTION,
                             embargo     : int = CV_EMBARGO_STEPS):
    """
    Train/val/test split for forward evaluation on a future calendar year.

    Train = all timesteps with year < eval_year (e.g. 2020–2025 for eval 2026).
    Test  = eval_year only. Validation carved from the tail of the train block.
    """
    df_train_full = df[df.index.year < eval_year]
    df_test       = df[df.index.year == eval_year]
    if len(df_train_full) < (LOOKBACK_STEPS + embargo + 100):
        raise ValueError(f"Insufficient training data before {eval_year}.")
    if len(df_test) < (LOOKBACK_STEPS + 50):
        raise ValueError(
            f"No usable test rows for {eval_year}. "
            f"Extend FORWARD_EVAL_END_DATE / re-download USGS data.")

    n_tr_full = len(df_train_full)
    n_val     = max(1, int(n_tr_full * val_fraction))
    split_pt  = n_tr_full - n_val
    df_train  = df_train_full.iloc[: max(0, split_pt - embargo)]
    df_val    = df_train_full.iloc[split_pt:]
    df_train  = df_train.iloc[: len(df_train) - embargo] if len(df_train) > embargo else df_train

    train_years = sorted(df_train_full.index.year.unique())
    print("=== Forward-evaluation split ===")
    print(f"  Train years : {train_years[0]}–{train_years[-1]}  "
          f"({len(df_train):,} train / {len(df_val):,} val rows)")
    print(f"  Test year   : {eval_year}  ({len(df_test):,} rows)")
    return df_train, df_val, df_test


def run_forward_eval_year(df_feat     : pd.DataFrame,
                          feat_cols   : list,
                          target_cols : list,
                          eval_year   : int = None):
    """
    Train (or load) lstm_forward on all years before eval_year; score eval_year.

    Use this for operational-style evaluation: model trained on 2020–2025,
    tested on 2026 once USGS data exist. Outputs → OUTPUT_DIR/forward_eval_{year}/.
    """
    eval_year = eval_year or FORWARD_EVAL_YEAR
    out_dir   = OUTPUT_DIR / f"forward_eval_{eval_year}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print(f"  FORWARD EVALUATION — TEST YEAR {eval_year}")
    print("=" * 70)

    df_tr, df_va, df_te = build_forward_eval_split(df_feat, eval_year)
    thresholds = build_stage_thresholds(df_tr, target_cols)
    flood_thr  = thresholds.significant_ft
    print_stage_threshold_diagnostics(df_tr, thresholds, label="forward train")

    tag = FORWARD_CHECKPOINT_TAG
    global FORCE_RETRAIN
    saved_force = FORCE_RETRAIN
    if FORWARD_FORCE_RETRAIN:
        FORCE_RETRAIN = False
    use_cache = USE_MODEL_CACHE and not FORWARD_FORCE_RETRAIN

    try:
        model, scaler, tr_losses, va_losses = train_final_model(
            HydroLSTM, LSTM_BEST_PARAMS, df_tr, df_va, feat_cols, target_cols,
            thresholds, f"LSTM-forward-{eval_year}",
            checkpoint_tag=tag, use_cache=use_cache)
    finally:
        FORCE_RETRAIN = saved_force

    pred_te = predict_continuous(model, scaler, df_te, feat_cols, target_cols)
    obs_te  = get_obs_dataframe(df_te, target_cols)
    metrics = compute_all_horizon_metrics(obs_te, pred_te, flood_thr,
                                          thresholds.extreme_ft)
    event   = compute_event_skill_table(obs_te, pred_te, flood_thr)

    metrics.to_csv(out_dir / "per_horizon_metrics.csv")
    event.to_csv(out_dir / "event_skill_lstm.csv")
    pred_te.to_csv(out_dir / "predictions.csv")
    obs_te.to_csv(out_dir / "observations.csv")

    print(f"\n  Forward eval @ t+150:")
    if "t+150min" in metrics.index:
        r = metrics.loc["t+150min"]
        print(f"    NSE={r['NSE']}  NSE_flood={r.get('NSE_flood')}  "
              f"Peak_RMSE={r.get('Peak_RMSE')}  Timing={r.get('Timing_min')} min")
    if 150 in event.index:
        e = event.loc[150]
        print(f"    CSI@9ft={e.get('CSI')}  POD={e.get('POD')}  "
              f"n_events={e.get('n_events')}")

    _detailed_outputs(
        [model], scaler, tr_losses, va_losses,
        None, None, [], [],
        df_tr, df_te, feat_cols, target_cols, flood_thr,
        tag=f"forward eval {eval_year}")

    # Move/copy key figures into forward_eval subfolder (plots already in OUTPUT_DIR)
    print(f"\n  Forward-eval tables → {out_dir.resolve()}/")
    print(f"  Figures (latest)    → {OUTPUT_DIR.resolve()}/")
    return {"metrics": metrics, "event": event, "model": model, "scaler": scaler}


def plot_cv_fold_skill(per_fold: dict, flood_thr: float):
    """
    v3 headline figure: forecast skill across rolling-origin folds.
    Left panel  = overall NSE vs lead time (per fold + mean±std).
    Right panel = flood-conditioned NSE vs lead time (the honest skill metric).
    Academic style consistent with the rest of the figure suite.
    """
    horizons = FORECAST_HORIZONS_MIN
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharex=True)
    panels = [("NSE", "Overall NSE", axes[0]),
              ("NSE_flood", f"NSE (stage >= {flood_thr:.0f} ft)", axes[1])]
    fold_ids = sorted(per_fold.keys())
    cmap = plt.cm.viridis(np.linspace(0.15, 0.85, len(fold_ids)))

    for metric, title, ax in panels:
        stack = []
        for c, fid in zip(cmap, fold_ids):
            m = per_fold[fid]["metrics_lstm"]
            vals = [m.loc[f"t+{h}min", metric] if f"t+{h}min" in m.index else np.nan
                    for h in horizons]
            stack.append(vals)
            ax.plot(horizons, vals, "-o", color=c, lw=1.4, ms=4, alpha=0.85,
                    label=f"Fold {fid} (test {per_fold[fid]['test_year']})")
        stack = np.array(stack, dtype=float)
        mean = np.nanmean(stack, axis=0); std = np.nanstd(stack, axis=0)
        ax.plot(horizons, mean, "-", color=C_OBS, lw=2.6, label="Mean ± std", zorder=5)
        ax.fill_between(horizons, mean - std, mean + std, color=C_OBS, alpha=0.12, zorder=0)
        ax.axhline(0.0, color="0.5", ls=":", lw=1)
        ax.set_xlabel("Forecast lead time (min)")
        ax.set_ylabel(metric)
        ax.set_title(title)
        ax.set_xticks(horizons)
        ax.grid(alpha=0.25)
    axes[1].legend(fontsize=8, loc="lower left")
    fig.suptitle("Rolling-origin cross-validation — HydroLSTM skill vs lead time\n"
                 f"NSE_flood computed at >= {flood_thr:.0f} ft significant stage "
                 f"(USGS 02087324)",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    path = OUTPUT_DIR / "cv_rolling_origin_skill.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path.name}")


def _detailed_outputs(lstm_model, lstm_scaler, lstm_tr_losses, lstm_val_losses,
                      tr_model, tr_scaler, tr_tr_losses, tr_val_losses,
                      df_tr, df_te, feat_cols, target_cols, FLOOD_THRESH,
                      tag=""):
    """
    Full inference + uncertainty + storm analysis + publication figures + CSV
    tables for ONE partition (the showcase fold, or the legacy chronological
    split). Supports deep-ensemble UQ (primary) and MC-Dropout (comparison).
    """
    lstm_models = lstm_model if isinstance(lstm_model, (list, tuple)) else [lstm_model]
    primary = lstm_models[0]
    have_tr = tr_model is not None
    print("\n" + "=" * 70)
    print(f"  DETAILED OUTPUTS{(' — ' + tag) if tag else ''} (INFERENCE + FIGURES)")
    print("=" * 70)

    pred_lstm_tr_df = predict_continuous(primary, lstm_scaler, df_tr, feat_cols, target_cols)
    obs_tr_df       = get_obs_dataframe(df_tr, target_cols)
    if USE_DEEP_ENSEMBLE and len(lstm_models) > 1:
        ens_te = ensemble_predict(lstm_models, lstm_scaler, df_te, feat_cols, target_cols)
        pred_lstm_te_df = ens_te["mean"]
    else:
        pred_lstm_te_df = predict_continuous(primary, lstm_scaler, df_te, feat_cols, target_cols)
    obs_te_df       = get_obs_dataframe(df_te, target_cols)

    obs_te_ser  = obs_te_df["t+150min"].dropna()
    pred_lstm_s = pred_lstm_te_df["t+150min"].reindex(obs_te_ser.index)
    persist_df  = compute_persistence_baseline(df_te, target_cols)
    persist_s   = persist_df["t+150min"].reindex(obs_te_ser.index)

    if have_tr:
        pred_tr_tr_df = predict_continuous(tr_model, tr_scaler, df_tr, feat_cols, target_cols)
        pred_tr_te_df = predict_continuous(tr_model, tr_scaler, df_te, feat_cols, target_cols)
        pred_tr_s     = pred_tr_te_df["t+150min"].reindex(obs_te_ser.index)
    else:
        pred_tr_te_df = pred_lstm_te_df          # placeholder; tr plots skipped
        pred_tr_s     = pred_lstm_s

    metrics_lstm = compute_all_horizon_metrics(obs_te_df, pred_lstm_te_df, FLOOD_THRESH)
    metrics_pers = compute_all_horizon_metrics(obs_te_df, persist_df,      FLOOD_THRESH)
    metrics_tr   = (compute_all_horizon_metrics(obs_te_df, pred_tr_te_df, FLOOD_THRESH)
                    if have_tr else pd.DataFrame())

    # Event-based skill
    event_lstm = compute_event_skill_table(obs_te_df, pred_lstm_te_df, FLOOD_THRESH)
    event_pers = compute_event_skill_table(obs_te_df, persist_df,      FLOOD_THRESH)
    try:
        event_lstm.to_csv(OUTPUT_DIR / "event_skill_lstm.csv")
        event_pers.to_csv(OUTPUT_DIR / "event_skill_persistence.csv")
        if have_tr:
            compute_event_skill_table(obs_te_df, pred_tr_te_df, FLOOD_THRESH)\
                .to_csv(OUTPUT_DIR / "event_skill_transformer.csv")
    except Exception as exc:
        print(f"  [WARN] event-skill CSV: {exc}")

    # Uncertainty — deep ensemble (primary) + MC-Dropout (comparison baseline)
    if USE_DEEP_ENSEMBLE and len(lstm_models) > 1:
        unc_lstm   = ensemble_predict(lstm_models, lstm_scaler, df_te, feat_cols, target_cols)
        unc_m_lstm = compute_uncertainty_metrics(obs_te_df, unc_lstm, FLOOD_THRESH)
        print("  Uncertainty: deep ensemble + heteroscedastic heads (primary UQ).")
    else:
        unc_lstm   = mc_dropout_predict(primary, lstm_scaler, df_te, feat_cols, target_cols, MC_SAMPLES)
        unc_m_lstm = compute_uncertainty_metrics(obs_te_df, unc_lstm, FLOOD_THRESH)

    if KEEP_MC_DROPOUT and USE_DEEP_ENSEMBLE and len(lstm_models) > 1:
        unc_mc = mc_dropout_predict(primary, lstm_scaler, df_te, feat_cols, target_cols, MC_SAMPLES)
        unc_m_mc = compute_uncertainty_metrics(obs_te_df, unc_mc, FLOOD_THRESH)
        try:
            cmp = unc_m_lstm.copy()
            cmp["method"] = "ensemble"
            mc = unc_m_mc.copy()
            mc["method"] = "mc_dropout"
            pd.concat([cmp, mc]).to_csv(OUTPUT_DIR / "uncertainty_comparison.csv")
            print("  Saved: uncertainty_comparison.csv (ensemble vs MC-Dropout PICP).")
        except Exception as exc:
            print(f"  [WARN] uncertainty comparison CSV: {exc}")
    if have_tr:
        unc_tr   = mc_dropout_predict(tr_model, tr_scaler, df_te, feat_cols, target_cols, MC_SAMPLES)
        unc_m_tr = compute_uncertainty_metrics(obs_te_df, unc_tr, FLOOD_THRESH)
    else:
        unc_tr, unc_m_tr = unc_lstm, unc_m_lstm

    # Storms
    storm_catalog = detect_storms(obs_te_ser, min_stage_ft=FLOOD_THRESH, min_sep_hr=18)
    per_storm_df  = compute_per_storm_metrics(
        storm_catalog, obs_te_ser, pred_lstm_te_df, pred_tr_te_df, FLOOD_THRESH)

    # ── Figures ───────────────────────────────────────────────────────────
    m_lstm_150 = metrics_lstm.loc["t+150min"].to_dict() if "t+150min" in metrics_lstm.index else {}
    m_tr_150   = metrics_tr.loc["t+150min"].to_dict()   if (have_tr and "t+150min" in metrics_tr.index) else {}
    m_pers_150 = metrics_pers.loc["t+150min"].to_dict() if "t+150min" in metrics_pers.index else {}

    obs_tr_s = obs_tr_df["t+150min"]
    pred_lstm_tr_s = pred_lstm_tr_df["t+150min"].reindex(obs_tr_s.index)

    tr_hist = (lstm_tr_losses[0] if isinstance(lstm_tr_losses, list)
               and lstm_tr_losses and isinstance(lstm_tr_losses[0], list)
               else lstm_tr_losses)
    va_hist = (lstm_val_losses[0] if isinstance(lstm_val_losses, list)
               and lstm_val_losses and isinstance(lstm_val_losses[0], list)
               else lstm_val_losses)

    plot_training_history(tr_hist, va_hist, "LSTM")
    plot_full_record(obs_tr_s, pred_lstm_tr_s, obs_te_ser, pred_lstm_s,
                     m_lstm_150, FLOOD_THRESH, "LSTM")
    plot_storm_zoom_panels(obs_te_ser, pred_lstm_te_df, unc_lstm,
                           storm_catalog, FLOOD_THRESH, "LSTM", n_panels=6)
    plot_gradient_saliency(primary, lstm_scaler, df_tr, feat_cols, target_cols)
    plot_monthly_rmse(obs_te_ser, pred_lstm_s, FLOOD_THRESH, "LSTM")

    if have_tr:
        pred_tr_tr_s = pred_tr_tr_df["t+150min"].reindex(obs_tr_s.index)
        plot_training_history(tr_tr_losses, tr_val_losses, "Transformer")
        plot_full_record(obs_tr_s, pred_tr_tr_s, obs_te_ser, pred_tr_s,
                         m_tr_150, FLOOD_THRESH, "Transformer")
        plot_storm_zoom_panels(obs_te_ser, pred_tr_te_df, unc_tr,
                               storm_catalog, FLOOD_THRESH, "Transformer", n_panels=6)
        plot_attention_rollout(tr_model, tr_scaler, df_tr, feat_cols, target_cols)
        plot_monthly_rmse(obs_te_ser, pred_tr_s, FLOOD_THRESH, "Transformer")
        plot_comparison_overlay(obs_te_ser, pred_lstm_s, pred_tr_s, persist_s,
                                m_lstm_150, m_tr_150, m_pers_150, FLOOD_THRESH)
        plot_scatter(obs_te_ser, pred_lstm_s, pred_tr_s, FLOOD_THRESH, m_lstm_150, m_tr_150)
        plot_metrics_bar(m_lstm_150, m_tr_150, m_pers_150)
        plot_flow_duration_curve(obs_te_ser, pred_lstm_s, pred_tr_s, FLOOD_THRESH)
        plot_timing_error_histogram(obs_te_ser, pred_lstm_s, pred_tr_s, FLOOD_THRESH)
        plot_comparison_storm_panels(obs_te_ser, pred_lstm_te_df, pred_tr_te_df,
                                     unc_lstm, unc_tr, storm_catalog, FLOOD_THRESH, n_panels=6)
        plot_horizon_skill(metrics_lstm, metrics_tr, metrics_pers)
        plot_uncertainty_calibration(unc_lstm, unc_tr, obs_te_df, FLOOD_THRESH, unc_m_lstm, unc_m_tr)
        plot_multihorizon_hydrograph(obs_te_ser, pred_lstm_te_df, pred_tr_te_df,
                                     storm_catalog, FLOOD_THRESH)

    try:
        save_all_csv_tables(
            metrics_lstm,
            metrics_tr if have_tr else metrics_lstm,
            metrics_pers, unc_m_lstm, unc_m_tr,
            storm_catalog, per_storm_df, df_tr, df_te, n_features=len(feat_cols))
    except Exception as exc:
        print(f"  [WARN] CSV export: {exc}")

    return {"metrics_lstm": metrics_lstm, "metrics_tr": metrics_tr,
            "metrics_pers": metrics_pers}


def main():
    """
    Main execution pipeline for the Crabtree Creek flash flood forecasting
    framework. Runs end-to-end from raw data retrieval to publication-ready
    figures and CSV tables.
    """
    # These are defined at module scope (SECTION 2). Because the optional
    # Optuna branch below *assigns* to them, Python would otherwise treat them
    # as function-locals for the whole of main() and raise UnboundLocalError
    # when OPTUNA_SKIP=True (the assignment branch is skipped). Declaring them
    # global makes the reads resolve to the SECTION 2 values and lets the
    # Optuna branch update them in place when enabled.
    global LSTM_BEST_PARAMS, TRANSFORMER_BEST_PARAMS

    print("\n" + "=" * 70)
    print("  STEP 1 — DATA LOADING")
    print("=" * 70)
    cache_feats = PREPROCESSED_DIR / "02_features_full.csv"
    cache_end   = PREPROCESSED_DIR / "data_end_date.txt"
    df_raw = None
    cache_ok = (USE_PREPROCESSED_CACHE and cache_feats.exists()
                and cache_end.exists()
                and cache_end.read_text().strip() == _data_end_date())
    if cache_ok:
        print(f"  Loading cached preprocessed features: {cache_feats}")
        print(f"  (data_end_date={cache_end.read_text().strip()})")
        df_feat = pd.read_csv(cache_feats, index_col=0, parse_dates=[0])
        feat_cols = [c for c in (PREPROCESSED_DIR / "06_feature_list.txt")
                     .read_text().splitlines() if c]
        target_cols = [f"t+{h}min" for h in FORECAST_HORIZONS_MIN]
    else:
        if USE_PREPROCESSED_CACHE and cache_feats.exists():
            print(f"  [INFO] Preprocessed cache stale (need {_data_end_date()}) "
                  f"— rebuilding features.")
        df_raw = build_raw_dataframe()

        # Optional: future-forcing ablation (perfect-prog / QPF / past-only).
        if RUN_FORCING_ABLATION:
            run_forcing_ablation(df_raw)

        print("\n" + "=" * 70)
        print("  STEP 2 — PHYSICS-INFORMED FEATURE ENGINEERING")
        print("=" * 70)
        df_feat, feat_cols, target_cols = build_feature_channels(df_raw)

    if RUN_FEATURE_PRUNING:
        print("\n" + "=" * 70)
        print("  FEATURE PRUNING (LightGBM / SHAP)")
        print("=" * 70)
        df_pr_tr, _, _ = chronological_split(df_feat)
        feat_cols = prune_features_shap_lightgbm(
            df_pr_tr, feat_cols, target_cols,
            top_k=FEATURE_PRUNING_TOP_K, min_keep=FEATURE_PRUNING_MIN_KEEP)

    # ── Persist a copy of the engineered dataset once (for inspection) ─────
    if SAVE_PREPROCESSED:
        try:
            _t, _v, _e = chronological_split(df_feat)
            save_preprocessed_data(df_raw, df_feat, feat_cols, target_cols, _t, _v, _e)
        except Exception as exc:
            print(f"  [WARN] preprocessed save skipped: {exc}")

    print(f"\n  Feature count : {len(feat_cols)}")
    print(f"  Lookback      : {LOOKBACK_MIN} min ({LOOKBACK_STEPS} steps)")
    print(f"  LSTM params   : {LSTM_BEST_PARAMS}")

    # ── Optional forward evaluation (e.g. train 2020–2025 → test 2026) ───────
    forward_result = None
    if RUN_FORWARD_EVAL:
        forward_result = run_forward_eval_year(
            df_feat, feat_cols, target_cols, FORWARD_EVAL_YEAR)
        if SKIP_ROLLING_CV:
            print("\n  ALL STEPS COMPLETE (forward eval only)")
            print(f"  Results saved to: {OUTPUT_DIR.resolve()}/")
            return {"forward_eval": forward_result}

    # =====================================================================
    # EVALUATION
    # =====================================================================
    if not USE_ROLLING_ORIGIN_CV:
        # ---- Legacy single chronological split --------------------------
        print("\n[EVAL MODE] Legacy chronological split (USE_ROLLING_ORIGIN_CV=False)")
        df_tr, df_va, df_te = chronological_split(df_feat)
        THRESHOLDS = build_stage_thresholds(df_tr, target_cols)
        FLOOD_THRESH = THRESHOLDS.significant_ft
        print_stage_threshold_diagnostics(df_tr, THRESHOLDS, label="legacy train")
        lstm_models, lstm_scaler, l_tr, l_va = train_deep_ensemble(
            HydroLSTM, LSTM_BEST_PARAMS, df_tr, df_va, feat_cols, target_cols,
            THRESHOLDS, "LSTM")
        lstm_model = lstm_models[0]
        if USE_RESIDUAL_CORRECTION:
            lstm_model = train_residual_corrector(
                lstm_model, df_tr, df_va, feat_cols, target_cols,
                THRESHOLDS, lstm_scaler, "LSTM")
            lstm_models = [lstm_model]
        tr_model = tr_scaler = t_tr = t_va = None
        if RUN_TRANSFORMER:
            tr_model, tr_scaler, t_tr, t_va = train_final_model(
                HydroTransformer, TRANSFORMER_BEST_PARAMS, df_tr, df_va,
                feat_cols, target_cols, THRESHOLDS, "Transformer")
        if RUN_GBM_BASELINE:
            g = gbm_flood_baseline(df_tr, df_te, feat_cols, target_cols, FLOOD_THRESH)
            if not g.empty:
                print("  [GBM baseline] flood-NSE: "
                      + "  ".join(f"t+{h}={g.loc[f't+{h}min','NSE_flood']}"
                                  for h in (60,120,180) if f"t+{h}min" in g.index))
        _detailed_outputs(lstm_models, lstm_scaler, l_tr, l_va,
                          tr_model, tr_scaler, t_tr, t_va,
                          df_tr, df_te, feat_cols, target_cols, FLOOD_THRESH,
                          tag="chronological")
        print("\n  ALL STEPS COMPLETE")
        return

    # ---- Rolling-origin (expanding-window) cross-validation -------------
    print("\n[EVAL MODE] Rolling-origin (expanding-window) cross-validation")
    folds = rolling_origin_folds(df_feat)
    if not folds:
        raise RuntimeError("No rolling-origin folds — check the data span / years.")
    showcase = folds[SHOWCASE_FOLD_INDEX]
    per_fold, gbm_rows = {}, []

    for f in folds:
        fid, ty = f["fold"], f["test_year"]
        print("\n" + "#" * 70)
        print(f"#  FOLD {fid}  —  HELD-OUT TEST YEAR {ty}")
        print("#" * 70)
        THRESHOLDS = build_stage_thresholds(f["df_train"], target_cols)
        FLOOD_THRESH = THRESHOLDS.significant_ft
        print_stage_threshold_diagnostics(f["df_train"], THRESHOLDS,
                                          label=f"fold {fid} train")

        lstm_models, lstm_scaler, l_tr, l_va = train_deep_ensemble(
            HydroLSTM, LSTM_BEST_PARAMS, f["df_train"], f["df_val"],
            feat_cols, target_cols, THRESHOLDS, f"LSTM-fold{fid}")
        lstm_model = lstm_models[0]
        if USE_RESIDUAL_CORRECTION:
            lstm_model = train_residual_corrector(
                lstm_model, f["df_train"], f["df_val"], feat_cols, target_cols,
                THRESHOLDS, lstm_scaler, f"LSTM-fold{fid}")
            lstm_models = [lstm_model]

        if USE_DEEP_ENSEMBLE and len(lstm_models) > 1:
            pred_te = ensemble_predict(lstm_models, lstm_scaler, f["df_test"],
                                       feat_cols, target_cols)["mean"]
        else:
            pred_te = predict_continuous(lstm_model, lstm_scaler, f["df_test"],
                                         feat_cols, target_cols)
        obs_te  = get_obs_dataframe(f["df_test"], target_cols)
        m_lstm  = compute_all_horizon_metrics(
            obs_te, pred_te, FLOOD_THRESH, THRESHOLDS.extreme_ft)
        ev_lstm = compute_event_skill_table(obs_te, pred_te, FLOOD_THRESH)
        ev_ext  = compute_event_skill_table(obs_te, pred_te, THRESHOLDS.extreme_ft)
        m_pers  = compute_all_horizon_metrics(
            obs_te, compute_persistence_baseline(f["df_test"], target_cols),
            FLOOD_THRESH, THRESHOLDS.extreme_ft)

        per_fold[fid] = {"test_year": ty, "flood_thr": FLOOD_THRESH,
                         "extreme_thr": THRESHOLDS.extreme_ft,
                         "elevated_thr": THRESHOLDS.elevated_ft,
                         "metrics_lstm": m_lstm, "metrics_pers": m_pers,
                         "event_lstm": ev_lstm, "event_extreme": ev_ext}

        for h in (60, 120, 180):
            k = f"t+{h}min"
            if k in m_lstm.index:
                print(f"    {k}: NSE={m_lstm.loc[k,'NSE']}  "
                      f"NSE_flood(>=9ft)={m_lstm.loc[k,'NSE_flood']}  "
                      f"NSE_extreme(>=12ft)={m_lstm.loc[k,'NSE_extreme']}  "
                      f"n_sig={m_lstm.loc[k,'n_significant']}  "
                      f"Peak_RMSE={m_lstm.loc[k,'Peak_RMSE']}  "
                      f"Timing={m_lstm.loc[k,'Timing_min']} min")
                if 150 in ev_lstm.index:
                    print(f"      CSI@9ft={ev_lstm.loc[150,'CSI']}  "
                          f"POD={ev_lstm.loc[150,'POD']}  "
                          f"n_events={ev_lstm.loc[150,'n_events']}")

        if RUN_GBM_BASELINE:
            g = gbm_flood_baseline(f["df_train"], f["df_test"], feat_cols,
                                   target_cols, FLOOD_THRESH)
            if not g.empty:
                for h in FORECAST_HORIZONS_MIN:
                    k = f"t+{h}min"
                    if k in g.index:
                        gbm_rows.append({"fold": fid, "test_year": ty,
                                         "horizon_min": h,
                                         "GBM_NSE": g.loc[k, "NSE"],
                                         "GBM_NSE_flood": g.loc[k, "NSE_flood"]})
                print("    [GBM baseline] flood-NSE: "
                      + "  ".join(f"t+{h}={g.loc[f't+{h}min','NSE_flood']}"
                                  for h in (60,120,180) if f"t+{h}min" in g.index))

        if f is showcase:
            print(f"\n  >> Showcase fold {fid} (test {ty}) — detailed publication figures")
            tr_model = tr_scaler = t_tr = t_va = None
            if RUN_TRANSFORMER:
                tr_model, tr_scaler, t_tr, t_va = train_final_model(
                    HydroTransformer, TRANSFORMER_BEST_PARAMS, f["df_train"],
                    f["df_val"], feat_cols, target_cols, THRESHOLDS,
                    f"Transformer-fold{fid}")
            _detailed_outputs(lstm_models, lstm_scaler, l_tr, l_va,
                              tr_model, tr_scaler, t_tr, t_va,
                              f["df_train"], f["df_test"], feat_cols, target_cols,
                              FLOOD_THRESH, tag=f"showcase fold {fid} (test {ty})")

    # ── Aggregate across folds (mean ± std) ───────────────────────────────
    print("\n" + "=" * 70)
    print("  ROLLING-ORIGIN CV — AGGREGATE SKILL (mean ± std across folds)")
    print("=" * 70)
    agg_rows = []
    for h in FORECAST_HORIZONS_MIN:
        k = f"t+{h}min"
        for metric in ["NSE", "NSE_flood", "NSE_extreme", "Peak_RMSE", "Timing_min",
                       "KGE", "PBIAS", "ETS", "Rise_RMSE", "ThrXing_min"]:
            vals = [per_fold[fid]["metrics_lstm"].loc[k, metric]
                    for fid in per_fold
                    if k in per_fold[fid]["metrics_lstm"].index]
            vals = [v for v in vals if pd.notna(v)]
            if vals:
                agg_rows.append({"horizon_min": h, "metric": metric,
                                 "mean": round(float(np.mean(vals)), 4),
                                 "std":  round(float(np.std(vals)), 4),
                                 "n_folds": len(vals)})
    agg_df = pd.DataFrame(agg_rows)
    agg_df.to_csv(OUTPUT_DIR / "cv_aggregate_metrics.csv", index=False)

    for h in FORECAST_HORIZONS_MIN:
        sub = agg_df[agg_df.horizon_min == h]
        def g(m):
            r = sub[sub.metric == m]
            return (f"{r['mean'].values[0]:+.3f}±{r['std'].values[0]:.3f}"
                    if len(r) else "n/a")
        print(f"  t+{h:3d}min | NSE {g('NSE')} | NSE_flood(9ft) {g('NSE_flood')} "
              f"| NSE_extreme(12ft) {g('NSE_extreme')} "
              f"| Peak_RMSE {g('Peak_RMSE')} | Timing {g('Timing_min')} min")

    fold_rows = []
    for fid in per_fold:
        m = per_fold[fid]["metrics_lstm"]
        for h in FORECAST_HORIZONS_MIN:
            k = f"t+{h}min"
            if k in m.index:
                fold_rows.append({"fold": fid,
                                  "test_year": per_fold[fid]["test_year"],
                                  "horizon_min": h, **m.loc[k].to_dict()})
    pd.DataFrame(fold_rows).to_csv(OUTPUT_DIR / "cv_per_fold_metrics.csv", index=False)
    if gbm_rows:
        pd.DataFrame(gbm_rows).to_csv(OUTPUT_DIR / "cv_gbm_baseline.csv", index=False)

    # Event detection skill at significant (9 ft) and extreme (12 ft) stages
    ev_rows = []
    for fid in per_fold:
        for label, evdf in [("significant_9ft", per_fold[fid].get("event_lstm")),
                            ("extreme_12ft", per_fold[fid].get("event_extreme"))]:
            if evdf is None or evdf.empty:
                continue
            for h in evdf.index:
                ev_rows.append({"fold": fid, "test_year": per_fold[fid]["test_year"],
                                "tier": label, "horizon_min": h,
                                **evdf.loc[h].to_dict()})
    if ev_rows:
        pd.DataFrame(ev_rows).to_csv(OUTPUT_DIR / "cv_event_detection_skill.csv",
                                     index=False)
        print("  Saved: cv_event_detection_skill.csv (CSI/POD @ 9 ft and 12 ft)")

    plot_cv_fold_skill(per_fold, SIGNIFICANT_STAGE_FT)

    print("\n" + "=" * 70)
    print("  ALL STEPS COMPLETE")
    print(f"  Results saved to: {OUTPUT_DIR.resolve()}/")
    if forward_result is not None:
        print(f"  Forward eval   : {OUTPUT_DIR / f'forward_eval_{FORWARD_EVAL_YEAR}'}/")
    print("=" * 70)
    out = {"per_fold": per_fold, "aggregate": agg_df}
    if forward_result is not None:
        out["forward_eval"] = forward_result
    return out



# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    results = main()
