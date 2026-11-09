"""
Single-feature fraud analysis with an HTML report.

Usage
-----
    from fraud_feature_analysis import analyse_features

    results = analyse_features(
        df,
        label_col="is_fraud",                        # 1 = fraud, 0 = genuine
        numeric_features=["amount_gbp", "txn_count_24h", "account_age_days"],
        categorical_features=["channel", "mcc", "country"],
        time_col="txn_timestamp",
        amount_col="amount_gbp",                     # optional, for value-based figures
        output_html="fraud_feature_report.html",
    )

Every other column in df (transaction ID, username, ...) is ignored and left untouched.
A feature may also be the amount column (e.g. amount as a numeric feature).

What the report covers, per feature
-----------------------------------
1. Data quality: missingness, distinct values, dominant values, rare categories,
   distribution summary for numeric features.
2. Risk by group: each category, or each equal-volume bin of a numeric feature,
   gets its volume, fraud rate with a 95% interval, lift, Weight of Evidence,
   share of fraud (and fraud value) captured, and a significance test corrected
   for the number of groups tested. Numeric features also get a shape check
   (does risk rise, fall, or form a U across the range?).
3. Overall strength: Information Value, Cramer's V, uncertainty coefficient,
   and Gini / ROC-AUC / PR-AUC / KS of the feature used on its own as a score.
   Scores are built out of fold, so features with many small categories cannot
   memorise the labels.
4. Change over time (walk-forward): for each period, the feature's risk pattern
   is learned on earlier periods and tested on that period. This gives a Gini
   per period with a bootstrap 95% interval, a trend label (Stable, Decaying,
   Emerging, Volatile) from a statistical test, and a diagnosis separating
   "the mix of values changed" from "which values are risky changed".

Before you run it
-----------------
* Labels must be mature: drop the most recent weeks whose fraud has not yet
  been reported. The report checks for the symptom (a falling fraud rate in
  the last period), but cannot fix it.
* Every feature must be known at authorisation time. The report flags
  symptoms of leakage but cannot prove its absence.
"""

# =============================================================================
# CONFIG - change these to suit your data
# =============================================================================

# --- Group size ----------------------------------------------------------------
# A category, or a bin of a numeric feature, must be at least this big.
# Smaller categories are merged into "(other)"; smaller numeric bins are merged
# with their neighbour. The LARGER of the two rules applies. Set one to 0 to
# use only the other. "(missing)" is never merged, because missingness is often
# predictive in fraud.
MIN_GROUP_PCT = 0.1               # % of all transactions (0.1 = 1 in 1,000)
MIN_GROUP_EXPECTED_FRAUDS = 10    # big enough to expect this many frauds at the overall fraud rate

N_BINS = 10                       # numeric features: start from this many equal-volume bins

# --- Time ----------------------------------------------------------------------
PERIOD = "M"                      # "W" week, "M" month, "Q" quarter
MIN_FRAUDS_PER_PERIOD = 30        # periods with fewer frauds are not scored (too noisy)
WALK_FORWARD_WINDOW = None        # None = learn from all earlier periods; 3 = only the 3 before
N_BOOTSTRAP = 500                 # resamples for the 95% intervals on each period's Gini

# --- Judging -------------------------------------------------------------------
GINI_BANDS = {"Weak": 0.05, "Moderate": 0.15, "Strong": 0.35}   # lower bound of each verdict
MIN_MEANINGFUL_GINI_CHANGE = 0.05 # smaller changes over time are not called a trend
TREND_ALPHA = 0.05                # significance level for trend / volatility tests
FDR_ALPHA = 0.05                  # false discovery rate for flagging risky / safe groups
IV_LEAKAGE_THRESHOLD = 0.5        # IV above this triggers a leakage warning
PSI_MODERATE, PSI_MAJOR = 0.10, 0.25
LABEL_MATURITY_RATIO = 0.7        # warn if the last period's fraud rate < 70% of the usual
OPTIMISM_WARN = 0.03              # warn if in-sample Gini/2 exceeds out-of-fold by this

# --- Scoring -------------------------------------------------------------------
ENCODING_SMOOTHING = 20           # shrinks small groups' fraud rates towards the overall rate
N_FOLDS = 5

# --- Report --------------------------------------------------------------------
REPORT_TITLE = "Single-feature fraud analysis"
CURRENCY = "£"
MAX_GROUPS_SHOWN = 25             # rows per feature in tables and charts (all kept in results)
RANDOM_STATE = 42

# =============================================================================

import base64
import html as _html
import io
import math
import warnings
from datetime import datetime
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import PercentFormatter  # noqa: E402

MISSING = "(missing)"
OTHER = "(other)"

# Colours (validated reference palette)
C_BLUE, C_ORANGE, C_RED = "#2a78d6", "#eb6834", "#e34948"
C_GREY, C_INK, C_INK2, C_GRID, C_SURFACE = "#9a9893", "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
C_MID = "#f0efec"
BLUE_RAMP = ["#f4f8fd", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


# =============================================================================
# 1. Validation
# =============================================================================

def validate_inputs(df, label_col, numeric_features, categorical_features, time_col, amount_col):
    if not isinstance(df, pd.DataFrame) or len(df) == 0:
        raise ValueError("df must be a non-empty pandas DataFrame.")

    for name, lst in (("numeric_features", numeric_features), ("categorical_features", categorical_features)):
        if len(set(lst)) != len(lst):
            raise ValueError(f"{name} contains duplicates.")
    overlap = set(numeric_features) & set(categorical_features)
    if overlap:
        raise ValueError(f"Features listed as both numeric and categorical: {sorted(overlap)}")
    if not numeric_features and not categorical_features:
        raise ValueError("Give at least one feature.")

    needed = [label_col, time_col, *numeric_features, *categorical_features]
    if amount_col:
        needed.append(amount_col)
    missing = [c for c in dict.fromkeys(needed) if c not in df.columns]
    if missing:
        raise KeyError(f"Columns not found in df: {missing}")

    bad = {label_col, time_col} & (set(numeric_features) | set(categorical_features))
    if bad:
        raise ValueError(f"The label and time columns cannot be features: {sorted(bad)}")

    y = df[label_col]
    if y.isna().any():
        raise ValueError(f"{int(y.isna().sum()):,} rows have no label. Unlabelled is not the same as genuine; "
                         "drop or resolve them first.")
    extra = set(pd.unique(y)) - {0, 1, True, False}
    if extra:
        raise ValueError(f"label_col must contain only 0/1. Found: {extra}")
    if y.sum() == 0 or y.sum() == len(y):
        raise ValueError("label_col must contain both fraud (1) and genuine (0) rows.")

    non_num = [c for c in numeric_features
               if not (pd.api.types.is_numeric_dtype(df[c]) or pd.api.types.is_bool_dtype(df[c]))]
    if non_num:
        raise TypeError(f"These numeric_features are not numeric: {non_num}. "
                        "Convert them, or list them as categorical.")

    t = pd.to_datetime(df[time_col], errors="coerce")
    if t.isna().any():
        raise ValueError(f"{int(t.isna().sum()):,} rows have a missing or unparseable {time_col}.")

    if amount_col:
        if not pd.api.types.is_numeric_dtype(df[amount_col]):
            raise TypeError(f"amount_col '{amount_col}' must be numeric.")
        if df[amount_col].isna().any():
            raise ValueError(f"amount_col '{amount_col}' has missing values.")


def min_group_size(n: int, base_rate: float) -> int:
    by_pct = math.ceil(MIN_GROUP_PCT / 100 * n) if MIN_GROUP_PCT > 0 else 0
    by_fraud = math.ceil(MIN_GROUP_EXPECTED_FRAUDS / base_rate) if MIN_GROUP_EXPECTED_FRAUDS > 0 else 0
    return max(1, by_pct, by_fraud)


# =============================================================================
# 2. Turning each feature into groups
# =============================================================================

def prepare_categorical(s: pd.Series, min_size: int) -> dict:
    """Groups = categories, with small ones merged into OTHER.
    Key = raw categories (used for scoring, so high-cardinality signal is not thrown away)."""
    codes, uniq = pd.factorize(s, use_na_sentinel=True)
    names = np.array([str(u) for u in uniq] + [MISSING], dtype=object)
    codes = np.where(codes < 0, len(uniq), codes)
    raw = names[codes]
    cnt = np.bincount(codes, minlength=len(names))
    counts = pd.Series(cnt, index=names)
    counts = counts[(counts > 0)].sort_values(ascending=False)
    rare = counts[(counts < min_size) & (counts.index != MISSING)]
    if len(rare):
        group_names = np.where(pd.Index(names).isin(rare.index), OTHER, names).astype(object)
        groups = group_names[codes]
    else:
        groups = raw

    n = len(raw)
    present = counts.drop(MISSING, errors="ignore")
    info = {
        "Distinct values": f"{len(present):,}",
        "Groups analysed": f"{len(np.unique(groups)):,}",
        "Small categories merged into (other)": f"{len(rare):,} ({_pct(rare.sum() / n, 1)} of transactions)",
        "Missing": _pct(counts.get(MISSING, 0) / n, 2),
        "Most common value": (f"{_esc(present.index[0])} ({_pct(present.iloc[0] / n, 1)})" if len(present) else "–"),
    }
    stats_ = {
        "n_distinct": int(len(present)),
        "missing_rate": float(counts.get(MISSING, 0) / n),
        "rare_volume_share": float(rare.sum() / n),
        "top_share": float(present.iloc[0] / n) if len(present) else 0.0,
        "top_value": str(present.index[0]) if len(present) else None,
    }
    return {"groups": groups, "key": raw, "order": None, "info": info, "stats": stats_, "edges": None}


def _fmt_num(v: float) -> str:
    if not np.isfinite(v):
        return str(v)
    a = abs(v)
    if a >= 1000:
        return f"{v:,.0f}"
    if a >= 1 or a == 0:
        return f"{v:,.2f}".rstrip("0").rstrip(".")
    return f"{v:.3g}"


def prepare_numeric(s: pd.Series, min_size: int) -> dict:
    """Groups = equal-volume bins, merged with neighbours until each is big enough."""
    x = pd.to_numeric(s, errors="coerce").astype(float).replace([np.inf, -np.inf], np.nan).to_numpy()
    miss = np.isnan(x)
    vals = x[~miss]
    n = len(x)
    groups = np.full(n, MISSING, dtype=object)
    edges: list = []
    order: list = []

    if len(vals):
        uniq = np.unique(vals)
        if len(uniq) <= N_BINS:
            edges = list(uniq[1:])                         # each distinct value its own bin
        else:
            q = np.quantile(vals, np.linspace(0, 1, N_BINS + 1))[1:-1]
            edges = [e for e in np.unique(q) if e > uniq[0]]
        idx = np.searchsorted(edges, vals, side="right")
        counts = list(np.bincount(idx, minlength=len(edges) + 1))

        while len(counts) > 1 and min(counts) < min_size:   # merge small bins into a neighbour
            i = int(np.argmin(counts))
            if i == 0:
                j = 1
            elif i == len(counts) - 1:
                j = i - 1
            else:
                j = i - 1 if counts[i - 1] <= counts[i + 1] else i + 1
            lo = min(i, j)
            counts[lo] += counts[lo + 1]
            del counts[lo + 1]
            del edges[lo]

        idx = np.searchsorted(edges, vals, side="right")
        b = pd.DataFrame({"b": idx, "v": vals}).groupby("b")["v"].agg(["min", "max"])
        labels = [f"= {_fmt_num(r['min'])}" if r["min"] == r["max"]
                  else f"{_fmt_num(r['min'])} to {_fmt_num(r['max'])}" for _, r in b.iterrows()]
        label_arr = np.array(labels, dtype=object)
        groups[~miss] = label_arr[idx]
        order = labels
    if miss.any():
        order = order + [MISSING]

    pcts = np.percentile(vals, [1, 25, 50, 75, 99]) if len(vals) else [np.nan] * 5
    info = {
        "Distinct values": f"{len(np.unique(vals)):,}",
        "Bins analysed": f"{len(order) - int(miss.any())} (+ missing)" if miss.any() else f"{len(order)}",
        "Missing": _pct(miss.mean(), 2),
        "Zero": _pct((vals == 0).sum() / n, 1) if len(vals) else "–",
        "Negative": _pct((vals < 0).sum() / n, 1) if len(vals) else "–",
        "Min / 1st pct / median / 99th pct / max": (
            " / ".join(_fmt_num(v) for v in [vals.min(), pcts[0], pcts[2], pcts[4], vals.max()]) if len(vals) else "–"),
    }
    vc = pd.Series(groups).value_counts(normalize=True)
    stats_ = {
        "n_distinct": int(len(np.unique(vals))),
        "missing_rate": float(miss.mean()),
        "rare_volume_share": 0.0,
        "top_share": float(vc.drop(MISSING, errors="ignore").max()) if len(vals) else 0.0,
        "top_value": str(vc.drop(MISSING, errors="ignore").idxmax()) if len(vals) else None,
    }
    return {"groups": groups, "key": groups, "order": order, "info": info, "stats": stats_,
            "edges": [float(e) for e in edges], "values": x}


# =============================================================================
# 3. Statistics helpers
# =============================================================================

def wilson_interval(k, n, z=1.96):
    k, n = np.asarray(k, float), np.asarray(n, float)
    p = k / n
    d = 1 + z**2 / n
    c = (p + z**2 / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / d
    return np.clip(c - h, 0, 1), np.clip(c + h, 0, 1)


def binomial_pvalues(k, n, p0):
    k, n = np.asarray(k), np.asarray(n)
    return np.minimum(1.0, 2 * np.minimum(stats.binom.cdf(k, n, p0), stats.binom.sf(k - 1, n, p0)))


def benjamini_hochberg(p):
    p = np.asarray(p, float)
    m = len(p)
    if m == 0:
        return p
    o = np.argsort(p)
    r = np.minimum.accumulate((p[o] * m / np.arange(1, m + 1))[::-1])[::-1]
    q = np.empty(m)
    q[o] = np.clip(r, 0, 1)
    return q


def psi_counts(ref_counts, cur_counts, eps=1e-4) -> float:
    """Population Stability Index between two count vectors over the same groups."""
    e = np.clip(ref_counts / max(ref_counts.sum(), 1), eps, None)
    a = np.clip(cur_counts / max(cur_counts.sum(), 1), eps, None)
    return float(((a - e) * np.log(a / e)).sum())


def information_value(frauds, genuine, smoothing=0.5) -> float:
    frauds, genuine = np.asarray(frauds, float), np.asarray(genuine, float)
    k = len(frauds)
    pf = (frauds + smoothing) / (frauds.sum() + smoothing * k)
    pg = (genuine + smoothing) / (genuine.sum() + smoothing * k)
    return float(((pf - pg) * np.log(pf / pg)).sum())


def auc_from_counts(f, g):
    """ROC-AUC from fraud / genuine counts per score level, levels sorted by ascending score.
    Works on 1-D arrays or 2-D (bootstrap replicates x levels)."""
    f, g = np.asarray(f, float), np.asarray(g, float)
    g_below = np.cumsum(g, axis=-1) - g
    return (f * (g_below + 0.5 * g)).sum(axis=-1) / (f.sum(axis=-1) * g.sum(axis=-1))


def counts_by_score(scores, y, weights=None):
    """Fraud and genuine counts (or weighted sums) per distinct score, ascending by score."""
    u, inv = np.unique(scores, return_inverse=True)
    w = np.ones(len(y)) if weights is None else weights
    f = np.bincount(inv, weights=w * y, minlength=len(u))
    g = np.bincount(inv, weights=w * (1 - y), minlength=len(u))
    return f, g


def ks_from_counts(f, g):
    tpr = np.cumsum(f[::-1]) / f.sum()
    fpr = np.cumsum(g[::-1]) / g.sum()
    return float(np.max(tpr - fpr))


def ap_from_counts(f, g):
    """Average precision (as in sklearn), from counts per score level."""
    tp, fp = np.cumsum(f[::-1]), np.cumsum(g[::-1])
    return float(np.sum((f[::-1] / f.sum()) * tp / (tp + fp)))


def gini_with_ci(scores, y, rng):
    """Gini with a stratified bootstrap 95% interval (fast: resamples counts, not rows)."""
    f, g = counts_by_score(scores, y)
    F, G = int(f.sum()), int(g.sum())
    point = 2 * auc_from_counts(f, g) - 1
    boots = []
    for start in range(0, N_BOOTSTRAP, 100):
        b = min(100, N_BOOTSTRAP - start)
        fb = rng.multinomial(F, f / F, size=b)
        gb = rng.multinomial(G, g / G, size=b)
        boots.append(2 * auc_from_counts(fb, gb) - 1)
    boots = np.concatenate(boots)
    return float(point), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)), float(boots.std())


def target_encode(train_codes, train_y, apply_codes, n_codes, smoothing=ENCODING_SMOOTHING):
    """Smoothed fraud rate per group, learned on train, applied to apply_codes.
    Codes are integers from pd.factorize; groups unseen in train get the train fraud rate."""
    prior = float(np.mean(train_y))
    frauds = np.bincount(train_codes, weights=train_y, minlength=n_codes)
    counts = np.bincount(train_codes, minlength=n_codes)
    enc = (frauds + smoothing * prior) / (counts + smoothing)
    return enc[apply_codes]


def oof_target_encode(codes, y, n_codes):
    scores = np.empty(len(y))
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    for tr, te in skf.split(np.zeros(len(y)), y):
        scores[te] = target_encode(codes[tr], y[tr], codes[te], n_codes)
    return scores


def band_label(v, bands):
    for upper, label in bands:
        if v < upper:
            return label
    return bands[-1][1]


IV_BANDS = [(0.02, "Useless"), (0.10, "Weak"), (0.30, "Medium"), (0.50, "Strong"), (np.inf, "Suspicious")]
PSI_BANDS = [(PSI_MODERATE, "Stable"), (PSI_MAJOR, "Moderate shift"), (np.inf, "Major shift")]


def gini_verdict(g):
    if g is None or not np.isfinite(g):
        return "Not enough data"
    for label in ("Strong", "Moderate", "Weak"):
        if g >= GINI_BANDS[label]:
            return label
    return "No useful signal"


# =============================================================================
# 4. Static analysis (whole period)
# =============================================================================

def group_table(groups, y, amount, order, base_rate) -> pd.DataFrame:
    d = pd.DataFrame({"g": groups, "y": y})
    agg = {"n": ("y", "size"), "frauds": ("y", "sum")}
    if amount is not None:
        d["amt"], d["famt"] = amount, amount * y
        agg.update(amount=("amt", "sum"), fraud_amount=("famt", "sum"))
    t = d.groupby("g", sort=False).agg(**agg)

    N, F = len(y), int(y.sum())
    t["genuine"] = t["n"] - t["frauds"]
    t["volume_share"] = t["n"] / N
    t["fraud_rate"] = t["frauds"] / t["n"]
    t["ci_low"], t["ci_high"] = wilson_interval(t["frauds"], t["n"])
    t["lift"] = t["fraud_rate"] / base_rate
    t["fraud_share"] = t["frauds"] / F
    k = len(t)
    pf = (t["frauds"] + 0.5) / (F + 0.5 * k)
    pg = (t["genuine"] + 0.5) / (N - F + 0.5 * k)
    t["woe"] = np.log(pf / pg)
    t["iv_part"] = (pf - pg) * t["woe"]
    if amount is not None:
        tot = t["fraud_amount"].sum()
        t["value_fraud_rate"] = np.where(t["amount"] > 0, t["fraud_amount"] / t["amount"].where(t["amount"] > 0), np.nan)
        t["fraud_value_share"] = t["fraud_amount"] / tot if tot > 0 else np.nan
    t["p_value"] = binomial_pvalues(t["frauds"].to_numpy(), t["n"].to_numpy(), base_rate)
    t["q_value"] = benjamini_hochberg(t["p_value"].to_numpy())
    expected = t["n"] * base_rate

    def sig(r, e):
        if e < 5 and r["frauds"] < 5:
            return "Too few frauds"
        if r["q_value"] < FDR_ALPHA:
            return "Higher risk" if r["fraud_rate"] > base_rate else "Lower risk"
        return "Not significant"

    t["signal"] = [sig(r, e) for (_, r), e in zip(t.iterrows(), expected)]
    if order is not None:
        t = t.reindex([o for o in order if o in t.index])
    else:
        t = t.sort_values("fraud_rate", ascending=False)
    t.index.name = "group"
    return t


def association(t: pd.DataFrame, N: int, base_rate: float) -> dict:
    out = {"iv": float(t["iv_part"].sum())}
    out["iv_band"] = band_label(out["iv"], IV_BANDS)
    if len(t) < 2:
        out.update(cramers_v=np.nan, uncertainty=np.nan)
        return out
    cont = t[["frauds", "genuine"]].to_numpy(float)
    chi2 = stats.chi2_contingency(cont, correction=False)[0]
    out["cramers_v"] = float(np.sqrt(chi2 / N))
    joint = cont / N
    pc, py = joint.sum(1, keepdims=True), joint.sum(0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        mi = float(np.where(joint > 0, joint * np.log2(joint / (pc * py)), 0).sum())
    hy = -(base_rate * np.log2(base_rate) + (1 - base_rate) * np.log2(1 - base_rate))
    out["uncertainty"] = mi / hy
    return out


def ranking_metrics(scores, y, amount) -> dict:
    """Ranking power of a score. Scores here have few distinct values, so everything is
    computed from counts per score level (fast, and handles ties exactly)."""
    f, g = counts_by_score(scores, y)
    if len(f) < 2:
        out = {"roc_auc": 0.5, "pr_auc": float(y.mean()), "ks": 0.0}
    else:
        out = {"roc_auc": float(auc_from_counts(f, g)), "pr_auc": ap_from_counts(f, g), "ks": ks_from_counts(f, g)}
    out["gini"] = 2 * out["roc_auc"] - 1
    out["pr_auc_lift"] = out["pr_auc"] / y.mean()
    if amount is not None and len(f) > 1 and amount.sum() > 0:
        fw, gw = counts_by_score(scores, y, amount)
        out["value_gini"] = 2 * float(auc_from_counts(fw, gw)) - 1
    else:
        out["value_gini"] = np.nan
    return out


def numeric_extras(x: np.ndarray, y: np.ndarray, t: pd.DataFrame) -> dict:
    """Direction, shape of the risk curve, and fraud-vs-genuine distributions."""
    m = ~np.isnan(x)
    xv, yv = x[m], y[m]
    out = {}
    if len(np.unique(xv)) > 1 and 0 < yv.sum() < len(yv):
        auc = roc_auc_score(yv, xv)
        out["raw_gini"] = abs(2 * auc - 1)
        out["direction"] = "Higher values are riskier" if auc >= 0.5 else "Lower values are riskier"
        ks = stats.ks_2samp(xv[yv == 1], xv[yv == 0])
        out["ks_stat"] = float(ks.statistic)
    else:
        out.update(raw_gini=np.nan, direction="–", ks_stat=np.nan)
    out["fraud_median"] = float(np.median(xv[yv == 1])) if yv.sum() else np.nan
    out["genuine_median"] = float(np.median(xv[yv == 0])) if (yv == 0).sum() else np.nan
    out["fraud_mean"] = float(xv[yv == 1].mean()) if yv.sum() else np.nan
    out["genuine_mean"] = float(xv[yv == 0].mean()) if (yv == 0).sum() else np.nan

    rates = t.drop(MISSING, errors="ignore")["fraud_rate"].to_numpy()
    if len(rates) >= 3:
        rho = stats.spearmanr(np.arange(len(rates)), rates)[0]
        changes = int((np.diff(np.sign(np.diff(rates))) != 0).sum())
        lo = int(np.argmin(rates))
        hi = int(np.argmax(rates))
        if rho >= 0.8:
            shape = "Rising" + (" steadily" if changes == 0 else ", with some noise")
        elif rho <= -0.8:
            shape = "Falling" + (" steadily" if changes == 0 else ", with some noise")
        elif 0 < lo < len(rates) - 1 and min(rates[0], rates[-1]) > 1.5 * rates[lo]:
            shape = "U-shaped: both extremes are riskier than the middle"
        elif 0 < hi < len(rates) - 1 and max(rates[0], rates[-1]) < rates[hi] / 1.5:
            shape = "Peaked: the middle of the range is riskiest"
        else:
            shape = "Irregular"
        out.update(shape=shape, shape_rho=float(rho))
    else:
        out.update(shape="Too few bins to judge", shape_rho=np.nan)
    return out


# =============================================================================
# 5. Change over time
# =============================================================================

def temporal_analysis(group_codes, group_names, key_codes, n_keys, y, pidx, period_labels,
                      overall: pd.DataFrame, base_rate: float, is_categorical: bool, rng) -> pd.DataFrame:
    """One row per period: walk-forward Gini (with 95% CI), within-period Gini, IV, PSI,
    rank consistency of risky groups, and share of unseen categories."""
    n_groups = len(group_names)
    ref_counts = np.bincount(group_codes[pidx == 0], minlength=n_groups)
    sig = overall[overall["signal"].isin(["Higher risk", "Lower risk"]) & (overall.index != OTHER)]["fraud_rate"]
    sig_idx = {name: i for i, name in enumerate(group_names) if name in sig.index}

    rows = []
    for t, label in enumerate(period_labels):
        m = pidx == t
        y_t = y[m]
        gc = group_codes[m]
        n_t, f_t = int(m.sum()), int(y_t.sum())
        cnt = np.bincount(gc, minlength=n_groups)
        row = {"period": label, "n": n_t, "frauds": f_t, "fraud_rate": f_t / n_t if n_t else np.nan,
               "psi": psi_counts(ref_counts, cnt) if n_t else np.nan,
               "iv": np.nan, "in_period_gini": np.nan, "rank_corr": np.nan,
               "wf_gini": np.nan, "wf_low": np.nan, "wf_high": np.nan, "wf_se": np.nan,
               "wf_pr_lift": np.nan, "unseen_share": np.nan}

        if 0 < f_t < n_t:
            fr = np.bincount(gc, weights=y_t, minlength=n_groups)
            present = cnt > 0
            row["iv"] = information_value(fr[present], (cnt - fr)[present])
            rates = np.divide(fr, cnt, out=np.zeros(n_groups), where=present)
            f, g = counts_by_score(rates[gc], y_t)
            row["in_period_gini"] = float(2 * auc_from_counts(f, g) - 1) if len(f) > 1 else 0.0
            names = [n for n, i in sig_idx.items() if cnt[i] * base_rate >= 5]
            if len(names) >= 3:
                row["rank_corr"] = float(stats.spearmanr([rates[sig_idx[n]] for n in names],
                                                         sig[names].to_numpy())[0])

        if t == 0:
            row["status"] = "No earlier data"
        else:
            train = pidx < t
            if WALK_FORWARD_WINDOW:
                train &= pidx >= t - WALK_FORWARD_WINDOW
            if f_t < MIN_FRAUDS_PER_PERIOD:
                row["status"] = f"Too few frauds (<{MIN_FRAUDS_PER_PERIOD})"
            elif y[train].sum() == 0:
                row["status"] = "No frauds in earlier data"
            else:
                kc_train, kc_test = key_codes[train], key_codes[m]
                scores = target_encode(kc_train, y[train], kc_test, n_keys)
                if len(np.unique(scores)) < 2:
                    row.update(wf_gini=0.0, wf_low=0.0, wf_high=0.0, wf_se=0.0, wf_pr_lift=1.0)
                else:
                    gi, lo, hi, se = gini_with_ci(scores, y_t, rng)
                    row.update(wf_gini=gi, wf_low=lo, wf_high=hi, wf_se=se,
                               wf_pr_lift=ap_from_counts(*counts_by_score(scores, y_t)) / y_t.mean())
                if is_categorical:
                    seen = np.bincount(kc_train, minlength=n_keys) > 0
                    row["unseen_share"] = float((~seen[kc_test]).mean())
                row["status"] = "Scored"
        rows.append(row)
    return pd.DataFrame(rows)


def trend_statistics(tdf: pd.DataFrame) -> dict:
    """Tests on the walk-forward Gini series.
    Trend: weighted regression of Gini on time (weights = 1/bootstrap variance), with the slope's
    uncertainty inflated when periods vary more than sampling noise allows.
    Volatility: do periods differ more than their intervals explain? (heterogeneity test), plus
    the size of that excess variation (between-period SD, 'tau')."""
    v = tdf[tdf["status"] == "Scored"]
    out = {"change": np.nan, "p_trend": np.nan, "p_volatility": np.nan, "tau": np.nan,
           "mean_wf_gini": np.nan, "n_scored": len(v)}
    if len(v):
        out["mean_wf_gini"] = float(np.average(v["wf_gini"], weights=v["frauds"]))
    if len(v) < 3:
        return out
    t = v.index.to_numpy(float)
    g = v["wf_gini"].to_numpy(float)
    w = 1 / np.clip(v["wf_se"].to_numpy(float), 0.005, None) ** 2
    k = len(g)
    gbar = np.sum(w * g) / w.sum()
    q = float(np.sum(w * (g - gbar) ** 2))
    out["p_volatility"] = float(stats.chi2.sf(q, k - 1))
    out["tau"] = float(math.sqrt(max(0.0, (q - (k - 1)) / (w.sum() - np.sum(w**2) / w.sum()))))

    tbar = np.sum(w * t) / w.sum()
    sxx = np.sum(w * (t - tbar) ** 2)
    slope = np.sum(w * (t - tbar) * (g - gbar)) / sxx
    resid = g - (gbar + slope * (t - tbar))
    phi = max(1.0, np.sum(w * resid**2) / (k - 2))
    out["p_trend"] = float(2 * stats.norm.sf(abs(slope / math.sqrt(phi / sxx))))
    out["change"] = float(slope * (t.max() - t.min()))
    return out


def assign_trend_labels(trends: dict) -> None:
    """Label each feature Stable / Decaying / Emerging / Volatile.
    p-values are corrected across features (Benjamini-Hochberg): testing many features at 5%
    would otherwise flag some by chance. A trend must also be large enough to matter."""
    feats = [f for f, tr in trends.items() if np.isfinite(tr["p_trend"])]
    q_tr = dict(zip(feats, benjamini_hochberg([trends[f]["p_trend"] for f in feats])))
    q_vol = dict(zip(feats, benjamini_hochberg([trends[f]["p_volatility"] for f in feats])))
    for f, tr in trends.items():
        tr["q_trend"], tr["q_volatility"] = q_tr.get(f, np.nan), q_vol.get(f, np.nan)
        if f not in q_tr:
            tr["label"] = "Not enough periods"
            continue
        if tr["q_trend"] < TREND_ALPHA and abs(tr["change"]) >= MIN_MEANINGFUL_GINI_CHANGE:
            label = "Decaying" if tr["change"] < 0 else "Emerging"
        elif tr["q_volatility"] < TREND_ALPHA and tr["tau"] >= MIN_MEANINGFUL_GINI_CHANGE / 2:
            label = "Volatile"
        else:
            label = "Stable"
        if label == "Stable" and tr["mean_wf_gini"] < GINI_BANDS["Weak"]:
            label = "Stable (no signal)"
        tr["label"] = label


def temporal_story(tdf: pd.DataFrame, trend: dict, is_categorical: bool) -> list[str]:
    """Plain-English reading of the time analysis."""
    what = "categories" if is_categorical else "value ranges"
    v = tdf[tdf["status"] == "Scored"]
    s = []
    lab = trend["label"]
    if lab == "Not enough periods":
        s.append(f"Only {len(v)} period(s) could be scored, so no trend can be judged. "
                 f"Use a shorter PERIOD or a longer history.")
    elif lab == "Decaying":
        ip = v["in_period_gini"].to_numpy()
        ip_change = np.polyfit(v.index.to_numpy(float), ip, 1)[0] * (v.index.max() - v.index.min()) if len(v) > 1 else 0
        s.append(f"Predictive power is falling: walk-forward Gini changed by {trend['change']:+.3f} "
                 f"from {v['period'].iloc[0]} to {v['period'].iloc[-1]} (trend p = {trend['p_trend']:.3g}, "
                 f"{trend['q_trend']:.3g} after correcting for the number of features).")
        if ip_change <= -MIN_MEANINGFUL_GINI_CHANGE / 2:
            s.append("The signal is weakening within each period too, so the feature itself is losing relevance.")
        else:
            s.append(f"Within each period the feature still separates fraud about as well as before, but what was "
                     f"learned from earlier periods transfers less well. The risky {what} have moved, which is "
                     f"typical when fraudsters adapt. The fraud-rate-by-period table shows where.")
    elif lab == "Emerging":
        s.append(f"Predictive power is rising: walk-forward Gini changed by {trend['change']:+.3f} "
                 f"(trend p = {trend['p_trend']:.3g}). Analyses or models built on older data understate "
                 f"this feature; give recent data more weight.")
    elif lab == "Volatile":
        lo = v.loc[v["wf_gini"].idxmin()]
        hi = v.loc[v["wf_gini"].idxmax()]
        s.append(f"Predictive power swings more than chance explains (p = {trend['p_volatility']:.3g}; "
                 f"typical swing ±{trend['tau']:.3f} beyond noise), "
                 f"from {lo['wf_gini']:.3f} in {lo['period']} to {hi['wf_gini']:.3f} in {hi['period']}, "
                 f"with no steady direction. Check what changed in those periods (campaigns, attacks, "
                 f"system or data changes).")
    elif lab == "Stable (no signal)":
        s.append("Consistently weak across periods.")
    else:
        s.append(f"Predictive power has been consistent across periods (average walk-forward Gini "
                 f"{trend['mean_wf_gini']:.3f}); differences between periods are within sampling noise.")

    max_psi = tdf["psi"].max()
    if np.isfinite(max_psi) and max_psi >= PSI_MODERATE:
        worst = tdf.loc[tdf["psi"].idxmax(), "period"]
        s.append(f"The mix of {what} has shifted ({band_label(max_psi, PSI_BANDS).lower()}, PSI {max_psi:.2f} "
                 f"in {worst} vs {tdf['period'].iloc[0]}). A mix shift alone does not reduce predictive power; "
                 f"compare with the Gini trend above.")
    rc = tdf["rank_corr"].dropna()
    if len(rc) and rc.min() < 0.5:
        s.append(f"In some periods the ordering of risky and safe {what} differs from the overall picture "
                 f"(lowest rank correlation {rc.min():.2f} in {tdf.loc[rc.idxmin(), 'period']}).")
    if is_categorical:
        last = tdf["unseen_share"].dropna()
        if len(last) and last.iloc[-1] > 0.05:
            s.append(f"{_pct(last.iloc[-1], 1)} of transactions in the latest period have a category never seen "
                     f"before; these get the average score.")
    skipped = (tdf["status"].str.startswith("Too few")).sum()
    if skipped:
        s.append(f"{skipped} period(s) were not scored because they had fewer than {MIN_FRAUDS_PER_PERIOD} frauds.")
    return s


# =============================================================================
# 6. Diagnosis
# =============================================================================

def diagnose(prep, t, assoc, oof, insample_gini, trend, tdf, n_frauds, is_categorical):
    warns, notes = [], []
    st = prep["stats"]

    if assoc["iv"] >= IV_LEAKAGE_THRESHOLD:
        warns.append(f"Information Value {assoc['iv']:.2f} is unusually high for a single fraud feature. "
                     "Confirm the value is known at authorisation time and is not written by the "
                     "investigation, chargeback or case-management process.")
    if MISSING in t.index:
        r = t.loc[MISSING]
        if r["signal"] == "Higher risk" and r["fraud_share"] > 0.2:
            warns.append(f"Missing values hold {_pct(r['fraud_share'], 0)} of all fraud at {r['lift']:.1f}x the "
                         "average rate. Find out why the field is missing; if it is filled in after "
                         "investigation, this is leakage.")
    if (insample_gini - oof["gini"]) / 2 > OPTIMISM_WARN:
        warns.append(f"Gini fitted on the same labels ({insample_gini:.3f}) is far above the out-of-fold Gini "
                     f"({oof['gini']:.3f}): many small categories let the encoding memorise labels. Trust the "
                     "out-of-fold figure, and group or smooth this feature before modelling.")
    if st["missing_rate"] > 0.5:
        warns.append(f"{_pct(st['missing_rate'], 0)} of values are missing.")
    if st["top_share"] > 0.95:
        warns.append(f"One group ({_esc(st['top_value'])}) holds {_pct(st['top_share'], 0)} of transactions, "
                     "leaving little room to separate fraud.")
    if st["n_distinct"] <= 1:
        warns.append("The feature has a single value.")
    if is_categorical:
        if st["rare_volume_share"] > 0.2:
            warns.append(f"{_pct(st['rare_volume_share'], 0)} of transactions fall in small categories merged into "
                         "(other), so the group table says little about them. Use a coarser hierarchy "
                         "(e.g. merchant → MCC → MCC group) or lower MIN_GROUP_PCT.")
        if st["n_distinct"] > 1000:
            notes.append(f"{st['n_distinct']:,} distinct values: expect unseen values in production and plan "
                         "an encoding that handles them.")

    if trend["label"] == "Decaying":
        warns.append(f"Predictive power is decaying over time (Gini change {trend['change']:+.3f}).")
    elif trend["label"] == "Volatile":
        warns.append("Predictive power swings between periods more than chance explains.")
    max_psi = tdf["psi"].max()
    if np.isfinite(max_psi) and max_psi >= PSI_MAJOR:
        warns.append(f"Major shift in the mix of values over time (PSI {max_psi:.2f}).")
    if n_frauds < 100:
        warns.append(f"Only {n_frauds} frauds in total: all figures are noisy.")

    ref =trend["mean_wf_gini"] if np.isfinite(trend["mean_wf_gini"]) else oof["gini"]
    verdict = gini_verdict(ref)
    if verdict not in ("No useful signal", "Not enough data"):
        if any("leakage" in w or "authorisation time" in w for w in warns):
            verdict += " – check for leakage"
        elif trend["label"] in ("Decaying", "Volatile"):
            verdict += f" – {trend['label'].lower()}"
    return warns, notes, verdict


# =============================================================================
# 7. Charts (matplotlib → embedded PNG)
# =============================================================================

plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": C_GRID, "axes.labelcolor": C_INK2, "xtick.color": C_INK2,
    "ytick.color": C_INK2, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": C_GRID, "grid.linewidth": 0.6, "axes.axisbelow": True, "figure.facecolor": C_SURFACE,
    "axes.facecolor": C_SURFACE, "legend.frameon": False, "axes.titlesize": 10, "axes.titlecolor": C_INK,
})


def _img(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight", facecolor=C_SURFACE)
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


SIGNAL_COLOURS = {"Higher risk": C_RED, "Lower risk": C_BLUE}


def groups_to_show(t: pd.DataFrame) -> list:
    """For categorical features with many groups: every significantly risky or safe group first,
    then the largest remaining groups, up to MAX_GROUPS_SHOWN. Returned riskiest first."""
    if len(t) <= MAX_GROUPS_SHOWN:
        return list(t.sort_values("fraud_rate", ascending=False).index)
    sig = t[t["signal"].isin(["Higher risk", "Lower risk"])].sort_values("q_value")
    keep = list(sig.index[:MAX_GROUPS_SHOWN])
    rest = t.drop(index=keep).sort_values("n", ascending=False)
    keep += list(rest.index[:MAX_GROUPS_SHOWN - len(keep)])
    return list(t.loc[keep].sort_values("fraud_rate", ascending=False).index)


def chart_groups(t: pd.DataFrame, ordered: bool, base_rate: float, feature: str) -> str:
    """Fraud rate per group with 95% CI (left/top) and volume share (right/bottom). No dual axis."""
    t = t.loc[groups_to_show(t)] if not ordered else t
    labels = [str(i) if len(str(i)) <= 28 else str(i)[:26] + "…" for i in t.index]
    cols = [SIGNAL_COLOURS.get(s, C_GREY) for s in t["signal"]]
    rate, lo, hi = t["fraud_rate"].to_numpy(), t["ci_low"].to_numpy(), t["ci_high"].to_numpy()
    err = np.vstack([rate - lo, hi - rate])

    if ordered:
        fig, (a1, a2) = plt.subplots(2, 1, sharex=True, figsize=(8.4, 4.8),
                                     gridspec_kw={"height_ratios": [3, 1.1], "hspace": 0.08})
        x = np.arange(len(t))
        has_missing = MISSING in t.index
        line_n = len(t) - int(has_missing)
        a1.plot(x[:line_n], rate[:line_n], color=C_GREY, lw=1.2, zorder=1)
        for i in range(len(t)):
            a1.errorbar(x[i], rate[i], yerr=err[:, i:i + 1], fmt="o", ms=6, color=cols[i],
                        elinewidth=1.4, capsize=0, zorder=2)
        a1.axhline(base_rate, ls="--", lw=1, color=C_INK2)
        a1.text(x[-1] + 0.4, base_rate, "average", va="bottom", ha="right", color=C_INK2, fontsize=8)
        a1.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=1))
        a1.set_ylabel("Fraud rate (95% CI)")
        a1.set_title(f"{feature}: fraud rate by bin")
        a2.bar(x, t["volume_share"], color="#b7d3f6", width=0.72)
        a2.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
        a2.set_ylabel("% of txns")
        a2.set_xticks(x)
        a2.set_xticklabels(labels, rotation=35, ha="right")
        a2.grid(axis="x", visible=False)
        a1.grid(axis="x", visible=False)
    else:
        k = len(t)
        fig, (a1, a2) = plt.subplots(1, 2, sharey=True, figsize=(8.4, 0.3 * k + 1.3),
                                     gridspec_kw={"width_ratios": [3, 1.1], "wspace": 0.06})
        yy = np.arange(k)
        for i in range(k):
            a1.errorbar(rate[i], yy[i], xerr=err[:, i:i + 1], fmt="o", ms=6, color=cols[i],
                        elinewidth=1.4, capsize=0)
        a1.axvline(base_rate, ls="--", lw=1, color=C_INK2)
        a1.set_yticks(yy)
        a1.set_yticklabels(labels)
        a1.invert_yaxis()
        a1.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=1))
        a1.set_xlabel("Fraud rate (95% CI); dashed = average")
        a1.set_title(f"{feature}: fraud rate by category", loc="left")
        a1.grid(axis="y", visible=False)
        a2.barh(yy, t["volume_share"], color="#b7d3f6", height=0.66)
        a2.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
        a2.set_xlabel("% of txns")
        a2.grid(axis="y", visible=False)

    from matplotlib.lines import Line2D
    handles = [Line2D([], [], marker="o", ls="", color=c, label=l) for l, c in
               [("Significantly riskier", C_RED), ("Significantly safer", C_BLUE), ("Not significant", C_GREY)]]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=3, fontsize=8)
    return _img(fig)


def chart_trend(tdf: pd.DataFrame, feature: str) -> str:
    fig, ax = plt.subplots(figsize=(8.4, 3.0))
    x = np.arange(len(tdf))
    v = tdf["status"] == "Scored"
    ax.fill_between(x[v], tdf.loc[v, "wf_low"], tdf.loc[v, "wf_high"], color=C_BLUE, alpha=0.16, lw=0)
    ax.plot(x[v], tdf.loc[v, "wf_gini"], "o-", color=C_BLUE, lw=2, ms=5, label="Walk-forward Gini (95% CI)")
    ax.plot(x, tdf["in_period_gini"], "s--", color=C_ORANGE, lw=1.4, ms=4, label="Within-period Gini (in-sample)")
    ax.axhline(0, color=C_INK2, lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(tdf["period"], rotation=0 if len(x) <= 12 else 45)
    ax.set_ylabel("Gini")
    ax.set_title(f"{feature}: predictive power over time", loc="left")
    ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), fontsize=8)
    ax.grid(axis="x", visible=False)
    return _img(fig)


def chart_small_multiples(temporal: dict, trends: dict, order: list, own_scale: Sequence[str] = ()) -> str:
    """One panel per feature on a common scale, so heights compare across panels. Features in
    own_scale (e.g. suspected leakage) get their own scale so they do not flatten the others."""
    n = len(order)
    ncol = min(4, n)
    nrow = math.ceil(n / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.6 * ncol, 1.9 * nrow), squeeze=False)
    def top(f):
        v = temporal[f]["wf_high"].to_numpy(float)
        return np.nanmax(v) if np.isfinite(v).any() else 0.0
    def bottom(f):
        v = temporal[f]["wf_low"].to_numpy(float)
        return np.nanmin(v) if np.isfinite(v).any() else 0.0
    common = [f for f in order if f not in own_scale] or order
    ymax = max([top(f) for f in common] + [0.1]) * 1.05
    ymin = min([bottom(f) for f in common] + [-0.05]) * 1.05
    for ax, f in zip(axes.flat, order):
        tdf = temporal[f]
        v = tdf["status"] == "Scored"
        x = np.arange(len(tdf))
        ax.fill_between(x[v], tdf.loc[v, "wf_low"], tdf.loc[v, "wf_high"], color=C_BLUE, alpha=0.16, lw=0)
        ax.plot(x[v], tdf.loc[v, "wf_gini"], "o-", color=C_BLUE, lw=1.8, ms=3)
        ax.axhline(0, color=C_INK2, lw=0.6)
        own = f in own_scale
        ax.set_ylim((min(-0.05, bottom(f) * 1.05), max(0.1, top(f) * 1.05)) if own else (ymin, ymax))
        ax.set_xticks([0, len(tdf) - 1])
        ax.set_xticklabels([tdf["period"].iloc[0], tdf["period"].iloc[-1]], fontsize=7)
        name = f if len(f) <= 22 else f[:20] + "…"
        ax.set_title(f"{name}\n{trends[f]['label']}" + ("  (own scale)" if own else ""), fontsize=8.5, loc="left")
        if f not in common:
            ax.set_facecolor("#f6f1f1")
        ax.grid(axis="x", visible=False)
    for ax in list(axes.flat)[n:]:
        ax.axis("off")
    fig.tight_layout()
    return _img(fig)


# =============================================================================
# 8. HTML helpers
# =============================================================================

def _esc(s) -> str:
    return _html.escape(str(s))


def _pct(x, dp=2) -> str:
    try:
        return "–" if x is None or not np.isfinite(x) else f"{x * 100:.{dp}f}%"
    except TypeError:
        return "–"


def _f(x, dp=3) -> str:
    try:
        return "–" if x is None or not np.isfinite(x) else f"{x:.{dp}f}"
    except TypeError:
        return "–"


def _hex_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def _mix(c1, c2, t):
    a, b = _hex_rgb(c1), _hex_rgb(c2)
    return "#" + "".join(f"{round(a[i] + (b[i] - a[i]) * t):02x}" for i in range(3))


def _ramp(t, ramp):
    t = float(np.clip(t, 0, 1)) * (len(ramp) - 1)
    i = min(int(t), len(ramp) - 2)
    return _mix(ramp[i], ramp[i + 1], t - i)


def _ink_for(bg):
    r, g, b = _hex_rgb(bg)
    return "#ffffff" if (0.299 * r + 0.587 * g + 0.114 * b) < 140 else C_INK


def gini_cell(v, title="", vmax=0.5):
    if v is None or not np.isfinite(v):
        return f'<td class="hm na" title="{_esc(title)}">–</td>'
    bg = _ramp(max(v, 0) / vmax, BLUE_RAMP)
    return f'<td class="hm" style="background:{bg};color:{_ink_for(bg)}" title="{_esc(title)}">{v:.3f}</td>'


def lift_cell(lift, text, title="", faded=False):
    if lift is None or not np.isfinite(lift):
        return f'<td class="hm na" title="{_esc(title)}">–</td>'
    z = float(np.clip(np.log2(max(lift, 1e-6)) / 2, -1, 1))
    if z >= 0:
        bg = _mix(C_MID, C_RED, min(z / 0.6, 1)) if z <= 0.6 else _mix(C_RED, "#9e2a2a", (z - 0.6) / 0.4)
    else:
        z = -z
        bg = _mix(C_MID, C_BLUE, min(z / 0.6, 1)) if z <= 0.6 else _mix(C_BLUE, "#184f95", (z - 0.6) / 0.4)
    cls = "hm faded" if faded else "hm"
    return f'<td class="{cls}" style="background:{bg};color:{_ink_for(bg)}" title="{_esc(title)}">{text}</td>'


TREND_CLASS = {"Stable": "ok", "Emerging": "info", "Decaying": "bad", "Volatile": "warn",
               "Stable (no signal)": "muted", "Not enough periods": "muted"}


def verdict_class(v: str) -> str:
    if "leakage" in v:
        return "bad"
    if v.startswith("Strong"):
        return "strong"
    if v.startswith("Moderate"):
        return "mod"
    if v.startswith("Weak"):
        return "weak"
    return "muted"


def group_table_html(t: pd.DataFrame, has_amount: bool, ordered: bool) -> str:
    shown = t if ordered else t.loc[groups_to_show(t)]
    head = ["Group", "Txns", "% of txns", "Frauds", "Fraud rate", "95% CI", "Lift", "WoE", "% of fraud"]
    if has_amount:
        head += ["Value fraud rate", "% of fraud value"]
    head += ["Signal"]
    rows = []
    for g, r in shown.iterrows():
        cells = [f"<td class='l'>{_esc(g)}</td>", f"<td>{int(r['n']):,}</td>", f"<td>{_pct(r['volume_share'], 1)}</td>",
                 f"<td>{int(r['frauds']):,}</td>"]
        cells.append(lift_cell(r["lift"], _pct(r["fraud_rate"], 3), f"lift {r['lift']:.2f}x"))
        cells += [f"<td class='mutedtxt'>{_pct(r['ci_low'], 3)} – {_pct(r['ci_high'], 3)}</td>",
                  f"<td>{r['lift']:.2f}x</td>", f"<td>{r['woe']:+.2f}</td>", f"<td>{_pct(r['fraud_share'], 1)}</td>"]
        if has_amount:
            cells += [f"<td>{_pct(r.get('value_fraud_rate'), 3)}</td>", f"<td>{_pct(r.get('fraud_value_share'), 1)}</td>"]
        cls = {"Higher risk": "sig-hi", "Lower risk": "sig-lo"}.get(r["signal"], "sig-no")
        cells.append(f"<td class='l'><span class='{cls}'>{_esc(r['signal'])}</span></td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    more = ""
    if len(t) > len(shown):
        more = (f"<p class='small'>Showing {len(shown)} of {len(t)} groups: all significantly risky or safe "
                f"ones, then the largest. "
                f"All groups are in <code>results['details']</code>.</p>")
    return ("<div class='tw'><table><thead><tr>" + "".join(f"<th>{h}</th>" for h in head) +
            "</tr></thead><tbody>" + "".join(rows) + "</tbody></table></div>" + more)


def drilldown_html(groups, y, pidx, period_labels, t: pd.DataFrame, base_rate, ordered: bool) -> str:
    """Fraud rate per group per period, coloured by lift vs the overall average."""
    d = pd.DataFrame({"g": groups, "y": y, "p": pidx})
    piv = d.groupby(["g", "p"])["y"].agg(["sum", "size"])
    shown = list(t.index) if ordered else groups_to_show(t)
    head = "<th class='l'>Group</th>" + "".join(f"<th>{_esc(p)}</th>" for p in period_labels)
    rows = []
    for g in shown:
        cells = [f"<td class='l'>{_esc(g)}</td>"]
        for p in range(len(period_labels)):
            if (g, p) in piv.index:
                s, n = piv.loc[(g, p)]
                rate = s / n
                faded = n * base_rate < 5
                tip = f"{int(s)} frauds / {int(n):,} txns" + (" – too few to judge" if faded else "")
                cells.append(lift_cell(rate / base_rate, _pct(rate, 2), tip, faded))
            else:
                cells.append("<td class='hm na'>–</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return ("<div class='tw'><table class='heat'><thead><tr>" + head + "</tr></thead><tbody>" +
            "".join(rows) + "</tbody></table></div>")


def period_table_html(tdf: pd.DataFrame, is_categorical: bool) -> str:
    head = ["Period", "Txns", "Frauds", "Fraud rate", "Walk-forward Gini", "95% CI", "PR-AUC lift",
            "Within-period Gini", "IV", "PSI vs first", "Rank corr."]
    if is_categorical:
        head.append("Unseen categories")
    head.append("Status")
    rows = []
    for _, r in tdf.iterrows():
        c = [f"<td class='l'>{_esc(r['period'])}</td>", f"<td>{int(r['n']):,}</td>", f"<td>{int(r['frauds']):,}</td>",
             f"<td>{_pct(r['fraud_rate'], 3)}</td>", gini_cell(r["wf_gini"]),
             f"<td class='mutedtxt'>{_f(r['wf_low'])} – {_f(r['wf_high'])}</td>" if np.isfinite(r["wf_low"]) else "<td>–</td>",
             f"<td>{_f(r['wf_pr_lift'], 2)}{'x' if np.isfinite(r['wf_pr_lift']) else ''}</td>",
             f"<td>{_f(r['in_period_gini'])}</td>", f"<td>{_f(r['iv'])}</td>",
             f"<td>{_f(r['psi'])}</td>", f"<td>{_f(r['rank_corr'], 2)}</td>"]
        if is_categorical:
            c.append(f"<td>{_pct(r['unseen_share'], 1)}</td>")
        c.append(f"<td class='l mutedtxt'>{_esc(r['status'])}</td>")
        rows.append("<tr>" + "".join(c) + "</tr>")
    return ("<div class='tw'><table><thead><tr>" + "".join(f"<th>{h}</th>" for h in head) +
            "</tr></thead><tbody>" + "".join(rows) + "</tbody></table></div>")


def kv_html(d: dict) -> str:
    return "<table class='kv'>" + "".join(f"<tr><th>{_esc(k)}</th><td>{v}</td></tr>" for k, v in d.items()) + "</table>"


CSS = """
:root{color-scheme:light;--bg:#f6f5f2;--surface:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--muted:#8a8984;
--line:#e4e3df;--blue:#2a78d6;--red:#d03b3b;--amber:#b87700;--green:#0a7f0a}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}
main{max-width:1180px;margin:0 auto;padding:28px 16px 80px}
h1{font-size:26px;margin:0 0 4px} h2{font-size:19px;margin:40px 0 10px;padding-top:8px;border-top:1px solid var(--line)}
h3{font-size:15px;margin:22px 0 8px} h4{font-size:13px;margin:16px 0 6px;color:var(--ink2);text-transform:uppercase;letter-spacing:.04em}
p{margin:6px 0 10px;max-width:900px} .small{font-size:12px;color:var(--ink2)} code{font-size:12px;background:#eceae5;padding:1px 4px;border-radius:3px}
.sub{color:var(--ink2);margin-bottom:18px}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:10px;margin:16px 0}
.card{background:var(--surface);border:1px solid var(--line);border-radius:8px;padding:12px 14px}
.card .k{font-size:12px;color:var(--ink2)} .card .v{font-size:20px;font-weight:600;margin-top:2px}
.box{background:var(--surface);border:1px solid var(--line);border-left:4px solid var(--amber);border-radius:6px;padding:10px 14px;margin:10px 0}
.box.bad{border-left-color:var(--red)} .box.ok{border-left-color:var(--green)} .box ul{margin:4px 0;padding-left:20px}
.tw{overflow-x:auto;margin:8px 0 6px;background:var(--surface);border:1px solid var(--line);border-radius:6px}
table{border-collapse:collapse;width:100%;font-size:12.5px;font-variant-numeric:tabular-nums}
th,td{padding:6px 9px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}
th{background:#f1f0ec;color:var(--ink2);font-weight:600;position:sticky;top:0}
td.l,th.l{text-align:left} tbody tr:hover td{background-image:linear-gradient(rgba(0,0,0,.035),rgba(0,0,0,.035))}
td.hm{text-align:center;min-width:64px} td.na{color:var(--muted);background:#f3f2ef} td.faded{opacity:.45}
.mutedtxt{color:var(--ink2)}
table.kv{width:auto;background:var(--surface);border:1px solid var(--line);border-radius:6px}
table.kv th{text-align:left;background:none;position:static;font-weight:500} table.kv td{text-align:left;white-space:normal}
.grid2{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);gap:18px} @media(max-width:860px){.grid2{grid-template-columns:1fr}}
img{max-width:100%;height:auto;display:block;background:var(--surface);border:1px solid var(--line);border-radius:6px}
.pill{display:inline-block;padding:1px 8px;border-radius:999px;font-size:11.5px;font-weight:600;border:1px solid transparent}
.pill.strong{background:#0d366b;color:#fff}.pill.mod{background:#3987e5;color:#fff}.pill.weak{background:#cde2fb;color:#0d366b}
.pill.muted{background:#ecebe7;color:var(--ink2)}.pill.bad{background:#fbe3e3;color:#9e2a2a;border-color:#f0b9b9}
.pill.warn{background:#fdf0d5;color:#7a4f00;border-color:#f3d48f}.pill.ok{background:#dff3df;color:#0a5f0a}.pill.info{background:#e3eefc;color:#184f95}
.sig-hi{color:#b02c2c;font-weight:600}.sig-lo{color:#1c5cab;font-weight:600}.sig-no{color:var(--muted)}
details.feat{background:var(--surface);border:1px solid var(--line);border-radius:8px;margin:10px 0;padding:0 16px}
details.feat>summary{cursor:pointer;padding:12px 0;font-weight:600;font-size:15px;list-style:none;display:flex;gap:10px;align-items:center;flex-wrap:wrap}
details.feat>summary::before{content:"▸";color:var(--muted)} details.feat[open]>summary::before{content:"▾"}
details.feat[open]{padding-bottom:14px}
.metrics{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:8px;margin:8px 0}
.metric{border:1px solid var(--line);border-radius:6px;padding:8px 10px;background:#fff}.metric .k{font-size:11.5px;color:var(--ink2)}.metric .v{font-size:17px;font-weight:600}
th.sortable{cursor:pointer} th.sortable:after{content:" ⇅";color:var(--muted);font-weight:400}
dl.gl{display:grid;grid-template-columns:200px 1fr;gap:6px 16px;background:var(--surface);border:1px solid var(--line);border-radius:6px;padding:12px 16px}
dl.gl dt{font-weight:600} dl.gl dd{margin:0;color:var(--ink2)} @media(max-width:640px){dl.gl{grid-template-columns:1fr}}
nav.toc{font-size:13px;margin:8px 0 0} nav.toc a{color:var(--blue);margin-right:14px;text-decoration:none}
"""

SORT_JS = """
document.querySelectorAll('table.sortable').forEach(function(tb){
  tb.querySelectorAll('th').forEach(function(th,i){
    th.classList.add('sortable');
    th.addEventListener('click',function(){
      var body=tb.tBodies[0], rows=Array.from(body.rows), asc=th.dataset.asc!=='1';
      rows.sort(function(a,b){
        var x=a.cells[i].dataset.v??a.cells[i].innerText, y=b.cells[i].dataset.v??b.cells[i].innerText;
        var nx=parseFloat(x), ny=parseFloat(y);
        if(!isNaN(nx)&&!isNaN(ny)) return asc?nx-ny:ny-nx;
        return asc?String(x).localeCompare(y):String(y).localeCompare(x);
      });
      th.dataset.asc=asc?'1':'0'; rows.forEach(function(r){body.appendChild(r)});
    });
  });
});
"""

GLOSSARY = [
    ("Gini", "2 × ROC-AUC − 1. How well the feature alone ranks fraud above genuine: 0 = no better than "
             "random, 1 = perfect. Here it is computed out of fold or walk-forward, never on the same labels "
             "used to learn the pattern."),
    ("Walk-forward Gini", "For each period, the fraud rate of every group is learned from earlier periods only, "
                          "then used to score that period. This mimics how the feature would be used in "
                          "production and is the basis of the verdict and the time analysis."),
    ("Within-period Gini", "Gini when each period's own fraud rates are used to score that period. It is "
                           "optimistic, but shows whether the signal exists in that period at all. If it stays "
                           "high while walk-forward Gini falls, the risky groups have moved."),
    ("95% CI", "Bootstrap 95% interval. Overlapping intervals between periods mean the difference may be noise."),
    ("Trend label", "Stable: differences between periods are within noise. Decaying / Emerging: a statistically "
                    f"significant downward / upward trend of at least {MIN_MEANINGFUL_GINI_CHANGE} Gini. Volatile: "
                    "periods differ more than noise explains, with no steady direction."),
    ("PR-AUC lift", "Average precision divided by the fraud rate. Focuses on the top of the ranking, which is "
                    "what matters when fraud is rare. 1 = random."),
    ("Value Gini", "Gini with each transaction weighted by its amount: ranking power in money rather than counts."),
    ("Fraud rate, 95% CI", "Wilson interval. Wide intervals mean too few transactions to be sure."),
    ("Lift", "Group fraud rate ÷ overall fraud rate. 3x = three times as risky as average. Heat-map cells are "
             "red above average, blue below, grey near average."),
    ("WoE", "ln(share of all fraud ÷ share of all genuine) for a group. Positive = riskier than average."),
    ("IV", "Information Value, the sum over groups of (%fraud − %genuine) × WoE. Below 0.02 useless, 0.02–0.1 weak, "
           f"0.1–0.3 medium, 0.3–0.5 strong, above {IV_LEAKAGE_THRESHOLD} suspicious (check for leakage)."),
    ("Signal", f"Whether a group's fraud rate differs from average after correcting for the number of groups "
               f"tested (Benjamini–Hochberg, false discovery rate {FDR_ALPHA})."),
    ("Cramér's V / uncertainty coefficient", "Strength of association (0 to 1). Values are small in fraud "
                                             "because fraud is rare; compare features with each other."),
    ("PSI", f"Population Stability Index: how much the mix of groups changed vs the first period. Below "
            f"{PSI_MODERATE} stable, {PSI_MODERATE}–{PSI_MAJOR} moderate, above {PSI_MAJOR} major."),
    ("Rank corr.", "Spearman correlation between each period's fraud rates and the overall fraud rates, for the "
                   "groups that are significantly risky or safe overall. 1 = the same groups are always riskiest."),
    ("Shape (numeric)", "Whether risk rises, falls, or forms a U across the value range. A U-shape or peak means a "
                        "linear model would miss the signal; bin the feature or use a tree-based model."),
]


# =============================================================================
# 9. Report assembly
# =============================================================================

def build_report(ctx: dict) -> str:
    S = ctx["summary"]
    H = []
    H.append(f"<!doctype html><html lang='en-GB'><head><meta charset='utf-8'>"
             f"<meta name='viewport' content='width=device-width,initial-scale=1'>"
             f"<title>{_esc(REPORT_TITLE)}</title><style>{CSS}</style></head><body><main>")
    H.append(f"<h1>{_esc(REPORT_TITLE)}</h1><div class='sub'>Generated {datetime.now():%d %b %Y %H:%M} · "
             f"{ctx['period_range']} · {len(S)} features ({ctx['n_num']} numeric, {ctx['n_cat']} categorical)</div>")
    H.append("<nav class='toc'><a href='#checks'>Checks</a><a href='#ranking'>Ranking</a>"
             "<a href='#time'>Over time</a><a href='#features'>Feature details</a><a href='#glossary'>How to read</a>"
             "<a href='#config'>Settings</a></nav>")

    cards = [("Transactions", f"{ctx['N']:,}"), ("Frauds", f"{ctx['F']:,}"), ("Fraud rate", _pct(ctx["base_rate"], 3))]
    if ctx["has_amount"]:
        cards += [("Fraud value", f"{CURRENCY}{ctx['fraud_value']:,.0f}"), ("Fraud share of value", _pct(ctx["fraud_value_share"], 3))]
    cards += [("Periods", f"{len(ctx['periods'])}"), ("Min. group size", f"{ctx['min_size']:,} txns")]
    H.append("<div class='cards'>" + "".join(f"<div class='card'><div class='k'>{k}</div><div class='v'>{v}</div></div>"
                                             for k, v in cards) + "</div>")

    # Checks
    H.append("<h2 id='checks'>Checks before you read the results</h2>")
    if ctx["global_warnings"]:
        H.append("<div class='box bad'><ul>" + "".join(f"<li>{w}</li>" for w in ctx["global_warnings"]) + "</ul></div>")
    else:
        H.append("<div class='box ok'>No data-level problems detected. Label maturity and leakage still need "
                 "your own check: this report can only spot their symptoms.</div>")
    pt = ctx["periods"]
    H.append("<h4>Volume and fraud by period</h4><div class='tw'><table><thead><tr><th class='l'>Period</th>"
             "<th>Txns</th><th>Frauds</th><th>Fraud rate</th><th class='l'>Note</th></tr></thead><tbody>" +
             "".join(f"<tr><td class='l'>{_esc(r['period'])}</td><td>{int(r['n']):,}</td><td>{int(r['frauds']):,}</td>"
                     f"<td>{_pct(r['fraud_rate'], 3)}</td><td class='l mutedtxt'>{_esc(r['note'])}</td></tr>"
                     for _, r in pt.iterrows()) + "</tbody></table></div>")

    # Ranking
    H.append("<h2 id='ranking'>Ranking</h2><p>Features are ranked by average walk-forward Gini: how well each "
             "one, on its own, ranks fraud in periods it has not seen. Click a column to sort.</p>")
    head = ["#", "Feature", "Type", "Verdict", "Trend", "Walk-forward Gini", "Out-of-fold Gini", "PR-AUC lift", "IV"]
    if ctx["has_amount"]:
        head.append("Value Gini")
    head += ["Max PSI", "Missing", "Warnings"]
    rows = []
    for i, r in S.iterrows():
        c = [f"<td data-v='{i + 1}'>{i + 1}</td>", f"<td class='l'><a href='#f-{i}'>{_esc(r['feature'])}</a></td>",
             f"<td class='l'>{r['type']}</td>",
             f"<td class='l'><span class='pill {verdict_class(r['verdict'])}'>{_esc(r['verdict'])}</span></td>",
             f"<td class='l'><span class='pill {TREND_CLASS.get(r['trend'], 'muted')}'>{_esc(r['trend'])}</span></td>",
             gini_cell(r["wf_gini"]).replace("<td", f"<td data-v='{r['wf_gini'] if np.isfinite(r['wf_gini']) else -9}'", 1),
             f"<td>{_f(r['oof_gini'])}</td>", f"<td>{_f(r['pr_auc_lift'], 2)}x</td>",
             f"<td>{_f(r['iv'])}</td>"]
        if ctx["has_amount"]:
            c.append(f"<td>{_f(r['value_gini'])}</td>")
        c += [f"<td>{_f(r['max_psi'], 2)}</td>", f"<td data-v='{r['missing_rate']}'>{_pct(r['missing_rate'], 1)}</td>",
              f"<td>{int(r['n_warnings'])}</td>"]
        rows.append("<tr>" + "".join(c) + "</tr>")
    H.append("<div class='tw'><table class='sortable'><thead><tr>" + "".join(f"<th>{h}</th>" for h in head) +
             "</tr></thead><tbody>" + "".join(rows) + "</tbody></table></div>")

    # Over time
    order = list(S["feature"])
    labels = list(ctx["periods"]["period"])
    H.append("<h2 id='time'>Predictive power over time</h2>")
    H.append("<p>Each period is scored with what was learned from earlier periods only. Darker = stronger. "
             "Hover a cell for its 95% interval. The first period has no earlier data, and periods with fewer "
             f"than {MIN_FRAUDS_PER_PERIOD} frauds are not scored.</p>")
    hrow = "<th class='l'>Feature</th><th class='l'>Trend</th>" + "".join(f"<th>{_esc(p)}</th>" for p in labels) + "<th>Average</th>"
    body = []
    for f in order:
        tdf = ctx["temporal"][f]
        tr = ctx["trends"][f]
        cells = [f"<td class='l'>{_esc(f)}</td>",
                 f"<td class='l'><span class='pill {TREND_CLASS.get(tr['label'], 'muted')}'>{_esc(tr['label'])}</span></td>"]
        for _, r in tdf.iterrows():
            tip = (f"95% CI {_f(r['wf_low'])} to {_f(r['wf_high'])}" if r["status"] == "Scored" else r["status"])
            cells.append(gini_cell(r["wf_gini"], tip))
        cells.append(gini_cell(tr["mean_wf_gini"], "fraud-weighted average"))
        body.append("<tr>" + "".join(cells) + "</tr>")
    H.append("<h4>Walk-forward Gini by period</h4><div class='tw'><table class='heat'><thead><tr>" + hrow +
             "</tr></thead><tbody>" + "".join(body) + "</tbody></table></div>")
    H.append(f"<h4>Trend per feature (shaded = 95% interval; panels share a scale except those marked 'own scale')</h4><img alt='Walk-forward Gini over time, one panel per feature' "
             f"src='{ctx['small_multiples']}'>")

    rk = ctx["rank_table"]
    if rk is not None and len(rk.columns):
        H.append("<h4>Rank by period (1 = most predictive that period)</h4><p class='small'>Shows whether the same "
                 "features stay on top. Large jumps mean the feature mix that matters is changing.</p>")
        H.append("<div class='tw'><table><thead><tr><th class='l'>Feature</th>" +
                 "".join(f"<th>{_esc(c)}</th>" for c in rk.columns) + "</tr></thead><tbody>" +
                 "".join("<tr><td class='l'>" + _esc(f) + "</td>" +
                         "".join(f"<td>{'–' if not np.isfinite(v) else int(v)}</td>" for v in rk.loc[f]) + "</tr>"
                         for f in rk.index) + "</tbody></table></div>")

    # Feature details
    H.append("<h2 id='features'>Feature details</h2><p>Click a feature to expand it.</p>")
    for i, f in enumerate(order):
        H.append(ctx["sections"][f].replace("{ANCHOR}", f"f-{i}"))

    # Glossary and config
    H.append("<h2 id='glossary'>How to read this report</h2><dl class='gl'>" +
             "".join(f"<dt>{_esc(k)}</dt><dd>{_esc(v)}</dd>" for k, v in GLOSSARY) + "</dl>")
    H.append("<p class='small'>What this report cannot tell you: whether a feature adds anything on top of the "
             "others (that needs a multivariate test), or whether it causes fraud. A weak feature on its own can "
             "still be valuable in combination.</p>")
    H.append("<h2 id='config'>Settings used</h2>" + kv_html(ctx["config"]))
    H.append(f"<script>{SORT_JS}</script></main></body></html>")
    return "".join(H)


def feature_section(f, ftype, prep, t, assoc, oof, insample_gini, extras, tdf, trend, story,
                    warns, notes, verdict, ctx_static) -> str:
    is_cat = ftype == "Categorical"
    has_amount = ctx_static["has_amount"]
    H = [f"<details class='feat' id='{{ANCHOR}}'><summary>{_esc(f)} "
         f"<span class='pill muted'>{ftype}</span>"
         f"<span class='pill {verdict_class(verdict)}'>{_esc(verdict)}</span>"
         f"<span class='pill {TREND_CLASS.get(trend['label'], 'muted')}'>{_esc(trend['label'])}</span></summary>"]

    if warns:
        H.append("<div class='box bad'><b>Warnings</b><ul>" + "".join(f"<li>{w}</li>" for w in warns) + "</ul></div>")
    if notes:
        H.append("<div class='box'><ul>" + "".join(f"<li>{n}</li>" for n in notes) + "</ul></div>")

    m = [("Walk-forward Gini", _f(trend["mean_wf_gini"])), ("Out-of-fold Gini", _f(oof["gini"])),
         ("ROC-AUC", _f(oof["roc_auc"])), ("PR-AUC lift", f"{_f(oof['pr_auc_lift'], 2)}x"), ("KS", _f(oof["ks"])),
         ("IV", f"{_f(assoc['iv'])} <span class='small'>({assoc['iv_band']})</span>")]
    if has_amount:
        m.append(("Value Gini", _f(oof["value_gini"])))
    m += [("Cramér's V", _f(assoc["cramers_v"], 4)), ("Uncertainty coef.", _f(assoc["uncertainty"], 4))]
    if not is_cat:
        m.append(("Raw-value Gini", _f(extras["raw_gini"])))
    H.append("<h3>Strength on its own</h3><div class='metrics'>" +
             "".join(f"<div class='metric'><div class='k'>{k}</div><div class='v'>{v}</div></div>" for k, v in m) + "</div>")
    if is_cat:
        H.append(f"<p class='small'>Fitted on the same labels the Gini would be {_f(insample_gini)}; the out-of-fold "
                 f"figure above is the honest one.</p>")

    H.append("<h3>Data quality</h3>" + kv_html(prep["info"]))
    if not is_cat:
        H.append("<h3>Shape and distribution</h3>")
        H.append(kv_html({
            "Risk across the range": _esc(extras["shape"]),
            "Direction (non-missing values)": _esc(extras["direction"]),
            "Median: fraud vs genuine": f"{_fmt_num(extras['fraud_median'])} vs {_fmt_num(extras['genuine_median'])}",
            "Mean: fraud vs genuine": f"{_fmt_num(extras['fraud_mean'])} vs {_fmt_num(extras['genuine_mean'])}",
            "KS distance between fraud and genuine distributions": _f(extras["ks_stat"]),
        }))
        if np.isfinite(extras["raw_gini"]) and oof["gini"] - extras["raw_gini"] > 0.05:
            H.append(f"<p class='small'>The binned Gini ({_f(oof['gini'])}) is clearly above the raw-value Gini "
                     f"({_f(extras['raw_gini'])}): risk does not move in one direction across the range, so use "
                     f"bins or a tree-based model rather than a linear term.</p>")
        if prep["edges"] is not None:
            H.append(f"<p class='small'>Bin edges: {', '.join(_fmt_num(e) for e in prep['edges']) or 'none'}</p>")

    H.append("<h3>Risk by " + ("category" if is_cat else "bin") + "</h3>")
    H.append(f"<img alt='Fraud rate by group for {_esc(f)}' src='{chart_groups(t, not is_cat, ctx_static['base_rate'], f)}'>")
    H.append(group_table_html(t, has_amount, not is_cat))

    H.append("<h3>Over time</h3>")
    H.append("<div class='box" + (" bad" if trend["label"] == "Decaying" else " ok" if trend["label"].startswith("Stable") else "") +
             "'>" + " ".join(_esc(s) for s in story) + "</div>")
    H.append(f"<img alt='Predictive power over time for {_esc(f)}' src='{chart_trend(tdf, f)}'>")
    H.append(period_table_html(tdf, is_cat))
    H.append("<h4>Fraud rate by " + ("category" if is_cat else "bin") + " and period</h4>"
             "<p class='small'>Red = riskier than the overall average, blue = safer. Faded cells have too few "
             "transactions to judge. A group changing colour across periods means its risk has moved.</p>")
    H.append(drilldown_html(ctx_static["groups"][f], ctx_static["y"], ctx_static["pidx"],
                            ctx_static["period_labels"], t, ctx_static["base_rate"], not is_cat))
    H.append("</details>")
    return "".join(H)


# =============================================================================
# 10. Main entry point
# =============================================================================

def analyse_features(
    df: pd.DataFrame,
    label_col: str,
    numeric_features: Sequence[str],
    categorical_features: Sequence[str],
    time_col: str,
    amount_col: Optional[str] = None,
    output_html: Optional[str] = "fraud_feature_report.html",
) -> dict:
    """
    Analyse the stand-alone predictive power of each feature for fraud, overall and over time,
    and write an HTML report. Tuning settings are in the CONFIG section at the top of this file.

    Parameters
    ----------
    df                   : one row per transaction. Columns not listed are ignored.
    label_col            : 1 = fraud, 0 = genuine.
    numeric_features     : quantitative features (amounts, counts, ages, scores...).
    categorical_features : qualitative features (channel, MCC, country...). Numeric codes such as
                           MCC or postcode district belong here, not in numeric_features.
    time_col             : transaction timestamp.
    amount_col           : optional transaction amount, adds value-based figures.
    output_html          : path of the report; None to skip writing it.

    Returns
    -------
    dict with "summary" (one row per feature), "details" (group table per feature),
    "temporal" (period table per feature), "trends", "periods" and "html_path".
    """
    numeric_features, categorical_features = list(numeric_features), list(categorical_features)
    validate_inputs(df, label_col, numeric_features, categorical_features, time_col, amount_col)
    rng = np.random.default_rng(RANDOM_STATE)

    d = df.reset_index(drop=True)
    y = d[label_col].astype(int).to_numpy()
    N, F = len(y), int(y.sum())
    base_rate = F / N
    amount = d[amount_col].abs().to_numpy(float) if amount_col else None
    min_size = min_group_size(N, base_rate)

    tvals = pd.to_datetime(d[time_col])
    per = tvals.dt.to_period(PERIOD)
    period_values = sorted(per.unique())
    period_labels = [str(p) for p in period_values]
    pidx = per.map({p: i for i, p in enumerate(period_values)}).to_numpy(int)

    # Period overview and data-level checks
    pt = pd.DataFrame({"p": pidx, "y": y}).groupby("p")["y"].agg(frauds="sum", n="size").reset_index()
    pt["period"] = [period_labels[i] for i in pt["p"]]
    pt["fraud_rate"] = pt["frauds"] / pt["n"]
    pt["note"] = ""
    global_warnings = []
    if F < 100:
        global_warnings.append(f"Only {F} frauds in total. Treat every figure as indicative.")
    low = pt["frauds"] < MIN_FRAUDS_PER_PERIOD
    pt.loc[low, "note"] = f"Fewer than {MIN_FRAUDS_PER_PERIOD} frauds: not scored"
    if low.any():
        global_warnings.append(f"{int(low.sum())} period(s) have fewer than {MIN_FRAUDS_PER_PERIOD} frauds and are not "
                               f"scored in the time analysis. Consider a longer PERIOD.")
    if len(pt) >= 3:
        usual = pt["fraud_rate"].iloc[:-1].median()
        last = pt["fraud_rate"].iloc[-1]
        if last < LABEL_MATURITY_RATIO * usual:
            pt.loc[pt.index[-1], "note"] = "Fraud rate well below usual: labels may be immature"
            global_warnings.append(f"The last period ({pt['period'].iloc[-1]}) has a fraud rate of {_pct(last, 3)}, "
                                   f"against a usual {_pct(usual, 3)}. This is the typical sign of fraud not yet "
                                   "reported. If so, drop that period and re-run; otherwise recent "
                                   "'genuine' transactions include undetected fraud and recent results are distorted.")
    if len(period_values) < 4:
        global_warnings.append(f"Only {len(period_values)} periods: trends cannot be judged reliably. "
                               "Use a shorter PERIOD or a longer history.")

    ctx_static = {"has_amount": amount is not None, "base_rate": base_rate, "y": y, "pidx": pidx,
                  "period_labels": period_labels, "groups": {}}
    summary_rows, details, temporal, trends, sections = [], {}, {}, {}, {}

    # Pass 1: all statistics
    work = {}
    for f in numeric_features + categorical_features:
        is_cat = f in categorical_features
        prep = prepare_categorical(d[f], min_size) if is_cat else prepare_numeric(d[f], min_size)
        groups, key = prep["groups"], prep["key"]
        ctx_static["groups"][f] = groups
        key_codes, key_names = pd.factorize(pd.Series(key), sort=False)
        group_codes, group_names = pd.factorize(pd.Series(groups), sort=False)
        n_keys = len(key_names)

        t = group_table(groups, y, amount, prep["order"], base_rate)
        assoc = association(t, N, base_rate)
        oof = ranking_metrics(oof_target_encode(key_codes, y, n_keys), y, amount)
        insample_gini = ranking_metrics(target_encode(key_codes, y, key_codes, n_keys), y, None)["gini"]
        extras = numeric_extras(prep["values"], y, t) if not is_cat else {}
        tdf = temporal_analysis(group_codes, list(group_names), key_codes, n_keys, y, pidx, period_labels,
                                t, base_rate, is_cat, rng)
        trends[f] = trend_statistics(tdf)
        details[f], temporal[f] = t, tdf
        work[f] = dict(is_cat=is_cat, prep=prep, t=t, assoc=assoc, oof=oof, insample_gini=insample_gini,
                       extras=extras, tdf=tdf)

    # Trend labels, corrected for the number of features tested
    assign_trend_labels(trends)

    # Pass 2: judgement and report sections
    for f, w in work.items():
        is_cat, prep, t, assoc, oof, tdf = w["is_cat"], w["prep"], w["t"], w["assoc"], w["oof"], w["tdf"]
        trend = trends[f]
        ftype = "Categorical" if is_cat else "Numeric"
        story = temporal_story(tdf, trend, is_cat)
        warns, notes, verdict = diagnose(prep, t, assoc, oof, w["insample_gini"], trend, tdf, F, is_cat)
        if output_html:
            sections[f] = feature_section(f, ftype, prep, t, assoc, oof, w["insample_gini"], w["extras"], tdf,
                                          trend, story, warns, notes, verdict, ctx_static)
        summary_rows.append({
            "feature": f, "type": ftype, "verdict": verdict, "trend": trend["label"],
            "wf_gini": trend["mean_wf_gini"], "oof_gini": oof["gini"], "oof_roc_auc": oof["roc_auc"],
            "pr_auc_lift": oof["pr_auc_lift"], "ks": oof["ks"], "iv": assoc["iv"], "iv_band": assoc["iv_band"],
            "cramers_v": assoc["cramers_v"], "uncertainty": assoc["uncertainty"], "value_gini": oof["value_gini"],
            "insample_gini": w["insample_gini"], "max_psi": float(tdf["psi"].max()),
            "gini_change": trend["change"], "p_trend": trend["p_trend"], "q_trend": trend["q_trend"],
            "p_volatility": trend["p_volatility"], "q_volatility": trend["q_volatility"], "tau": trend["tau"],
            "missing_rate": prep["stats"]["missing_rate"], "n_groups": len(t),
            "n_warnings": len(warns), "warnings": " | ".join(warns),
        })

    S = pd.DataFrame(summary_rows)
    S["_sort"] = S["wf_gini"].fillna(S["oof_gini"])
    S = S.sort_values("_sort", ascending=False).drop(columns="_sort").reset_index(drop=True)

    leaky = S[S["warnings"].str.contains("authorisation time|leakage", regex=True)]["feature"].tolist()
    if leaky:
        global_warnings.append("Possible leakage in: " + ", ".join(_esc(x) for x in leaky) +
                               ". Check these before trusting any ranking that includes them.")

    wf = pd.DataFrame({f: temporal[f].set_index("period")["wf_gini"] for f in S["feature"]}).T
    wf = wf.loc[:, wf.notna().any()]
    rank_table = wf.rank(ascending=False, method="min") if len(wf.columns) else None

    html_path = None
    if output_html:
        cfg_names = ["MIN_GROUP_PCT", "MIN_GROUP_EXPECTED_FRAUDS", "N_BINS", "PERIOD", "MIN_FRAUDS_PER_PERIOD",
                     "WALK_FORWARD_WINDOW", "N_BOOTSTRAP", "GINI_BANDS", "MIN_MEANINGFUL_GINI_CHANGE", "TREND_ALPHA",
                     "FDR_ALPHA", "IV_LEAKAGE_THRESHOLD", "PSI_MODERATE", "PSI_MAJOR", "LABEL_MATURITY_RATIO",
                     "ENCODING_SMOOTHING", "N_FOLDS", "RANDOM_STATE"]
        config = {k: _esc(globals()[k]) for k in cfg_names}
        config["Resulting minimum group size"] = f"{min_size:,} transactions"
        ctx = {
            "summary": S, "N": N, "F": F, "base_rate": base_rate, "has_amount": amount is not None,
            "fraud_value": float((amount * y).sum()) if amount is not None else 0.0,
            "fraud_value_share": float((amount * y).sum() / amount.sum()) if amount is not None and amount.sum() > 0 else np.nan,
            "periods": pt, "min_size": min_size, "global_warnings": global_warnings,
            "period_range": f"{tvals.min():%d %b %Y} to {tvals.max():%d %b %Y}",
            "n_num": len(numeric_features), "n_cat": len(categorical_features),
            "temporal": temporal, "trends": trends, "rank_table": rank_table, "sections": sections,
            "small_multiples": chart_small_multiples(temporal, trends, list(S["feature"]), leaky),
            "config": config,
        }
        with open(output_html, "w", encoding="utf-8") as fh:
            fh.write(build_report(ctx))
        html_path = output_html

    return {"summary": S, "details": details, "temporal": temporal, "trends": trends,
            "periods": pt, "global_warnings": global_warnings, "html_path": html_path}