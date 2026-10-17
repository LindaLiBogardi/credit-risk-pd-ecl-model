"""Collect the facts the model documentation is allowed to state, each with its source.

Only aggregated results are used: the dashboard files in app/data/ and a short list
of results recorded in the notebooks. No client-level data is read.

Run from the project root:

    python -m src.model_doc.facts
"""

import hashlib
import json
from pathlib import Path

import pandas as pd

from src.ecl import SICR_PD_THRESHOLD

APP_DATA = Path("app/data")
FACTS_FILE = Path("docs/facts.json")
SCENARIOS = ["Base", "Adverse", "Severe"]

# Results recorded in the notebooks: key -> (value, display text, source)
NOTEBOOK_FACTS = {
    # Data and exploration
    "n_applications": (307511, "307,511", "notebooks/01_explore.ipynb"),
    "n_raw_columns": (122, "122", "notebooks/01_explore.ipynb"),
    "default_rate": (0.0807, "8.07%", "notebooks/01_explore.ipynb"),
    "n_long_list": (23, "23", "notebooks/01_explore.ipynb"),
    # Variable selection settings
    "max_bins": (8, "8", "notebooks/03_woe_iv.ipynb"),
    "min_bin_share": (0.05, "5%", "notebooks/03_woe_iv.ipynb"),
    "iv_threshold": (0.02, "0.02", "notebooks/03_woe_iv.ipynb"),
    "corr_threshold": (0.6, "0.6", "notebooks/03_woe_iv.ipynb"),
    # Validation
    "train_gini": (0.492, "0.492", "notebooks/05_validation.ipynb"),
    "psi_train_test": (0.0001, "0.0001", "notebooks/05_validation.ipynb"),
    # ECL and staging
    "ecl_12m_unstaged": (5.97e9, "5.97 bn", "notebooks/06_ecl.ipynb"),
    "coverage_12m": (0.0324, "3.24%", "notebooks/06_ecl.ipynb"),
    "staging_uplift": (0.126, "+12.6%", "notebooks/07_staging.ipynb"),
    "stage2_share_of_ecl": (0.31, "31%", "notebooks/07_staging.ipynb"),
    "sicr_threshold_low": (0.10, "10%", "notebooks/07_staging.ipynb"),
    "sicr_threshold_high": (0.30, "30%", "notebooks/07_staging.ipynb"),
    "ecl_sicr_10pct": (8.2e9, "8.2 bn", "notebooks/07_staging.ipynb"),
    "ecl_sicr_30pct": (6.2e9, "6.2 bn", "notebooks/07_staging.ipynb"),
    # Stress testing
    "pd_target_adverse": (0.11, "11%", "notebooks/08_stress_test.ipynb"),
    "pd_target_severe": (0.14, "14%", "notebooks/08_stress_test.ipynb"),
    "lgd_stressed": (0.75, "75%", "notebooks/08_stress_test.ipynb"),
    "lgd_75_uplift": (0.67, "+67%", "notebooks/08_stress_test.ipynb"),
    # Challenger, fairness, bureau
    "xgb_test_gini": (0.531, "0.531", "notebooks/09_xgboost_challenger.ipynb"),
    "xgb_gini_gain": (0.041, "+0.041", "notebooks/09_xgboost_challenger.ipynb"),
    "gender_proxy_auc": (0.720, "0.720", "notebooks/10_fairness.ipynb"),
    "n_bureau_passed_iv": (4, "4", "notebooks/11_bureau_features.ipynb"),
    "bureau_gini_gain": ("0.002-0.004", "0.002-0.004", "notebooks/11_bureau_features.ipynb"),
}


def fact(value, display: str, source: str) -> dict:
    """One documented fact: the raw value, the exact text to print, and its source."""
    return {"value": value, "display": display, "source": source}


def pct(x: float, decimals: int = 2) -> str:
    return f"{x * 100:.{decimals}f}%"


def signed_pct(x: float, decimals: int = 0) -> str:
    return f"{x * 100:+.{decimals}f}%"


def bn(x: float) -> str:
    return f"{x / 1e9:.2f} bn"


def slug(text: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in text.lower()).strip("_")


def metric_facts(data_dir: Path) -> dict:
    src = str(data_dir / "metrics.json")
    m = json.loads((data_dir / "metrics.json").read_text())
    s = m["scaling"]
    return {
        "test_gini": fact(m["test_gini"], f"{m['test_gini']:.3f}", src),
        "test_auc": fact(m["test_auc"], f"{m['test_auc']:.3f}", src),
        "n_final_variables": fact(m["n_variables"], str(m["n_variables"]), src),
        "lgd": fact(m["lgd"], pct(m["lgd"], 0), src),
        "base_score": fact(s["base_score"], f"{s['base_score']:g}", src),
        "base_odds": fact(s["base_odds"], f"{s['base_odds']:g}:1", src),
        "pdo": fact(s["pdo"], f"{s['pdo']:g}", src),
        "total_ead": fact(m["total_ead"], bn(m["total_ead"]), src),
        "n_portfolio_loans": fact(m["loans"], f"{m['loans']:,}", src),
        "sicr_threshold": fact(SICR_PD_THRESHOLD, pct(SICR_PD_THRESHOLD, 0), "src/ecl.py"),
    }


def ecl_facts(data_dir: Path) -> dict:
    src = str(data_dir / "ecl_by_scenario_stage.csv")
    df = pd.read_csv(data_dir / "ecl_by_scenario_stage.csv")
    totals = df.groupby("scenario").agg(
        ecl=("ecl", "sum"), loans=("loans", "sum"), weight=("weight", "first")
    )
    stage2_loans = df[df["stage"] == 2].set_index("scenario")["loans"]

    facts = {}
    for name in SCENARIOS:
        key = name.lower()
        ecl, weight = totals.loc[name, "ecl"], totals.loc[name, "weight"]
        share = stage2_loans.get(name, 0) / totals.loc[name, "loans"]
        facts[f"ecl_{key}"] = fact(ecl, bn(ecl), src)
        facts[f"weight_{key}"] = fact(weight, pct(weight, 0), src)
        facts[f"stage2_share_{key}"] = fact(share, pct(share, 1), src)

    base = totals.loc["Base", "ecl"]
    weighted = float((totals["ecl"] * totals["weight"]).sum())
    facts["ecl_weighted"] = fact(weighted, bn(weighted), src)
    facts["ecl_weighted_vs_base"] = fact(weighted / base - 1, signed_pct(weighted / base - 1), src)
    for name in ["Adverse", "Severe"]:
        change = totals.loc[name, "ecl"] / base - 1
        facts[f"ecl_{name.lower()}_vs_base"] = fact(change, signed_pct(change), src)
    return facts


def calibration_facts(data_dir: Path) -> dict:
    src = str(data_dir / "calibration.csv")
    calib = pd.read_csv(data_dir / "calibration.csv")
    gap_pp = float((calib["observed"] - calib["predicted"]).abs().max() * 100)
    return {"calib_max_gap_pp": fact(gap_pp, f"{gap_pp:.2f} percentage points", src)}


def decile_facts(data_dir: Path) -> dict:
    src = str(data_dir / "ecl_by_decile.csv")
    deciles = pd.read_csv(data_dir / "ecl_by_decile.csv")
    top = float(deciles.loc[deciles["PD_DECILE"].idxmax(), "ecl_share"])
    return {"top_decile_ecl_share": fact(top, pct(top, 0), src)}


def fairness_facts(data_dir: Path) -> dict:
    src = str(data_dir / "fairness.csv")
    fair = pd.read_csv(data_dir / "fairness.csv")
    facts = {}
    for row in fair.itertuples():
        key = f"{slug(row.attribute)}_{slug(str(row.group))}"
        ratio, observed, predicted = row.ratio_vs_highest, row.observed_default, row.predicted_pd
        facts[f"approval_ratio_{key}"] = fact(ratio, f"{ratio:.2f}", src)
        facts[f"observed_default_{key}"] = fact(observed, pct(observed), src)
        facts[f"predicted_pd_{key}"] = fact(predicted, pct(predicted), src)
    failing = fair.loc[fair["ratio_vs_highest"] < 0.8, "group"].astype(str).tolist()
    text = ", ".join(failing) if failing else "none"
    facts["groups_failing_80_rule"] = fact(failing, text, src)
    return facts


def build_facts(data_dir: Path = APP_DATA) -> dict:
    """All facts, computed ones first, then those recorded in the notebooks."""
    facts = {}
    for collect in (metric_facts, ecl_facts, calibration_facts, decile_facts, fairness_facts):
        facts.update(collect(data_dir))
    for key, (value, display, source) in NOTEBOOK_FACTS.items():
        facts[key] = fact(value, display, source)
    return facts


def fingerprint(facts: dict) -> str:
    """Short hash that identifies this exact set of facts in the document header."""
    text = json.dumps(facts, sort_keys=True, default=str)
    return hashlib.sha256(text.encode()).hexdigest()[:12]


def main() -> None:
    facts = build_facts()
    FACTS_FILE.parent.mkdir(exist_ok=True)
    FACTS_FILE.write_text(json.dumps(facts, indent=2, default=float))
    for key, f in facts.items():
        print(f"{key:40s} {f['display']:28s} {f['source']}")
    print(f"\n{len(facts)} facts written to {FACTS_FILE} (fingerprint {fingerprint(facts)})")


if __name__ == "__main__":
    main()