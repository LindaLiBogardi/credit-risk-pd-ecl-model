"""Export small, aggregated result files for the Streamlit dashboard.

Run from the project root, after the notebooks have saved their outputs:

    python -m src.export_dashboard_data

Only aggregated results are written to app/data/; no individual loan records.
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

from src.ecl import LGD_UNSECURED, assign_stage, compute_ecl, lifetime_pd
from src.stress import calibrate_delta, shift_pd
from src.woe import load_binners, woe_transform

RAW = Path("data/raw")
PROCESSED = Path("data/processed")
MODELS = Path("models")
OUT = Path("app/data")

# name: (target average PD, weight); None = base PDs unchanged
SCENARIOS = {"Base": (None, 0.60), "Adverse": (0.11, 0.30), "Severe": (0.14, 0.10)}


def score_test_set(saved: dict, binners: dict) -> pd.DataFrame:
    """PD and score for every test client."""
    test = pd.read_parquet(PROCESSED / "test.parquet")
    X = woe_transform(test, binners)[saved["final_vars"]]
    test["PD"] = saved["model"].predict_proba(X)[:, 1]
    test["SCORE"] = saved["offset"] - saved["factor"] * saved["model"].decision_function(X)
    return test


def export_validation(test: pd.DataFrame) -> float:
    """ROC curve points and calibration by PD decile. Returns the test AUC."""
    y, pd_test = test["TARGET"], test["PD"]
    fpr, tpr, _ = roc_curve(y, pd_test)
    idx = np.linspace(0, len(fpr) - 1, 200).astype(int)
    pd.DataFrame({"fpr": fpr[idx], "tpr": tpr[idx]}).to_csv(OUT / "roc_curve.csv", index=False)

    calib = pd.DataFrame({
        "decile": pd.qcut(pd_test, 10, labels=False) + 1,
        "predicted": pd_test,
        "observed": y,
    }).groupby("decile").mean()
    calib.to_csv(OUT / "calibration.csv")
    return roc_auc_score(y, pd_test)


def export_ecl() -> dict:
    """ECL by scenario and stage (LGD 45%), plus the ECL by PD decile."""
    portfolio = pd.read_parquet(PROCESSED / "portfolio_staged.parquet")
    term_years = portfolio["CREDIT_TERM"].fillna(portfolio["CREDIT_TERM"].median()) / 12
    ead = portfolio["EAD"].to_numpy()

    rows = []
    for name, (target, weight) in SCENARIOS.items():
        delta = 0.0 if target is None else calibrate_delta(portfolio["PD"], target)
        pd_s = shift_pd(portfolio["PD"], delta)
        stage = assign_stage(pd_s)
        ecl_12m = compute_ecl(pd_s, portfolio["EAD"]).to_numpy()
        ecl_life = compute_ecl(lifetime_pd(pd_s, term_years), portfolio["EAD"]).to_numpy()
        ecl = np.where(stage == 2, ecl_life, ecl_12m)
        for st in (1, 2):
            mask = stage == st
            rows.append({"scenario": name, "weight": weight, "stage": st,
                         "loans": int(mask.sum()), "ead": float(ead[mask].sum()),
                         "ecl": float(ecl[mask].sum())})
    pd.DataFrame(rows).to_csv(OUT / "ecl_by_scenario_stage.csv", index=False)

    pd.read_csv(MODELS / "ecl_by_decile.csv").to_csv(OUT / "ecl_by_decile.csv", index=False)
    return {"total_ead": float(ead.sum()), "loans": len(portfolio)}


def export_fairness(test: pd.DataFrame) -> None:
    """Approval rates (80% rule) and calibration by gender and age group."""
    gender = pd.read_csv(RAW / "application_train.csv", usecols=["SK_ID_CURR", "CODE_GENDER"])
    df = test.merge(gender, on="SK_ID_CURR", how="left")
    df = df[df["CODE_GENDER"].isin(["F", "M"])]
    df["AGE_GROUP"] = pd.cut(df["AGE_YEARS"], bins=[0, 30, 50, 100],
                             labels=["Under 30", "30-50", "Over 50"])
    df["APPROVED"] = df["SCORE"] >= df["SCORE"].quantile(0.20)

    tables = []
    for group in ["CODE_GENDER", "AGE_GROUP"]:
        t = df.groupby(group, observed=True).agg(
            clients=("TARGET", "size"),
            observed_default=("TARGET", "mean"),
            predicted_pd=("PD", "mean"),
            approval_rate=("APPROVED", "mean"),
        ).reset_index().rename(columns={group: "group"})
        t["ratio_vs_highest"] = t["approval_rate"] / t["approval_rate"].max()
        t.insert(0, "attribute", "Gender" if group == "CODE_GENDER" else "Age group")
        tables.append(t)
    pd.concat(tables).to_csv(OUT / "fairness.csv", index=False)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    saved = joblib.load(MODELS / "pd_model.joblib")
    binners = load_binners(MODELS / "binners.joblib")

    pd.read_csv(MODELS / "scorecard.csv").to_csv(OUT / "scorecard.csv", index=False)

    test = score_test_set(saved, binners)
    auc = export_validation(test)
    portfolio_info = export_ecl()
    export_fairness(test)

    metrics = {
        "test_auc": auc,
        "test_gini": 2 * auc - 1,
        "n_variables": len(saved["final_vars"]),
        "lgd": LGD_UNSECURED,
        "scaling": saved["scaling"],
        **portfolio_info,
    }
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"Exported dashboard data to {OUT}/ (test Gini {metrics['test_gini']:.3f})")


if __name__ == "__main__":
    main()