"""Markdown tables for the model documentation, built by code from the facts and app/data/.

Tables are deterministic: the LLM never writes or edits a number in a table.
"""

import re
from pathlib import Path

import pandas as pd

from src.model_doc.facts import APP_DATA, SCENARIOS

VARIABLE_LABELS = {
    "EXT_SOURCE_3": "External credit score 3",
    "EXT_SOURCE_2": "External credit score 2",
    "EXT_SOURCE_1": "External credit score 1",
    "DAYS_EMPLOYED": "Employment length (days, negative)",
    "OCCUPATION_TYPE": "Occupation",
    "ORGANIZATION_TYPE": "Employer industry",
    "NAME_EDUCATION_TYPE": "Education",
    "REGION_RATING_CLIENT_W_CITY": "Region rating (1 = best)",
    "AMT_GOODS_PRICE": "Price of goods financed",
    "CREDIT_GOODS_RATIO": "Loan amount / goods price",
    "CREDIT_TERM": "Number of payments",
    "OWN_CAR_AGE": "Age of own car (years)",
}


def md_table(headers: list[str], rows: list[list]) -> str:
    """A Markdown table; pipe characters in cells are escaped."""

    def cell(x) -> str:
        return str(x).replace("|", "\\|")

    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    lines += ["| " + " | ".join(cell(x) for x in row) + " |" for row in rows]
    return "\n".join(lines)


def performance_table(facts: dict) -> str:
    d = {k: v["display"] for k, v in facts.items()}
    rows = [
        ["Gini, test set", d["test_gini"]],
        ["AUC, test set", d["test_auc"]],
        ["Gini, training set", d["train_gini"]],
        ["PSI, training vs test", d["psi_train_test"]],
        ["Largest calibration gap across PD deciles", d["calib_max_gap_pp"]],
        ["Gini of XGBoost challenger, test set", d["xgb_test_gini"]],
    ]
    return md_table(["Measure", "Value"], rows)


def scenario_table(facts: dict) -> str:
    d = {k: v["display"] for k, v in facts.items()}
    rows = []
    for name in SCENARIOS:
        key = name.lower()
        change = "" if name == "Base" else d[f"ecl_{key}_vs_base"]
        rows.append([name, d[f"weight_{key}"], d[f"stage2_share_{key}"], d[f"ecl_{key}"], change])
    rows.append(["Probability-weighted", "", "", d["ecl_weighted"], d["ecl_weighted_vs_base"]])
    headers = ["Scenario", "Weight", "Loans in Stage 2", "ECL", "Change vs base"]
    return md_table(headers, rows)


def fairness_table(data_dir: Path) -> str:
    fair = pd.read_csv(data_dir / "fairness.csv")
    rows = [
        [
            r.attribute,
            r.group,
            f"{r.observed_default:.2%}",
            f"{r.predicted_pd:.2%}",
            f"{r.ratio_vs_highest:.2f}",
            "Yes" if r.ratio_vs_highest >= 0.8 else "No",
        ]
        for r in fair.itertuples()
    ]
    headers = ["Attribute", "Group", "Observed default", "Predicted PD",
               "Approval ratio vs highest", "Passes 80% rule"]
    return md_table(headers, rows)


def clean_bin(label) -> str:
    """Readable bin label: a list of categories becomes 'A, B, C'.

    optbinning stores categorical bins as the printed form of an array, e.g.
    "<ArrowStringArray>\\n['Accountants', 'IT staff']\\nLength: 2, dtype: str".
    """
    text = str(label)
    categories = re.findall(r"'([^']*)'", text)
    if categories:
        return ", ".join(categories)
    return " ".join(text.split())


def scorecard_table(data_dir: Path) -> str:
    card = pd.read_csv(data_dir / "scorecard.csv")
    card = card[card["bin"] != "Special"]  # no clients in these bins
    rows = []
    for variable, group in card.groupby("variable", sort=False):
        label = VARIABLE_LABELS.get(variable, variable)
        for i, r in enumerate(group.itertuples()):
            rows.append([label if i == 0 else "", clean_bin(r.bin), int(r.points)])
    return md_table(["Variable", "Bin", "Points"], rows)


def facts_table(facts: dict) -> str:
    rows = [[f"`{key}`", f["display"], f"`{f['source']}`"] for key, f in facts.items()]
    return md_table(["Fact", "Value", "Source"], rows)


def build_tables(facts: dict, data_dir: Path = APP_DATA) -> dict:
    """All tables, by name, ready to insert into the document."""
    return {
        "performance": performance_table(facts),
        "scenarios": scenario_table(facts),
        "fairness": fairness_table(data_dir),
        "scorecard": scorecard_table(data_dir),
        "facts": facts_table(facts),
    }


if __name__ == "__main__":
    from src.model_doc.facts import build_facts

    for name, table in build_tables(build_facts()).items():
        print(f"\n## {name}\n\n{table}")

