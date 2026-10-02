"""Download the Home Credit Default Risk files used in this project.

Requires a Kaggle account that has accepted the competition rules:
https://www.kaggle.com/competitions/home-credit-default-risk

Usage (from the project root):
    python src/download_data.py
"""
import shutil
from pathlib import Path

import kagglehub

COMPETITION = "home-credit-default-risk"
FILES = [
    "application_train.csv",
    "bureau.csv",
    "HomeCredit_columns_description.csv",
]
RAW_DIR = Path("data/raw")


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        target = RAW_DIR / name
        if target.exists():
            print(f"Already present: {target}")
            continue
        cached = kagglehub.competition_download(COMPETITION, path=name)
        shutil.copy(cached, target)
        print(f"Downloaded: {target}")


if __name__ == "__main__":
    main()