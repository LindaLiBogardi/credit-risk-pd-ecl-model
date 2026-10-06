## Data

This project uses the [Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk)
dataset from Kaggle. The data is not included in this repository, in line with
Kaggle's competition rules.

| File | Description |
|---|---|
| `application_train.csv` | One row per loan application, with applicant data and the default flag (`TARGET`) |
| `bureau.csv` | Clients' previous loans at other lenders, as reported to a credit bureau |
| `HomeCredit_columns_description.csv` | Description of every column in the dataset |

To download the data:
1. Create a Kaggle account and accept the competition rules on the Data tab.
2. Configure your Kaggle API credentials.
3. Run `python src/download_data.py` from the project root.

Files are saved to `data/raw/`.

Limitations: 

1. TARGET in application_train.csv is Home Credit's own definition of payment difficulties, not the regulatory definition of default under CRR Article 178. In a bank, the target would follow the regulatory default definition.

## Results

**PD model:** logistic regression scorecard with 12 variables, selected from 122 through an expert long list, WoE/IV screening, correlation checks and backward elimination. Scaled to 600 points at odds of 50:1, with 20 points to double the odds.

| Validation test (test set) | Result |
|---|---|
| Discrimination | Gini 0.490 (AUC 0.745); training Gini 0.492 |
| Calibration | Mean PD 8.07% vs observed 8.07%; within about 0.3 percentage points in every PD decile |
| Stability | PSI 0.0001 (train vs test) |

The strongest drivers are the three external credit scores, followed by employment, organisation type, loan structure and education. Full details are in `notebooks/04_pd_model.ipynb` and `notebooks/05_validation.ipynb`.

**Main limitation:** the data covers a single period, so no out-of-time validation was possible.