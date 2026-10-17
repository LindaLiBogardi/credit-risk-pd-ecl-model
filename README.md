# Credit Risk PD Model and IFRS 9 ECL

![CI](https://github.com/LindaLiBogardi/credit-risk-pd-ecl-model/actions/workflows/ci.yml/badge.svg)

A probability of default (PD) scorecard and an IFRS 9 expected credit loss (ECL) calculation, built on the Home Credit Default Risk dataset. The project follows the workflow used in bank model development: an expert long list of risk drivers, WoE/IV binning, a logistic regression scorecard, independent-style validation, staging, macro stress testing and fairness testing, all delivered through issues, pull requests and CI/CD.

**Live dashboard:** [credit-risk-pd-ecl.streamlit.app](https://credit-risk-pd-ecl.streamlit.app/)

## Key results

| Area | Result |
|---|---|
| PD model | Logistic regression scorecard, 12 variables selected from 122 |
| Discrimination | Test Gini 0.490 (AUC 0.745); training Gini 0.492 |
| Calibration | Mean PD 8.07% vs observed 8.07%; within about 0.3 percentage points in every PD decile |
| Stability | PSI 0.0001 (train vs test) |
| 12-month ECL | 5.97 bn on 184 bn exposure; coverage 3.24% |
| With staging | 6.73 bn (+12.6%); Stage 2 holds 7.3% of loans but 31% of ECL |
| Probability-weighted ECL | 8.28 bn, 23% above the base scenario |
| XGBoost challenger | Test Gini 0.531 (+0.041); scorecard kept for explainability |
| Fairness | Gender passes the 80% rule; clients under 30 fail it (ratio 0.67) |

Amounts are in the dataset's undisclosed currency unit.

## Method

### 1. Data and exploration
307,511 loan applications with 122 columns from `application_train.csv`. The default rate (TARGET = 1) is 8.07%. Exploration found 41 columns more than 50% missing, a placeholder value of 365243 in `DAYS_EMPLOYED` meaning "not employed" (18% of rows, almost all pensioners), and informative missingness.

Before any testing, an **expert long list** of 23 candidate risk drivers was written, each with an expected effect and a business reason. Gender, the application weekday and 47 building variables were excluded up front.

### 2. Cleaning, features and split
The placeholder was replaced with a missing value plus a flag, and four affordability ratios were derived (credit/income, annuity/income, credit term, credit/goods). The data was split 70/30, stratified on TARGET with a fixed seed. Everything learned from data after this point uses the training set only.

### 3. Variable selection
Optimal binning (optbinning) with at most 8 bins, at least 5% of the data per bin, and a monotonic or peak/valley trend. Selection steps:
- IV of at least 0.02
- WoE correlation below 0.6 within each pair
- business review of every variable's bins
- backward elimination with 5-fold cross-validation (threshold 0.001 Gini)

The expected effects were compared with the evidence for every variable. A notable finding: the affordability ratios were weak (IV below 0.02), most likely because of **selection bias**, since the data contains only approved loans.

### 4. PD model and scorecard
Logistic regression on WoE values, all 12 coefficients negative as expected. Scaled to 600 points at good:bad odds of 50:1, with 20 points to double the odds. On the test set, the observed default rate falls steadily across score bands, and clients scoring 590–610 default at 2.2%, matching the 2% implied by the scaling.

The final variables: the three external credit scores, employment length, occupation, employer industry, education, regional rating, goods price, credit/goods ratio, credit term and car age.

### 5. IFRS 9 ECL, staging and stress testing
- **ECL = PD × LGD × EAD**, with LGD fixed at 45% and EAD equal to the loan amount.
- **Staging:** a 12-month PD above 20% moves a loan to Stage 2, with lifetime ECL from a constant annual default rate over the remaining term. Moving the threshold from 10% to 30% changes total ECL from 8.2 to 6.2 bn, showing how much the SICR rule matters.
- **Stress test:** base, adverse and severe scenarios (average PD 8.07%, 11% and 14%; weights 60/30/10) via a log-odds shift that preserves the ranking. ECL rises faster than PD because of stage migration: +45% in the adverse and +95% in the severe scenario. About three quarters of the increase comes from higher PDs, and the rest from loans moving to Stage 2.
- **LGD sensitivity:** raising LGD from 45% to 75% increases the probability-weighted ECL by 67%, an effect comparable to the severe scenario.

### 6. Challenger, bureau data and fairness
- **XGBoost challenger** on the cleaned raw features: Gini 0.531. All 12 scorecard variables are among its top 20 features. Its extra power comes from interactions and from variables excluded for good reasons, such as an undocumented document flag and social-circle default counts.
- **Credit bureau features** (`bureau.csv`): four passed IV screening, but added only 0.002–0.004 Gini, most likely because the external scores already contain bureau information. Not adopted.
- **Fairness:** gender is excluded from the model, but the 12 variables still predict it with AUC 0.720, mainly through occupation and employer industry. The model over-predicts risk for women and for clients under 30. Correcting the gender gap would require using gender, which anti-discrimination rules prohibit: a known trade-off between fairness definitions.

## Limitations

- **TARGET** is Home Credit's definition of early payment difficulties, not the regulatory default definition (CRR Article 178).
- **No time dimension:** the data covers a single period, so out-of-time validation, real staging (no origination PD) and Stage 3 are not possible.
- **LGD and EAD** are assumptions: there is no recovery data, and no credit conversion factor for revolving loans.
- **Scenario PDs** are assumed; a bank would derive them from a satellite model linking default rates to macro variables.
- **Selection bias:** only approved applications are observed.
- **Fairness:** only gender and age are available for testing, and the 80% rule is a US screening heuristic, not an EU legal standard.

## Project structure

    credit-risk-pd-ecl-model/
    ├── notebooks/
    │   ├── 01_explore.ipynb               # EDA, key findings, expert long list
    │   ├── 02_data_cleaning.ipynb         # cleaning, features, train/test split
    │   ├── 03_woe_iv.ipynb                # binning, IV screening, variable selection
    │   ├── 04_pd_model.ipynb              # logistic regression and scorecard
    │   ├── 05_validation.ipynb            # discrimination, calibration, stability
    │   ├── 06_ecl.ipynb                   # 12-month ECL
    │   ├── 07_staging.ipynb               # staging and lifetime ECL
    │   ├── 08_stress_test.ipynb           # scenarios and LGD sensitivity
    │   ├── 09_xgboost_challenger.ipynb    # challenger model
    │   ├── 10_fairness.ipynb              # proxy test, 80% rule, group calibration
    │   └── 11_bureau_features.ipynb       # credit bureau experiment
    ├── src/                               # reusable functions (cleaning, WoE, ECL, stress, ...)
    ├── tests/                             # unit tests run in CI
    ├── app/                               # Streamlit dashboard and aggregated data
    └── .github/workflows/                 # CI (ruff, pytest) and manual CD to Azure

## How to run

1. Create an environment with Python 3.11 and install dependencies: `pip install -r requirements.txt`. On macOS, XGBoost also needs `brew install libomp`.
2. Create a Kaggle account, accept the [competition rules](https://www.kaggle.com/competitions/home-credit-default-risk), and set up your Kaggle API token.
3. Download the data: `python src/download_data.py`
4. Run the notebooks in order from the `notebooks/` folder.
5. Export the dashboard data and start the app from the project root:
   ```
   python -m src.export_dashboard_data
   streamlit run app/streamlit_app.py
   ```

The raw data is not included in this repository, in line with Kaggle's competition rules. The dashboard uses only aggregated results.

## Development workflow

Work was planned as GitHub issues on a project board and delivered through feature branches and pull requests. **CI** runs ruff and pytest on every pull request. The dashboard is hosted on **Streamlit Community Cloud** and redeploys automatically from `main`. A manual **CD** workflow can also deploy it to Azure App Service, with dependencies built on the GitHub runner; the Azure free tier's CPU quota proved too small for a Streamlit app, which is why Community Cloud is the main host.

In a bank, the model would also go through independent validation, model risk governance and a parallel run against the existing model before production, and deployment would use OpenID Connect rather than a publish profile.

**Model documentation:** [model documentation](docs/model_documentation.md), drafted with the Claude API and reviewed by the model owner; every number is traced to its source.