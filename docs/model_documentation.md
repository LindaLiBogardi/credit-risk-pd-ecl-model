# Model documentation: PD scorecard and IFRS 9 ECL

> **DRAFT for human review.** Prose drafted with the Claude API; every number is inserted by code from `docs/facts.json`. See Appendix C.

| | |
|---|---|
| Generated | 2026-10-12 |
| LLM | `claude-sonnet-5-5` (39,745 input, 13,792 output tokens) |
| Code version | git commit `51c1dab` |
| Facts fingerprint | `4ab2be0b813d` |

## Open points for the reviewer

- **Data:** [REVIEWER: the rationale for excluding the application weekday and the building-related variables is not in the project context. Add it if it is to be documented.]
- **Data:** [REVIEWER: treatment of other missing values, outliers and categorical level grouping is not described in the context. Add it if applicable.]
- **Data:** [REVIEWER: the split proportions, the seed value and the resulting training and test row counts are not in the facts provided. Add them.]
- **Methodology:** [REVIEWER: number of folds, selection metric and stopping rule are not documented.]
- **Methodology:** [REVIEWER: regularisation, solver settings and the train/test split ratio are not documented.]
- **Model performance:** [REVIEWER: the bank's PSI interpretation thresholds are not specified in the project context, so none were applied.]
- **IFRS 9 expected credit loss, staging and stress testing:** [REVIEWER: confirm that the staging uplift, the Stage 2 share of ECL and the SICR sensitivity figures refer to the base scenario.]
- **Challenger model and additional data:** [REVIEWER: XGBoost hyperparameters, tuning procedure and the number of input features are not in the provided context; add them if the validators need to reproduce the challenger.]
- **Challenger model and additional data:** [REVIEWER: confirm the dataset on which the bureau gain was measured, and whether a formal materiality criterion for adopting additional features was set.]
- **Fairness:** [REVIEWER: the approval cut-off or the share of applicants approved under this policy is not stated in the context.]
- **Fairness:** [REVIEWER: whether the result for the group failing the 80% rule requires action, such as a policy review or a documented acceptance, is a governance decision not covered by the context.]

## 1. Executive summary

The model is a probability of default (PD) scorecard with an IFRS 9 expected credit loss (ECL) calculation for retail loans. It was developed on the Home Credit Default Risk dataset (307,511 applications). It is a logistic regression on WoE values with 12 variables, all with the expected sign. On the held-out test set the Gini was 0.490 and the AUC 0.745. The largest gap between predicted and observed default rates in the calibration check was 0.29 percentage points. An XGBoost challenger reached a test Gini of 0.531, but the scorecard was retained for explainability.

With an LGD of 45% and total EAD of 184.21 bn, the base-scenario ECL was 6.73 bn. The probability-weighted ECL was 8.28 bn (+23% against base), and the severe scenario gave 13.12 bn. Moving the SICR threshold from 30% to 10% changed ECL from 6.2 bn to 8.2 bn.

The model over-predicts risk for women and for clients under 30. The group failing the 80% rule was Under 30, with an approval ratio of 0.67.

The main limitations are:
- The target is not the Article 178 default definition.
- There is no time dimension, so there is no out-of-time validation and no Stage 3.
- LGD, EAD and scenario PDs are assumptions.
- Only approved loans are observed.

## 2. Purpose and scope

This document describes a retail credit risk model made of a probability of default (PD) scorecard and an IFRS 9 expected credit loss (ECL) calculation. The scorecard estimates the 12-month PD of a loan application. It has two intended uses. The first is application scoring, where PDs are converted to points with a base score of 600 at base good:bad odds of 50:1, and 20 points (PDO) double the odds. The second is as the PD input to the IFRS 9 calculation, ECL = PD x LGD x EAD.

**This is a portfolio project built on public data. It is not a production model.** It has not been independently validated, approved under any model risk governance process, or used for any credit decision or financial reporting.

The model was developed on the Home Credit Default Risk dataset, which contains 307,511 loan applications and 122 raw columns, with an observed target rate of 8.07%. The target flags early payment difficulties as defined by Home Credit. It is not the regulatory default definition in CRR Article 178, so the PD estimated here refers to that event only. The data contains only approved loans and has no dates.

The scope of the work is as follows:

- Development of a logistic regression scorecard on WoE values with 12 final variables, selected from a long list of 23 candidate risk drivers.
- Measurement of discrimination, calibration, stability and fairness on the held-out test set. The test Gini is 0.490.
- Calculation of ECL, staging and stress testing on the full portfolio of 307,511 loans, with a total EAD of 184.21 bn. LGD is a fixed assumption of 45%, and EAD equals the loan amount.
- Comparison with an XGBoost challenger model and an assessment of credit bureau features.

The following are outside the scope of this work: out-of-time validation, Stage 3, staging based on an origination PD, and any scenario PDs derived from a macroeconomic satellite model. LGD and EAD are assumptions, because the data contains no recovery or credit conversion factor information. Staging uses an absolute PD threshold as a proxy for a significant increase in credit risk.

## 3. Data

The analysis used the Home Credit Default Risk dataset (Kaggle). Each row is a loan application, and the dataset contains 307,511 applications described by 122 raw columns. The overall default rate is 8.07%. The data contains only approved loans and covers one period without dates. It therefore has no time dimension, and applicants who were rejected are not observed. This is a source of selection bias.

The target flags early payment difficulties as defined by Home Credit. It is not the regulatory default definition in CRR Article 178, so the PD estimates are not regulatory PDs.

Before any statistical testing, an expert long list of 23 candidate risk drivers was written. Each candidate has an expected direction of effect and a business reason. These expectations were later compared with the evidence for every variable. Three groups of variables were excluded up front:

- Gender, which is not used as a model input. It was joined back later for fairness testing only.
- The application weekday.
- Building-related variables.

[REVIEWER: the rationale for excluding the application weekday and the building-related variables is not in the project context. Add it if it is to be documented.]

The following cleaning and feature steps were applied:

- Employment length contains a placeholder value that marks clients who are not employed, almost all of them pensioners. The placeholder was replaced by a missing value and a separate flag.
- Affordability ratios were derived: credit/income, annuity/income, credit term and credit/goods price.

[REVIEWER: treatment of other missing values, outliers and categorical level grouping is not described in the context. Add it if applicable.]

The data was split into a training set and a test set. The split was stratified on the target and used a fixed seed. Everything learned from data, including the binning, the variable selection and the model coefficients, used the training set only. The test set was held out for discrimination, calibration, stability and fairness measurement.

[REVIEWER: the split proportions, the seed value and the resulting training and test row counts are not in the facts provided. Add them.]

## 4. Methodology

The binning, selection and modelling steps are described below. Everything learned from data, including bin boundaries, WoE values, variable selection and coefficients, used the training set only.

**Binning.** Each candidate variable was binned with optimal binning (optbinning), using a maximum of 8 bins, a minimum bin share of 5% of the data, and a monotonic or peak/valley trend constraint. WoE and IV were computed per variable from these bins.

**Variable selection.** The starting point was the expert long list of 23 candidate drivers, each written with an expected direction of effect and a business reason before any statistical testing. These candidates were drawn from the 122 raw columns. Variables were then screened in four steps:

- An IV threshold of 0.02: variables below it were dropped.
- A WoE correlation threshold of 0.6: for variable pairs above it, only one variable was retained.
- A business review of the bins of every variable.
- Backward elimination with cross-validation. [REVIEWER: number of folds, selection metric and stopping rule are not documented.]

For every variable, the expected direction of effect was compared with the direction in the binned data. The final model has 12 variables:

- external credit score 1/2/3
- employment length
- occupation
- employer industry
- education
- regional rating
- goods price
- credit/goods ratio
- credit term
- car age

**Selection bias.** The derived affordability ratios were weak. The credit/income and annuity/income ratios are not among the final variables. The most likely explanation is selection bias, because only approved loans are observed, so applicants with poor affordability are largely absent. This explanation was not tested, since rejected applications are not in the data.

**Logistic regression.** A logistic regression was fitted on the WoE-transformed values of the final variables. All coefficients have the expected sign. [REVIEWER: regularisation, solver settings and the train/test split ratio are not documented.]

**Score scaling.** The log-odds output was converted to points with a linear transformation. A base score of 600 corresponds to good:bad odds of 50:1, and an increase of 20 points (PDO) doubles the odds.

## 5. Model performance

Discrimination was measured on the held-out test set. The Gini was 0.490 and the AUC was 0.745. The training-set Gini was 0.492, marginally above the test value. The gap between the two is negligible, so the check gave no indication of overfitting. This is consistent with the model's structure: a logistic regression on WoE values with 12 variables, fitted on binned inputs with a minimum bin share. On the test set, the observed default rate falls steadily across score bands, so the score ranks clients in the intended order.

Calibration was assessed by comparing the mean predicted PD with the observed default rate in each PD decile of the test set. The largest gap in any decile was 0.29 percentage points. Predicted PDs therefore stay close to observed default rates across the whole score range, with no decile showing a material deviation. This applies to the target as defined by Home Credit for early payment difficulties, not to a regulatory default definition.

Stability was measured with the population stability index (PSI) between the training and test score distributions. The PSI was 0.0001, which indicates virtually no shift between the two samples. Both samples come from a stratified random split of a single undated period, so this result confirms the split is consistent but says nothing about stability over time. Out-of-time validation was not possible because the data has no time dimension.

[REVIEWER: the bank's PSI interpretation thresholds are not specified in the project context, so none were applied.]

| Measure | Value |
|---|---|
| Gini, test set | 0.490 |
| AUC, test set | 0.745 |
| Gini, training set | 0.492 |
| PSI, training vs test | 0.0001 |
| Largest calibration gap across PD deciles | 0.29 percentage points |
| Gini of XGBoost challenger, test set | 0.531 |

## 6. IFRS 9 expected credit loss, staging and stress testing

ECL was calculated as PD x LGD x EAD on the full portfolio of 307,511 loans. The LGD was a fixed assumption of 45% and EAD was set equal to the loan amount, giving a total EAD of 184.21 bn. No recovery data or credit conversion factor was available, so both LGD and EAD are assumptions rather than estimates.

**12-month ECL and staging.** Without staging, the 12-month ECL was 5.97 bn, a coverage ratio of 3.24%. A loan moved to Stage 2 when its 12-month PD exceeded the SICR threshold of 20%. Stage 2 loans received lifetime ECL based on a constant annual default rate over the remaining term. Staging raised the ECL by +12.6% relative to the unstaged figure, and Stage 2 accounted for 31% of the staged ECL. No Stage 3 was modelled, because the target is not the Article 178 default definition and the data has no time dimension.

**SICR sensitivity.** No origination PD was available, so the absolute PD threshold is a proxy for a significant increase in credit risk. Moving the threshold to 10% gave an ECL of 8.2 bn, and moving it to 30% gave 6.2 bn. The base-scenario ECL at the 20% threshold was 6.73 bn. The result is therefore materially sensitive to this rule. [REVIEWER: confirm that the staging uplift, the Stage 2 share of ECL and the SICR sensitivity figures refer to the base scenario.]

**Scenarios.** Three scenarios were used: base, adverse and severe, with assumed probability weights of 60%, 30% and 10%. Scenario PDs were produced by a log-odds shift that preserves the ranking of clients. The shift was calibrated so that the average PD reached the scenario target, 11% for the adverse scenario and 14% for the severe scenario. The targets and weights are assumptions and were not derived from a macroeconomic satellite model.

The ECL was 6.73 bn in the base scenario, 9.76 bn in the adverse scenario (+45% versus base) and 13.12 bn in the severe scenario (+95% versus base). The probability-weighted ECL was 8.28 bn, which is +23% versus base.

**Stage migration.** The Stage 2 share of loans rose from 7.3% in the base scenario to 14.1% in the adverse scenario and 22.2% in the severe scenario. ECL therefore rose faster than PD. Most of the increase came from higher PDs and the remainder from migration to Stage 2.

**LGD sensitivity.** Raising the LGD assumption from 45% to 75% increased the probability-weighted ECL by +67%. This is comparable in size to the effect of the severe scenario.

The scenario results follow in the table below.

| Scenario | Weight | Loans in Stage 2 | ECL | Change vs base |
|---|---|---|---|---|
| Base | 60% | 7.3% | 6.73 bn |  |
| Adverse | 30% | 14.1% | 9.76 bn | +45% |
| Severe | 10% | 22.2% | 13.12 bn | +95% |
| Probability-weighted |  |  | 8.28 bn | +23% |

## 7. Challenger model and additional data

An XGBoost model was trained on the cleaned raw features as a challenger to the scorecard. Its test Gini was 0.531, against 0.490 for the scorecard, a gain of +0.041. All scorecard variables were among the challenger's most important features. Its additional discriminatory power came from two sources:

- interactions between variables, which a logistic regression on WoE values does not capture;
- variables that were deliberately excluded from the scorecard, such as an undocumented document flag and counts of defaults in the client's social circle.

The exclusions were made for reasons documented earlier in the development, so the gain is not evidence that the scorecard is mis-specified. The scorecard was kept because of its explainability: each variable contributes a transparent number of points per bin, and every coefficient has the expected sign. [REVIEWER: XGBoost hyperparameters, tuning procedure and the number of input features are not in the provided context; add them if the validators need to reproduce the challenger.]

Aggregated credit bureau features were tested as an additional data source. 4 of them passed the IV screening. Added to the model, they raised the Gini by only 0.002-0.004. The most likely explanation is that the external credit scores already contain bureau information, so the bureau features add little that is new. This explanation was not tested directly. The bureau features were not adopted, because the gain did not justify the additional data dependency and model complexity. [REVIEWER: confirm the dataset on which the bureau gain was measured, and whether a formal materiality criterion for adopting additional features was set.]

## 8. Fairness

Gender is excluded from the model but was joined back to the test set for fairness testing. The model variables still predict gender: a proxy model reached an AUC of 0.720, mainly through occupation and employer industry. Excluding gender therefore does not remove its information from the score. Age and gender are the only protected characteristics available, so no other attributes could be tested.

Approval rates were compared under a policy that accepts the highest-scoring applicants. [REVIEWER: the approval cut-off or the share of applicants approved under this policy is not stated in the context.] The 80% rule divides each group's approval rate by that of the highest group. For gender, women had a ratio of 1.00 and men 0.90. For age, the ratios were 0.67 for under 30, 0.89 for 30-50 and 1.00 for over 50. The group failing the 80% rule was: Under 30.

Calibration was checked by group by comparing the mean predicted PD with the observed default rate on the test set.

- Women: observed 7.01%, predicted 7.47%.
- Men: observed 10.14%, predicted 9.24%.
- Under 30: observed 11.29%, predicted 12.51%.
- 30-50: observed 8.66%, predicted 8.20%.
- Over 50: observed 5.77%, predicted 5.94%.

The model over-predicts risk for women and for clients under 30 relative to their observed default rates. It under-predicts risk for men. The under 30 group has both the lowest approval ratio and an over-predicted PD, so part of its lower approval rate is not supported by its observed default rate.

The gender gap in calibration cannot be corrected without using gender in the model, which anti-discrimination rules prohibit. This is a known trade-off between fairness definitions: equal approval rates, equal calibration across groups and the exclusion of protected attributes cannot all be satisfied at once when the groups' observed default rates differ. The model was not adjusted to favour any one definition.

The test has the following limits:

- The 80% rule is a US screening heuristic, not an EU legal standard. It was used as an indicator only.
- Only gender and age were available, and age was tested in bands.
- Only approved loans are observed, so the results describe approved clients and not all applicants.

[REVIEWER: whether the result for the group failing the 80% rule requires action, such as a policy review or a documented acceptance, is a governance decision not covered by the context.]

| Attribute | Group | Observed default | Predicted PD | Approval ratio vs highest | Passes 80% rule |
|---|---|---|---|---|---|
| Gender | F | 7.01% | 7.47% | 1.00 | Yes |
| Gender | M | 10.14% | 9.24% | 0.90 | Yes |
| Age group | Under 30 | 11.29% | 12.51% | 0.67 | No |
| Age group | 30-50 | 8.66% | 8.20% | 0.89 | Yes |
| Age group | Over 50 | 5.77% | 5.94% | 1.00 | Yes |

## 9. Limitations and model risk

- The target flags early payment difficulties as defined by Home Credit and is not the regulatory default definition in Article 178, so the PD output is not a regulatory PD.
- The data covers a single period without dates, so no out-of-time validation was possible and the stability result (0.0001 between training and test) reflects a random split only.
- Without an origination PD, staging relies on an absolute 12-month PD threshold of 20% as a proxy for a significant increase in credit risk, and ECL ranges from 6.2 bn at a threshold of 30% to 8.2 bn at 10%.
- No Stage 3 could be modelled because the data has no time dimension and no regulatory default flag.
- LGD is a fixed assumption of 45% and EAD is set equal to the loan amount, because the data has no recovery information and no credit conversion factor for revolving loans.
- Scenario PDs (average targets of 11% for the adverse and 14% for the severe scenario) and the scenario weights are assumed rather than derived from a macroeconomic satellite model.
- Only approved loans are observed, so the scorecard is subject to selection bias, which most likely explains the weak performance of the affordability ratios.
- Fairness testing was limited to gender and age, the model over-predicts risk for women and for clients under 30, and the 80% rule is a US screening heuristic rather than an EU legal standard.
- The scorecard was kept for explainability although the XGBoost challenger reached a higher test Gini (0.531 against 0.490), so some predictive power is not captured.
- Before use in a bank, the model would require independent validation, model risk governance and a parallel run against the existing model.

## Appendix A. Scorecard

| Variable | Bin | Points |
|---|---|---|
| External credit score 3 | (-inf, 0.18) | 19 |
|  | [0.18, 0.31) | 31 |
|  | [0.31, 0.41) | 41 |
|  | [0.41, 0.50) | 49 |
|  | [0.50, 0.57) | 55 |
|  | [0.57, 0.64) | 59 |
|  | [0.64, 0.70) | 64 |
|  | [0.70, inf) | 69 |
|  | Missing | 43 |
| External credit score 2 | (-inf, 0.15) | 22 |
|  | [0.15, 0.26) | 33 |
|  | [0.26, 0.39) | 38 |
|  | [0.39, 0.49) | 44 |
|  | [0.49, 0.62) | 49 |
|  | [0.62, 0.67) | 55 |
|  | [0.67, 0.73) | 61 |
|  | [0.73, inf) | 71 |
|  | Missing | 45 |
| External credit score 1 | (-inf, 0.27) | 36 |
|  | [0.27, 0.40) | 44 |
|  | [0.40, 0.53) | 49 |
|  | [0.53, 0.62) | 53 |
|  | [0.62, 0.70) | 56 |
|  | [0.70, inf) | 62 |
|  | Missing | 45 |
| Employment length (days, negative) | (-inf, -5443.50) | 55 |
|  | [-5443.50, -3887.50) | 51 |
|  | [-3887.50, -3075.50) | 49 |
|  | [-3075.50, -2447.50) | 48 |
|  | [-2447.50, -1785.50) | 46 |
|  | [-1785.50, -1297.50) | 45 |
|  | [-1297.50, -982.50) | 43 |
|  | [-982.50, inf) | 42 |
|  | Missing | 52 |
| Occupation | Accountants, IT staff, Managers | 50 |
|  | High skill tech staff, Private service staff, HR staff, Core staff, Secretaries, Medicine staff, Realty agents | 49 |
|  | Sales staff, Cleaning staff | 45 |
|  | Cooking staff, Laborers, Security staff | 44 |
|  | Drivers, Waiters/barmen staff, Low-skill Laborers | 42 |
|  | Missing | 49 |
| Employer industry | Trade: type 4, Trade: type 5, Trade: type 6, Industry: type 12, Military, Religion, Industry: type 6, Security Ministries, Police, University, Insurance, Bank, XNA, Transport: type 1, School, Culture, Services | 53 |
|  | Trade: type 2, Industry: type 10, Industry: type 9, Government | 49 |
|  | Medicine, Hotel, Emergency, Kindergarten | 49 |
|  | Transport: type 2, Electricity, Industry: type 7, Other, Industry: type 5, Business Entity Type 1 | 48 |
|  | Mobile, Housing, Postal, Telecom, Business Entity Type 2, Industry: type 11, Industry: type 4 | 46 |
|  | Industry: type 2, Transport: type 4, Business Entity Type 3, Advertising, Trade: type 7, Legal Services, Trade: type 1 | 44 |
|  | Trade: type 3, Self-employed, Industry: type 3 | 42 |
|  | Security, Agriculture, Realtor, Restaurant, Industry: type 8, Industry: type 1, Construction, Cleaning, Industry: type 13, Transport: type 3 | 40 |
|  | Missing | 46 |
| Education | Academic degree, Higher education | 53 |
|  | Incomplete higher, Secondary / secondary special, Lower secondary | 45 |
|  | Missing | 46 |
| Region rating (1 = best) | (-inf, 1.50) | 52 |
|  | [1.50, 2.50) | 47 |
|  | [2.50, inf) | 43 |
|  | Missing | 46 |
| Price of goods financed | (-inf, 138321.00) | 50 |
|  | [138321.00, 282703.50) | 46 |
|  | [282703.50, 407362.50) | 42 |
|  | [407362.50, 450117.00) | 38 |
|  | [450117.00, 677889.00) | 47 |
|  | [677889.00, 902371.50) | 50 |
|  | [902371.50, 1125056.25) | 52 |
|  | [1125056.25, inf) | 55 |
|  | Missing | 46 |
| Loan amount / goods price | (-inf, 1.10) | 50 |
|  | [1.10, 1.13) | 48 |
|  | [1.13, 1.16) | 48 |
|  | [1.16, 1.21) | 44 |
|  | [1.21, 1.32) | 42 |
|  | [1.32, inf) | 39 |
|  | Missing | 47 |
| Number of payments | (-inf, 10.78) | 57 |
|  | [10.78, inf) | 45 |
|  | Missing | 46 |
| Age of own car (years) | (-inf, 3.50) | 52 |
|  | [3.50, 8.50) | 53 |
|  | [8.50, 12.50) | 48 |
|  | [12.50, 16.50) | 45 |
|  | [16.50, inf) | 44 |
|  | Missing | 45 |

## Appendix B. Facts and sources

| Fact | Value | Source |
|---|---|---|
| `test_gini` | 0.490 | `app/data/metrics.json` |
| `test_auc` | 0.745 | `app/data/metrics.json` |
| `n_final_variables` | 12 | `app/data/metrics.json` |
| `lgd` | 45% | `app/data/metrics.json` |
| `base_score` | 600 | `app/data/metrics.json` |
| `base_odds` | 50:1 | `app/data/metrics.json` |
| `pdo` | 20 | `app/data/metrics.json` |
| `total_ead` | 184.21 bn | `app/data/metrics.json` |
| `n_portfolio_loans` | 307,511 | `app/data/metrics.json` |
| `sicr_threshold` | 20% | `src/ecl.py` |
| `ecl_base` | 6.73 bn | `app/data/ecl_by_scenario_stage.csv` |
| `weight_base` | 60% | `app/data/ecl_by_scenario_stage.csv` |
| `stage2_share_base` | 7.3% | `app/data/ecl_by_scenario_stage.csv` |
| `ecl_adverse` | 9.76 bn | `app/data/ecl_by_scenario_stage.csv` |
| `weight_adverse` | 30% | `app/data/ecl_by_scenario_stage.csv` |
| `stage2_share_adverse` | 14.1% | `app/data/ecl_by_scenario_stage.csv` |
| `ecl_severe` | 13.12 bn | `app/data/ecl_by_scenario_stage.csv` |
| `weight_severe` | 10% | `app/data/ecl_by_scenario_stage.csv` |
| `stage2_share_severe` | 22.2% | `app/data/ecl_by_scenario_stage.csv` |
| `ecl_weighted` | 8.28 bn | `app/data/ecl_by_scenario_stage.csv` |
| `ecl_weighted_vs_base` | +23% | `app/data/ecl_by_scenario_stage.csv` |
| `ecl_adverse_vs_base` | +45% | `app/data/ecl_by_scenario_stage.csv` |
| `ecl_severe_vs_base` | +95% | `app/data/ecl_by_scenario_stage.csv` |
| `calib_max_gap_pp` | 0.29 percentage points | `app/data/calibration.csv` |
| `top_decile_ecl_share` | 28% | `app/data/ecl_by_decile.csv` |
| `approval_ratio_gender_f` | 1.00 | `app/data/fairness.csv` |
| `observed_default_gender_f` | 7.01% | `app/data/fairness.csv` |
| `predicted_pd_gender_f` | 7.47% | `app/data/fairness.csv` |
| `approval_ratio_gender_m` | 0.90 | `app/data/fairness.csv` |
| `observed_default_gender_m` | 10.14% | `app/data/fairness.csv` |
| `predicted_pd_gender_m` | 9.24% | `app/data/fairness.csv` |
| `approval_ratio_age_group_under_30` | 0.67 | `app/data/fairness.csv` |
| `observed_default_age_group_under_30` | 11.29% | `app/data/fairness.csv` |
| `predicted_pd_age_group_under_30` | 12.51% | `app/data/fairness.csv` |
| `approval_ratio_age_group_30_50` | 0.89 | `app/data/fairness.csv` |
| `observed_default_age_group_30_50` | 8.66% | `app/data/fairness.csv` |
| `predicted_pd_age_group_30_50` | 8.20% | `app/data/fairness.csv` |
| `approval_ratio_age_group_over_50` | 1.00 | `app/data/fairness.csv` |
| `observed_default_age_group_over_50` | 5.77% | `app/data/fairness.csv` |
| `predicted_pd_age_group_over_50` | 5.94% | `app/data/fairness.csv` |
| `groups_failing_80_rule` | Under 30 | `app/data/fairness.csv` |
| `n_applications` | 307,511 | `notebooks/01_explore.ipynb` |
| `n_raw_columns` | 122 | `notebooks/01_explore.ipynb` |
| `default_rate` | 8.07% | `notebooks/01_explore.ipynb` |
| `n_long_list` | 23 | `notebooks/01_explore.ipynb` |
| `max_bins` | 8 | `notebooks/03_woe_iv.ipynb` |
| `min_bin_share` | 5% | `notebooks/03_woe_iv.ipynb` |
| `iv_threshold` | 0.02 | `notebooks/03_woe_iv.ipynb` |
| `corr_threshold` | 0.6 | `notebooks/03_woe_iv.ipynb` |
| `train_gini` | 0.492 | `notebooks/05_validation.ipynb` |
| `psi_train_test` | 0.0001 | `notebooks/05_validation.ipynb` |
| `ecl_12m_unstaged` | 5.97 bn | `notebooks/06_ecl.ipynb` |
| `coverage_12m` | 3.24% | `notebooks/06_ecl.ipynb` |
| `staging_uplift` | +12.6% | `notebooks/07_staging.ipynb` |
| `stage2_share_of_ecl` | 31% | `notebooks/07_staging.ipynb` |
| `sicr_threshold_low` | 10% | `notebooks/07_staging.ipynb` |
| `sicr_threshold_high` | 30% | `notebooks/07_staging.ipynb` |
| `ecl_sicr_10pct` | 8.2 bn | `notebooks/07_staging.ipynb` |
| `ecl_sicr_30pct` | 6.2 bn | `notebooks/07_staging.ipynb` |
| `pd_target_adverse` | 11% | `notebooks/08_stress_test.ipynb` |
| `pd_target_severe` | 14% | `notebooks/08_stress_test.ipynb` |
| `lgd_stressed` | 75% | `notebooks/08_stress_test.ipynb` |
| `lgd_75_uplift` | +67% | `notebooks/08_stress_test.ipynb` |
| `xgb_test_gini` | 0.531 | `notebooks/09_xgboost_challenger.ipynb` |
| `xgb_gini_gain` | +0.041 | `notebooks/09_xgboost_challenger.ipynb` |
| `gender_proxy_auc` | 0.720 | `notebooks/10_fairness.ipynb` |
| `n_bureau_passed_iv` | 4 | `notebooks/11_bureau_features.ipynb` |
| `bureau_gini_gain` | 0.002-0.004 | `notebooks/11_bureau_features.ipynb` |

## Appendix C. How this document was produced

This document was produced by `src/model_doc/generate.py`:

1. **Facts** (`facts.py`): every number the document may state is collected by code
   from the aggregated results in `app/data/` and the notebooks, with its source
   (Appendix B). No client-level data is used.
2. **Tables** (`tables.py`): all tables are built by code from the facts.
3. **Prose** (`narrative.py`): the Claude API drafts each section from a qualitative
   project description and the list of facts. It never writes a number; it writes a
   placeholder such as `{{test_gini}}`.
4. **Check** (`check.py`): code rejects unknown placeholders and numbers typed
   directly in the text, then replaces each placeholder with its value. Open points
   the model could not resolve are marked `[REVIEWER: ...]` and listed at the top.
5. **Human review**: the model owner reviews the draft, resolves the open points and
   records the changes in the review notes. The raw drafts with placeholders are kept
   in `docs/model_documentation_drafts.json` for traceability.


## Review notes

_To be completed by the model owner: what was checked, what was changed and why._
