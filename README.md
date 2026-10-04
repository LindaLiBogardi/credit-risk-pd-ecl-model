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