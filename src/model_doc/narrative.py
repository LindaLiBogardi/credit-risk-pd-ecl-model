"""Draft the prose sections of the model documentation with the Claude API.

Claude never types a number. It writes placeholders such as {{test_gini}}, and code
fills them from the facts (see check.py). Only aggregated facts and a qualitative
project description are sent to the API; no client-level data.

Try one section from the project root:

    python -m src.model_doc.narrative performance
"""

import os
import sys

import anthropic

MODEL = os.environ.get("MODEL_DOC_MODEL", "claude-sonnet-5-5")
MAX_TOKENS = 8000  # a ceiling, not a target; only tokens used are billed

SYSTEM_PROMPT = """\
You draft sections of an internal model development document for a retail credit
risk model: a probability of default (PD) scorecard and an IFRS 9 expected credit
loss (ECL) calculation. Readers are model validators and a model risk committee.

Rules:
1. Use only the PROJECT CONTEXT and the FACTS you are given. Do not add benchmarks,
   industry comparisons, regulatory requirements or claims that are not in them.
2. Never write a number yourself, in digits or in words. Every quantity must be a
   placeholder with a fact key in double braces, e.g. "The test Gini is {{test_gini}}."
   Use only keys from the FACTS list. Code replaces each placeholder with its value.
   Exceptions you may write as plain text: "IFRS 9", "Article 178", "12-month",
   "Stage 1", "Stage 2", "Stage 3", "80% rule", "external credit score 1/2/3", and
   the age group names "under 30", "30-50" and "over 50".
3. If a point needs information that is not in the context or facts, do not guess.
   Write a note for the reviewer instead: [REVIEWER: what is missing].
4. Style: formal, concise and neutral, in the third person and past tense for work
   done. No marketing language. Plain Markdown paragraphs; use a short bullet list
   only when listing items. Do not write headings; they are added by code.
   Keep each section under about 350 words unless the section asks for less.
5. Return only the section text, with no preamble.
"""

PROJECT_CONTEXT = """\
Data: the Home Credit Default Risk dataset (Kaggle), one row per loan application.
The target flags early payment difficulties as defined by Home Credit; it is not the
regulatory default definition in CRR Article 178. The data contains only approved
loans and covers a single period, without dates.

Exploration and long list: before any statistical testing, an expert long list of
candidate risk drivers was written, each with an expected direction of effect and a
business reason. Gender, the application weekday and building-related variables were
excluded up front. A placeholder value in employment length marks clients who are not
employed (almost all pensioners); it was replaced by a missing value plus a flag.
Affordability ratios were derived (credit/income, annuity/income, credit term,
credit/goods price). The data was split into a training and a test set, stratified on
the target with a fixed seed; everything learned from data used the training set only.

Variable selection: optimal binning (optbinning) with a maximum number of bins, a
minimum share of data per bin and a monotonic or peak/valley trend. Variables were
kept if their information value (IV) passed a threshold, if their WoE correlation with
other variables was below a threshold, after a business review of every variable's
bins, and after backward elimination with cross-validation. The expected direction of
effect was compared with the evidence for every variable. The affordability ratios
were weak, most likely because of selection bias: only approved loans are observed.

PD model: logistic regression on WoE values; all coefficients have the expected sign.
Final variables: the three external credit scores, employment length, occupation,
employer industry, education, regional rating, goods price, credit/goods ratio, credit
term and car age. Points are scaled so that a base score corresponds to base good:bad
odds, and a fixed number of points (PDO) doubles the odds. On the test set the
observed default rate falls steadily across score bands.

Populations: discrimination, calibration, stability and fairness are measured on the
held-out test set. ECL, staging and stress testing are calculated on the full
portfolio of all loans in the dataset.

ECL: ECL = PD x LGD x EAD, with a fixed LGD assumption and EAD equal to the loan
amount. Staging: a loan moves to Stage 2 when its 12-month PD exceeds the SICR
threshold; Stage 2 loans get lifetime ECL from a constant annual default rate over the
remaining term. Because there is no origination PD, this absolute threshold is a proxy
for a significant increase in credit risk. Moving the threshold changes ECL
materially, which shows how much the SICR rule matters.

Stress testing: base, adverse and severe scenarios with assumed probability weights.
Scenario PDs come from a log-odds shift that preserves the ranking of clients,
calibrated so the average PD reaches each scenario's target. ECL rises faster than PD
because loans migrate to Stage 2; most of the increase comes from higher PDs and the
rest from stage migration. LGD sensitivity: a higher LGD assumption raises the
probability-weighted ECL by an amount comparable to the severe scenario.

Challenger: an XGBoost model on the cleaned raw features has a higher Gini. All
scorecard variables are among its most important features; its extra power comes from
interactions and from variables excluded for good reasons, such as an undocumented
document flag and counts of defaults in the client's social circle. The scorecard was
kept for explainability.

Credit bureau data: aggregated bureau features passed IV screening but added very
little Gini, most likely because the external credit scores already contain bureau
information. They were not adopted.

Fairness: gender is excluded from the model but was joined back for testing. The model
variables still predict gender well (proxy AUC), mainly through occupation and
employer industry. Approval rates were compared for an approval policy that accepts
the highest-scoring applicants; the 80% rule compares each group's approval rate with
the highest group's. The model over-predicts risk for women and for clients under 30
relative to their observed default rates. Correcting the gender gap would require
using gender, which anti-discrimination rules prohibit: a known trade-off between
fairness definitions. The 80% rule is a US screening heuristic, not an EU legal
standard; only gender and age are available for testing.

Limitations: target is not the regulatory default definition; no time dimension, so
no out-of-time validation, no origination PD for real staging and no Stage 3; LGD and
EAD are assumptions (no recovery data, no credit conversion factor for revolving
loans); scenario PDs are assumed rather than derived from a macroeconomic satellite
model; selection bias from approved loans only. In a bank, the model would also go
through independent validation, model risk governance and a parallel run against the
existing model before use.
"""

# id: (heading, what the section should cover)
SECTIONS = {
    "summary": (
        "Executive summary",
        "At most about 200 words: what the model is for, the main performance and "
        "calibration results, the ECL results with staging and scenarios, the key "
        "fairness finding and the most important limitations.",
    ),
    "purpose": (
        "Purpose and scope",
        "What the model estimates and how it is used (12-month PD for application "
        "scoring and as input to IFRS 9 ECL). State clearly that this is a portfolio "
        "project on public data, not a production model.",
    ),
    "data": (
        "Data",
        "Dataset size, number of raw columns, default rate, target definition, "
        "exclusions, the expert long list, cleaning, derived features and the split.",
    ),
    "methodology": (
        "Methodology",
        "Binning settings, the selection steps and thresholds, the selection-bias "
        "finding, the logistic regression and the score scaling.",
    ),
    "performance": (
        "Model performance",
        "Discrimination on test and training data (overfitting check), calibration "
        "across PD deciles, stability (PSI). A table with these results follows "
        "the text, so interpret rather than list every number.",
    ),
    "ecl": (
        "IFRS 9 expected credit loss, staging and stress testing",
        "12-month ECL and coverage, the effect of staging, the sensitivity to the "
        "SICR threshold, the scenarios and probability-weighted ECL, stage migration "
        "and the LGD sensitivity. A scenario table follows the text.",
    ),
    "challenger": (
        "Challenger model and additional data",
        "The XGBoost challenger and why the scorecard was kept; the credit bureau "
        "experiment and why it was not adopted.",
    ),
    "fairness": (
        "Fairness",
        "The proxy test, approval-rate comparison (80% rule), group calibration, the "
        "trade-off between fairness definitions and the limits of the test. A table "
        "by group follows the text.",
    ),
    "limitations": (
        "Limitations and model risk",
        "The limitations and what a bank would do before using such a model. "
        "Present them as a short bullet list with one sentence each.",
    ),
}


def facts_for_prompt(facts: dict) -> str:
    """Facts as 'key = value (source)' lines, so Claude knows what each key holds."""
    return "\n".join(f"{key} = {f['display']} (source: {f['source']})" for key, f in facts.items())


def build_prompt(section_id: str, facts: dict, feedback: list[str] | None = None) -> str:
    heading, guidance = SECTIONS[section_id]
    prompt = (
        f"PROJECT CONTEXT\n{PROJECT_CONTEXT}\n"
        f"FACTS\n{facts_for_prompt(facts)}\n\n"
        f"SECTION TO WRITE: {heading}\n{guidance}"
    )
    if feedback:
        prompt += (
            "\n\nYOUR PREVIOUS DRAFT FAILED THE AUTOMATED CHECK:\n- "
            + "\n- ".join(feedback)
            + "\nWrite the section again. Use only fact keys from the FACTS list, and no "
            "numbers outside placeholders. If a quantity has no fact key, leave it out."
        )
    return prompt


def draft_section(
    client: anthropic.Anthropic, section_id: str, facts: dict, feedback: list[str] | None = None
) -> dict:
    """Draft one section. Returns the raw text (with placeholders) and token usage."""
    response = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": build_prompt(section_id, facts, feedback)}],
    )
    if response.stop_reason != "end_turn":
        raise RuntimeError(f"Section '{section_id}' stopped early: {response.stop_reason}")
    text = "".join(block.text for block in response.content if block.type == "text")
    return {
        "text": text.strip(),
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
    }


if __name__ == "__main__":
    from src.model_doc.facts import build_facts

    section = sys.argv[1] if len(sys.argv) > 1 else "performance"
    result = draft_section(anthropic.Anthropic(), section, build_facts())
    print(result["text"])
    print(f"\n[{MODEL}: {result['input_tokens']} input, {result['output_tokens']} output tokens]")