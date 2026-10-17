"""Tests for the model documentation pipeline (no API calls)."""

import pytest

from src.model_doc.check import (
    check_section,
    fill_placeholders,
    reviewer_notes,
    stray_numbers,
    unknown_placeholders,
)
from src.model_doc.tables import clean_bin, md_table

FACTS = {
    "test_gini": {"value": 0.49, "display": "0.490", "source": "app/data/metrics.json"},
    "approval_ratio_age_group_under_30": {"value": 0.67, "display": "0.67", "source": "x"},
}


def test_fill_placeholders_uses_display_values():
    text = "The test Gini is {{test_gini}}."
    assert fill_placeholders(text, FACTS) == "The test Gini is 0.490."


def test_fill_placeholders_rejects_unknown_key():
    with pytest.raises(KeyError):
        fill_placeholders("Gini {{invented_gini}}", FACTS)


def test_unknown_placeholders_found():
    text = "{{test_gini}} and {{invented_gini}}"
    assert unknown_placeholders(text, FACTS) == ["invented_gini"]


def test_digits_inside_placeholders_are_not_stray():
    text = "The ratio for under 30 is {{approval_ratio_age_group_under_30}}."
    assert stray_numbers(text) == []


def test_typed_numbers_are_stray():
    assert stray_numbers("The Gini is 0.49 and coverage 3.2%.") == ["0.49", "3.2%"]


def test_allowed_terms_are_not_stray():
    text = (
        "Under IFRS 9, Stage 2 loans get lifetime ECL; the 12-month PD and "
        "external credit scores 1, 2 and 3 are inputs; see Article 178 and the 80% rule."
    )
    assert stray_numbers(text) == []


def test_reviewer_notes_are_collected_and_not_checked():
    text = "Text. [REVIEWER: confirm the PSI threshold of 0.25.]"
    assert reviewer_notes(text) == ["[REVIEWER: confirm the PSI threshold of 0.25.]"]
    assert check_section(text, FACTS) == []


def test_check_section_reports_problems():
    problems = check_section("Gini {{nope}} is 0.5.", FACTS)
    assert problems == ["unknown fact key: nope", "number typed directly: 0.5"]


def test_clean_bin_turns_array_text_into_names():
    label = "<ArrowStringArray>\n['Accountants', 'IT staff']\nLength: 2, dtype: str"
    assert clean_bin(label) == "Accountants, IT staff"
    assert clean_bin("[0.18, 0.31)") == "[0.18, 0.31)"


def test_md_table_escapes_pipes():
    table = md_table(["A"], [["x|y"]])
    assert "x\\|y" in table