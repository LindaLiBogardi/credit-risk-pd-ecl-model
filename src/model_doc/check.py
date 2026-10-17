"""Check Claude's draft and fill in the numbers.

Claude writes placeholders such as {{test_gini}} instead of numbers. This module:
- finds placeholders whose key is not a known fact,
- finds digits typed directly in the text (outside an allowed list of terms),
- collects the [REVIEWER: ...] notes for the human reviewer,
- replaces each placeholder with the fact's display value.

No API calls, so it is tested in CI.
"""

import re

PLACEHOLDER = re.compile(r"\{\{\s*([A-Za-z0-9_]+)\s*\}\}")
REVIEWER_NOTE = re.compile(r"\[REVIEWER:[^\]]*\]")
NUMBER = re.compile(r"\d+(?:[.,]\d+)*%?")

# Terms that contain digits but are names, not results
ALLOWED_TERMS = [
    r"IFRS 9",
    r"Article 178",
    r"12-month",
    r"Stage [123]",
    r"80% rule",
    r"external credit scores? [123](?:(?:,| and| or|, and|, or|/) ?[123])*",
    r"under 30",
    r"30-50",
    r"over 50",
]
ALLOWED = re.compile("|".join(f"(?:{t})" for t in ALLOWED_TERMS), re.IGNORECASE)


def unknown_placeholders(text: str, facts: dict) -> list[str]:
    """Placeholder keys that are not in the facts."""
    return sorted({key for key in PLACEHOLDER.findall(text) if key not in facts})


def stray_numbers(text: str) -> list[str]:
    """Numbers typed directly in the text instead of as placeholders."""
    text = PLACEHOLDER.sub(" ", text)
    text = REVIEWER_NOTE.sub(" ", text)
    text = ALLOWED.sub(" ", text)
    return NUMBER.findall(text)


def reviewer_notes(text: str) -> list[str]:
    return REVIEWER_NOTE.findall(text)


def check_section(text: str, facts: dict) -> list[str]:
    """Problems found in one drafted section; an empty list means it passed."""
    problems = [f"unknown fact key: {k}" for k in unknown_placeholders(text, facts)]
    problems += [f"number typed directly: {n}" for n in stray_numbers(text)]
    return problems


def fill_placeholders(text: str, facts: dict) -> str:
    """Replace every {{key}} with the fact's display value."""

    def value(match: re.Match) -> str:
        key = match.group(1)
        if key not in facts:
            raise KeyError(f"Unknown fact key in draft: {key}")
        return facts[key]["display"]

    return PLACEHOLDER.sub(value, text)