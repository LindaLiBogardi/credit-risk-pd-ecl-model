"""Generate the model documentation draft: docs/model_documentation.md.

Pipeline: facts (code) -> tables (code) -> prose with placeholders (Claude API)
-> check (code) -> fill placeholders (code) -> Markdown document marked DRAFT.

Run from the project root (needs ANTHROPIC_API_KEY, about $0.20 per run):

    python -m src.model_doc.generate          # refuses to overwrite a reviewed document
    python -m src.model_doc.generate --force  # overwrite
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from src.model_doc.check import check_section, fill_placeholders, reviewer_notes
from src.model_doc.facts import FACTS_FILE, build_facts, fingerprint
from src.model_doc.narrative import MODEL, SECTIONS, draft_section
from src.model_doc.tables import build_tables

DOC_FILE = Path("docs/model_documentation.md")
DRAFTS_FILE = Path("docs/model_documentation_drafts.json")
MAX_ATTEMPTS = 3

# Table shown after a section's text
TABLE_AFTER = {"performance": "performance", "ecl": "scenarios", "fairness": "fairness"}

HOW_PRODUCED = """\
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
"""


def git_commit() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, check=True)
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def draft_checked_section(client, section_id: str, facts: dict) -> dict:
    """Draft a section; if the check fails, redraft with the problems as feedback."""
    problems = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        result = draft_section(client, section_id, facts, feedback=problems)
        problems = check_section(result["text"], facts)
        if not problems:
            return result
        print(f"  attempt {attempt} failed the check: {problems}")
    sys.exit(f"Section '{section_id}' failed the check {MAX_ATTEMPTS} times; no document written.")


def build_document(drafts: dict, facts: dict, tables: dict, usage: dict) -> str:
    today = datetime.now(timezone.utc).date().isoformat()
    notes = [(sid, note) for sid, d in drafts.items() for note in reviewer_notes(d["text"])]

    parts = [
        "# Model documentation: PD scorecard and IFRS 9 ECL",
        "> **DRAFT for human review.** Prose drafted with the Claude API; every number "
        "is inserted by code from `docs/facts.json`. See Appendix C.",
        "| | |\n|---|---|\n"
        f"| Generated | {today} |\n"
        f"| LLM | `{MODEL}` ({usage['input']:,} input, {usage['output']:,} output tokens) |\n"
        f"| Code version | git commit `{git_commit()}` |\n"
        f"| Facts fingerprint | `{fingerprint(facts)}` |",
        "## Open points for the reviewer",
    ]
    if notes:
        parts.append("\n".join(f"- **{SECTIONS[sid][0]}:** {note}" for sid, note in notes))
    else:
        parts.append("None.")

    for number, (sid, (heading, _)) in enumerate(SECTIONS.items(), start=1):
        parts.append(f"## {number}. {heading}")
        parts.append(fill_placeholders(drafts[sid]["text"], facts))
        if sid in TABLE_AFTER:
            parts.append(tables[TABLE_AFTER[sid]])

    parts += [
        "## Appendix A. Scorecard", tables["scorecard"],
        "## Appendix B. Facts and sources", tables["facts"],
        "## Appendix C. How this document was produced", HOW_PRODUCED,
        "## Review notes",
        "_To be completed by the model owner: what was checked, what was changed and why._",
    ]
    return "\n\n".join(parts) + "\n"


def generate(client, force: bool = False) -> None:
    if DOC_FILE.exists() and not force:
        sys.exit(f"{DOC_FILE} exists and may contain review edits. Use --force to overwrite.")

    facts = build_facts()
    tables = build_tables(facts)
    drafts, usage = {}, {"input": 0, "output": 0}
    for sid in SECTIONS:
        print(f"Drafting: {SECTIONS[sid][0]}")
        drafts[sid] = draft_checked_section(client, sid, facts)
        usage["input"] += drafts[sid]["input_tokens"]
        usage["output"] += drafts[sid]["output_tokens"]

    DOC_FILE.parent.mkdir(exist_ok=True)
    FACTS_FILE.write_text(json.dumps(facts, indent=2, default=float))
    DRAFTS_FILE.write_text(json.dumps({sid: d["text"] for sid, d in drafts.items()}, indent=2))
    DOC_FILE.write_text(build_document(drafts, facts, tables, usage))
    print(f"\nWrote {DOC_FILE} ({usage['input']:,} input, {usage['output']:,} output tokens)")


def main() -> None:
    import anthropic

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--force", action="store_true", help="overwrite an existing document")
    generate(anthropic.Anthropic(), force=parser.parse_args().force)


if __name__ == "__main__":
    main()