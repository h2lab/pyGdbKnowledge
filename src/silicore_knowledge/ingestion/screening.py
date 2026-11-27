# SPDX-FileCopyrightText: 2026 H2Lab Development Team
# SPDX-License-Identifier: Apache-2.0

"""Screening of untrusted content and metadata (ANSSI R22, R25).

Indexed passages are later given to a language model: text that tries to address the model
(instruction overrides, fake chat roles, bidirectional text tricks) is a poisoning signal.
Detection is heuristic and deliberately narrow; a hit sends the document to quarantine for
human review rather than silently dropping text.
"""

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from ..storage.models import Chunk

_PATTERNS: dict[str, re.Pattern[str]] = {
    "instruction_override": re.compile(
        r"\b(ignore|disregard|forget)\s+(all\s+|any\s+|the\s+|your\s+)*"
        r"(previous|prior|above|earlier|preceding)\s+(instructions?|prompts?|rules?|context)\b"
        r"|\bignore[zr]?\s+(toutes\s+)?les\s+instructions\s+(précédentes|ci-dessus)",
        re.IGNORECASE,
    ),
    "role_hijack": re.compile(
        r"\byou\s+are\s+now\s+(a|an|in|the)\b|\bnew\s+instructions\s*:", re.IGNORECASE
    ),
    "prompt_disclosure": re.compile(
        r"\b(reveal|print|show|repeat)\s+(me\s+)?(your|the)\s+(system\s+)?(prompt|instructions)\b",
        re.IGNORECASE,
    ),
    "chat_markup": re.compile(
        r"<\|?\s*/?\s*(im_start|im_end|system|assistant)\s*\|?>|\[/?INST\]", re.IGNORECASE
    ),
    "concealment": re.compile(
        r"\bdo\s+not\s+(tell|inform|mention\s+(this\s+)?to)\s+the\s+user\b", re.IGNORECASE
    ),
    # Bidirectional overrides and isolates are never legitimate in a datasheet.
    "bidi_control": re.compile("[\u202a-\u202e\u2066-\u2069]"),
}


@dataclass(frozen=True, slots=True)
class Finding:
    """A screening hit: the rule that matched and where (never the matched text)."""

    rule: str
    chunk_id: str | None = None


def screen_text(text: str) -> list[str]:
    """Return the names of the rules matched by ``text``."""
    return [name for name, pattern in _PATTERNS.items() if pattern.search(text)]


def screen_chunks(chunks: Sequence[Chunk]) -> list[Finding]:
    """Screen every chunk (section path included)."""
    findings: list[Finding] = []
    for chunk in chunks:
        findings.extend(
            Finding(rule, chunk.chunk_id)
            for rule in screen_text(f"{chunk.section_path}\n{chunk.text}")
        )
    return findings


def screen_metadata(values: Iterable[str | None]) -> list[str]:
    """Return the rules matched by operator-supplied metadata values."""
    rules: list[str] = []
    for value in values:
        if value:
            rules.extend(rule for rule in screen_text(value) if rule not in rules)
    return rules
