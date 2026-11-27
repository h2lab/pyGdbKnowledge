# SPDX-FileCopyrightText: 2026 H2Lab Development Team
# SPDX-License-Identifier: Apache-2.0

"""Tests for the concept registry and controlled query expansion."""

from pathlib import Path

import pytest
import yaml

from silicore_knowledge.retrieval.concepts import (
    MAX_EXPANSION_TERMS,
    MAX_MATCHED_CONCEPTS,
    Concept,
    ConceptError,
    ConceptRegistry,
    load_concepts,
)

REGISTRY = ConceptRegistry(
    [
        Concept(
            id="firmware-extraction",
            terms=("firmware extraction", "readout protection", "RDP"),
            related=("debug-access",),
        ),
        Concept(id="debug-access", terms=("JTAG", "SWD", "debug port")),
        Concept(id="clocks", terms=("PLL", "clock tree")),
    ]
)


def test_match_uses_word_boundaries_and_phrases() -> None:
    """Terms match whole words and whitespace-separated phrases, case-insensitively."""
    assert [c.id for c in REGISTRY.match("enable the jtag probe")] == ["debug-access"]
    assert [c.id for c in REGISTRY.match("readout  protection level")] == ["firmware-extraction"]
    assert REGISTRY.match("cardpiler rdps") == ()  # no substring matches
    assert REGISTRY.match("nothing relevant") == ()


def test_expansion_adds_sibling_and_related_terms_only() -> None:
    """Expansion holds the other terms of matched and related concepts, not the query's."""
    terms = REGISTRY.expansion_terms("prevent firmware extraction")
    assert "firmware extraction" not in terms
    assert {"readout protection", "RDP", "JTAG", "SWD", "debug port"} == set(terms)
    # Related links are one hop only and not symmetric.
    assert set(REGISTRY.expansion_terms("swd pins")) == {"JTAG", "debug port"}
    assert REGISTRY.expansion_terms("nothing relevant") == ()


def test_expansion_is_bounded() -> None:
    """Matched concepts and expansion terms are capped."""
    concepts = [
        Concept(id=f"c{i}", terms=(f"alpha{i}", *(f"term{i}x{j}" for j in range(8))))
        for i in range(6)
    ]
    registry = ConceptRegistry(concepts)
    query = " ".join(f"alpha{i}" for i in range(6))
    assert len(registry.match(query)) == MAX_MATCHED_CONCEPTS
    assert len(registry.expansion_terms(query)) == MAX_EXPANSION_TERMS


def test_triggers_narrow_matching_without_restricting_retrieval_terms() -> None:
    """A concept can match intent wording and still expand to its technical vocabulary."""
    registry = ConceptRegistry(
        [
            Concept(
                id="dma-transfer",
                terms=("DMA", "direct memory access", "DMA stream"),
                triggers=("transfer data without CPU intervention",),
            )
        ]
    )
    assert registry.match("DMA stream FIFO control register") == ()
    assert set(registry.expansion_terms("transfer data without CPU intervention")) == {
        "DMA",
        "direct memory access",
        "DMA stream",
    }


def test_inconsistent_registries_are_rejected() -> None:
    """Duplicate ids and unknown related ids raise ConceptError."""
    concept = Concept(id="a", terms=("x",))
    with pytest.raises(ConceptError, match="duplicate"):
        ConceptRegistry([concept, concept])
    with pytest.raises(ConceptError, match="unknown"):
        ConceptRegistry([Concept(id="a", terms=("x",), related=("ghost",))])


def test_load_concepts_validates_the_file(tmp_path: Path) -> None:
    """A valid file loads; malformed or unreadable files raise ConceptError."""
    path = tmp_path / "c.yaml"
    path.write_text(
        yaml.safe_dump({"concepts": [{"id": "a", "terms": ["x"], "label": "A"}]}),
        encoding="utf-8",
    )
    assert len(load_concepts(path)) == 1
    path.write_text(yaml.safe_dump({"concepts": [{"id": "a"}]}), encoding="utf-8")
    with pytest.raises(ConceptError, match="invalid"):
        load_concepts(path)
    with pytest.raises(ConceptError, match="cannot read"):
        load_concepts(tmp_path / "missing.yaml")


def test_repository_concepts_are_valid() -> None:
    """The registry shipped in config/ must stay loadable and useful for intent queries."""
    registry = load_concepts(Path(__file__).parent.parent / "config" / "concepts.yaml")
    terms = registry.expansion_terms("prevent firmware extraction from the microcontroller")
    assert any("readout" in term.lower() for term in terms)
    assert any(term.upper() == "JTAG" for term in terms)


def test_repository_concepts_avoid_expanding_precise_register_queries() -> None:
    """Specific RCC register questions stay exact; intent phrases still activate concepts."""
    registry = load_concepts(Path(__file__).parent.parent / "config" / "concepts.yaml")
    assert registry.expansion_terms("RCC AHB1 peripheral clock enable register") == ()
    assert (
        registry.expansion_terms("RCC peripherals independent clock configuration register") == ()
    )
    assert (
        registry.expansion_terms("DMA stream x FIFO control register channel selection CHSEL") == ()
    )
    assert registry.expansion_terms("secure maintenance firmware update") == ()
    assert registry.expansion_terms("stop mode") == ()
    assert registry.expansion_terms("configure the system clock")
    assert {"JTAG", "SWD"} <= set(registry.expansion_terms("lock the debug interface"))
    assert registry.expansion_terms("update the firmware in the field")
    assert registry.expansion_terms("extend the battery life")
    assert "DMA" in registry.expansion_terms("transfer data without CPU intervention")


def test_repository_concepts_map_physical_intrusion_to_hardware_security_terms() -> None:
    """A physical-intrusion intent expands to the terms used by the security guide."""
    registry = load_concepts(Path(__file__).parent.parent / "config" / "concepts.yaml")
    terms = registry.expansion_terms(
        "Detect an intruder opening the enclosure before components are removed"
    )
    assert {"physical attack surface", "forced opening", "active detection and reaction"} <= set(
        terms
    )


def test_repository_concepts_map_modified_firmware_to_secure_boot_terms() -> None:
    """A firmware-integrity intent expands to secure-boot evidence vocabulary."""
    registry = load_concepts(Path(__file__).parent.parent / "config" / "concepts.yaml")
    terms = registry.expansion_terms("prevent modified firmware from executing after reset")
    assert set(terms) == {"image integrity", "signature verification"}
    assert "image integrity" not in registry.expansion_terms("secure boot chain of trust")
