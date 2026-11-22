# SPDX-FileCopyrightText: 2026 H2Lab Development Team
# SPDX-License-Identifier: Apache-2.0

"""CMSIS-SVD parsing and chunking: one section per peripheral, registers and bit fields as text.

SVD files are untrusted XML. Documents carrying a DOCTYPE or entity declaration are refused
(no entity expansion attacks), and file size, peripherals, registers and nesting are bounded.
Chunks use the same ``Bits 7:4 NAME: ...`` notation as the datasheets so that both corpora
answer the same questions.
"""

import hashlib
import re
from collections.abc import Iterator
from pathlib import Path
from xml.etree import ElementTree

from ..config import IngestionSettings
from ..storage.models import Chunk, Section
from .chunking import ChunkedDocument
from .extract import clean_text

SVD_MAGIC = b"<device"
MAX_SVD_BYTES = 64 * 1024 * 1024
_MAX_PERIPHERALS = 2000
_MAX_REGISTERS = 5000
_MAX_DEPTH = 8
_MAX_ENUMS = 16
_FORBIDDEN = (b"<!doctype", b"<!entity")
_SPACES_RE = re.compile(r"\s+")
_RANGE_RE = re.compile(r"^\[(\d+):(\d+)\]$")


class SvdError(Exception):
    """Raised when an SVD file is unsafe, malformed or exceeds the bounds."""


def _int(text: str | None) -> int | None:
    if text is None:
        return None
    value = text.strip().lower()
    try:
        if value.startswith("0x"):
            return int(value[2:], 16)
        if value.startswith("#"):
            return int(value[1:], 2)
        return int(value, 10)
    except ValueError:
        return None


def _text(element: ElementTree.Element, tag: str) -> str | None:
    child = element.find(tag)
    if child is None or child.text is None:
        return None
    value = _SPACES_RE.sub(" ", clean_text(child.text)).strip()
    return value or None


def _hex(value: int | None, width: int = 0) -> str:
    return "?" if value is None else f"0x{value:0{width}X}"


def _bits(field: ElementTree.Element) -> tuple[int, int] | None:
    """Return ``(msb, lsb)`` of a field in any of the three SVD notations."""
    offset, width = _int(_text(field, "bitOffset")), _int(_text(field, "bitWidth"))
    if offset is not None:
        return offset + (width or 1) - 1, offset
    lsb, msb = _int(_text(field, "lsb")), _int(_text(field, "msb"))
    if lsb is not None and msb is not None:
        return msb, lsb
    match = _RANGE_RE.match(_text(field, "bitRange") or "")
    if match:
        return int(match.group(1)), int(match.group(2))
    return None


def _field_line(field: ElementTree.Element, default_access: str | None) -> str | None:
    name, bits = _text(field, "name"), _bits(field)
    if name is None or bits is None:
        return None
    msb, lsb = bits
    span = f"{msb}:{lsb}" if msb != lsb else str(msb)
    access = _text(field, "access") or default_access
    line = f"Bits {span} {name}" + (f" ({access})" if access else "")
    description = _text(field, "description")
    if description:
        line += f": {description}"
    values = []
    for enum in field.findall("enumeratedValues/enumeratedValue")[:_MAX_ENUMS]:
        label, number = _text(enum, "name"), _text(enum, "value")
        if label and number:
            values.append(f"{number}={label}")
    if values:
        line += f" [values: {', '.join(values)}]"
    return line


class _Walker:
    """Flattens the registers and clusters of one peripheral into text blocks."""

    def __init__(self, peripheral: str, base: int | None, size: int | None, access: str | None):
        self.peripheral = peripheral
        self.base = base
        self.size = size
        self.access = access
        self.count = 0

    def blocks(
        self, parent: ElementTree.Element, prefix: str, origin: int, depth: int
    ) -> Iterator[list[str]]:
        if depth > _MAX_DEPTH:
            raise SvdError(f"clusters nested deeper than {_MAX_DEPTH} in {self.peripheral}")
        for node in parent:
            if node.tag == "register":
                self.count += 1
                if self.count > _MAX_REGISTERS:
                    raise SvdError(f"more than {_MAX_REGISTERS} registers in {self.peripheral}")
                yield self._register(node, prefix, origin)
            elif node.tag == "cluster":
                name = _text(node, "name") or "cluster"
                offset = _int(_text(node, "addressOffset")) or 0
                yield from self.blocks(node, f"{prefix}{name}.", origin + offset, depth + 1)

    def _register(self, node: ElementTree.Element, prefix: str, origin: int) -> list[str]:
        name = f"{prefix}{_text(node, 'name') or '?'}"
        offset = origin + (_int(_text(node, "addressOffset")) or 0)
        address = None if self.base is None else self.base + offset
        size = _int(_text(node, "size")) or self.size
        access = _text(node, "access") or self.access
        head = f"Register {self.peripheral}_{name} (offset {_hex(offset)}, address {_hex(address)}"
        if size:
            head += f", size {size}"
        if access:
            head += f", access {access}"
        reset = _int(_text(node, "resetValue"))
        if reset is not None:
            head += f", reset {_hex(reset, (size or 32) // 4)}"
        dim = _int(_text(node, "dim"))
        if dim:
            head += f", array of {dim} every {_hex(_int(_text(node, 'dimIncrement')))}"
        head += ")"
        description = _text(node, "description")
        if description:
            head += f": {description}"
        lines = [head]
        for field in node.findall("fields/field"):
            line = _field_line(field, access)
            if line:
                lines.append("  " + line)
        return lines


def _peripherals(root: ElementTree.Element) -> list[ElementTree.Element]:
    found = root.findall("peripherals/peripheral")
    if len(found) > _MAX_PERIPHERALS:
        raise SvdError(f"more than {_MAX_PERIPHERALS} peripherals")
    return found


def parse_svd(path: Path) -> ElementTree.Element:
    """Read and parse an SVD file defensively.

    Raises
    ------
    SvdError
        If the file is too large, unreadable, declares a DOCTYPE or entities, is not XML, or
        its root element is not ``device``.
    """
    try:
        if path.stat().st_size > MAX_SVD_BYTES:
            raise SvdError("SVD file is too large")
        data = path.read_bytes()
    except OSError as exc:
        raise SvdError(f"cannot read the SVD file: {exc.strerror}") from exc
    lowered = data.lower()
    if any(token in lowered for token in _FORBIDDEN):
        raise SvdError("DOCTYPE and entity declarations are not accepted")
    try:
        root = ElementTree.fromstring(data)  # noqa: S314 - DTDs rejected above
    except ElementTree.ParseError as exc:
        raise SvdError(f"invalid XML: {exc}") from exc
    if root.tag != "device":
        raise SvdError("not an SVD file (root element is not 'device')")
    return root


def _pack(blocks: list[list[str]], max_chars: int) -> list[list[str]]:
    """Group register blocks into chunks; an oversized block is split, header repeated."""
    packed: list[list[str]] = []
    current: list[str] = []
    size = 0
    for block in blocks:
        pieces = [block]
        if sum(len(line) + 1 for line in block) > max_chars and len(block) > 1:
            head, rest = block[0], block[1:]
            pieces, group, used = [], [head], len(head) + 1
            for line in rest:
                if used + len(line) + 1 > max_chars and len(group) > 1:
                    pieces.append(group)
                    group, used = [head + " (continued)"], len(head) + 13
                group.append(line)
                used += len(line) + 1
            pieces.append(group)
        for piece in pieces:
            piece_size = sum(len(line) + 1 for line in piece)
            if current and size + piece_size > max_chars:
                packed.append(current)
                current, size = [], 0
            current.extend(piece)
            size += piece_size
    if current:
        packed.append(current)
    return packed


def chunk_svd(document_id: str, path: Path, settings: IngestionSettings) -> ChunkedDocument:
    """Build sections and chunks from an SVD file.

    One section per peripheral (``s<ordinal>``); chunk ids and hashes are stable for the same
    file. A peripheral without registers of its own (``derivedFrom``) reuses the registers of
    its base, with its own name and base address.

    Raises
    ------
    SvdError
        If the file is unsafe or malformed (see ``parse_svd``).
    """
    root = parse_svd(path)
    device = _text(root, "name") or path.stem
    default_size, default_access = _int(_text(root, "size")), _text(root, "access")
    elements = _peripherals(root)
    by_name = {_text(e, "name"): e for e in elements}
    sections: list[Section] = []
    chunks: list[Chunk] = []
    for ordinal, element in enumerate(elements):
        name = _text(element, "name")
        if name is None:
            continue
        base_name = element.get("derivedFrom")
        source = element
        if element.find("registers") is None and base_name in by_name:
            source = by_name[base_name]
        base = _int(_text(element, "baseAddress"))
        head = [f"Peripheral {name} of {device} (base address {_hex(base, 8)})"]
        description = _text(element, "description") or _text(source, "description")
        if description:
            head[0] += f": {description}"
        if base_name:
            head.append(f"Same registers as {base_name}.")
        for interrupt in element.findall("interrupt"):
            irq, number = _text(interrupt, "name"), _text(interrupt, "value")
            if irq and number:
                head.append(f"Interrupt {irq} = {number}")
        walker = _Walker(
            name, base, _int(_text(source, "size")) or default_size,
            _text(source, "access") or default_access,
        )  # fmt: skip
        registers = source.find("registers")
        blocks = [head]
        if registers is not None:
            blocks.extend(walker.blocks(registers, "", 0, 0))
        section_id = f"s{ordinal:04d}"
        path_text = f"{device} > {name}"
        sections.append(
            Section(
                document_id=document_id,
                section_id=section_id,
                title=name,
                path=path_text,
                ordinal=ordinal,
            )
        )
        for index, lines in enumerate(_pack(blocks, settings.chunk_max_chars)):
            text = "\n".join(lines)
            chunks.append(
                Chunk(
                    chunk_id=f"{document_id}:{section_id}:c{index:03d}",
                    document_id=document_id,
                    ordinal=len(chunks),
                    text=text,
                    content_hash=hashlib.sha256(text.encode()).hexdigest(),
                    section_id=section_id,
                    section_path=path_text,
                    peripheral=name.upper(),
                )
            )
    return ChunkedDocument(tuple(sections), tuple(chunks))
