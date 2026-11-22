# SPDX-FileCopyrightText: 2026 H2Lab Development Team
# SPDX-License-Identifier: Apache-2.0

"""Generate small PDF files for tests (reportlab)."""

from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib.pdfencrypt import StandardEncryption
from reportlab.pdfgen.canvas import Canvas


def make_pdf(
    path: Path,
    bodies: list[str],
    toc: list[list[object]] | None = None,
    *,
    title: str | None = None,
    running_header: bool = False,
    password: str | None = None,
) -> Path:
    """Write a PDF with one page per body; ``toc`` holds ``[level, title, page]`` entries."""
    encrypt = StandardEncryption(password, ownerPassword="owner") if password else None
    canvas = Canvas(str(path), pagesize=A4, encrypt=encrypt)
    if title:
        canvas.setTitle(title)
    for number, body in enumerate(bodies, start=1):
        for index, (level, name, page) in enumerate(toc or []):
            if page == number:
                key = f"b{index}"
                canvas.bookmarkPage(key)
                canvas.addOutlineEntry(str(name), key, level=int(str(level)) - 1)
        if running_header:
            canvas.drawString(72, 800, f"DS999 Rev 3 - page {number}/{len(bodies)}")
        for line_no, line in enumerate(body.splitlines()):
            canvas.drawString(72, 720 - 14 * line_no, line)
        if running_header:
            canvas.drawString(72, 40, "ACME Confidential")
        canvas.showPage()
    canvas.save()
    return path
