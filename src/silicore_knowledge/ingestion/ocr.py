# SPDX-FileCopyrightText: 2026 H2Lab Development Team
# SPDX-License-Identifier: Apache-2.0

"""Optional OCR of scanned PDF files through an external ``ocrmypdf`` process.

OCR processes untrusted files with a large toolchain, so it runs in a separate process
(no shell, bounded time, scratch files only) and never modifies the stored original.
"""

import shutil
import subprocess
from pathlib import Path

from ..config import OcrSettings
from .extract import ExtractedDocument


class OcrError(Exception):
    """Raised when OCR is unavailable or fails."""


def needs_ocr(document: ExtractedDocument, settings: OcrSettings) -> bool:
    """Tell whether the extracted text is too sparse to be a real text layer."""
    total = sum(len(page.text) for page in document.pages)
    return total < settings.min_chars_per_page * document.page_count


def ocr_pdf(source: Path, destination: Path, settings: OcrSettings) -> None:
    """Write an OCR'd copy of ``source`` to ``destination``.

    Parameters
    ----------
    source : Path
        Original PDF.
    destination : Path
        Output PDF with a text layer; must not exist.
    settings : OcrSettings
        Language and time limit.

    Raises
    ------
    OcrError
        If ``ocrmypdf`` is missing, times out or fails.
    """
    binary = shutil.which("ocrmypdf")
    if binary is None:
        raise OcrError("ocrmypdf is not installed; install the 'ocr' extra and tesseract")
    command = [
        binary,
        "--skip-text",
        "--output-type",
        "pdf",
        "--jobs",
        "1",
        "-l",
        settings.language,
        str(source),
        str(destination),
    ]
    try:
        completed = subprocess.run(  # noqa: S603  fixed argument list, no shell
            command,
            check=False,
            capture_output=True,
            timeout=settings.timeout_s,
            stdin=subprocess.DEVNULL,
        )
    except subprocess.TimeoutExpired as exc:
        destination.unlink(missing_ok=True)
        raise OcrError(f"OCR exceeded {settings.timeout_s} seconds") from exc
    except OSError as exc:
        raise OcrError(f"cannot run ocrmypdf: {exc.strerror}") from exc
    if completed.returncode != 0 or not destination.is_file():
        destination.unlink(missing_ok=True)
        raise OcrError(f"OCR failed with exit code {completed.returncode}")
