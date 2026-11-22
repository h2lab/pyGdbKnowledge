# SPDX-FileCopyrightText: 2026 H2Lab Development Team
# SPDX-License-Identifier: Apache-2.0

"""Smoke tests for the command-line entry point."""

from pyGdbKnowledge.cli import main


def test_main_returns_zero() -> None:
    """The skeleton entry point exits successfully."""
    assert main([]) == 0
