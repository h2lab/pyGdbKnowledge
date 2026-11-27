# SPDX-FileCopyrightText: 2026 H2Lab Development Team
# SPDX-License-Identifier: Apache-2.0

"""Local knowledge server for pyGdbToolkit."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("silicore-knowledge")
except PackageNotFoundError:  # running from a source tree that is not installed
    __version__ = "0+unknown"
