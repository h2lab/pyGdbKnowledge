# SPDX-FileCopyrightText: 2026 H2Lab Development Team
# SPDX-License-Identifier: Apache-2.0

# Build: docker build -t pygdbknowledge .
# Build without the semantic stack (lexical search only): --build-arg EXTRAS=pdf
ARG PYTHON_VERSION=3.13

FROM python:${PYTHON_VERSION}-slim AS builder
ARG EXTRAS=pdf,vector
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_CACHE_DIR=1
RUN python -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH
WORKDIR /build
COPY pyproject.toml README.md ./
COPY LICENSES ./LICENSES
COPY src ./src
# setuptools_scm cannot read git metadata in the image: the version is set explicitly.
ARG VERSION=0.0.0
ENV SETUPTOOLS_SCM_PRETEND_VERSION=${VERSION}
# CPU-only torch keeps the image small; the service never needs a GPU.
RUN pip install --extra-index-url https://download.pytorch.org/whl/cpu ".[${EXTRAS}]"

FROM python:${PYTHON_VERSION}-slim
RUN useradd --system --uid 10001 --no-create-home --shell /usr/sbin/nologin knowledge \
    && mkdir -p /var/lib/pygdbknowledge /models \
    && chown knowledge /var/lib/pygdbknowledge
COPY --from=builder /opt/venv /opt/venv
COPY --chown=10001:0 --chmod=0400 config/settings.yaml /etc/pygdbknowledge/settings.yaml
ENV PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/models \
    HF_HUB_OFFLINE=1 \
    PYGDBKNOWLEDGE_CONFIG=/etc/pygdbknowledge/settings.yaml \
    PYGDBKNOWLEDGE__SERVER__HOST=0.0.0.0
USER knowledge
VOLUME ["/var/lib/pygdbknowledge"]
EXPOSE 8081
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD ["python", "-c", "import urllib.request as u; u.urlopen('http://127.0.0.1:8080/api/v1/health', timeout=3)"]
ENTRYPOINT ["pyGdbKnowledge"]
CMD ["serve"]
