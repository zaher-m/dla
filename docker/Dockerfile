# ---------------------------------------------------------------------------
# DLA Benchmark Harness — dedicated, extensible container
#
# Base: an NVIDIA CUDA runtime image
#   chosen because it already ships an aarch64 CUDA build of PyTorch 2.11 +
#   torchvision.  Public
#   PyPI/pytorch.org wheels for this arch+CUDA combination are large and slow
#   to fetch, so inheriting them costs zero download.
#
# Design: the image provides only the *shared substrate* (system libraries,
# fonts, Java, uv, the harness).  Each benchmarked system gets its own uv
# virtualenv created by harness/setup/<system>.sh into /work/benchmark/envs,
# which is a bind-mounted host directory.  Adding a new repo/tool later means
# dropping in one setup script + one adapter — no image rebuild required.
# ---------------------------------------------------------------------------
ARG BASE_IMAGE=nvcr.io/nvidia/vllm:26.02-py3
FROM ${BASE_IMAGE}

LABEL org.opencontainers.image.title="dla-suite" \
      org.opencontainers.image.description="Document layout analysis: run every layout model over a PDF and compare them side by side" \
      org.opencontainers.image.source="https://github.com/your-org/dla-suite" \
      org.opencontainers.image.licenses="Apache-2.0"

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# --- system substrate ------------------------------------------------------
# poppler/mupdf/ghostscript : PDF rasterisation used by nearly every adapter
# libatk/libnss/libgbm/...    : headless Chromium, for the report smoke tests
#                               (harness/setup/shot.sh). Without these the
#                               browser downloads fine and then fails to start
#                               with a missing .so, which is a confusing way to
#                               discover a packaging gap.
# libgl/libglib/libjpeg/libpng : OpenCV + torchvision image codecs
# fonts-*                   : Arabic-capable rendering for visualisations
# openjdk + maven           : LAREX (Java/Spring) build & run
# build-essential/cmake/ninja : source builds (detectron2, C extensions)
RUN apt-get update && apt-get install -y --no-install-recommends \
        poppler-utils ghostscript mupdf-tools qpdf \
        libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
        libjpeg-turbo8 libpng16-16 libtiff6 libopenjp2-7 \
        libmagic1 libcairo2 libpango-1.0-0 libgdk-pixbuf-2.0-0 \
        fonts-dejavu-core fonts-dejavu-extra \
        fonts-noto-core fonts-noto-extra fonts-kacst fonts-hosny-amiri \
        openjdk-17-jdk-headless maven \
        build-essential cmake ninja-build pkg-config \
        git git-lfs curl wget unzip jq ca-certificates procps gosu \
        libatk1.0-0t64 libatk-bridge2.0-0t64 libatspi2.0-0t64 libcups2t64 \
        libxcomposite1 libxdamage1 libxfixes3 libxrandr2 libxkbcommon0 \
        libasound2t64 libnss3 libnspr4 libdrm2 libgbm1 \
    && rm -rf /var/lib/apt/lists/* \
    && fc-cache -f

# --- python tooling --------------------------------------------------------
# uv creates the per-system virtualenvs. The harness's own dependencies are
# installed system-wide as well, so a freshly cloned repository serves its UI
# and can render a report *before* any model environment exists — the interface
# then tells you what is missing instead of failing to start.
RUN python -m pip install --no-cache-dir "uv>=0.5" && uv --version
RUN python -m pip install --no-cache-dir \
        "pymupdf" "pillow" "pyyaml" "shapely" "opencv-python-headless" \
        "fastapi" "uvicorn[standard]" "python-multipart" \
    && python -c "import fitz, fastapi, uvicorn, cv2, shapely; print('harness deps ok')"

# --- runtime layout --------------------------------------------------------
# Every path below points into the bind-mounted working tree, so caches,
# virtualenvs and weights survive an image rebuild and can be inspected from
# the host. Nothing large is ever baked into the image.
WORKDIR /work
ENV DLA_ROOT=/work \
    HF_HOME=/hf-cache \
    HUGGINGFACE_HUB_CACHE=/hf-cache/hub \
    TRANSFORMERS_CACHE=/hf-cache/hub \
    TORCH_HOME=/work/assets/cache/torch \
    MODELSCOPE_CACHE=/work/assets/cache/modelscope \
    UV_CACHE_DIR=/work/assets/cache/uv \
    UV_LINK_MODE=copy \
    PYTHONPATH=/work/harness \
    MPLCONFIGDIR=/work/assets/cache/mpl \
    OMP_NUM_THREADS=8

EXPOSE 8080

COPY entrypoint.sh /usr/local/bin/dla-entrypoint
RUN chmod +x /usr/local/bin/dla-entrypoint

ENTRYPOINT ["/usr/local/bin/dla-entrypoint"]
CMD ["serve"]
