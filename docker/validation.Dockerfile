# ---------------------------------------------------------------------------
# The validation package, alone.
#
# Not how the pipeline runs it.  A validation stage inside a job is a
# subprocess in the main container like every other stage, because that is what
# keeps `python -m validation.stage --workspace ...` runnable by hand exactly as
# the pipeline runs it, and a stage that has to cross a container boundary
# loses that.
#
# This image exists for two other reasons.
#
#   It is the deployable unit for anything built on top.  Deciding whether a
#   layout is safe to write to a store needs PyMuPDF, numpy and a YAML file --
#   no GPU, no model weights, no CUDA.  The main image is an NVIDIA vLLM base
#   measured in tens of gigabytes; this one is measured in hundreds of
#   megabytes and runs anywhere.
#
#   It is how the dependency boundary is enforced.  Only harness/validation and
#   config/checks.yaml are copied, so an `import core` anywhere in the package
#   fails the import check below and the build stops.  A boundary that is only
#   a convention is a boundary that erodes.
#
#   docker build -f docker/validation.Dockerfile -t dla-validation:1 .
#   docker run --rm -v "$PWD:/data" dla-validation:1 \
#       --pdf /data/report.pdf --page 4 --layout /data/regions.json
# ---------------------------------------------------------------------------
FROM python:3.12-slim

LABEL org.opencontainers.image.title="dla-validation" \
      org.opencontainers.image.description="Decide whether a page's layout is safe to write downstream: accept, escalate to a reviewer, or defer" \
      org.opencontainers.image.licenses="Apache-2.0"

ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app/harness

# numpy is pinned to match the harness environment (harness/setup/harness.sh):
# the checks compare medians and percentiles across the two, and a version skew
# that changes a tie-break changes a decision.
RUN pip install --no-cache-dir \
        "pymupdf" "numpy<2.3" "pyyaml" "pillow" \
 && rm -rf /root/.cache

WORKDIR /app

# The layout mirrors the repository, because `load_thresholds` and
# `load_policy` resolve config/checks.yaml relative to the package. Same code
# path in the image and in the tree, so a threshold cannot silently differ.
COPY harness/validation /app/harness/validation
COPY config/checks.yaml /app/config/checks.yaml
# The learned table score. Without it C6-02 silently falls back to the
# structural test alone, which is the behaviour the model was fitted to replace.
COPY config/table_model.json /app/config/table_model.json
# The learned figure-vs-content score.  Without it every shaded table's text
# leaves the body and the coverage family stops seeing 8% of the corpus, with
# no error -- `selftest` asserts both models load for exactly that reason.
COPY config/graphic_model.json /app/config/graphic_model.json

# The boundary check.  `core` is not in this image; if the package reaches for
# it, this line fails and the build does too.
RUN python - <<'PY'
import importlib, pkgutil, sys
sys.path.insert(0, "/app/harness")
import validation
bad = []
for m in pkgutil.iter_modules(validation.__path__):
    try:
        importlib.import_module("validation." + m.name)
    except Exception as e:
        bad.append(f"{m.name}: {type(e).__name__}: {e}")
if bad:
    sys.exit("validation does not stand alone:\n  " + "\n  ".join(bad))
from validation import selftest
import tempfile
with tempfile.TemporaryDirectory() as t:
    n, _ = selftest.run(t)
print(f"selftest: {len(n)} cases")
print(f"validation standalone: {len(list(pkgutil.iter_modules(validation.__path__)))} modules")
PY

ENTRYPOINT ["python", "-m", "validation.api"]
CMD ["--help"]
