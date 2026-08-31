# The container

One image, one service. The image is a **substrate**: system libraries, fonts, a CUDA build of
PyTorch, and `uv`. It contains no model, no model dependency and no project data. Everything else is
bind-mounted from the working tree at run time.

That split is the reason a dozen mutually incompatible repositories can live in one project, and the
reason adding a model never rebuilds the image.

```bash
make build      # docker compose --project-directory . -f docker/compose.yaml build
make up         # start the service
make shell      # a shell inside it
make down
```

## What is in the image

| | |
|---|---|
| Base | an NVIDIA CUDA runtime image that already ships PyTorch and torchvision. Override with `DLA_BASE_IMAGE`. |
| PDF | poppler-utils, ghostscript, mupdf-tools, qpdf |
| Imaging | libgl, libglib, libjpeg-turbo, libpng, libtiff, libopenjp2 |
| Fonts | DejaVu, Noto (core + extra), **KACST and Amiri** so Arabic renders correctly in visualisations |
| Browser libs | libatk, libnss, libgbm and friends — only so headless Chromium can start for the report smoke tests |
| Java | OpenJDK 17 + Maven, for JVM-based tools |
| Build | build-essential, cmake, ninja, pkg-config — detectron2, mmcv and CUDA kernels compile here |
| Python | `uv` only |

The base image is chosen for one reason: it already ships an aarch64 CUDA build of PyTorch for the
target hardware. Fetching that from PyPI takes tens of minutes and several gigabytes, and getting it
wrong costs GPU support silently. On x86 you will want a different base — set `DLA_BASE_IMAGE` and
rebuild; nothing else in the image is architecture-specific.

## What is deliberately *not* in the image

Every benchmarked system's dependencies. Docling wants current `transformers`, PDF-Extract-Kit's
vendored LayoutLMv3 wants an old one, detectron2 pins `iopath<0.1.10`, RoDLA needs mmcv 1.x and M2Doc
needs mmcv 2.x. There is no resolution of that into one environment that does not silently break
something.

So each system gets its own virtualenv under `assets/envs/`, created by `harness/setup/<env>.sh`
into the bind-mounted working tree. They survive image rebuilds, are reproducible from the scripts,
and are inspectable from the host.

## Mounts

| Host | Container | Contents |
|---|---|---|
| `./` | `/work` | the whole project: code, config, `assets/`, `data/`, `benchmark/` |
| `${DLA_HF_CACHE}` | `/hf-cache` | Hugging Face cache. Point it at a shared directory if you have one — several systems pull the same backbones. |

Paths in `compose.yaml` are relative to the **project directory**, which the Makefile sets to the
repository root with `--project-directory .`. That is also what makes a root `.env` apply.

## File ownership

`entrypoint.sh` creates a user matching `HOST_UID`/`HOST_GID` and drops to it, so nothing the
container writes into the bind mount ends up owned by root on the host. The Makefile passes both.

## Commands

The entrypoint dispatches on its first argument:

| Command | Effect |
|---|---|
| `serve` (default) | Start the web application. If the harness environment is not built yet it says so and idles, so you can exec in and build it. |
| `setup [args]` | `scripts/bootstrap.sh` — build environments and fetch weights. |
| anything else | Run it verbatim as the host user. |

## Running a command inside a running container

```bash
./docker/dx  'python -m core.runner --list'
./docker/dxbg /work/benchmark/logs/run.log 'python -m core.runner --profile full'
```

`dx` runs in the foreground; `dxbg` detaches and writes to a log, appending a final `EXIT=<code>`
line so a poller can tell "still running" from "finished and failed". Both honour `DLA_CONTAINER`.

## Without a GPU

Delete the `deploy.resources.reservations.devices` block from `compose.yaml` and use the `cpu`
profile:

```bash
make up
make analyse FILE=doc.pdf PROFILE=cpu
```

Seven systems run on CPU end to end. Their timings are CPU timings and the report labels them as
such — they are not comparable head-to-head with GPU numbers.
