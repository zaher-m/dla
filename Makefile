# DLA Suite — everything you need in one place.
#
#   make build     build the container image
#   make setup     build the harness + model environments and fetch weights
#   make up        start the web UI on http://localhost:8080
#   make logs      follow it
#   make shell     a shell inside the container
#   make analyse FILE=report.pdf     run one PDF from the command line
#   make corpus DIR=pdfs/            run a whole directory
#
# Everything runs inside the container; nothing is installed on the host.

SHELL := /bin/bash

# Local overrides live in .env (gitignored; see .env.example). Loading it here
# as well as letting compose interpolate it keeps `make status` and the compose
# port mapping in agreement.
-include .env
export

# --project-directory . makes the repository root the compose project directory,
# so the root .env is used for interpolation — compose would otherwise look for
# one beside the compose file.
COMPOSE        := docker compose --project-directory . -f docker/compose.yaml
CONTAINER      ?= $(or $(DLA_CONTAINER),dla)
PORT           ?= $(or $(DLA_SERVER_PORT),8080)
PROFILE        ?= $(or $(DLA_RUN_PROFILE),balanced)
SETUP_PROFILE  ?= $(PROFILE)
FILE           ?=
STAGE          ?=
VALIDATION_IMAGE ?= $(or $(DLA_VALIDATION_IMAGE),dla-validation:1)

export HOST_UID := $(shell id -u)
export HOST_GID := $(shell id -g)
export DLA_CONTAINER := $(CONTAINER)
export DLA_SERVER_PORT := $(PORT)

# Run a command in the container, as the invoking user.
EXEC = $(COMPOSE) exec -u $(HOST_UID):$(HOST_GID) -e PYTHONPATH=/work/harness -T $(CONTAINER)
PY   = /work/assets/envs/harness/bin/python

.DEFAULT_GOAL := help
.PHONY: help build setup setup-env up down restart logs shell status list \
        analyse corpus stage report clean-jobs clean-workspace doctor test \
        verify test-ui validate validate-selftest validation-image validate-page

help:  ## show this help
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
	 | awk 'BEGIN{FS=":.*?## "};{printf "  \033[1m%-16s\033[0m %s\n",$$1,$$2}'

build:  ## build the container image
	$(COMPOSE) build

up:  ## start the web UI (http://localhost:$(PORT))
	$(COMPOSE) up -d
	@echo "DLA Suite → http://localhost:$(PORT)"

down:  ## stop and remove the container
	$(COMPOSE) down

restart: down up  ## restart the service

logs:  ## follow the service log
	$(COMPOSE) logs -f --tail=100

shell:  ## interactive shell inside the container
	$(COMPOSE) exec -u $(HOST_UID):$(HOST_GID) $(CONTAINER) bash

status:  ## service + queue health
	@curl -fsS http://localhost:$(PORT)/api/health | $(or $(shell command -v jq),cat)

setup:  ## build environments and fetch weights for SETUP_PROFILE (default: balanced)
	$(COMPOSE) up -d
	$(EXEC) bash scripts/bootstrap.sh --profile $(SETUP_PROFILE)

setup-env:  ## build one environment: make setup-env ENV=docling
	@test -n "$(ENV)" || { echo "usage: make setup-env ENV=<name>"; exit 2; }
	$(EXEC) bash harness/setup/$(ENV).sh

list:  ## registry: every system, and whether its environment is built
	$(EXEC) $(PY) -m core.runner --list --profile $(PROFILE)

analyse:  ## analyse one PDF: make analyse FILE=path/to/report.pdf
	@test -n "$(FILE)" || { echo "usage: make analyse FILE=<pdf>"; exit 2; }
	$(EXEC) $(PY) -m core.pipeline --input "$(FILE)" --profile $(PROFILE)

corpus:  ## analyse a directory of PDFs into one workspace
	@test -n "$(DIR)" || { echo "usage: make corpus DIR=<dir> [WORKSPACE=<dir>]"; exit 2; }
	$(EXEC) $(PY) -m core.pipeline --corpus "$(DIR)" \
	  --workspace "$(or $(WORKSPACE),benchmark)" --profile $(PROFILE) --resume

stage:  ## re-run one pipeline stage: make stage STAGE=metrics
	@test -n "$(STAGE)" || { echo "usage: make stage STAGE=<metrics|consensus|package|report|...>"; exit 2; }
	$(EXEC) $(PY) -m core.$(STAGE)

report:  ## rebuild the HTML report from existing results in WORKSPACE
	$(EXEC) $(PY) -m core.package_report --workspace "$(or $(WORKSPACE),benchmark)"
	$(EXEC) $(PY) -m core.build_report --workspace "$(or $(WORKSPACE),benchmark)"

doctor:  ## check paths, GPU, and which environments are present
	$(EXEC) $(PY) -m core.paths
	@$(EXEC) $(PY) -c "import torch;print('cuda:',torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')" || true

test:  ## end-to-end smoke test: the first PDF in samples/, three pages, fast profile
	@test -n "$(firstword $(wildcard samples/*.pdf))" \
	  || { echo "put a PDF in samples/ first (see samples/README.md)"; exit 2; }
	$(EXEC) env DLA_SELECTION_MAX_PAGES=3 $(PY) -m core.pipeline \
	  --input "$(firstword $(wildcard samples/*.pdf))" --profile fast

validate:  ## decide every page of a workspace: make validate WORKSPACE=benchmark
	$(EXEC) $(PY) -m validation.stage \
	  --workspace $(or $(WORKSPACE),benchmark) --corpus $(or $(CORPUS),data/corpus_flat)

validate-selftest:  ## run the validation self-test: pages built in memory, no corpus
	$(EXEC) $(PY) -m validation.selftest

# The validation package builds as its own image: 300 MB against the suite's
# 16 GB, no GPU, no weights. It is what other software runs, and its build
# fails if the package has grown a dependency on the harness.
validation-image:  ## build the standalone validation image
	docker build -f docker/validation.Dockerfile -t $(VALIDATION_IMAGE) .

validate-page:  ## decide one page: make validate-page PDF=f.pdf PAGE=4 LAYOUT=regions.json
	@test -n "$(PDF)" -a -n "$(PAGE)" -a -n "$(LAYOUT)" \
	  || { echo "usage: make validate-page PDF=f.pdf PAGE=4 LAYOUT=regions.json"; exit 2; }
	docker run --rm -v "$(CURDIR):/w:ro" $(VALIDATION_IMAGE) \
	  --pdf /w/$(PDF) --page $(PAGE) --layout /w/$(LAYOUT)

verify:  ## render a built report in a headless browser and assert it works
	$(EXEC) /work/assets/envs/shot/bin/python scripts/dev/verify_report.py \
	  "$(or $(REPORT),benchmark/reports/index.html)"

test-ui:  ## drive the running UI end to end: upload, wait, open the report
	$(EXEC) /work/assets/envs/shot/bin/python scripts/dev/ui_smoke.py \
	  --url http://127.0.0.1:$(PORT) --profile fast

clean-jobs:  ## delete every finished analysis under data/jobs
	@read -p "Delete all job workspaces under data/jobs? [y/N] " a; [ "$$a" = y ] || exit 1; \
	 rm -rf data/jobs/* && echo "removed"

clean-workspace:  ## delete a workspace's computed outputs (keeps weights and envs)
	@w="$(or $(WORKSPACE),benchmark)"; read -p "Delete $$w/{raw,normalized}_outputs, metrics, working? [y/N] " a; \
	 [ "$$a" = y ] || exit 1; \
	 rm -rf "$$w"/raw_outputs/* "$$w"/normalized_outputs/* "$$w"/metrics/* "$$w"/working/* && echo "removed"
