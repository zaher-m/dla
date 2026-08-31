# Contributing

The most useful contribution is **another model**, especially one trained on a distribution nothing
here has seen. See [`docs/adding-a-model.md`](docs/adding-a-model.md) — three files, no image
rebuild.

## Setting up

```bash
make build
make setup SETUP_PROFILE=fast     # a small install is enough to develop against
make up
make test                          # three pages, five models, end to end
```

Work inside the container (`make shell`). Nothing needs to be installed on the host.

## The rules that are not style preferences

**Never fabricate a result.** A system that cannot run is reported as `env_missing`, `crashed`,
`blocked` or `attempted_invalid`, with a diagnosis. It is never scored, never estimated, and never
silently dropped from a comparison. If you cannot make something work, a documented failure is a
genuine contribution.

**Record every deviation from upstream.** If a model needed a patched kernel, an unpinned dependency
or a different inference path than its own README describes, say so in the setup script and in the
adapter's `deviations` field. Those notes are the difference between a port that can be maintained
and a pile of pins nobody dares touch.

**Import `core.adapter_base` before constructing a model.** It applies the runtime guards. On some
hardware a wrong answer is not an error, and skipping this has already produced numbers that looked
plausible and were not — see the note in `harness/adapters/kraken_blla.py`.

**Emit coordinates in 300 dpi page pixels.** If the model resizes internally, scale back. A scale
error looks exactly like a bad model.

**Map class names from the dataset's own definitions, not from what the English word suggests.** The
study mapped three D4LA classes wrong that way: D4LA's `Footer` is documented as "the footnote of the
document", not a page footer. Mark every mapping `exact`, `approximate` or `ambiguous`.

**Get `repo` right in the registry.** Consensus is deduplicated by it — one repository, one vote.
Three configurations sharing a `repo` value vote once; splitting them lets one family out-vote the
field and inflate its own agreement score.

## Before opening a pull request

```bash
make list                 # your system appears, environment builds
make test                 # the pipeline still completes end to end
make verify               # the report renders, in both themes, at every width
make test-ui              # the UI drives an upload to a finished report
make analyse FILE=samples/2021-3_AR.pdf PROFILE=fast
```

`make verify` and `make test-ui` need the optional headless browser:
`make setup-env ENV=shot`.

Open the report and **look at the overlays** before reading any number. A model that scores well and
draws nonsense is a mapping or scaling bug, and the orbit view shows it in a second.

If you changed anything in `harness/report/`, rebuild both variants and check both render:

```bash
make report                                        # full
make analyse FILE=samples/BODAchievements_2012.pdf # viewer
```

## Style

Match what is there. Comments explain **why**, not what — a comment that restates the code is worse
than none, and a comment that records a decision or a trap is worth ten lines of code. Prefer a
paragraph at the top of a module explaining what it is for and what it deliberately does not do.

Keep configuration in `config/dla.yaml` with an environment override. No new hardcoded paths:
`harness/core/paths.py` is the only module that resolves one.

## Reporting a problem

Include the job id, the stage that failed, and the stage log — `data/jobs/<id>/logs/stage_<name>.log`
and, for a model failure, `data/jobs/<id>/logs/run_<system>.log`. `status.json` from the job
directory is usually enough on its own.
