# samples

Put PDFs here to use them as a corpus:

```bash
make analyse FILE=samples/your-document.pdf
make corpus  DIR=samples WORKSPACE=benchmark
make test                                     # uses the first PDF in this directory
```

Nothing in this directory is tracked. `paths.corpus` in `config/dla.yaml` points here, and
`DLA_CORPUS` overrides it, so a corpus can live anywhere.
