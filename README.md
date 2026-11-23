# payo-bench
Payo Benchmark: Gemma vs. SEA-LION on Filipino emotion recognition and empathic advice generation, validated by Filipino readers.

**What to do next:** follow [`RUNBOOK.md`](RUNBOOK.md). Every remaining step is either filling in a sheet or running one command.

## Repository layout

The repo reads top to bottom in pipeline order. Each medallion layer sits inside the stage that produces it.

```
00-data/bronze/            Raw r/adviceph dump (immutable)
01-preprocessing/silver/   Template posts paired with their top-voted reply
02-llm/gold/               Frozen benchmark artifacts + Gemma / SEA-LION outputs
02-llm/work/               Sheets for annotators and raters (gitignored; contain post text)
03-evaluation/             Workflow commands, statistics, notebooks
```

| Layer  | Location                   | Rule                                        |
| ------ | -------------------------- | ------------------------------------------- |
| Bronze | `00-data/bronze/`          | Never edited                                |
| Silver | `01-preprocessing/silver/` | Regenerable from bronze by code             |
| Gold   | `02-llm/gold/`             | Versioned and frozen; what the paper reports on |

Data files are not in git. Each layer has a committed `MANIFEST.json` (sha256 + row counts) for checking your copy. See [`00-data/README.md`](00-data/README.md) for the full data contract.
